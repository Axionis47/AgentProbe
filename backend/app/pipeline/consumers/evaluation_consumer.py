"""Consumes EvaluationScoreCompleted events.

Two side effects per event:

1. Backfill the score onto the conversation's Chroma record so the
   similarity-search API can filter by quality (e.g. "find similar
   conversations that scored below 6 on model_judge").
2. If this was the last conversation to finish evaluating in the run,
   roll up metrics and mark the run completed.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog
from sqlalchemy import func, select

from app.db.chromadb_client import ChromaDBClient
from app.db.session import async_session_factory
from app.evaluation.aggregation import aggregate_metric_values
from app.models.conversation import Conversation
from app.models.eval_run import EvalRun
from app.models.evaluation import Evaluation
from app.models.metric import Metric
from app.pipeline.consumers.base import BaseConsumer
from app.pipeline.events import EventEnvelope, MetricsAggregatedEvent
from app.pipeline.producer import KafkaProducer
from app.pipeline.topics import EVALUATION_SCORE_COMPLETED, METRICS_AGGREGATED

logger = structlog.get_logger()


class EvaluationCompletedConsumer(BaseConsumer):
    """Backfills Chroma metadata then checks for run completion."""

    def __init__(self) -> None:
        super().__init__(topic=EVALUATION_SCORE_COMPLETED)

    def handle_event(self, envelope: EventEnvelope) -> None:
        payload = envelope.payload
        eval_run_id = payload.get("eval_run_id")

        if not eval_run_id:
            return

        # Best-effort metadata update on the Chroma record.
        try:
            _update_chroma_score(
                conversation_id=str(payload.get("conversation_id", "")),
                evaluator_type=str(payload.get("evaluator_type", "")),
                overall_score=payload.get("overall_score"),
            )
        except Exception as exc:  # noqa: BLE001 - similarity is a side feature
            logger.warning(
                "chroma_score_update_failed",
                conversation_id=payload.get("conversation_id"),
                error=str(exc),
            )

        asyncio.run(self._check_and_aggregate(str(eval_run_id)))

    async def _check_and_aggregate(self, eval_run_id: str) -> None:
        """Check if all conversations have evaluations; if so, aggregate metrics."""
        async with async_session_factory() as session:
            # Count completed conversations in this run
            conv_count_result = await session.execute(
                select(func.count(Conversation.id)).where(
                    Conversation.eval_run_id == eval_run_id,
                    Conversation.status == "completed",
                )
            )
            total_conversations = conv_count_result.scalar() or 0

            if total_conversations == 0:
                logger.debug("no_completed_conversations", eval_run_id=eval_run_id)
                return

            # Count conversations that have at least one evaluation
            evaluated_count_result = await session.execute(
                select(func.count(func.distinct(Evaluation.conversation_id))).where(
                    Evaluation.conversation_id.in_(
                        select(Conversation.id).where(
                            Conversation.eval_run_id == eval_run_id,
                            Conversation.status == "completed",
                        )
                    )
                )
            )
            evaluated_count = evaluated_count_result.scalar() or 0

            if evaluated_count < total_conversations:
                logger.debug(
                    "evaluation_incomplete",
                    eval_run_id=eval_run_id,
                    evaluated=evaluated_count,
                    total=total_conversations,
                )
                return

            # All evaluated — mark eval_run completed if not already done
            result = await session.execute(
                select(EvalRun).where(EvalRun.id == eval_run_id)
            )
            eval_run = result.scalar_one_or_none()
            if eval_run and eval_run.status == "running_evaluation":
                eval_run.status = "completed"
                eval_run.completed_at = datetime.now(timezone.utc)
                await session.commit()
                logger.info(
                    "eval_run_completed_by_consumer",
                    eval_run_id=eval_run_id,
                    evaluated=evaluated_count,
                    total=total_conversations,
                )

            # Aggregate metrics
            logger.info(
                "aggregating_metrics",
                eval_run_id=eval_run_id,
                conversation_count=total_conversations,
            )

            # Load all metrics for conversations in this run
            conv_ids_subquery = select(Conversation.id).where(
                Conversation.eval_run_id == eval_run_id,
            )
            metrics_result = await session.execute(
                select(Metric).where(
                    Metric.conversation_id.in_(conv_ids_subquery)
                )
            )
            metrics = metrics_result.scalars().all()

            # Group by metric_name
            metric_groups: dict[str, list[float]] = {}
            for m in metrics:
                metric_groups.setdefault(m.metric_name, []).append(m.value)

            # Aggregate and publish events
            try:
                producer = KafkaProducer()
                for name, values in metric_groups.items():
                    agg = aggregate_metric_values(name, values)
                    event = MetricsAggregatedEvent(
                        eval_run_id=eval_run_id,
                        metric_name=agg.metric_name,
                        mean=agg.mean,
                        median=agg.median,
                        std_dev=agg.std_dev,
                        min_val=agg.min_val,
                        max_val=agg.max_val,
                        sample_count=agg.sample_count,
                    )
                    producer.produce(METRICS_AGGREGATED, event.to_envelope(), key=eval_run_id)

                producer.flush(timeout=5.0)
            except Exception as e:
                logger.error("metrics_aggregation_publish_failed", error=str(e))


def _update_chroma_score(
    conversation_id: str,
    evaluator_type: str,
    overall_score: float | int | None,
) -> None:
    """Tag the conversation's Chroma record with this evaluator's score.

    Stores under metadata key ``score_<evaluator_type>`` so the API can filter
    by any evaluator independently. Multiple events per conversation just
    overwrite their own slot. Skips silently if any required field is missing.
    """
    if not conversation_id or not evaluator_type or overall_score is None:
        return

    collection = ChromaDBClient.get_conversations_collection()
    collection.update(
        ids=[conversation_id],
        metadatas=[{f"score_{evaluator_type}": float(overall_score)}],
    )
    logger.debug(
        "chroma_score_backfilled",
        conversation_id=conversation_id,
        evaluator=evaluator_type,
        score=overall_score,
    )
