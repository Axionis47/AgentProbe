"""Consumes EvaluationScoreCompleted events and triggers metric aggregation."""

from __future__ import annotations

import asyncio

import structlog
from sqlalchemy import func, select

from app.db.session import async_session_factory
from app.evaluation.aggregation import aggregate_metric_values
from app.models.conversation import Conversation
from app.models.evaluation import Evaluation
from app.models.metric import Metric
from app.pipeline.consumers.base import BaseConsumer
from app.pipeline.events import EventEnvelope, MetricsAggregatedEvent
from app.pipeline.producer import KafkaProducer
from app.pipeline.topics import EVALUATION_SCORE_COMPLETED, METRICS_AGGREGATED

logger = structlog.get_logger()


class EvaluationCompletedConsumer(BaseConsumer):
    """Checks if all conversations in a run are evaluated, then aggregates metrics."""

    def __init__(self) -> None:
        super().__init__(topic=EVALUATION_SCORE_COMPLETED)

    def handle_event(self, envelope: EventEnvelope) -> None:
        """Check completion and aggregate metrics if all conversations are evaluated."""
        payload = envelope.payload
        eval_run_id = payload.get("eval_run_id")

        if not eval_run_id:
            return

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

            # Count conversations that have at least one evaluation
            evaluated_count_result = await session.execute(
                select(func.count(func.distinct(Evaluation.conversation_id))).where(
                    Evaluation.conversation_id.in_(
                        select(Conversation.id).where(
                            Conversation.eval_run_id == eval_run_id,
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

            # All evaluated — aggregate metrics
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
