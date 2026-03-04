"""Evaluation orchestrator — runs all evaluators and stores results.

Mirrors the AgentSimulationService pattern: load from DB → run evaluators → store results.
"""

from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.engine.llm_client import LLMClient
from app.evaluation.automated_metrics import AutomatedMetricsCalculator
from app.evaluation.model_judge import ModelJudgeEvaluator
from app.evaluation.rubric_grader import RubricGraderEvaluator
from app.evaluation.types import DEFAULT_DIMENSIONS, EvaluationResult, MetricValue, RubricDimension
from app.models.conversation import Conversation
from app.models.evaluation import Evaluation
from app.models.metric import Metric
from app.models.rubric import Rubric

logger = structlog.get_logger()


class EvaluationService:
    """Orchestrates evaluation of a single conversation."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.llm_client = LLMClient()

    async def evaluate_conversation(
        self,
        conversation_id: str,
        rubric_id: str | None = None,
    ) -> list[Evaluation]:
        """Run all evaluators on a conversation and store results.

        1. Load conversation from DB
        2. Load rubric dimensions (or use defaults)
        3. Run ModelJudgeEvaluator → store Evaluation
        4. Run RubricGraderEvaluator → store Evaluation
        5. Run AutomatedMetricsCalculator → store Metrics
        6. Return list of Evaluation records
        """
        # Load conversation
        result = await self.db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Load rubric dimensions
        dimensions = await self._load_dimensions(rubric_id)

        turns: list[dict[str, Any]] = conversation.turns or []
        evaluations: list[Evaluation] = []

        # 1. Model Judge
        try:
            judge = ModelJudgeEvaluator(llm_client=self.llm_client)
            judge_result = await judge.evaluate(turns, dimensions)
            eval_record = await self._store_evaluation(
                conversation_id=conversation_id,
                rubric_id=rubric_id,
                result=judge_result,
            )
            evaluations.append(eval_record)
            logger.info(
                "model_judge_completed",
                conversation_id=conversation_id,
                overall_score=judge_result.overall_score,
            )
        except Exception as e:
            logger.error(
                "model_judge_failed",
                conversation_id=conversation_id,
                error=str(e),
            )

        # 2. Rubric Grader
        try:
            grader = RubricGraderEvaluator()
            grader_result = await grader.evaluate(turns, dimensions)
            eval_record = await self._store_evaluation(
                conversation_id=conversation_id,
                rubric_id=rubric_id,
                result=grader_result,
            )
            evaluations.append(eval_record)
            logger.info(
                "rubric_grader_completed",
                conversation_id=conversation_id,
                overall_score=grader_result.overall_score,
            )
        except Exception as e:
            logger.error(
                "rubric_grader_failed",
                conversation_id=conversation_id,
                error=str(e),
            )

        # 3. Automated Metrics
        try:
            calculator = AutomatedMetricsCalculator()
            metric_values = calculator.compute_all(
                turns=turns,
                turn_count=conversation.turn_count,
                total_tokens=conversation.total_tokens,
                total_input_tokens=conversation.total_input_tokens,
                total_output_tokens=conversation.total_output_tokens,
                total_latency_ms=conversation.total_latency_ms,
                status=conversation.status,
            )
            await self._store_metrics(conversation_id, metric_values)
            logger.info(
                "automated_metrics_completed",
                conversation_id=conversation_id,
                metric_count=len(metric_values),
            )
        except Exception as e:
            logger.error(
                "automated_metrics_failed",
                conversation_id=conversation_id,
                error=str(e),
            )

        await self.db.flush()

        # Emit Kafka events for completed evaluations (best-effort)
        for eval_record in evaluations:
            try:
                from app.pipeline.events import EvaluationScoreCompletedEvent
                from app.pipeline.producer import KafkaProducer
                from app.pipeline.topics import EVALUATION_SCORE_COMPLETED

                event = EvaluationScoreCompletedEvent(
                    eval_run_id=conversation.eval_run_id,
                    conversation_id=conversation_id,
                    evaluation_id=eval_record.id,
                    evaluator_type=eval_record.evaluator_type,
                    overall_score=eval_record.overall_score or 0.0,
                    dimension_scores=eval_record.scores,
                )
                producer = KafkaProducer()
                producer.produce(
                    EVALUATION_SCORE_COMPLETED,
                    event.to_envelope(),
                    key=conversation_id,
                )
            except Exception as kafka_err:
                logger.warning("kafka_eval_event_failed", error=str(kafka_err))

        return evaluations

    async def _load_dimensions(
        self, rubric_id: str | None,
    ) -> list[RubricDimension]:
        """Load rubric dimensions from DB or use defaults."""
        if not rubric_id:
            return DEFAULT_DIMENSIONS

        result = await self.db.execute(
            select(Rubric).where(Rubric.id == rubric_id)
        )
        rubric = result.scalar_one_or_none()
        if not rubric or not rubric.dimensions:
            return DEFAULT_DIMENSIONS

        return [
            RubricDimension(
                name=d["name"],
                description=d.get("description", ""),
                weight=d.get("weight", 1.0),
                criteria=d.get("criteria", []),
            )
            for d in rubric.dimensions
        ]

    async def _store_evaluation(
        self,
        conversation_id: str,
        rubric_id: str | None,
        result: EvaluationResult,
    ) -> Evaluation:
        """Persist an evaluation result to the database."""
        eval_record = Evaluation(
            conversation_id=conversation_id,
            evaluator_type=result.evaluator_type,
            rubric_id=rubric_id,
            scores=result.scores,
            overall_score=result.overall_score,
            reasoning=result.reasoning,
            per_turn_scores=result.per_turn_scores,
            metadata_=result.metadata,
        )
        self.db.add(eval_record)
        await self.db.flush()
        return eval_record

    async def _store_metrics(
        self,
        conversation_id: str,
        metric_values: list[MetricValue],
    ) -> None:
        """Persist computed metrics to the database."""
        for mv in metric_values:
            metric = Metric(
                conversation_id=conversation_id,
                metric_name=mv.name,
                value=mv.value,
                unit=mv.unit,
                metadata_=mv.metadata,
            )
            self.db.add(metric)
        await self.db.flush()
