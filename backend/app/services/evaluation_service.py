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
from app.models.eval_run import EvalRun
from app.models.evaluation import Evaluation
from app.models.metric import Metric
from app.models.rubric import Rubric
from app.models.scenario import Scenario

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

        # 2.5 Reference Evaluator (only if scenario has expected_response fields)
        try:
            scenario = await self._load_scenario_for_conversation(conversation)
            if scenario and self._has_reference_answers(scenario):
                from app.evaluation.reference_evaluator import ReferenceEvaluator

                ref_evaluator = ReferenceEvaluator()
                enriched_turns = self._enrich_turns_with_references(turns, scenario)
                ref_result = await ref_evaluator.evaluate(enriched_turns, dimensions)
                eval_record = await self._store_evaluation(
                    conversation_id=conversation_id,
                    rubric_id=rubric_id,
                    result=ref_result,
                )
                evaluations.append(eval_record)
                logger.info(
                    "reference_evaluator_completed",
                    conversation_id=conversation_id,
                    overall_score=ref_result.overall_score,
                )
        except Exception as e:
            logger.error("reference_evaluator_failed", conversation_id=conversation_id, error=str(e))

        # 2.7 Trajectory Evaluator (only if scenario has expected_tool_sequence)
        try:
            if not scenario:
                scenario = await self._load_scenario_for_conversation(conversation)
            if scenario and self._has_expected_trajectory(scenario):
                from app.evaluation.trajectory_evaluator import TrajectoryEvaluator

                expected_seq = (scenario.constraints or {}).get("expected_tool_sequence", [])
                traj_evaluator = TrajectoryEvaluator(expected_tool_sequence=expected_seq)
                traj_result = await traj_evaluator.evaluate(turns, dimensions)
                eval_record = await self._store_evaluation(
                    conversation_id=conversation_id,
                    rubric_id=rubric_id,
                    result=traj_result,
                )
                evaluations.append(eval_record)
                logger.info(
                    "trajectory_evaluator_completed",
                    conversation_id=conversation_id,
                    overall_score=traj_result.overall_score,
                )
        except Exception as e:
            logger.error("trajectory_evaluator_failed", conversation_id=conversation_id, error=str(e))

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

    # ------------------------------------------------------------------
    # Scenario helpers for reference + trajectory evaluators
    # ------------------------------------------------------------------

    async def _load_scenario_for_conversation(
        self, conversation: Conversation,
    ) -> Scenario | None:
        """Load the scenario associated with a conversation's eval run."""
        result = await self.db.execute(
            select(EvalRun).where(EvalRun.id == conversation.eval_run_id)
        )
        eval_run = result.scalar_one_or_none()
        if not eval_run:
            return None

        result = await self.db.execute(
            select(Scenario).where(Scenario.id == eval_run.scenario_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _has_reference_answers(scenario: Scenario) -> bool:
        """Check if any turn in the scenario template has expected_response."""
        turns = scenario.turns_template or []
        return any(t.get("expected_response") for t in turns)

    @staticmethod
    def _has_expected_trajectory(scenario: Scenario) -> bool:
        """Check if the scenario constraints contain expected_tool_sequence."""
        constraints = scenario.constraints or {}
        seq = constraints.get("expected_tool_sequence", [])
        return isinstance(seq, list) and len(seq) > 0

    @staticmethod
    def _enrich_turns_with_references(
        actual_turns: list[dict[str, Any]],
        scenario: Scenario,
    ) -> list[dict[str, Any]]:
        """Copy expected_response from scenario template into actual turns."""
        template = scenario.turns_template or []
        enriched = []
        for i, turn in enumerate(actual_turns):
            enriched_turn = dict(turn)
            if i < len(template) and "expected_response" in template[i]:
                enriched_turn["expected_response"] = template[i]["expected_response"]
            enriched.append(enriched_turn)
        return enriched
