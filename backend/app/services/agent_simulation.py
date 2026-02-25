"""High-level simulation service.

Loads configs from DB, creates engine components, runs simulations,
and stores results. This is the bridge between the API and the engine.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.engine.adversarial import AdversarialStrategy, NoOpAdversarial
from app.engine.environment import SimulationEnvironment
from app.engine.llm_client import LLMClient
from app.engine.persona import AgentPersona, UserPersona
from app.engine.scenario_runner import ScenarioRunner
from app.engine.tool_simulator import ToolSimulator
from app.engine.types import ConversationResult
from app.engine.user_simulator import UserSimulator
from app.models.agent_config import AgentConfig
from app.models.conversation import Conversation
from app.models.eval_run import EvalRun
from app.models.scenario import Scenario

logger = structlog.get_logger()


class AgentSimulationService:
    """Orchestrates full evaluation run: load config → run N conversations → store results."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.llm_client = LLMClient()

    async def run_eval(self, eval_run_id: str) -> None:
        """Execute all conversations for an eval run."""
        # Load eval run
        result = await self.db.execute(select(EvalRun).where(EvalRun.id == eval_run_id))
        eval_run = result.scalar_one_or_none()
        if not eval_run:
            raise ValueError(f"Eval run {eval_run_id} not found")

        # Update status
        eval_run.status = "running_simulation"
        eval_run.started_at = datetime.now(timezone.utc)
        await self.db.flush()

        # Load agent config
        result = await self.db.execute(
            select(AgentConfig).where(AgentConfig.id == eval_run.agent_config_id)
        )
        agent_config = result.scalar_one()

        # Load scenario
        result = await self.db.execute(
            select(Scenario).where(Scenario.id == eval_run.scenario_id)
        )
        scenario = result.scalar_one()

        # Build personas
        agent_persona = AgentPersona.from_db(agent_config)
        user_persona = UserPersona.from_dict(
            scenario.user_persona,
            model=agent_config.model,  # Use same-tier model for user sim
        )

        # Build environment from scenario constraints
        environment = SimulationEnvironment.from_dict(scenario.constraints)

        # Extract initial message from turns template
        initial_message = ""
        turns_template = scenario.turns_template
        if turns_template and isinstance(turns_template, list) and len(turns_template) > 0:
            first = turns_template[0]
            if isinstance(first, dict):
                initial_message = first.get("content", first.get("content_template", ""))

        logger.info(
            "simulation_starting",
            eval_run_id=eval_run_id,
            agent=agent_persona.name,
            num_conversations=eval_run.num_conversations,
        )

        try:
            for seq_num in range(eval_run.num_conversations):
                await self._run_single_conversation(
                    eval_run=eval_run,
                    agent_persona=agent_persona,
                    user_persona=user_persona,
                    environment=environment,
                    initial_message=initial_message,
                    sequence_num=seq_num,
                )

            eval_run.status = "running_evaluation"
            eval_run.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            eval_run.status = "failed"
            eval_run.error_message = str(e)
            eval_run.completed_at = datetime.now(timezone.utc)
            logger.error("simulation_failed", eval_run_id=eval_run_id, error=str(e))

        await self.db.flush()

    async def _run_single_conversation(
        self,
        eval_run: EvalRun,
        agent_persona: AgentPersona,
        user_persona: UserPersona,
        environment: SimulationEnvironment,
        initial_message: str,
        sequence_num: int,
    ) -> Conversation:
        """Run a single multi-turn conversation and store it."""
        # Create conversation record
        conv = Conversation(
            eval_run_id=eval_run.id,
            sequence_num=sequence_num,
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(conv)
        await self.db.flush()

        # Build components
        user_sim = UserSimulator(
            llm_client=self.llm_client,
            persona=user_persona,
            initial_message=initial_message,
        )
        tool_sim = ToolSimulator(environment=environment)

        adversarial: AdversarialStrategy | NoOpAdversarial
        if environment.adversarial_turns:
            adversarial = AdversarialStrategy(environment)
        else:
            adversarial = NoOpAdversarial()

        runner = ScenarioRunner(
            llm_client=self.llm_client,
            agent_persona=agent_persona,
            user_simulator=user_sim,
            tool_simulator=tool_sim,
            environment=environment,
            adversarial=adversarial,
        )

        # Run the conversation
        conv_result: ConversationResult = await runner.run()

        # Store results
        conv.turns = [asdict(t) for t in conv_result.turns]
        conv.turn_count = conv_result.turn_count
        conv.total_tokens = conv_result.total_tokens
        conv.total_input_tokens = conv_result.total_input_tokens
        conv.total_output_tokens = conv_result.total_output_tokens
        conv.total_latency_ms = conv_result.total_latency_ms
        conv.status = "completed" if conv_result.status != "failed" else "failed"
        conv.error_message = conv_result.error_message
        conv.completed_at = datetime.now(timezone.utc)
        conv.metadata_ = {"simulation_status": conv_result.status}

        await self.db.flush()

        logger.info(
            "conversation_completed",
            conversation_id=conv.id,
            sequence_num=sequence_num,
            turns=conv_result.turn_count,
            status=conv_result.status,
        )

        # Emit Kafka event (best-effort — failure must not break simulation)
        try:
            from app.pipeline.events import ConversationCompletedEvent
            from app.pipeline.producer import KafkaProducer
            from app.pipeline.topics import CONVERSATION_COMPLETED

            event = ConversationCompletedEvent(
                eval_run_id=eval_run.id,
                conversation_id=conv.id,
                turn_count=conv_result.turn_count,
                total_tokens=conv_result.total_tokens,
                total_latency_ms=conv_result.total_latency_ms,
                status=conv.status,
            )
            producer = KafkaProducer()
            producer.produce(CONVERSATION_COMPLETED, event.to_envelope(), key=conv.id)
        except Exception as kafka_err:
            logger.warning("kafka_event_failed", error=str(kafka_err))

        return conv
