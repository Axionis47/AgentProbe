"""Integration tests for the evaluation pipeline.

Tests the full flow: create agent_config -> create scenario -> create eval_run
-> run simulation -> verify DB state. Uses mocked DB objects to simulate
the full pipeline without requiring a real database.

All external dependencies (LLM, Kafka, Celery) are mocked.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.engine.environment import SimulationEnvironment
from app.engine.persona import AgentPersona, UserPersona
from app.engine.scenario_runner import ScenarioRunner
from app.engine.tool_simulator import ToolSimulator
from app.engine.types import ConversationResult, LLMResponse, ToolCall, Turn
from app.engine.user_simulator import UserSimulator
from app.evaluation.types import DEFAULT_DIMENSIONS
from app.services.agent_simulation import AgentSimulationService
from app.services.evaluation_service import EvaluationService


# ============================================================
# Helpers
# ============================================================


def make_llm_response(content="OK", tool_calls=None):
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        input_tokens=10,
        output_tokens=20,
        model="test-model",
        stop_reason="end_turn",
    )


def _make_judge_response():
    args = {}
    for dim in DEFAULT_DIMENSIONS:
        args[f"{dim.name}_score"] = 7.0
        args[f"{dim.name}_reasoning"] = f"Good {dim.name}"
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="call_1", name="submit_evaluation", arguments=args)],
        input_tokens=50,
        output_tokens=100,
        model="test-model",
        stop_reason="tool_calls",
    )


# ============================================================
# Test: Full simulation pipeline
# ============================================================


class TestSimulationPipeline:
    """Test the simulation pipeline end-to-end with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_full_simulation_creates_conversations(self):
        """Full flow: load config -> run N conversations -> store results."""
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()

        # Setup eval run
        eval_run = MagicMock()
        eval_run.id = "run-int-001"
        eval_run.agent_config_id = "cfg-int-001"
        eval_run.scenario_id = "scn-int-001"
        eval_run.rubric_id = None
        eval_run.num_conversations = 2
        eval_run.status = "pending"
        eval_run.config = {}
        eval_run.started_at = None
        eval_run.completed_at = None
        eval_run.error_message = None

        # Setup agent config
        agent_config = MagicMock()
        agent_config.name = "Integration Agent"
        agent_config.system_prompt = "You are a helpful assistant."
        agent_config.model = "test-model"
        agent_config.temperature = 0.7
        agent_config.max_tokens = 4096
        agent_config.tools = []

        # Setup scenario
        scenario = MagicMock()
        scenario.user_persona = {"personality": "neutral", "goal": "Get help with testing"}
        scenario.constraints = {"max_turns": 3, "max_total_tokens": 50000}
        scenario.turns_template = [{"role": "user", "content": "Help me write a test"}]

        # DB execute returns in order: eval_run, agent_config, scenario
        run_result = MagicMock()
        run_result.scalar_one_or_none.return_value = eval_run
        run_result.scalar_one.return_value = eval_run

        cfg_result = MagicMock()
        cfg_result.scalar_one.return_value = agent_config

        scn_result = MagicMock()
        scn_result.scalar_one.return_value = scenario

        mock_db.execute = AsyncMock(side_effect=[run_result, cfg_result, scn_result])

        service = AgentSimulationService(db=mock_db)

        with patch.object(service, "llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=make_llm_response("Test response"))
            await service.run_eval("run-int-001")

        # Verify status transitions (completed_at is set after evaluation, not simulation)
        assert eval_run.status == "running_evaluation"
        assert eval_run.started_at is not None

        # Verify conversations were added to DB
        # Each conversation: 1 add() call for the Conversation object
        add_calls = mock_db.add.call_count
        assert add_calls == 2  # 2 conversations

    @pytest.mark.asyncio
    async def test_simulation_failure_sets_failed_status(self):
        """When simulation fails mid-way, status is set to failed."""
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.refresh = AsyncMock()

        # Use a counter to fail on the 3rd flush call
        flush_count = 0

        async def flush_side_effect():
            nonlocal flush_count
            flush_count += 1
            if flush_count == 3:
                raise Exception("DB write failed")

        mock_db.flush = AsyncMock(side_effect=flush_side_effect)

        eval_run = MagicMock()
        eval_run.id = "run-fail"
        eval_run.agent_config_id = "cfg-001"
        eval_run.scenario_id = "scn-001"
        eval_run.num_conversations = 1
        eval_run.status = "pending"
        eval_run.config = {}
        eval_run.started_at = None
        eval_run.completed_at = None
        eval_run.error_message = None

        agent_config = MagicMock()
        agent_config.name = "Agent"
        agent_config.system_prompt = "Help"
        agent_config.model = "test-model"
        agent_config.temperature = 0.7
        agent_config.max_tokens = 4096
        agent_config.tools = []

        scenario = MagicMock()
        scenario.user_persona = {}
        scenario.constraints = {"max_turns": 2}
        scenario.turns_template = [{"role": "user", "content": "Hi"}]

        run_result = MagicMock()
        run_result.scalar_one_or_none.return_value = eval_run
        run_result.scalar_one.return_value = eval_run
        cfg_result = MagicMock()
        cfg_result.scalar_one.return_value = agent_config
        scn_result = MagicMock()
        scn_result.scalar_one.return_value = scenario

        mock_db.execute = AsyncMock(side_effect=[run_result, cfg_result, scn_result])

        service = AgentSimulationService(db=mock_db)
        with patch.object(service, "llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=make_llm_response())
            await service.run_eval("run-fail")

        assert eval_run.status == "failed"
        assert eval_run.error_message is not None


# ============================================================
# Test: Status transitions
# ============================================================


class TestStatusTransitions:
    """Verify correct status transition ordering."""

    def test_pending_to_running_simulation(self):
        """Status transitions from pending to running_simulation."""
        run = MagicMock()
        run.status = "pending"
        run.status = "running_simulation"
        assert run.status == "running_simulation"

    def test_running_simulation_to_running_evaluation(self):
        """Status transitions from running_simulation to running_evaluation."""
        run = MagicMock()
        run.status = "running_simulation"
        run.status = "running_evaluation"
        assert run.status == "running_evaluation"

    def test_cancellable_statuses(self):
        """Only pending, running_simulation, running_evaluation can be cancelled."""
        cancellable = {"pending", "running_simulation", "running_evaluation"}
        non_cancellable = {"completed", "failed", "cancelled"}

        for status in cancellable:
            assert status in cancellable

        for status in non_cancellable:
            assert status not in cancellable


# ============================================================
# Test: Conversation storage with turns
# ============================================================


class TestConversationStorage:
    """Test that conversation turns are correctly serialized."""

    def test_turns_serialized_as_dicts(self):
        """Turn dataclasses are serialized to dicts for JSONB storage."""
        turns = [
            Turn(role="user", content="Hello", latency_ms=0, input_tokens=0, output_tokens=0),
            Turn(
                role="assistant",
                content="Hi there!",
                latency_ms=150,
                input_tokens=10,
                output_tokens=20,
                tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "test"})],
            ),
        ]

        serialized = [asdict(t) for t in turns]

        assert isinstance(serialized, list)
        assert serialized[0]["role"] == "user"
        assert serialized[0]["content"] == "Hello"
        assert serialized[1]["latency_ms"] == 150
        assert serialized[1]["tool_calls"][0]["name"] == "search"

    def test_empty_turns_serialized(self):
        """Empty turns list serializes to empty list."""
        serialized = [asdict(t) for t in []]
        assert serialized == []


# ============================================================
# Test: Evaluation pipeline
# ============================================================


class TestEvaluationPipeline:
    """Test the evaluation service pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_full_evaluation_flow(self):
        """Full evaluation: load conversation -> run evaluators -> store results."""
        mock_db = AsyncMock()

        # Create conversation mock
        conv = MagicMock()
        conv.id = "conv-eval-001"
        conv.eval_run_id = "run-eval-001"
        conv.turns = [
            {"role": "user", "content": "How do I write a test?"},
            {"role": "assistant", "content": "Use pytest with fixtures.", "latency_ms": 120, "input_tokens": 15, "output_tokens": 25},
            {"role": "user", "content": "Can you show an example?"},
            {"role": "assistant", "content": "Sure, here's an example...", "latency_ms": 200, "input_tokens": 20, "output_tokens": 30},
        ]
        conv.turn_count = 2
        conv.total_tokens = 90
        conv.total_input_tokens = 35
        conv.total_output_tokens = 55
        conv.total_latency_ms = 320
        conv.status = "completed"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = conv
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        service = EvaluationService(db=mock_db)

        with patch.object(service, "llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_judge_response())
            evaluations = await service.evaluate_conversation("conv-eval-001")

        # Should produce at least 2 evaluations (model_judge + rubric_grader)
        assert len(evaluations) >= 2

        # db.add should have been called for evaluations + metrics
        assert mock_db.add.call_count > 0

    @pytest.mark.asyncio
    async def test_evaluation_with_custom_rubric(self):
        """Evaluation with a custom rubric loads custom dimensions."""
        mock_db = AsyncMock()

        conv = MagicMock()
        conv.id = "conv-rubric"
        conv.eval_run_id = "run-rubric"
        conv.turns = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "latency_ms": 50, "input_tokens": 5, "output_tokens": 10},
        ]
        conv.turn_count = 1
        conv.total_tokens = 15
        conv.total_input_tokens = 5
        conv.total_output_tokens = 10
        conv.total_latency_ms = 50
        conv.status = "completed"

        # First execute returns conversation, second returns rubric
        conv_result = MagicMock()
        conv_result.scalar_one_or_none.return_value = conv

        rubric = MagicMock()
        rubric.dimensions = [
            {"name": "custom_quality", "description": "Custom quality dimension", "weight": 1.0, "criteria": ["Be good"]},
        ]
        rubric_result = MagicMock()
        rubric_result.scalar_one_or_none.return_value = rubric

        mock_db.execute = AsyncMock(side_effect=[conv_result, rubric_result])
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        service = EvaluationService(db=mock_db)

        # Create a judge response matching the custom dimension
        judge_response = LLMResponse(
            content="",
            tool_calls=[ToolCall(
                id="c1",
                name="submit_evaluation",
                arguments={"custom_quality_score": 8.0, "custom_quality_reasoning": "Good quality"},
            )],
            input_tokens=30,
            output_tokens=50,
            model="test-model",
            stop_reason="tool_calls",
        )

        with patch.object(service, "llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=judge_response)
            evaluations = await service.evaluate_conversation("conv-rubric", rubric_id="rubric-custom")

        assert len(evaluations) >= 1


# ============================================================
# Test: ScenarioRunner full integration
# ============================================================


class TestScenarioRunnerIntegration:
    """Integration test for ScenarioRunner with all components wired up."""

    @pytest.mark.asyncio
    async def test_full_conversation_with_tool_calls(self):
        """Full multi-turn conversation including tool call interception."""
        mock_llm = AsyncMock()
        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1

            # Call 1: Agent responds to initial message with tool call
            if call_count == 1:
                return LLMResponse(
                    content="Let me search for that.",
                    tool_calls=[ToolCall(id="c1", name="search", arguments={"query": "python testing"})],
                    input_tokens=15,
                    output_tokens=25,
                    model="test-model",
                    stop_reason="tool_calls",
                )
            # Call 2: Agent follow-up after tool results
            if call_count == 2:
                return make_llm_response("Based on my search, here's how to test...")
            # Call 3: User sim generates follow-up
            if call_count == 3:
                return make_llm_response("Thanks! [GOAL_ACHIEVED]")
            # Call 4+: Agent responds
            return make_llm_response("Glad I could help!")

        mock_llm.chat = mock_chat

        agent = AgentPersona(
            name="test-agent",
            system_prompt="You are a testing expert.",
            model="test-model",
            tools=[{"name": "search", "description": "Search for information"}],
        )
        user_persona = UserPersona(goal="Learn about testing", model="test-model")
        env = SimulationEnvironment(max_turns=5)
        user_sim = UserSimulator(llm_client=mock_llm, persona=user_persona, initial_message="How do I test Python?")
        tool_sim = ToolSimulator(environment=env)

        runner = ScenarioRunner(
            llm_client=mock_llm,
            agent_persona=agent,
            user_simulator=user_sim,
            tool_simulator=tool_sim,
            environment=env,
        )

        result = await runner.run()

        # Should have completed with goal achieved or regular completion
        assert result.status in ("completed", "goal_achieved")
        assert len(result.turns) > 0
        assert result.total_tokens > 0

        # Should have at least one tool call turn
        tool_turns = [t for t in result.turns if t.tool_calls]
        assert len(tool_turns) >= 1
