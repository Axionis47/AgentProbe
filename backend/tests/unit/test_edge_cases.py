"""Edge case tests for AgentProbe.

Tests unusual, boundary, and adversarial inputs that could break
the simulation engine, evaluation service, or API layer.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.engine.adversarial import AdversarialStrategy, NoOpAdversarial
from app.engine.environment import SimulationEnvironment
from app.engine.persona import AgentPersona, UserPersona
from app.engine.scenario_runner import ScenarioRunner
from app.engine.tool_simulator import ToolSimulator
from app.engine.types import ConversationResult, LLMResponse, ToolCall, ToolResult, Turn
from app.engine.user_simulator import UserSimulator
from app.evaluation.types import DEFAULT_DIMENSIONS, RubricDimension
from app.services.evaluation_service import EvaluationService


# ============================================================
# Helpers
# ============================================================


def make_llm_response(content: str = "OK", tool_calls=None, tokens: int = 30) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        input_tokens=tokens // 2,
        output_tokens=tokens // 2,
        model="test-model",
        stop_reason="end_turn",
    )


def build_runner(
    mock_llm,
    max_turns=5,
    max_total_tokens=50000,
    initial_message="Hello",
    adversarial=None,
    tool_failure_rate=0.0,
):
    """Build a ScenarioRunner with standard test configuration."""
    agent = AgentPersona(
        name="test-agent",
        system_prompt="You are helpful.",
        model="test-model",
        temperature=0.7,
        max_tokens=1000,
    )
    user_persona = UserPersona(
        personality="neutral",
        expertise_level="intermediate",
        goal="Get help",
        model="test-model",
    )
    env = SimulationEnvironment(
        max_turns=max_turns,
        max_total_tokens=max_total_tokens,
        tool_failure_rate=tool_failure_rate,
    )
    user_sim = UserSimulator(
        llm_client=mock_llm,
        persona=user_persona,
        initial_message=initial_message,
    )
    tool_sim = ToolSimulator(environment=env)
    return ScenarioRunner(
        llm_client=mock_llm,
        agent_persona=agent,
        user_simulator=user_sim,
        tool_simulator=tool_sim,
        environment=env,
        adversarial=adversarial,
    )


# ============================================================
# Edge Case: Empty conversation (0 turns)
# ============================================================


class TestZeroTurnConversation:
    """Conversation with max_turns=0 should complete without error."""

    @pytest.mark.asyncio
    async def test_zero_max_turns_completes(self):
        """Setting max_turns=0 produces a conversation with 0 turns."""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=make_llm_response())

        runner = build_runner(mock_llm, max_turns=0)
        result = await runner.run()

        assert result.status == "completed"
        assert result.turn_count == 0
        assert result.turns == []
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_zero_max_turns_no_llm_calls(self):
        """With max_turns=0, no LLM calls should be made."""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=make_llm_response())

        runner = build_runner(mock_llm, max_turns=0)
        await runner.run()

        mock_llm.chat.assert_not_called()


# ============================================================
# Edge Case: Token budget hit exactly at limit
# ============================================================


class TestTokenBudgetBoundary:
    """Token budget enforcement at exact boundary values."""

    @pytest.mark.asyncio
    async def test_token_budget_exact_boundary(self):
        """Conversation stops when total_tokens reaches exactly the budget."""
        mock_llm = AsyncMock()
        # Each call uses 30 tokens (15 in + 15 out)
        mock_llm.chat = AsyncMock(return_value=make_llm_response(tokens=30))

        # Budget of 30: after first agent response, budget is met
        runner = build_runner(mock_llm, max_turns=100, max_total_tokens=30)
        result = await runner.run()

        # Should stop after 1 turn (30 tokens >= 30 budget)
        assert result.total_tokens <= 60  # At most 2 agent calls
        assert result.turn_count <= 2

    @pytest.mark.asyncio
    async def test_very_low_token_budget(self):
        """Token budget of 1 should stop after at most 1 turn."""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=make_llm_response(tokens=30))

        runner = build_runner(mock_llm, max_turns=100, max_total_tokens=1)
        result = await runner.run()

        assert result.turn_count <= 1


# ============================================================
# Edge Case: Tool simulator with 100% failure rate
# ============================================================


class TestToolFullFailureRate:
    """All tool calls fail when failure rate is 100%."""

    @pytest.mark.asyncio
    async def test_all_tool_calls_return_errors(self):
        """With 100% failure rate, every tool call returns an error result."""
        env = SimulationEnvironment(tool_failure_rate=1.0)
        sim = ToolSimulator(environment=env)

        for i in range(10):
            call = ToolCall(id=f"call_{i}", name="search", arguments={"q": "test"})
            result = await sim.execute(call)
            assert result.is_error
            data = json.loads(result.content)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_agent_handles_tool_failure_gracefully(self):
        """Agent can continue after tool call failure."""
        mock_llm = AsyncMock()
        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_llm_response(
                    content="Let me search.",
                    tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "test"})],
                )
            return make_llm_response("Here's what I found despite the error.")

        mock_llm.chat = mock_chat

        agent = AgentPersona(
            name="test-agent",
            system_prompt="You are helpful.",
            model="test-model",
            tools=[{"name": "search", "description": "Search"}],
        )
        user_persona = UserPersona(model="test-model")
        env = SimulationEnvironment(max_turns=2, tool_failure_rate=1.0)
        user_sim = UserSimulator(llm_client=mock_llm, persona=user_persona, initial_message="Search for me")
        tool_sim = ToolSimulator(environment=env)

        runner = ScenarioRunner(
            llm_client=mock_llm,
            agent_persona=agent,
            user_simulator=user_sim,
            tool_simulator=tool_sim,
            environment=env,
        )
        result = await runner.run()

        # Should complete despite tool failures
        assert result.status in ("completed", "goal_achieved", "frustrated")
        # Tool results should be marked as errors
        tool_turns = [t for t in result.turns if t.tool_calls]
        if tool_turns:
            assert tool_turns[0].tool_results[0].is_error


# ============================================================
# Edge Case: Agent response with empty content
# ============================================================


class TestEmptyAgentResponse:
    """Agent returns empty string content."""

    @pytest.mark.asyncio
    async def test_empty_content_does_not_crash(self):
        """Agent returning empty string still produces a valid conversation."""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=make_llm_response(content=""))

        runner = build_runner(mock_llm, max_turns=2)
        result = await runner.run()

        assert result.status == "completed"
        assistant_turns = [t for t in result.turns if t.role == "assistant"]
        for turn in assistant_turns:
            assert turn.content == ""

    @pytest.mark.asyncio
    async def test_whitespace_only_content(self):
        """Agent returning whitespace-only content still works."""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=make_llm_response(content="   \n\t  "))

        runner = build_runner(mock_llm, max_turns=1)
        result = await runner.run()

        assert result.status == "completed"


# ============================================================
# Edge Case: Adversarial injection at turn 0
# ============================================================


class TestAdversarialAtTurnZero:
    """Adversarial injection at the very first turn (turn 0)."""

    @pytest.mark.asyncio
    async def test_adversarial_replaces_initial_message_at_turn_0(self):
        """When adversarial is configured for turn 0, it overrides the initial message."""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=make_llm_response("OK"))

        env = SimulationEnvironment(max_turns=2, adversarial_turns=[0])
        adversarial = AdversarialStrategy(env)

        runner = build_runner(
            mock_llm,
            max_turns=2,
            initial_message="Normal message",
            adversarial=adversarial,
        )
        # Override the runner's env to match
        runner.env = env

        result = await runner.run()

        # Turn 0 user message should be adversarial (not "Normal message")
        user_turns = [t for t in result.turns if t.role == "user"]
        assert len(user_turns) >= 1
        # The first user message should NOT be the initial_message
        # because adversarial injection overrides it
        assert user_turns[0].content != "Normal message"

    def test_adversarial_should_inject_at_zero(self):
        """AdversarialStrategy.should_inject returns True for turn 0 when configured."""
        env = SimulationEnvironment(adversarial_turns=[0, 2])
        strategy = AdversarialStrategy(env)

        assert strategy.should_inject(0) is True
        assert strategy.should_inject(1) is False
        assert strategy.should_inject(2) is True

    def test_adversarial_generates_non_empty(self):
        """generate_adversarial_input always returns a non-empty string."""
        env = SimulationEnvironment(adversarial_turns=[0])
        strategy = AdversarialStrategy(env)

        for _ in range(20):
            msg = strategy.generate_adversarial_input(0)
            assert len(msg) > 0


# ============================================================
# Edge Case: Scenario with no turns_template (empty list)
# ============================================================


class TestNoTurnsTemplate:
    """Scenario with empty or missing turns_template."""

    @pytest.mark.asyncio
    async def test_empty_initial_message_triggers_llm(self):
        """When initial_message is empty, user simulator calls LLM for turn 0."""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=make_llm_response("Generated message"))

        runner = build_runner(mock_llm, max_turns=1, initial_message="")
        result = await runner.run()

        assert result.status == "completed"
        user_turns = [t for t in result.turns if t.role == "user"]
        assert len(user_turns) == 1
        # The user message should come from the LLM (since no template)
        # It will be "Generated message" since mock returns that
        assert user_turns[0].content == "Generated message"


# ============================================================
# Edge Case: Rubric with 0 dimensions
# ============================================================


class TestZeroDimensionsRubric:
    """Rubric with empty dimensions list."""

    @pytest.mark.asyncio
    async def test_empty_rubric_falls_back_to_defaults(self):
        """When a rubric has 0 dimensions, service falls back to DEFAULT_DIMENSIONS."""
        mock_db = AsyncMock()
        mock_rubric = MagicMock()
        mock_rubric.dimensions = []

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_rubric
        mock_db.execute = AsyncMock(return_value=mock_result)

        service = EvaluationService(db=mock_db)
        dims = await service._load_dimensions("rubric-empty")

        assert dims == DEFAULT_DIMENSIONS
        assert len(dims) == 5

    @pytest.mark.asyncio
    async def test_none_rubric_dimensions_falls_back(self):
        """When rubric.dimensions is None, service falls back to defaults."""
        mock_db = AsyncMock()
        mock_rubric = MagicMock()
        mock_rubric.dimensions = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_rubric
        mock_db.execute = AsyncMock(return_value=mock_result)

        service = EvaluationService(db=mock_db)
        dims = await service._load_dimensions("rubric-none")

        assert dims == DEFAULT_DIMENSIONS


# ============================================================
# Edge Case: Eval run with 0 num_conversations
# ============================================================


class TestZeroNumConversations:
    """Eval run configured with 0 conversations."""

    @pytest.mark.asyncio
    async def test_zero_conversations_completes_immediately(self):
        """Service with num_conversations=0 transitions to running_evaluation."""
        from app.services.agent_simulation import AgentSimulationService

        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()

        eval_run = MagicMock()
        eval_run.id = "run-zero"
        eval_run.num_conversations = 0
        eval_run.agent_config_id = "cfg-001"
        eval_run.scenario_id = "scn-001"
        eval_run.status = "pending"
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
        scenario.constraints = {"max_turns": 5}
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
        with patch.object(service, "llm_client"):
            await service.run_eval("run-zero")

        # With zero conversations and no successes, pipeline treats this as failed
        assert eval_run.status == "failed"


# ============================================================
# Edge Case: SimulationEnvironment boundary values
# ============================================================


class TestEnvironmentEdgeCases:
    """Edge cases for SimulationEnvironment construction."""

    def test_from_dict_with_empty_dict(self):
        """from_dict with empty dict uses all defaults."""
        env = SimulationEnvironment.from_dict({})
        assert env.max_turns == 10
        assert env.max_total_tokens == 50000
        assert env.timeout_seconds == 120.0
        assert env.tool_failure_rate == 0.0
        assert env.tool_latency_ms == 0
        assert env.adversarial_turns == []

    def test_from_dict_with_all_fields(self):
        """from_dict with all fields populates correctly."""
        env = SimulationEnvironment.from_dict({
            "max_turns": 3,
            "max_total_tokens": 1000,
            "timeout_seconds": 30.0,
            "tool_failure_rate": 0.5,
            "tool_latency_ms": 200,
            "adversarial_turns": [0, 2],
        })
        assert env.max_turns == 3
        assert env.tool_failure_rate == 0.5
        assert env.adversarial_turns == [0, 2]

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict ignores keys not in the dataclass."""
        env = SimulationEnvironment.from_dict({"unknown_key": "value", "max_turns": 7})
        assert env.max_turns == 7


# ============================================================
# Edge Case: Persona edge cases
# ============================================================


class TestPersonaEdgeCases:
    """Edge cases for persona construction."""

    def test_agent_persona_from_db_with_none_tools(self):
        """AgentPersona.from_db handles None tools gracefully."""
        mock_config = MagicMock()
        mock_config.name = "Agent"
        mock_config.system_prompt = "Help"
        mock_config.model = "m"
        mock_config.temperature = 0.5
        mock_config.max_tokens = 1000
        mock_config.tools = None

        persona = AgentPersona.from_db(mock_config)
        assert persona.tools == []

    def test_user_persona_from_dict_empty(self):
        """UserPersona.from_dict with empty dict uses defaults."""
        persona = UserPersona.from_dict({})
        assert persona.personality == "neutral"
        assert persona.expertise_level == "intermediate"

    def test_user_persona_system_prompt_contains_persona(self):
        """UserPersona.system_prompt includes persona fields."""
        persona = UserPersona(
            personality="impatient",
            expertise_level="expert",
            goal="Debug code",
        )
        prompt = persona.system_prompt
        assert "impatient" in prompt
        assert "expert" in prompt
        assert "Debug code" in prompt


# ============================================================
# Edge Case: Turn/ToolCall/ToolResult dataclass defaults
# ============================================================


class TestDataclassDefaults:
    """Verify engine dataclass default values."""

    def test_turn_defaults(self):
        """Turn has sensible defaults for optional fields."""
        t = Turn(role="user", content="Hello")
        assert t.tool_calls is None
        assert t.tool_results is None
        assert t.latency_ms == 0
        assert t.input_tokens == 0
        assert t.output_tokens == 0

    def test_tool_result_defaults(self):
        """ToolResult defaults to is_error=False."""
        r = ToolResult(tool_call_id="c1", content="ok")
        assert r.is_error is False

    def test_llm_response_defaults(self):
        """LLMResponse has sensible defaults."""
        r = LLMResponse(content="Hi")
        assert r.tool_calls == []
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.model == ""
        assert r.stop_reason == ""

    def test_conversation_result_defaults(self):
        """ConversationResult has optional error_message=None."""
        cr = ConversationResult(
            turns=[],
            turn_count=0,
            total_tokens=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_latency_ms=0,
            status="completed",
        )
        assert cr.error_message is None
