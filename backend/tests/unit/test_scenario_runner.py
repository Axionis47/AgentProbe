"""Unit tests for ScenarioRunner.

All LLM calls are mocked — no real API calls needed.
Tests verify the orchestration logic, not the LLM output.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.engine.adversarial import AdversarialStrategy, NoOpAdversarial
from app.engine.environment import SimulationEnvironment
from app.engine.persona import AgentPersona, UserPersona
from app.engine.scenario_runner import ScenarioRunner
from app.engine.tool_simulator import ToolSimulator
from app.engine.types import LLMResponse, ToolCall
from app.engine.user_simulator import UserSimulator


def make_llm_response(content: str = "OK", tool_calls: list | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        input_tokens=10,
        output_tokens=20,
        model="test-model",
        stop_reason="end_turn",
    )


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    client = AsyncMock()
    client.chat = AsyncMock(return_value=make_llm_response("Test response"))
    return client


@pytest.fixture
def agent_persona() -> AgentPersona:
    return AgentPersona(
        name="test-agent",
        system_prompt="You are a helpful assistant.",
        model="test-model",
        temperature=0.7,
        max_tokens=1000,
    )


@pytest.fixture
def user_persona() -> UserPersona:
    return UserPersona(
        personality="neutral",
        expertise_level="intermediate",
        goal="Get help with a task",
        model="test-model",
    )


@pytest.fixture
def environment() -> SimulationEnvironment:
    return SimulationEnvironment(max_turns=5, max_total_tokens=50000)


def build_runner(
    mock_llm: AsyncMock,
    agent: AgentPersona,
    user_persona: UserPersona,
    env: SimulationEnvironment,
    initial_message: str = "Hello, I need help",
    adversarial: AdversarialStrategy | NoOpAdversarial | None = None,
) -> ScenarioRunner:
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


class TestScenarioRunnerBasic:
    """Test basic conversation flow."""

    @pytest.mark.asyncio
    async def test_completes_5_turn_conversation(
        self, mock_llm_client: AsyncMock, agent_persona: AgentPersona,
        user_persona: UserPersona, environment: SimulationEnvironment,
    ) -> None:
        """2.1: ScenarioRunner completes a 5-turn conversation with mocked Claude."""
        runner = build_runner(mock_llm_client, agent_persona, user_persona, environment)
        result = await runner.run()

        assert result.turn_count == 5
        assert result.status == "completed"
        assert len(result.turns) == 10  # 5 user + 5 assistant
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_records_latency_per_turn(
        self, mock_llm_client: AsyncMock, agent_persona: AgentPersona,
        user_persona: UserPersona, environment: SimulationEnvironment,
    ) -> None:
        runner = build_runner(mock_llm_client, agent_persona, user_persona, environment)
        result = await runner.run()

        assistant_turns = [t for t in result.turns if t.role == "assistant"]
        for turn in assistant_turns:
            assert turn.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_records_tokens_per_turn(
        self, mock_llm_client: AsyncMock, agent_persona: AgentPersona,
        user_persona: UserPersona, environment: SimulationEnvironment,
    ) -> None:
        runner = build_runner(mock_llm_client, agent_persona, user_persona, environment)
        result = await runner.run()

        assert result.total_input_tokens > 0
        assert result.total_output_tokens > 0


class TestGoalTermination:
    """Test early termination on [GOAL_ACHIEVED]."""

    @pytest.mark.asyncio
    async def test_goal_achieved_stops_early(
        self, mock_llm_client: AsyncMock, agent_persona: AgentPersona,
        user_persona: UserPersona, environment: SimulationEnvironment,
    ) -> None:
        """2.3: [GOAL_ACHIEVED] terminates conversation early."""
        # First turn: normal. Second turn: user sim returns goal achieved.
        call_count = 0

        async def mock_chat(**kwargs: object) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            # User sim calls (odd calls in this pattern)
            return make_llm_response("Thanks! That solved it. [GOAL_ACHIEVED]")

        mock_llm_client.chat = mock_chat

        # Use initial message for turn 0 so user sim is only called from turn 1
        runner = build_runner(
            mock_llm_client, agent_persona, user_persona, environment,
            initial_message="Help me with Python",
        )
        result = await runner.run()

        assert result.status == "goal_achieved"
        assert result.turn_count < environment.max_turns

    @pytest.mark.asyncio
    async def test_frustrated_stops_early(
        self, mock_llm_client: AsyncMock, agent_persona: AgentPersona,
        user_persona: UserPersona, environment: SimulationEnvironment,
    ) -> None:
        call_count = 0

        async def mock_chat(**kwargs: object) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            return make_llm_response("[FRUSTRATED] This isn't helping at all.")

        mock_llm_client.chat = mock_chat

        runner = build_runner(
            mock_llm_client, agent_persona, user_persona, environment,
            initial_message="Fix my code",
        )
        result = await runner.run()

        assert result.status == "frustrated"


class TestEnvironmentConstraints:
    """Test environment constraint enforcement."""

    @pytest.mark.asyncio
    async def test_max_turns_enforced(
        self, mock_llm_client: AsyncMock, agent_persona: AgentPersona,
        user_persona: UserPersona,
    ) -> None:
        """2.4: Environment constraints enforced (max turns)."""
        env = SimulationEnvironment(max_turns=3, max_total_tokens=50000)
        runner = build_runner(mock_llm_client, agent_persona, user_persona, env)
        result = await runner.run()

        assert result.turn_count == 3

    @pytest.mark.asyncio
    async def test_token_budget_enforced(
        self, mock_llm_client: AsyncMock, agent_persona: AgentPersona,
        user_persona: UserPersona,
    ) -> None:
        # Set very low token budget
        env = SimulationEnvironment(max_turns=100, max_total_tokens=50)

        # Each response uses 30 tokens (10 input + 20 output)
        runner = build_runner(mock_llm_client, agent_persona, user_persona, env)
        result = await runner.run()

        assert result.total_tokens <= 100  # Should stop within ~2 turns


class TestToolCalls:
    """Test tool call interception and simulation."""

    @pytest.mark.asyncio
    async def test_tool_calls_intercepted(
        self, agent_persona: AgentPersona,
        user_persona: UserPersona, environment: SimulationEnvironment,
    ) -> None:
        """2.2: Tool calls are intercepted and simulated."""
        mock_llm = AsyncMock()

        # Call sequence with max_turns=2 and initial_message:
        # Turn 0: initial_message (no LLM) → agent call #1 (return tool call here)
        # Tool followup: agent call #2
        # Turn 1: user sim call #3 → agent call #4
        call_count = 0

        async def mock_chat(**kwargs: object) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First agent turn — return tool call
                return make_llm_response(
                    content="Let me search for that.",
                    tool_calls=[ToolCall(id="call_1", name="search", arguments={"query": "python error"})],
                )
            return make_llm_response("Here's what I found...")

        mock_llm.chat = mock_chat

        agent_persona.tools = [
            {"name": "search", "description": "Search for info", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}
        ]

        env = SimulationEnvironment(max_turns=2)
        runner = build_runner(mock_llm, agent_persona, user_persona, env)
        result = await runner.run()

        # Should have turns with tool calls
        tool_turns = [t for t in result.turns if t.tool_calls]
        assert len(tool_turns) >= 1
        assert tool_turns[0].tool_results is not None
        assert not tool_turns[0].tool_results[0].is_error


class TestAdversarialInjection:
    """Test adversarial message injection."""

    @pytest.mark.asyncio
    async def test_adversarial_injection_at_configured_turns(
        self, mock_llm_client: AsyncMock, agent_persona: AgentPersona,
        user_persona: UserPersona,
    ) -> None:
        """2.5: Adversarial mode injects hostile user messages at configured turns."""
        env = SimulationEnvironment(
            max_turns=5,
            adversarial_turns=[1, 3],
        )
        adversarial = AdversarialStrategy(env)

        runner = build_runner(
            mock_llm_client, agent_persona, user_persona, env,
            adversarial=adversarial,
        )
        result = await runner.run()

        user_turns = [t for t in result.turns if t.role == "user"]

        # Turns 1 and 3 should be adversarial (not the normal user sim output)
        # Turn 0 uses initial_message, turns 1 and 3 are adversarial
        assert len(user_turns) == 5

    @pytest.mark.asyncio
    async def test_no_adversarial_when_disabled(
        self, mock_llm_client: AsyncMock, agent_persona: AgentPersona,
        user_persona: UserPersona, environment: SimulationEnvironment,
    ) -> None:
        noop = NoOpAdversarial()
        assert not noop.should_inject(0)
        assert not noop.should_inject(5)


class TestErrorHandling:
    """Test error handling and failure modes."""

    @pytest.mark.asyncio
    async def test_llm_error_results_in_failed_status(
        self, agent_persona: AgentPersona,
        user_persona: UserPersona, environment: SimulationEnvironment,
    ) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("API connection failed"))

        runner = build_runner(mock_llm, agent_persona, user_persona, environment)
        result = await runner.run()

        assert result.status == "failed"
        assert result.error_message is not None
        assert "API connection failed" in result.error_message
