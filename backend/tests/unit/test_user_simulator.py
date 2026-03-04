"""Unit tests for UserSimulator."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.engine.persona import UserPersona
from app.engine.types import LLMResponse, Turn
from app.engine.user_simulator import UserSimulator


def make_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, input_tokens=5, output_tokens=10, model="test")


@pytest.fixture
def mock_llm() -> AsyncMock:
    client = AsyncMock()
    client.chat = AsyncMock(return_value=make_response("I need help with Python"))
    return client


@pytest.fixture
def persona() -> UserPersona:
    return UserPersona(
        personality="confused",
        expertise_level="novice",
        goal="Debug a Python error",
        model="test-model",
    )


class TestUserSimulator:
    @pytest.mark.asyncio
    async def test_uses_initial_message_for_turn_0(
        self, mock_llm: AsyncMock, persona: UserPersona,
    ) -> None:
        sim = UserSimulator(
            llm_client=mock_llm,
            persona=persona,
            initial_message="Help me with my code",
        )

        result = await sim.generate([], turn_index=0)

        assert result == "Help me with my code"
        mock_llm.chat.assert_not_called()  # Should NOT call LLM for turn 0

    @pytest.mark.asyncio
    async def test_calls_llm_for_subsequent_turns(
        self, mock_llm: AsyncMock, persona: UserPersona,
    ) -> None:
        sim = UserSimulator(
            llm_client=mock_llm,
            persona=persona,
            initial_message="Help me",
        )

        history = [
            Turn(role="user", content="Help me"),
            Turn(role="assistant", content="Sure, what do you need?"),
        ]

        result = await sim.generate(history, turn_index=1)

        assert result == "I need help with Python"
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_calls_llm_when_no_initial_message(
        self, mock_llm: AsyncMock, persona: UserPersona,
    ) -> None:
        sim = UserSimulator(
            llm_client=mock_llm,
            persona=persona,
            initial_message="",
        )

        result = await sim.generate([], turn_index=0)

        assert result == "I need help with Python"
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_prompt_contains_persona_info(
        self, persona: UserPersona,
    ) -> None:
        prompt = persona.system_prompt

        assert "confused" in prompt
        assert "novice" in prompt
        assert "Debug a Python error" in prompt
        assert "[GOAL_ACHIEVED]" in prompt
        assert "[FRUSTRATED]" in prompt
