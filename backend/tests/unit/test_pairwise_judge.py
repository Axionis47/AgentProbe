"""Tests for pairwise judge evaluator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.engine.types import LLMResponse, ToolCall
from app.evaluation.pairwise_judge import PairwiseJudgeEvaluator, PairwiseResult
from app.evaluation.types import RubricDimension


DIMENSIONS = [
    RubricDimension(name="helpfulness", description="How helpful", weight=0.5, criteria=[]),
    RubricDimension(name="accuracy", description="How accurate", weight=0.5, criteria=[]),
]

TURNS_A = [
    {"role": "user", "content": "Help me"},
    {"role": "assistant", "content": "Sure, here's the answer."},
]

TURNS_B = [
    {"role": "user", "content": "Help me"},
    {"role": "assistant", "content": "I cannot help."},
]


def _make_comparison_response(winner: str = "a", confidence: float = 0.9) -> LLMResponse:
    args = {
        "winner": winner,
        "confidence": confidence,
        "reasoning": "Agent A was more helpful.",
        "helpfulness_preference": "a",
        "accuracy_preference": "draw",
    }
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc1", name="submit_comparison", arguments=args)],
    )


class TestPairwiseJudge:
    @pytest.mark.asyncio
    async def test_compare_returns_winner(self):
        mock_client = AsyncMock()
        mock_client.chat.return_value = _make_comparison_response(winner="a")

        evaluator = PairwiseJudgeEvaluator(llm_client=mock_client, model="test-model")
        result = await evaluator.compare(TURNS_A, TURNS_B, DIMENSIONS)

        assert isinstance(result, PairwiseResult)
        assert result.winner in ("a", "b", "draw")
        assert result.match_id  # Not empty

    @pytest.mark.asyncio
    async def test_compare_draw(self):
        mock_client = AsyncMock()
        mock_client.chat.return_value = _make_comparison_response(winner="draw")

        evaluator = PairwiseJudgeEvaluator(llm_client=mock_client, model="test-model")
        result = await evaluator.compare(TURNS_A, TURNS_B, DIMENSIONS)
        assert result.winner == "draw"

    @pytest.mark.asyncio
    async def test_dimension_preferences_parsed(self):
        mock_client = AsyncMock()
        mock_client.chat.return_value = _make_comparison_response()

        evaluator = PairwiseJudgeEvaluator(llm_client=mock_client, model="test-model")
        result = await evaluator.compare(TURNS_A, TURNS_B, DIMENSIONS)

        # Preferences should be present (may be swapped if position was flipped)
        assert len(result.dimension_preferences) == 2
        for dim_name, pref in result.dimension_preferences.items():
            assert pref in ("a", "b", "draw")

    @pytest.mark.asyncio
    async def test_match_id_generated(self):
        mock_client = AsyncMock()
        mock_client.chat.return_value = _make_comparison_response()

        evaluator = PairwiseJudgeEvaluator(llm_client=mock_client, model="test-model")
        result = await evaluator.compare(TURNS_A, TURNS_B, DIMENSIONS)

        assert isinstance(result.match_id, str)
        assert len(result.match_id) > 10  # UUID string

    def test_tool_schema_correct(self):
        evaluator = PairwiseJudgeEvaluator(llm_client=MagicMock(), model="test")
        tool = evaluator._build_comparison_tool(DIMENSIONS)

        props = tool["function"]["parameters"]["properties"]
        assert "winner" in props
        assert "confidence" in props
        assert "reasoning" in props
        assert "helpfulness_preference" in props
        assert "accuracy_preference" in props
        assert props["winner"]["enum"] == ["a", "b", "draw"]
