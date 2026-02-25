"""Unit tests for the model judge evaluator."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.engine.types import LLMResponse, ToolCall
from app.evaluation.model_judge import ModelJudgeEvaluator
from app.evaluation.types import DEFAULT_DIMENSIONS


def _make_judge_tool_response(dimensions: list) -> LLMResponse:
    """Build a mock LLM response with tool call containing scores."""
    args = {}
    for dim in dimensions:
        args[f"{dim.name}_score"] = 7.5
        args[f"{dim.name}_reasoning"] = f"Good {dim.name}"

    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="call_judge_1",
                name="submit_evaluation",
                arguments=args,
            ),
        ],
        input_tokens=100,
        output_tokens=200,
        model="test-model",
        stop_reason="tool_calls",
    )


def _make_conversation() -> list[dict]:
    return [
        {"role": "user", "content": "Help me with Python"},
        {"role": "assistant", "content": "Sure, I can help with Python."},
    ]


class TestModelJudge:

    @pytest.mark.asyncio
    async def test_tool_call_scores_parsed(self) -> None:
        """Model returns tool_call → scores extracted correctly."""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=_make_judge_tool_response(DEFAULT_DIMENSIONS))

        judge = ModelJudgeEvaluator(llm_client=mock_llm, model="test-model")
        result = await judge.evaluate(_make_conversation(), DEFAULT_DIMENSIONS)

        for dim in DEFAULT_DIMENSIONS:
            assert dim.name in result.scores
            assert result.scores[dim.name] == 7.5

    @pytest.mark.asyncio
    async def test_weighted_overall_computed(self) -> None:
        """Overall score is weighted average of dimension scores."""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=_make_judge_tool_response(DEFAULT_DIMENSIONS))

        judge = ModelJudgeEvaluator(llm_client=mock_llm, model="test-model")
        result = await judge.evaluate(_make_conversation(), DEFAULT_DIMENSIONS)

        # All scores are 7.5, so weighted average = 7.5
        assert abs(result.overall_score - 7.5) < 0.01

    @pytest.mark.asyncio
    async def test_tool_schema_includes_all_dimensions(self) -> None:
        """The scoring tool schema has properties for every dimension."""
        judge = ModelJudgeEvaluator(llm_client=AsyncMock(), model="test-model")
        tool = judge._build_scoring_tool(DEFAULT_DIMENSIONS)

        props = tool["function"]["parameters"]["properties"]
        for dim in DEFAULT_DIMENSIONS:
            assert f"{dim.name}_score" in props
            assert f"{dim.name}_reasoning" in props

    @pytest.mark.asyncio
    async def test_evaluator_type(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=_make_judge_tool_response(DEFAULT_DIMENSIONS))

        judge = ModelJudgeEvaluator(llm_client=mock_llm, model="test-model")
        result = await judge.evaluate(_make_conversation(), DEFAULT_DIMENSIONS)
        assert result.evaluator_type == "model_judge"
