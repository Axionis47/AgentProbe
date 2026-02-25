"""Unit tests for the evaluation service orchestrator.

All LLM calls and DB operations are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.engine.types import LLMResponse, ToolCall
from app.evaluation.types import DEFAULT_DIMENSIONS
from app.services.evaluation_service import EvaluationService


def _make_mock_conversation() -> MagicMock:
    """Create a mock Conversation DB object."""
    conv = MagicMock()
    conv.id = "conv-test-123"
    conv.eval_run_id = "run-test-123"
    conv.turns = [
        {"role": "user", "content": "Help me"},
        {
            "role": "assistant",
            "content": "Sure, I can help.",
            "latency_ms": 100,
            "input_tokens": 10,
            "output_tokens": 20,
        },
    ]
    conv.turn_count = 1
    conv.total_tokens = 30
    conv.total_input_tokens = 10
    conv.total_output_tokens = 20
    conv.total_latency_ms = 100
    conv.status = "completed"
    return conv


def _make_judge_response() -> LLMResponse:
    """Mock LLM response for model judge."""
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


class TestEvaluationService:

    @pytest.mark.asyncio
    async def test_evaluate_creates_evaluations(self) -> None:
        """evaluate_conversation should create judge + grader evaluations."""
        mock_db = AsyncMock()
        mock_conv = _make_mock_conversation()

        # Mock DB query to return our conversation
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_conv
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        service = EvaluationService(db=mock_db)

        # Mock the LLM client for model judge
        with patch.object(service, "llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_judge_response())

            evaluations = await service.evaluate_conversation("conv-test-123")

        # Should have 2 evaluations (judge + grader)
        assert len(evaluations) == 2

    @pytest.mark.asyncio
    async def test_default_dimensions_used_when_no_rubric(self) -> None:
        """When rubric_id is None, DEFAULT_DIMENSIONS should be used."""
        service = EvaluationService(db=AsyncMock())
        dimensions = await service._load_dimensions(None)
        assert dimensions == DEFAULT_DIMENSIONS

    @pytest.mark.asyncio
    async def test_metrics_stored(self) -> None:
        """Automated metrics should be stored via db.add()."""
        mock_db = AsyncMock()
        mock_conv = _make_mock_conversation()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_conv
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        service = EvaluationService(db=mock_db)

        with patch.object(service, "llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_judge_response())

            await service.evaluate_conversation("conv-test-123")

        # db.add should be called for: 2 evaluations + 8 metrics = 10 times
        assert mock_db.add.call_count == 10
