"""Unit tests for the rubric grader evaluator."""

import pytest

from app.evaluation.rubric_grader import RubricGraderEvaluator
from app.evaluation.types import DEFAULT_DIMENSIONS, RubricDimension


def _make_conversation() -> list[dict]:
    return [
        {"role": "user", "content": "How do I sort a list in Python?"},
        {
            "role": "assistant",
            "content": (
                "You can sort a list in Python using the built-in `sorted()` function "
                "or the `.sort()` method.\n\n"
                "1. `sorted(my_list)` — returns a new sorted list\n"
                "2. `my_list.sort()` — sorts in place\n\n"
                "Both support a `key` parameter for custom sorting."
            ),
            "latency_ms": 150,
            "input_tokens": 20,
            "output_tokens": 50,
        },
        {"role": "user", "content": "Can you show me an example?"},
        {
            "role": "assistant",
            "content": (
                "Here's an example:\n\n"
                "```python\n"
                "numbers = [3, 1, 4, 1, 5]\n"
                "sorted_numbers = sorted(numbers)\n"
                "print(sorted_numbers)  # [1, 1, 3, 4, 5]\n"
                "```"
            ),
            "latency_ms": 120,
            "input_tokens": 30,
            "output_tokens": 40,
        },
    ]


class TestRubricGrader:

    @pytest.mark.asyncio
    async def test_returns_scores_for_all_dimensions(self) -> None:
        grader = RubricGraderEvaluator()
        result = await grader.evaluate(_make_conversation(), DEFAULT_DIMENSIONS)

        for dim in DEFAULT_DIMENSIONS:
            assert dim.name in result.scores
            assert 0.0 <= result.scores[dim.name] <= 10.0

    @pytest.mark.asyncio
    async def test_overall_is_weighted_average(self) -> None:
        grader = RubricGraderEvaluator()
        result = await grader.evaluate(_make_conversation(), DEFAULT_DIMENSIONS)

        # Manually compute weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        for dim in DEFAULT_DIMENSIONS:
            if dim.name in result.scores:
                weighted_sum += result.scores[dim.name] * dim.weight
                total_weight += dim.weight

        expected = round(weighted_sum / total_weight, 2)
        assert abs(result.overall_score - expected) < 0.01

    @pytest.mark.asyncio
    async def test_evaluator_type(self) -> None:
        grader = RubricGraderEvaluator()
        result = await grader.evaluate(_make_conversation(), DEFAULT_DIMENSIONS)
        assert result.evaluator_type == "rubric_grader"

    @pytest.mark.asyncio
    async def test_tool_usage_grading(self) -> None:
        turns = [
            {"role": "user", "content": "Search for Python docs"},
            {
                "role": "assistant",
                "content": "Let me search for that.",
                "tool_calls": [{"id": "c1", "name": "search", "arguments": {"q": "python"}}],
                "tool_results": [{"tool_call_id": "c1", "content": "Python docs", "is_error": False}],
            },
        ]
        grader = RubricGraderEvaluator()
        dims = [RubricDimension(name="tool_usage", description="Tool use", weight=1.0, criteria=[])]
        result = await grader.evaluate(turns, dims)

        assert result.scores["tool_usage"] == 10.0  # 1 call, 100% success
