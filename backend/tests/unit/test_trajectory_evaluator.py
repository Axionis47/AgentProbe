"""Tests for trajectory evaluator."""

import pytest

from app.evaluation.trajectory_evaluator import TrajectoryEvaluator


def _make_turns_with_tools(tool_names: list[str]) -> list[dict]:
    """Helper — create turns with the given tool call sequence."""
    turns = [{"role": "user", "content": "help me"}]
    for name in tool_names:
        turns.append({
            "role": "assistant",
            "content": f"Using {name}",
            "tool_calls": [{"name": name, "arguments": {}}],
        })
    return turns


class TestExtractToolSequence:
    def test_extracts_tools(self):
        turns = _make_turns_with_tools(["search", "lookup", "reset"])
        seq = TrajectoryEvaluator._extract_tool_sequence(turns)
        assert seq == ["search", "lookup", "reset"]

    def test_no_tools(self):
        turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        assert TrajectoryEvaluator._extract_tool_sequence(turns) == []


class TestTrajectoryMetrics:
    def test_perfect_match(self):
        expected = ["search", "lookup", "reset"]
        actual = ["search", "lookup", "reset"]
        assert TrajectoryEvaluator._sequence_match_ratio(actual, expected) == 1.0
        assert TrajectoryEvaluator._precision(actual, expected) == 1.0
        assert TrajectoryEvaluator._recall(actual, expected) == 1.0
        assert TrajectoryEvaluator._order_score(actual, expected) == 1.0
        assert TrajectoryEvaluator._unnecessary_action_count(actual, expected) == 0

    def test_extra_tools_penalized(self):
        expected = ["search", "reset"]
        actual = ["search", "debug", "reset", "cleanup"]
        assert TrajectoryEvaluator._precision(actual, expected) == 0.5  # 2/4
        assert TrajectoryEvaluator._recall(actual, expected) == 1.0
        assert TrajectoryEvaluator._unnecessary_action_count(actual, expected) == 2

    def test_missing_tools(self):
        expected = ["search", "lookup", "reset"]
        actual = ["search"]
        assert TrajectoryEvaluator._recall(actual, expected) == pytest.approx(1 / 3)

    def test_wrong_order(self):
        expected = ["a", "b", "c"]
        actual = ["c", "b", "a"]  # reversed
        order = TrajectoryEvaluator._order_score(actual, expected)
        assert order < 1.0

    def test_no_actual_tools(self):
        assert TrajectoryEvaluator._precision([], ["a"]) == 0.0
        assert TrajectoryEvaluator._recall([], ["a"]) == 0.0


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_perfect_trajectory(self):
        expected = ["search", "lookup", "reset"]
        evaluator = TrajectoryEvaluator(expected_tool_sequence=expected)
        turns = _make_turns_with_tools(expected)
        result = await evaluator.evaluate(turns, [])
        assert result.overall_score == 10.0
        assert result.evaluator_type == "trajectory"

    @pytest.mark.asyncio
    async def test_no_expected_sequence(self):
        evaluator = TrajectoryEvaluator(expected_tool_sequence=[])
        turns = _make_turns_with_tools(["search"])
        result = await evaluator.evaluate(turns, [])
        assert result.overall_score == 0.0

    @pytest.mark.asyncio
    async def test_evaluator_type(self):
        evaluator = TrajectoryEvaluator(expected_tool_sequence=["a"])
        turns = _make_turns_with_tools(["a"])
        result = await evaluator.evaluate(turns, [])
        assert result.evaluator_type == "trajectory"
