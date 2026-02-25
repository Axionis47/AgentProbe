"""Trajectory evaluation — compare actual tool usage against expected sequence.

Implements EvaluatorProtocol.  Pure Python, no LLM dependency.
Evaluates whether the agent used the correct tools in the correct order.
"""

from __future__ import annotations

from typing import Any

from app.evaluation.types import EvaluationResult, RubricDimension


class TrajectoryEvaluator:
    """Evaluates tool-call trajectories against an expected sequence.

    The expected_tool_sequence is injected at construction time (sourced from
    ``Scenario.constraints.expected_tool_sequence``).
    """

    def __init__(self, expected_tool_sequence: list[str]) -> None:
        self.expected_tool_sequence = expected_tool_sequence

    async def evaluate(
        self,
        conversation_turns: list[dict[str, Any]],
        rubric_dimensions: list[RubricDimension],
    ) -> EvaluationResult:
        actual = self._extract_tool_sequence(conversation_turns)
        expected = self.expected_tool_sequence

        if not expected:
            return EvaluationResult(
                evaluator_type="trajectory",
                scores={},
                overall_score=0.0,
                reasoning="No expected tool sequence defined.",
            )

        seq_match = self._sequence_match_ratio(actual, expected)
        prec = self._precision(actual, expected)
        rec = self._recall(actual, expected)
        order = self._order_score(actual, expected)
        unnecessary = self._unnecessary_action_count(actual, expected)

        # Overall: average of the four ratio metrics, scaled to 0-10
        overall = round(((seq_match + prec + rec + order) / 4.0) * 10.0, 2)

        scores = {
            "sequence_match_ratio": round(seq_match, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "order_score": round(order, 4),
            "unnecessary_actions": float(unnecessary),
        }

        return EvaluationResult(
            evaluator_type="trajectory",
            scores=scores,
            overall_score=overall,
            reasoning=(
                f"Actual tools: {actual}. Expected: {expected}. "
                f"Sequence match={seq_match:.3f}, precision={prec:.3f}, "
                f"recall={rec:.3f}, order={order:.3f}, "
                f"unnecessary={unnecessary}."
            ),
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tool_sequence(turns: list[dict[str, Any]]) -> list[str]:
        """Extract ordered list of tool names called across all turns."""
        tools: list[str] = []
        for turn in turns:
            for tc in turn.get("tool_calls", []):
                name = tc.get("name") or tc.get("function", {}).get("name", "")
                if name:
                    tools.append(name)
        return tools

    @staticmethod
    def _lcs_length(seq_a: list[str], seq_b: list[str]) -> int:
        """Longest common subsequence length via DP."""
        m, n = len(seq_a), len(seq_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    @staticmethod
    def _sequence_match_ratio(actual: list[str], expected: list[str]) -> float:
        """LCS length / len(expected). 1.0 = all expected tools called in order."""
        if not expected:
            return 1.0
        return TrajectoryEvaluator._lcs_length(actual, expected) / len(expected)

    @staticmethod
    def _precision(actual: list[str], expected: list[str]) -> float:
        """Correct tools / total tools called.  Penalizes unnecessary actions."""
        if not actual:
            return 0.0
        expected_set = set(expected)
        correct = sum(1 for t in actual if t in expected_set)
        return correct / len(actual)

    @staticmethod
    def _recall(actual: list[str], expected: list[str]) -> float:
        """Correct tools / expected tools.  Penalizes missing actions."""
        if not expected:
            return 1.0
        actual_set = set(actual)
        found = sum(1 for t in expected if t in actual_set)
        return found / len(expected)

    @staticmethod
    def _order_score(actual: list[str], expected: list[str]) -> float:
        """Kendall-tau-like rank correlation for shared tools.

        For tools present in both sequences, counts concordant vs discordant
        pairs based on their relative ordering.  Returns value in [0, 1].
        """
        expected_set = set(expected)
        shared = [t for t in actual if t in expected_set]
        if len(shared) < 2:
            return 1.0 if shared else 0.0

        # Build expected rank map
        rank_map = {t: i for i, t in enumerate(expected)}
        ranks = [rank_map.get(t, 0) for t in shared]

        concordant = 0
        total = 0
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                total += 1
                if ranks[i] < ranks[j]:
                    concordant += 1

        return concordant / total if total > 0 else 1.0

    @staticmethod
    def _unnecessary_action_count(actual: list[str], expected: list[str]) -> int:
        """Count tools called that are not in the expected sequence."""
        expected_set = set(expected)
        return sum(1 for t in actual if t not in expected_set)
