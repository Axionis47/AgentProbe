"""Reference-based evaluation — compare agent output to gold-standard answers.

Implements EvaluatorProtocol.  Uses pure-Python string similarity metrics
(ROUGE-1, ROUGE-L approximations, exact match) — no external NLP deps.
"""

from __future__ import annotations

import re
from typing import Any

from app.evaluation.types import EvaluationResult, RubricDimension


class ReferenceEvaluator:
    """Compares assistant responses against expected_response fields in turns."""

    async def evaluate(
        self,
        conversation_turns: list[dict[str, Any]],
        rubric_dimensions: list[RubricDimension],
    ) -> EvaluationResult:
        pairs = self._extract_pairs(conversation_turns)

        if not pairs:
            return EvaluationResult(
                evaluator_type="reference_based",
                scores={"token_overlap": 0.0, "lcs_ratio": 0.0, "exact_match": 0.0},
                overall_score=0.0,
                reasoning="No reference answers available in scenario.",
            )

        overlaps, lcs_ratios, exacts = [], [], []
        for actual, expected in pairs:
            overlaps.append(self._token_overlap(actual, expected))
            lcs_ratios.append(self._lcs_ratio(actual, expected))
            exacts.append(self._exact_match(actual, expected))

        avg_overlap = sum(overlaps) / len(overlaps)
        avg_lcs = sum(lcs_ratios) / len(lcs_ratios)
        avg_exact = sum(exacts) / len(exacts)

        overall = (0.4 * avg_overlap + 0.4 * avg_lcs + 0.2 * avg_exact) * 10.0

        return EvaluationResult(
            evaluator_type="reference_based",
            scores={
                "token_overlap": round(avg_overlap, 4),
                "lcs_ratio": round(avg_lcs, 4),
                "exact_match": round(avg_exact, 4),
            },
            overall_score=round(overall, 2),
            reasoning=(
                f"Evaluated {len(pairs)} reference pair(s). "
                f"Token overlap={avg_overlap:.3f}, LCS ratio={avg_lcs:.3f}, "
                f"Exact match={avg_exact:.3f}."
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pairs(
        turns: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        """Extract (actual_response, expected_response) pairs.

        Walks through turns.  For each user turn with ``expected_response``,
        finds the **next** assistant turn's content.
        """
        pairs: list[tuple[str, str]] = []
        for i, turn in enumerate(turns):
            expected = turn.get("expected_response")
            if expected and turn.get("role") == "user":
                # Find next assistant turn
                for j in range(i + 1, len(turns)):
                    if turns[j].get("role") == "assistant":
                        pairs.append((turns[j].get("content", ""), expected))
                        break
        return pairs

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return ReferenceEvaluator._normalize(text).split()

    @staticmethod
    def _token_overlap(actual: str, expected: str) -> float:
        """ROUGE-1 F1 approximation (unigram set overlap)."""
        actual_tokens = set(ReferenceEvaluator._tokenize(actual))
        expected_tokens = set(ReferenceEvaluator._tokenize(expected))

        if not actual_tokens or not expected_tokens:
            return 0.0

        overlap = actual_tokens & expected_tokens
        precision = len(overlap) / len(actual_tokens)
        recall = len(overlap) / len(expected_tokens)

        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    @staticmethod
    def _lcs_ratio(actual: str, expected: str) -> float:
        """ROUGE-L approximation via longest common subsequence of tokens."""
        a_tokens = ReferenceEvaluator._tokenize(actual)
        e_tokens = ReferenceEvaluator._tokenize(expected)

        if not a_tokens or not e_tokens:
            return 0.0

        m, n = len(a_tokens), len(e_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a_tokens[i - 1] == e_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_len = dp[m][n]
        return lcs_len / max(m, n)

    @staticmethod
    def _exact_match(actual: str, expected: str) -> float:
        """1.0 if normalized strings match, 0.0 otherwise."""
        return 1.0 if ReferenceEvaluator._normalize(actual) == ReferenceEvaluator._normalize(expected) else 0.0
