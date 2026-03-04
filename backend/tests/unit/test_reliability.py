"""Tests for interrater reliability (Krippendorff's alpha)."""

import pytest

from app.evaluation.reliability import (
    ReliabilityResult,
    compute_reliability,
    krippendorffs_alpha,
    pairwise_correlations,
)


class TestKrippendorffsAlpha:
    def test_perfect_agreement(self):
        """All raters give the same scores → alpha = 1.0."""
        matrix = [
            [5.0, 5.0, 5.0],
            [8.0, 8.0, 8.0],
            [3.0, 3.0, 3.0],
        ]
        assert krippendorffs_alpha(matrix) == 1.0

    def test_no_variance_same_values(self):
        """All values identical across all items and raters."""
        matrix = [[5.0, 5.0], [5.0, 5.0]]
        # d_e = 0 → alpha = 1.0 (everyone agrees)
        assert krippendorffs_alpha(matrix) == 1.0

    def test_missing_values_handled(self):
        """None entries are skipped."""
        matrix = [
            [5.0, None, 5.0],
            [3.0, 3.0, None],
            [7.0, 7.0, 7.0],
        ]
        alpha = krippendorffs_alpha(matrix)
        assert isinstance(alpha, float)

    def test_single_item(self):
        """Only one item rated by multiple raters."""
        matrix = [[5.0, 5.0, 5.0]]
        assert krippendorffs_alpha(matrix) == 1.0

    def test_empty_matrix(self):
        assert krippendorffs_alpha([]) == 0.0

    def test_disagreement_reduces_alpha(self):
        """Moderate disagreement → alpha between 0 and 1."""
        matrix = [
            [5.0, 6.0],
            [8.0, 7.0],
            [3.0, 4.0],
            [9.0, 8.0],
        ]
        alpha = krippendorffs_alpha(matrix)
        assert 0 < alpha < 1.0

    def test_strong_disagreement(self):
        """Systematic disagreement → alpha near 0 or negative."""
        matrix = [
            [1.0, 10.0],
            [10.0, 1.0],
            [1.0, 10.0],
            [10.0, 1.0],
        ]
        alpha = krippendorffs_alpha(matrix)
        assert alpha < 0.5


class TestComputeReliability:
    def test_returns_result(self):
        evals = {
            "conv1": [{"helpfulness": 5.0, "accuracy": 6.0}, {"helpfulness": 5.0, "accuracy": 6.0}],
            "conv2": [{"helpfulness": 8.0, "accuracy": 7.0}, {"helpfulness": 8.0, "accuracy": 7.0}],
        }
        result = compute_reliability(evals, ["helpfulness", "accuracy"])
        assert isinstance(result, ReliabilityResult)
        assert result.num_items == 2
        assert result.num_raters == 2
        assert result.alpha == 1.0  # Perfect agreement

    def test_per_dimension_alpha(self):
        evals = {
            "c1": [{"h": 5.0}, {"h": 5.0}],
            "c2": [{"h": 8.0}, {"h": 8.0}],
        }
        result = compute_reliability(evals, ["h"])
        assert "h" in result.per_dimension_alpha
        assert result.per_dimension_alpha["h"] == 1.0

    def test_single_rater(self):
        evals = {"c1": [{"h": 5.0}]}
        result = compute_reliability(evals, ["h"])
        assert result.alpha == 0.0
        assert result.num_raters == 1


class TestPairwiseCorrelations:
    def test_two_raters(self):
        evals = {
            "c1": [{"h": 5.0}, {"h": 6.0}],
            "c2": [{"h": 8.0}, {"h": 9.0}],
            "c3": [{"h": 3.0}, {"h": 4.0}],
        }
        results = pairwise_correlations(evals, "h")
        assert len(results) == 1
        assert results[0]["rater_a"] == 0
        assert results[0]["rater_b"] == 1
        assert results[0]["pearson_r"] == 1.0  # Perfect linear
