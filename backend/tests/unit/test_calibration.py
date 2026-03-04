"""Tests for calibration analysis."""

import math

import pytest

from app.evaluation.calibration import (
    CalibrationBin,
    CalibrationMetrics,
    calibration_curve,
    calibration_metrics,
    pearson_r,
    spearman_rho,
)


class TestPearsonR:
    def test_perfect_positive(self):
        assert pearson_r([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert pearson_r([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)

    def test_zero_variance(self):
        assert pearson_r([5, 5, 5], [1, 2, 3]) == 0.0

    def test_uncorrelated(self):
        r = pearson_r([1, 2, 3, 4], [1, 3, 2, 4])
        assert -1 <= r <= 1


class TestSpearmanRho:
    def test_perfect_rank_correlation(self):
        assert spearman_rho([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)

    def test_reverse_rank(self):
        assert spearman_rho([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)

    def test_handles_ties(self):
        rho = spearman_rho([1, 1, 2, 3], [3, 3, 1, 2])
        assert -1 <= rho <= 1


class TestCalibrationMetrics:
    def test_identical_scores(self):
        result = calibration_metrics([5.0, 7.0, 3.0], [5.0, 7.0, 3.0])
        assert result.pearson_r == pytest.approx(1.0)
        assert result.mae == 0.0
        assert result.rmse == 0.0
        assert result.bias == 0.0
        assert result.n == 3

    def test_positive_bias(self):
        result = calibration_metrics([5.0, 6.0, 7.0], [6.0, 7.0, 8.0])
        assert result.bias == pytest.approx(1.0)  # model consistently +1

    def test_mae_and_rmse(self):
        result = calibration_metrics([5.0, 5.0], [7.0, 3.0])
        assert result.mae == pytest.approx(2.0)
        assert result.rmse == pytest.approx(2.0)

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            calibration_metrics([5.0], [5.0])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            calibration_metrics([1, 2], [1])

    def test_returns_dataclass(self):
        result = calibration_metrics([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
        assert isinstance(result, CalibrationMetrics)


class TestCalibrationCurve:
    def test_basic_bins(self):
        human = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        model = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        bins = calibration_curve(human, model, num_bins=5)
        assert len(bins) <= 5
        for b in bins:
            assert isinstance(b, CalibrationBin)
            assert b.count > 0
            # Perfect calibration: avg_human ≈ avg_model
            assert abs(b.avg_human - b.avg_model) < 0.5

    def test_empty_input(self):
        assert calibration_curve([], []) == []

    def test_constant_model_scores(self):
        bins = calibration_curve([1, 2, 3], [5, 5, 5], num_bins=5)
        assert len(bins) == 1
        assert bins[0].count == 3
