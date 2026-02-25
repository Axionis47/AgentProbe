"""Unit tests for metric aggregation, z-score calibration, weighted averaging."""

from app.evaluation.aggregation import (
    aggregate_metric_values,
    weighted_dimension_average,
    z_score_calibrate,
)


class TestAggregateMetricValues:

    def test_basic_aggregation(self) -> None:
        agg = aggregate_metric_values("latency", [10.0, 20.0, 30.0, 40.0, 50.0])
        assert agg.metric_name == "latency"
        assert agg.mean == 30.0
        assert agg.median == 30.0
        assert agg.min_val == 10.0
        assert agg.max_val == 50.0
        assert agg.sample_count == 5
        assert agg.std_dev > 0

    def test_empty_values(self) -> None:
        agg = aggregate_metric_values("empty", [])
        assert agg.sample_count == 0
        assert agg.mean == 0.0
        assert agg.median == 0.0

    def test_single_value(self) -> None:
        agg = aggregate_metric_values("single", [42.0])
        assert agg.mean == 42.0
        assert agg.median == 42.0
        assert agg.std_dev == 0.0
        assert agg.sample_count == 1


class TestZScoreCalibrate:

    def test_normalizes_distribution(self) -> None:
        scores = [2.0, 4.0, 6.0, 8.0, 10.0]
        z_scores = z_score_calibrate(scores)
        assert len(z_scores) == 5
        # Mean of z-scores should be ~0
        mean_z = sum(z_scores) / len(z_scores)
        assert abs(mean_z) < 0.01

    def test_identical_values_returns_originals(self) -> None:
        scores = [5.0, 5.0, 5.0, 5.0]
        result = z_score_calibrate(scores)
        assert result == scores  # std_dev = 0 → return originals

    def test_single_value_returns_original(self) -> None:
        assert z_score_calibrate([7.0]) == [7.0]

    def test_empty_returns_empty(self) -> None:
        assert z_score_calibrate([]) == []


class TestWeightedDimensionAverage:

    def test_basic_weighted_average(self) -> None:
        scores = {"helpfulness": 8.0, "safety": 10.0}
        weights = {"helpfulness": 0.6, "safety": 0.4}
        result = weighted_dimension_average(scores, weights)
        # (8.0 * 0.6 + 10.0 * 0.4) / (0.6 + 0.4) = 8.8
        assert abs(result - 8.8) < 0.01

    def test_missing_dimensions_ignored(self) -> None:
        scores = {"helpfulness": 8.0}
        weights = {"helpfulness": 0.6, "safety": 0.4}
        result = weighted_dimension_average(scores, weights)
        # Only helpfulness present: 8.0 * 0.6 / 0.6 = 8.0
        assert abs(result - 8.0) < 0.01

    def test_empty_scores(self) -> None:
        assert weighted_dimension_average({}, {"a": 0.5}) == 0.0
