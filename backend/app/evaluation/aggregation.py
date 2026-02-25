"""Pure math: metric aggregation, z-score calibration, weighted averaging."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class AggregatedMetric:
    """Statistical summary of a single metric across conversations."""

    metric_name: str
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    sample_count: int


def aggregate_metric_values(name: str, values: list[float]) -> AggregatedMetric:
    """Compute descriptive statistics for a list of metric values."""
    if not values:
        return AggregatedMetric(
            metric_name=name,
            mean=0.0,
            median=0.0,
            std_dev=0.0,
            min_val=0.0,
            max_val=0.0,
            sample_count=0,
        )

    n = len(values)
    mean = statistics.mean(values)
    median = statistics.median(values)
    std_dev = statistics.stdev(values) if n >= 2 else 0.0

    return AggregatedMetric(
        metric_name=name,
        mean=round(mean, 4),
        median=round(median, 4),
        std_dev=round(std_dev, 4),
        min_val=round(min(values), 4),
        max_val=round(max(values), 4),
        sample_count=n,
    )


def z_score_calibrate(scores: list[float]) -> list[float]:
    """Normalize scores to z-scores (mean=0, std=1).

    If all scores are identical (std_dev=0), returns the original scores unchanged.
    """
    if len(scores) < 2:
        return scores

    mean = statistics.mean(scores)
    std_dev = statistics.stdev(scores)

    if std_dev == 0:
        return scores

    return [round((x - mean) / std_dev, 4) for x in scores]


def weighted_dimension_average(
    scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Compute weighted average of dimension scores.

    Dimensions missing from scores are ignored.
    Weights are re-normalized to sum to 1.0 over present dimensions.
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for dim_name, score in scores.items():
        w = weights.get(dim_name, 0.0)
        weighted_sum += score * w
        total_weight += w

    if total_weight == 0:
        return 0.0

    return round(weighted_sum / total_weight, 4)
