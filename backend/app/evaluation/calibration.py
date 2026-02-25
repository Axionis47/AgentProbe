"""Calibration analysis — model judge vs human score agreement.

Pure math module.  Measures how well automated model-judge scores
predict human evaluation scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CalibrationMetrics:
    """Statistical comparison between human and model scores."""

    pearson_r: float
    spearman_rho: float
    mae: float  # Mean absolute error
    rmse: float  # Root mean squared error
    bias: float  # mean(model - human), positive = model scores higher
    n: int  # Number of paired observations


@dataclass
class CalibrationBin:
    """One bin in a calibration curve."""

    bin_center: float
    avg_human: float
    avg_model: float
    count: int


def calibration_metrics(
    human_scores: list[float],
    model_scores: list[float],
) -> CalibrationMetrics:
    """Compute agreement metrics between paired human and model scores."""
    n = len(human_scores)
    if n != len(model_scores):
        raise ValueError(f"Length mismatch: {n} human vs {len(model_scores)} model")
    if n < 2:
        raise ValueError(f"Need at least 2 paired observations, got {n}")

    mae = sum(abs(h - m) for h, m in zip(human_scores, model_scores)) / n
    rmse = math.sqrt(sum((h - m) ** 2 for h, m in zip(human_scores, model_scores)) / n)
    bias = sum(m - h for h, m in zip(human_scores, model_scores)) / n

    return CalibrationMetrics(
        pearson_r=round(pearson_r(human_scores, model_scores), 4),
        spearman_rho=round(spearman_rho(human_scores, model_scores), 4),
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        bias=round(bias, 4),
        n=n,
    )


def pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient — pure Python."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation — converts to ranks then uses Pearson."""
    return pearson_r(_to_ranks(x), _to_ranks(y))


def calibration_curve(
    human_scores: list[float],
    model_scores: list[float],
    num_bins: int = 10,
) -> list[CalibrationBin]:
    """Compute calibration curve by binning model scores.

    For each bin of model scores, computes average human and model score.
    Perfect calibration: avg_human ≈ avg_model in every bin.
    """
    if not human_scores or not model_scores:
        return []

    min_score = min(model_scores)
    max_score = max(model_scores)

    if max_score == min_score:
        return [CalibrationBin(
            bin_center=round(min_score, 2),
            avg_human=round(sum(human_scores) / len(human_scores), 4),
            avg_model=round(min_score, 4),
            count=len(human_scores),
        )]

    bin_width = (max_score - min_score) / num_bins
    bins: dict[int, list[tuple[float, float]]] = {}

    for h, m in zip(human_scores, model_scores):
        idx = min(int((m - min_score) / bin_width), num_bins - 1)
        bins.setdefault(idx, []).append((h, m))

    result = []
    for idx in sorted(bins.keys()):
        pairs = bins[idx]
        center = min_score + (idx + 0.5) * bin_width
        avg_h = sum(p[0] for p in pairs) / len(pairs)
        avg_m = sum(p[1] for p in pairs) / len(pairs)
        result.append(CalibrationBin(
            bin_center=round(center, 2),
            avg_human=round(avg_h, 4),
            avg_model=round(avg_m, 4),
            count=len(pairs),
        ))

    return result


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _to_ranks(values: list[float]) -> list[float]:
    """Convert values to average ranks (handles ties)."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    return ranks
