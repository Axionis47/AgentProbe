"""Interrater reliability — Krippendorff's alpha for interval data.

Pure math module.  Measures agreement among multiple human evaluators
scoring the same conversations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class ReliabilityResult:
    """Results of interrater reliability analysis."""

    alpha: float  # Krippendorff's alpha (-1 to 1, >0.8 = good)
    num_items: int  # Number of items (conversations) rated
    num_raters: int  # Number of raters
    per_dimension_alpha: dict[str, float] = field(default_factory=dict)


def krippendorffs_alpha(
    ratings_matrix: list[list[float | None]],
) -> float:
    """Compute Krippendorff's alpha for interval data.

    Args:
        ratings_matrix: rows = items, columns = raters.
            None means that rater did not rate that item.

    Returns:
        Alpha coefficient.  1.0 = perfect agreement, 0.0 = chance,
        negative = systematic disagreement.

    Algorithm (interval metric):
        1. Collect all value pairs from items rated by 2+ raters.
        2. Observed disagreement D_o = mean of squared differences within items.
        3. Expected disagreement D_e = mean of squared differences across all values.
        4. alpha = 1 - D_o / D_e.
    """
    if not ratings_matrix:
        return 0.0

    # Collect observed pairs within items
    observed_diffs_sq: list[float] = []
    all_values: list[float] = []

    for row in ratings_matrix:
        values = [v for v in row if v is not None]
        all_values.extend(values)
        if len(values) < 2:
            continue
        for a, b in combinations(values, 2):
            observed_diffs_sq.append((a - b) ** 2)

    if not observed_diffs_sq or len(all_values) < 2:
        return 0.0

    # Observed disagreement
    d_o = sum(observed_diffs_sq) / len(observed_diffs_sq)

    # Expected disagreement: all possible pairs across the entire dataset
    expected_diffs_sq: list[float] = []
    for a, b in combinations(all_values, 2):
        expected_diffs_sq.append((a - b) ** 2)

    if not expected_diffs_sq:
        return 0.0

    d_e = sum(expected_diffs_sq) / len(expected_diffs_sq)

    if d_e == 0:
        return 1.0  # No variance → perfect agreement

    return round(1.0 - d_o / d_e, 4)


def compute_reliability(
    evaluations_by_conversation: dict[str, list[dict[str, float]]],
    dimensions: list[str],
) -> ReliabilityResult:
    """Compute reliability from grouped evaluations.

    Args:
        evaluations_by_conversation: Maps conversation_id → list of score dicts.
            Each score dict maps dimension_name → score value.
        dimensions: Dimension names to analyze.

    Returns:
        ReliabilityResult with overall and per-dimension alpha.
    """
    conv_ids = list(evaluations_by_conversation.keys())
    max_raters = max((len(v) for v in evaluations_by_conversation.values()), default=0)

    if max_raters < 2:
        return ReliabilityResult(alpha=0.0, num_items=len(conv_ids), num_raters=max_raters)

    # Overall alpha: use overall_score if present, else average across dimensions
    overall_matrix: list[list[float | None]] = []
    for cid in conv_ids:
        evals = evaluations_by_conversation[cid]
        row: list[float | None] = []
        for i in range(max_raters):
            if i < len(evals):
                scores = evals[i]
                # Use mean of all dimensions as overall
                vals = [scores.get(d, 0.0) for d in dimensions if d in scores]
                row.append(sum(vals) / len(vals) if vals else None)
            else:
                row.append(None)
        overall_matrix.append(row)

    overall_alpha = krippendorffs_alpha(overall_matrix)

    # Per-dimension alpha
    per_dim: dict[str, float] = {}
    for dim in dimensions:
        dim_matrix: list[list[float | None]] = []
        for cid in conv_ids:
            evals = evaluations_by_conversation[cid]
            row: list[float | None] = []
            for i in range(max_raters):
                if i < len(evals) and dim in evals[i]:
                    row.append(evals[i][dim])
                else:
                    row.append(None)
            dim_matrix.append(row)
        per_dim[dim] = krippendorffs_alpha(dim_matrix)

    return ReliabilityResult(
        alpha=overall_alpha,
        num_items=len(conv_ids),
        num_raters=max_raters,
        per_dimension_alpha=per_dim,
    )


def pairwise_correlations(
    evaluations_by_conversation: dict[str, list[dict[str, float]]],
    dimension: str,
) -> list[dict]:
    """Compute Pearson correlation between all rater pairs for one dimension.

    Returns list of {rater_a: int, rater_b: int, pearson_r: float, n: int}.
    """
    conv_ids = list(evaluations_by_conversation.keys())
    max_raters = max((len(v) for v in evaluations_by_conversation.values()), default=0)

    results = []
    for ra, rb in combinations(range(max_raters), 2):
        xs, ys = [], []
        for cid in conv_ids:
            evals = evaluations_by_conversation[cid]
            if ra < len(evals) and rb < len(evals):
                va = evals[ra].get(dimension)
                vb = evals[rb].get(dimension)
                if va is not None and vb is not None:
                    xs.append(va)
                    ys.append(vb)

        if len(xs) >= 2:
            r = _pearson(xs, ys)
            results.append({"rater_a": ra, "rater_b": rb, "pearson_r": round(r, 4), "n": len(xs)})

    return results


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation — pure Python."""
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
