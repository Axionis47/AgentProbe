"""Automated metrics computed from conversation data — no LLM needed.

These are objective, deterministic metrics derived from conversation structure,
token counts, latency measurements, and tool usage patterns.
"""

from __future__ import annotations

import statistics
from typing import Any

from app.evaluation.types import MetricValue


class AutomatedMetricsCalculator:
    """Computes all automated metrics from a completed conversation."""

    def compute_all(
        self,
        turns: list[dict[str, Any]],
        turn_count: int,
        total_tokens: int,
        total_input_tokens: int,
        total_output_tokens: int,
        total_latency_ms: int,
        status: str,
    ) -> list[MetricValue]:
        """Compute all automated metrics from conversation data."""
        metrics: list[MetricValue] = []

        # Token metrics
        tokens_per_turn = total_tokens / turn_count if turn_count > 0 else 0.0
        metrics.append(MetricValue(
            name="tokens_per_turn",
            value=round(tokens_per_turn, 2),
            unit="tokens",
        ))

        output_input_ratio = (
            total_output_tokens / total_input_tokens
            if total_input_tokens > 0
            else 0.0
        )
        metrics.append(MetricValue(
            name="output_input_ratio",
            value=round(output_input_ratio, 4),
            unit="ratio",
        ))

        # Latency metrics
        assistant_turns = [t for t in turns if t.get("role") == "assistant"]
        latencies = [t.get("latency_ms", 0) for t in assistant_turns]

        avg_latency = statistics.mean(latencies) if latencies else 0.0
        metrics.append(MetricValue(
            name="avg_latency_ms",
            value=round(avg_latency, 2),
            unit="ms",
        ))

        if latencies:
            sorted_lat = sorted(latencies)
            p95_idx = max(0, int(len(sorted_lat) * 0.95) - 1)
            p95_latency = sorted_lat[p95_idx]
        else:
            p95_latency = 0.0
        metrics.append(MetricValue(
            name="p95_latency_ms",
            value=round(p95_latency, 2),
            unit="ms",
        ))

        # Resolution metrics
        metrics.append(MetricValue(
            name="turns_to_resolution",
            value=float(turn_count),
            unit="turns",
        ))

        metrics.append(MetricValue(
            name="conversation_completed",
            value=1.0 if status in ("completed", "goal_achieved") else 0.0,
            unit="boolean",
        ))

        # Tool usage metrics
        all_tool_calls: list[dict[str, Any]] = []
        all_tool_results: list[dict[str, Any]] = []
        for t in turns:
            if t.get("tool_calls"):
                all_tool_calls.extend(t["tool_calls"])
            if t.get("tool_results"):
                all_tool_results.extend(t["tool_results"])

        metrics.append(MetricValue(
            name="tool_call_count",
            value=float(len(all_tool_calls)),
            unit="count",
        ))

        if all_tool_results:
            successes = sum(1 for r in all_tool_results if not r.get("is_error", False))
            tool_success_rate = successes / len(all_tool_results)
        else:
            tool_success_rate = 1.0  # No tools called = no failures
        metrics.append(MetricValue(
            name="tool_success_rate",
            value=round(tool_success_rate, 4),
            unit="ratio",
        ))

        return metrics
