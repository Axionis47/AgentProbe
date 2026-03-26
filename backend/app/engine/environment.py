"""Simulation environment constraints.

Controls conversation boundaries, failure injection, and test conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import settings


@dataclass
class SimulationEnvironment:
    """Defines constraints and conditions for a simulation run."""

    # Conversation limits
    max_turns: int = settings.max_turns
    max_total_tokens: int = settings.max_total_tokens
    timeout_seconds: float = settings.simulation_timeout_seconds

    # Tool behavior
    tool_failure_rate: float = 0.0  # 0.0 to 1.0
    tool_latency_ms: int = 0

    # Adversarial injection
    adversarial_turns: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationEnvironment:
        return cls(
            max_turns=data.get("max_turns", settings.max_turns),
            max_total_tokens=data.get("max_total_tokens", settings.max_total_tokens),
            timeout_seconds=data.get("timeout_seconds", settings.simulation_timeout_seconds),
            tool_failure_rate=data.get("tool_failure_rate", 0.0),
            tool_latency_ms=data.get("tool_latency_ms", 0),
            adversarial_turns=data.get("adversarial_turns", []),
        )
