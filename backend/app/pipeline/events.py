"""Typed Kafka event schemas with versioned envelope."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from uuid_extensions import uuid7


@dataclass
class EventEnvelope:
    """Versioned envelope wrapping all Kafka events."""

    version: int
    event_type: str
    payload: dict[str, object]

    def serialize(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> EventEnvelope:
        raw = json.loads(data.decode("utf-8"))
        return cls(
            version=raw["version"],
            event_type=raw["event_type"],
            payload=raw["payload"],
        )


@dataclass
class ConversationCompletedEvent:
    """Emitted when a simulation conversation finishes."""

    eval_run_id: str
    conversation_id: str
    turn_count: int
    total_tokens: int
    total_latency_ms: int
    status: str  # "completed" | "failed"
    event_id: str = field(default_factory=lambda: str(uuid7()))
    event_type: str = "agent.conversation.completed"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_envelope(self) -> EventEnvelope:
        return EventEnvelope(
            version=1,
            event_type=self.event_type,
            payload=asdict(self),
        )


@dataclass
class EvaluationScoreCompletedEvent:
    """Emitted when a single conversation evaluation finishes."""

    eval_run_id: str
    conversation_id: str
    evaluation_id: str
    evaluator_type: str  # "model_judge" | "rubric_grader" | "human"
    overall_score: float
    dimension_scores: dict[str, float]
    event_id: str = field(default_factory=lambda: str(uuid7()))
    event_type: str = "evaluation.score.completed"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_envelope(self) -> EventEnvelope:
        return EventEnvelope(
            version=1,
            event_type=self.event_type,
            payload=asdict(self),
        )


@dataclass
class MetricsAggregatedEvent:
    """Emitted when metrics are aggregated for an eval run."""

    eval_run_id: str
    metric_name: str
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    sample_count: int
    event_id: str = field(default_factory=lambda: str(uuid7()))
    event_type: str = "metrics.aggregated"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_envelope(self) -> EventEnvelope:
        return EventEnvelope(
            version=1,
            event_type=self.event_type,
            payload=asdict(self),
        )
