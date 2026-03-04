from datetime import datetime

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    eval_run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("eval_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence_num: Mapped[int] = mapped_column(Integer, nullable=False)
    turns: Mapped[dict] = mapped_column(JSONB, nullable=False, default=list)  # type: ignore[assignment]
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)  # type: ignore[assignment]
    started_at: Mapped[datetime | None] = mapped_column()
    completed_at: Mapped[datetime | None] = mapped_column()

    # Relationships
    eval_run: Mapped["EvalRun"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        back_populates="conversations",
    )
    evaluations: Mapped[list["Evaluation"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        back_populates="conversation",
        cascade="all, delete-orphan",
    )
    metrics: Mapped[list["Metric"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        back_populates="conversation",
        cascade="all, delete-orphan",
    )
