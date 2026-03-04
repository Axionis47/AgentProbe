from datetime import datetime

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class EvalRun(Base):
    __tablename__ = "eval_runs"

    name: Mapped[str | None] = mapped_column(String(255))
    agent_config_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("agent_configs.id", ondelete="CASCADE"),
        nullable=False,
    )
    scenario_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("scenarios.id", ondelete="CASCADE"),
        nullable=False,
    )
    rubric_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("rubrics.id"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="pending",
    )
    num_conversations: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    config: Mapped[dict] = mapped_column(JSONB, default=dict)  # type: ignore[assignment]
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column()
    completed_at: Mapped[datetime | None] = mapped_column()
    created_by: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=True,
    )

    # Relationships
    agent_config: Mapped["AgentConfig"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        back_populates="eval_runs",
    )
    scenario: Mapped["Scenario"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        back_populates="eval_runs",
    )
    rubric: Mapped["Rubric | None"] = relationship()  # type: ignore[name-defined]  # noqa: F821
    conversations: Mapped[list["Conversation"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        back_populates="eval_run",
        cascade="all, delete-orphan",
    )
