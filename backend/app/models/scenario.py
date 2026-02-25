from datetime import datetime

from sqlalchemy import Boolean, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Scenario(Base):
    __tablename__ = "scenarios"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(100))
    turns_template: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[assignment]
    user_persona: Mapped[dict] = mapped_column(JSONB, default=dict)  # type: ignore[assignment]
    constraints: Mapped[dict] = mapped_column(JSONB, default=dict)  # type: ignore[assignment]
    difficulty: Mapped[str] = mapped_column(String(20), default="medium")
    tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)  # type: ignore[assignment]
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    eval_runs: Mapped[list["EvalRun"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        back_populates="scenario",
    )
