from datetime import datetime

from sqlalchemy import Boolean, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class AgentConfig(Base):
    __tablename__ = "agent_configs"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False, default="claude-sonnet-4-20250514")
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    max_tokens: Mapped[int] = mapped_column(Integer, default=4096)
    tools: Mapped[dict] = mapped_column(JSONB, default=list)  # type: ignore[assignment]
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)  # type: ignore[assignment]
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    eval_runs: Mapped[list["EvalRun"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        back_populates="agent_config",
    )
