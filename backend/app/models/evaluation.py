from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Evaluation(Base):
    __tablename__ = "evaluations"

    conversation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    evaluator_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
    )
    evaluator_id: Mapped[str | None] = mapped_column(String(255))
    rubric_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("rubrics.id"),
        nullable=True,
    )
    scores: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[assignment]
    overall_score: Mapped[float | None] = mapped_column(Float)
    reasoning: Mapped[str | None] = mapped_column(Text)
    per_turn_scores: Mapped[dict | None] = mapped_column(JSONB)  # type: ignore[assignment]
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)  # type: ignore[assignment]

    # Relationships
    conversation: Mapped["Conversation"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        back_populates="evaluations",
    )
