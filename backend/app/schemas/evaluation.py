from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HumanEvaluationCreate(BaseModel):
    conversation_id: str
    rubric_id: str | None = None
    scores: dict[str, Any]
    overall_score: float = Field(..., ge=0.0, le=10.0)
    reasoning: str | None = None
    per_turn_scores: list[dict[str, Any]] | None = None
    evaluator_id: str | None = None


class EvaluationResponse(BaseModel):
    id: str
    conversation_id: str
    evaluator_type: str
    evaluator_id: str | None
    rubric_id: str | None
    scores: dict[str, Any]
    overall_score: float | None
    reasoning: str | None
    per_turn_scores: list[dict[str, Any]] | None
    metadata: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class EvaluationListResponse(BaseModel):
    total: int
    items: list[EvaluationResponse]
