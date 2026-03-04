"""Evaluation API endpoints."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.conversation import Conversation
from app.models.evaluation import Evaluation
from app.schemas.evaluation import (
    EvaluationResponse,
    HumanEvaluationCreate,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post(
    "/human",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_human_evaluation(
    payload: HumanEvaluationCreate,
    db: AsyncSession = Depends(get_db),
) -> Evaluation:
    """Submit a human evaluation for a conversation."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == payload.conversation_id)
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {payload.conversation_id} not found",
        )

    evaluation = Evaluation(
        conversation_id=payload.conversation_id,
        evaluator_type="human",
        evaluator_id=payload.evaluator_id,
        rubric_id=payload.rubric_id,
        scores=payload.scores,
        overall_score=payload.overall_score,
        reasoning=payload.reasoning,
        per_turn_scores=payload.per_turn_scores,
    )
    db.add(evaluation)
    await db.flush()
    await db.refresh(evaluation)
    return evaluation
