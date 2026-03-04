from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.db.session import get_db
from app.models.conversation import Conversation
from app.models.evaluation import Evaluation
from app.models.metric import Metric
from app.schemas.conversation import ConversationListResponse, ConversationResponse
from app.schemas.evaluation import EvaluationListResponse, EvaluationResponse
from app.schemas.metric import MetricListResponse, MetricResponse

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    eval_run_id: str | None = None,
    status: str | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> ConversationListResponse:
    query = select(Conversation)
    count_query = select(func.count(Conversation.id))

    if eval_run_id:
        query = query.where(Conversation.eval_run_id == eval_run_id)
        count_query = count_query.where(Conversation.eval_run_id == eval_run_id)
    if status:
        query = query.where(Conversation.status == status)
        count_query = count_query.where(Conversation.status == status)

    total = (await db.execute(count_query)).scalar_one()
    result = await db.execute(
        query.order_by(Conversation.sequence_num).offset(offset).limit(limit)
    )
    items = [ConversationResponse.model_validate(r) for r in result.scalars().all()]

    return ConversationListResponse(total=total, offset=offset, limit=limit, items=items)


@router.get("/{conv_id}", response_model=ConversationResponse)
async def get_conversation(
    conv_id: str,
    db: AsyncSession = Depends(get_db),
) -> ConversationResponse:
    result = await db.execute(select(Conversation).where(Conversation.id == conv_id))
    conv = result.scalar_one_or_none()
    if not conv:
        raise NotFoundError("Conversation", conv_id)
    return ConversationResponse.model_validate(conv)


@router.get("/{conv_id}/evaluations", response_model=EvaluationListResponse)
async def get_conversation_evaluations(
    conv_id: str,
    db: AsyncSession = Depends(get_db),
) -> EvaluationListResponse:
    result = await db.execute(
        select(Evaluation)
        .where(Evaluation.conversation_id == conv_id)
        .order_by(Evaluation.created_at)
    )
    items = [EvaluationResponse.model_validate(r) for r in result.scalars().all()]
    return EvaluationListResponse(total=len(items), items=items)


@router.get("/{conv_id}/metrics", response_model=MetricListResponse)
async def get_conversation_metrics(
    conv_id: str,
    db: AsyncSession = Depends(get_db),
) -> MetricListResponse:
    result = await db.execute(
        select(Metric)
        .where(Metric.conversation_id == conv_id)
        .order_by(Metric.metric_name)
    )
    items = [MetricResponse.model_validate(r) for r in result.scalars().all()]
    return MetricListResponse(total=len(items), items=items)
