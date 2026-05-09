import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.db.chromadb_client import ChromaDBClient
from app.db.session import get_db
from app.models.conversation import Conversation
from app.models.evaluation import Evaluation
from app.models.metric import Metric
from app.schemas.conversation import (
    ConversationListResponse,
    ConversationResponse,
    SimilarConversationItem,
    SimilarConversationsResponse,
)
from app.schemas.evaluation import EvaluationListResponse, EvaluationResponse
from app.schemas.metric import MetricListResponse, MetricResponse

logger = structlog.get_logger()

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


@router.get("/{conv_id}/similar", response_model=SimilarConversationsResponse)
async def get_similar_conversations(
    conv_id: str,
    limit: int = Query(default=5, ge=1, le=20),
    same_scenario: bool = Query(default=False),
    min_score: float | None = Query(default=None, ge=0.0, le=10.0),
    max_score: float | None = Query(default=None, ge=0.0, le=10.0),
    score_evaluator: str = Query(default="model_judge"),
    db: AsyncSession = Depends(get_db),
) -> SimilarConversationsResponse:
    """Find the N conversations most similar to this one.

    Similarity is cosine over the conversation embedding. Optional metadata
    filters: same_scenario restricts matches to conversations from the same
    scenario; min/max_score filter on the named evaluator's score (defaults
    to model_judge — the most useful "is this run good?" signal).
    """
    source_result = await db.execute(select(Conversation).where(Conversation.id == conv_id))
    source = source_result.scalar_one_or_none()
    if source is None:
        raise NotFoundError("Conversation", conv_id)

    where: dict = {}
    if same_scenario:
        run_lookup = await db.execute(
            select(Conversation.eval_run_id).where(Conversation.id == conv_id)
        )
        # Look up scenario via the eval_run; cheaper to filter by scenario_id
        # which the consumer wrote into chroma metadata directly.
        from app.models.eval_run import EvalRun
        eval_run = (
            await db.execute(select(EvalRun).where(EvalRun.id == run_lookup.scalar_one()))
        ).scalar_one()
        where["scenario_id"] = str(eval_run.scenario_id)

    score_key = f"score_{score_evaluator}"
    score_clauses: list[dict] = []
    if min_score is not None:
        score_clauses.append({score_key: {"$gte": float(min_score)}})
    if max_score is not None:
        score_clauses.append({score_key: {"$lte": float(max_score)}})
    if score_clauses:
        # Combine with any same_scenario filter using $and
        all_clauses = score_clauses + ([{"scenario_id": where["scenario_id"]}] if "scenario_id" in where else [])
        where = {"$and": all_clauses} if len(all_clauses) > 1 else all_clauses[0]

    try:
        collection = ChromaDBClient.get_conversations_collection()
        # Pull (limit + 1) so we can drop the source row itself if it lands in results.
        result = collection.query(
            query_texts=None,  # we want similarity by ID, so use the stored embedding via .get
            n_results=limit + 1,
            where=where if where else None,
            query_embeddings=_load_source_embedding(collection, conv_id),
        )
    except Exception as exc:  # noqa: BLE001 - graceful empty if chroma is unavailable
        logger.warning("chroma_query_failed", conversation_id=conv_id, error=str(exc))
        return SimilarConversationsResponse(source_conversation_id=conv_id, items=[])

    matches = _shape_chroma_matches(result, exclude_id=conv_id, limit=limit)
    if not matches:
        return SimilarConversationsResponse(source_conversation_id=conv_id, items=[])

    # Hydrate the matched conversation rows from Postgres.
    matched_ids = [m["id"] for m in matches]
    convs_result = await db.execute(select(Conversation).where(Conversation.id.in_(matched_ids)))
    convs_by_id = {c.id: c for c in convs_result.scalars().all()}

    items: list[SimilarConversationItem] = []
    for m in matches:
        c = convs_by_id.get(m["id"])
        if c is None:
            # Stale Chroma record pointing at a conversation that's been deleted.
            continue
        items.append(
            SimilarConversationItem(
                conversation=ConversationResponse.model_validate(c),
                similarity=m["similarity"],
                metadata=m["metadata"],
            )
        )

    return SimilarConversationsResponse(source_conversation_id=conv_id, items=items)


def _load_source_embedding(collection, conv_id: str) -> list[list[float]] | None:
    """Pull the source conversation's stored embedding so we can query by it."""
    got = collection.get(ids=[conv_id], include=["embeddings"])
    embs = got.get("embeddings") if isinstance(got, dict) else None
    if not embs or embs[0] is None:
        return None
    return [list(embs[0])]


def _shape_chroma_matches(result: dict, exclude_id: str, limit: int) -> list[dict]:
    """Turn Chroma's column-major result into a list of {id, similarity, metadata}."""
    ids_outer = result.get("ids") or []
    if not ids_outer:
        return []
    ids = ids_outer[0]
    distances = (result.get("distances") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]

    shaped: list[dict] = []
    for i, mid in enumerate(ids):
        if mid == exclude_id:
            continue
        dist = distances[i] if i < len(distances) else 0.0
        # Cosine similarity: 1 - distance; clamp to [0, 1] for safety.
        sim = max(0.0, min(1.0, 1.0 - float(dist)))
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        shaped.append({"id": mid, "similarity": sim, "metadata": meta})
        if len(shaped) >= limit:
            break
    return shaped
