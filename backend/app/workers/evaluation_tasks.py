"""Celery tasks for running evaluations asynchronously."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.models.conversation import Conversation
from app.models.eval_run import EvalRun
from app.models.evaluation import Evaluation
from app.services.evaluation_service import EvaluationService
from app.workers.celery_app import celery_app

logger = structlog.get_logger()


def _make_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create a fresh engine + session factory per task invocation."""
    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
    )
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@celery_app.task(bind=True, name="evaluate_conversation", max_retries=2, default_retry_delay=30)
def evaluate_conversation(self: object, conversation_id: str, rubric_id: str | None = None) -> dict[str, str]:
    """Evaluate a single conversation.

    Wraps async EvaluationService in asyncio.run() for Celery compatibility.
    After evaluation, checks whether all conversations in the eval run are
    evaluated and transitions the eval run to 'completed' if so.
    """
    logger.info("evaluation_task_started", conversation_id=conversation_id)

    async def _run() -> str:
        session_factory = _make_session_factory()
        async with session_factory() as session:
            try:
                service = EvaluationService(db=session)
                await service.evaluate_conversation(conversation_id, rubric_id)
                await session.commit()
            except Exception as exc:
                await session.rollback()
                logger.error(
                    "evaluation_task_error",
                    conversation_id=conversation_id,
                    error=str(exc),
                )
                raise

            # Check if all conversations in the eval run are now evaluated
            eval_run_id = await _check_eval_run_completion(session, conversation_id)
            return eval_run_id or ""

    try:
        eval_run_id = asyncio.run(_run())
    except Exception as exc:
        logger.error(
            "evaluation_task_failed",
            conversation_id=conversation_id,
            error=str(exc),
        )
        return {"status": "failed", "conversation_id": conversation_id, "error": str(exc)[:500]}

    logger.info("evaluation_task_completed", conversation_id=conversation_id)
    return {"status": "completed", "conversation_id": conversation_id, "eval_run_id": eval_run_id}


async def _check_eval_run_completion(session: object, conversation_id: str) -> str | None:
    """Check if all conversations in the eval run are evaluated; if so, mark completed.

    Returns the eval_run_id if found, else None.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    assert isinstance(session, AsyncSession)

    # Find the eval_run_id for this conversation
    result = await session.execute(
        select(Conversation.eval_run_id).where(Conversation.id == conversation_id)
    )
    row = result.first()
    if not row:
        return None
    eval_run_id = row[0]

    # Count completed conversations in this run
    conv_count_result = await session.execute(
        select(func.count(Conversation.id)).where(
            Conversation.eval_run_id == eval_run_id,
            Conversation.status == "completed",
        )
    )
    total_conversations = conv_count_result.scalar() or 0

    # Count conversations that have at least one evaluation
    evaluated_count_result = await session.execute(
        select(func.count(func.distinct(Evaluation.conversation_id))).where(
            Evaluation.conversation_id.in_(
                select(Conversation.id).where(
                    Conversation.eval_run_id == eval_run_id,
                    Conversation.status == "completed",
                )
            )
        )
    )
    evaluated_count = evaluated_count_result.scalar() or 0

    if evaluated_count >= total_conversations and total_conversations > 0:
        result = await session.execute(
            select(EvalRun).where(EvalRun.id == eval_run_id)
        )
        eval_run = result.scalar_one_or_none()
        if eval_run and eval_run.status == "running_evaluation":
            eval_run.status = "completed"
            eval_run.completed_at = datetime.utcnow()
            await session.commit()
            logger.info(
                "eval_run_completed",
                eval_run_id=eval_run_id,
                evaluated=evaluated_count,
                total=total_conversations,
            )
    else:
        logger.debug(
            "eval_run_incomplete",
            eval_run_id=eval_run_id,
            evaluated=evaluated_count,
            total=total_conversations,
        )

    return eval_run_id


@celery_app.task(bind=True, name="evaluate_all_conversations", max_retries=1)
def evaluate_all_conversations(self: object, eval_run_id: str, rubric_id: str | None = None) -> dict[str, str]:
    """Fan-out: dispatch evaluate_conversation for every completed conversation in an eval run."""
    logger.info("evaluate_all_started", eval_run_id=eval_run_id)

    async def _dispatch() -> int:
        session_factory = _make_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                select(Conversation.id).where(
                    Conversation.eval_run_id == eval_run_id,
                    Conversation.status == "completed",
                )
            )
            conv_ids = [row[0] for row in result.all()]

            for conv_id in conv_ids:
                evaluate_conversation.delay(conv_id, rubric_id)

            return len(conv_ids)

    try:
        count = asyncio.run(_dispatch())
    except Exception as exc:
        logger.error(
            "evaluate_all_failed",
            eval_run_id=eval_run_id,
            error=str(exc),
        )
        return {"status": "failed", "eval_run_id": eval_run_id, "error": str(exc)[:500]}

    logger.info("evaluate_all_dispatched", eval_run_id=eval_run_id, conversation_count=count)
    return {"status": "dispatched", "eval_run_id": eval_run_id, "count": str(count)}
