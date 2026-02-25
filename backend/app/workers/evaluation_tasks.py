"""Celery tasks for running evaluations asynchronously."""

from __future__ import annotations

import asyncio

import structlog

from app.db.session import async_session_factory
from app.services.evaluation_service import EvaluationService
from app.workers.celery_app import celery_app

logger = structlog.get_logger()


@celery_app.task(bind=True, name="evaluate_conversation", max_retries=2, default_retry_delay=30)
def evaluate_conversation(self: object, conversation_id: str, rubric_id: str | None = None) -> dict[str, str]:
    """Evaluate a single conversation.

    Wraps async EvaluationService in asyncio.run() for Celery compatibility.
    """
    logger.info("evaluation_task_started", conversation_id=conversation_id)

    async def _run() -> None:
        async with async_session_factory() as session:
            service = EvaluationService(db=session)
            await service.evaluate_conversation(conversation_id, rubric_id)
            await session.commit()

    asyncio.run(_run())

    logger.info("evaluation_task_completed", conversation_id=conversation_id)
    return {"status": "completed", "conversation_id": conversation_id}


@celery_app.task(bind=True, name="evaluate_all_conversations", max_retries=1)
def evaluate_all_conversations(self: object, eval_run_id: str, rubric_id: str | None = None) -> dict[str, str]:
    """Fan-out: dispatch evaluate_conversation for every completed conversation in an eval run."""
    from app.models.conversation import Conversation
    from sqlalchemy import select

    logger.info("evaluate_all_started", eval_run_id=eval_run_id)

    async def _dispatch() -> int:
        async with async_session_factory() as session:
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

    count = asyncio.run(_dispatch())

    logger.info("evaluate_all_dispatched", eval_run_id=eval_run_id, conversation_count=count)
    return {"status": "dispatched", "eval_run_id": eval_run_id, "count": str(count)}
