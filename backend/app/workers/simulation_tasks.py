"""Celery tasks for running agent simulations asynchronously."""

from __future__ import annotations

import asyncio
from datetime import datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.models.eval_run import EvalRun
from app.services.agent_simulation import AgentSimulationService
from app.workers.celery_app import celery_app

logger = structlog.get_logger()


def _make_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create a fresh engine + session factory per task invocation.

    Celery forks workers, so the global engine from app.db.session
    is bound to the parent's event loop and cannot be reused.
    """
    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
    )
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@celery_app.task(bind=True, name="run_simulation", max_retries=2, default_retry_delay=30)
def run_simulation(self: object, eval_run_id: str) -> dict[str, str]:
    """Execute all conversations for an eval run.

    This is a Celery task that wraps the async simulation service.
    Celery workers are sync, so we use asyncio.run() to bridge.
    """
    logger.info("simulation_task_started", eval_run_id=eval_run_id)

    async def _run() -> str:
        session_factory = _make_session_factory()
        async with session_factory() as session:
            try:
                service = AgentSimulationService(db=session)
                await service.run_eval(eval_run_id)
                await session.commit()

                # Emit Kafka events AFTER commit so consumers see committed data
                service.emit_pending_kafka_events()

                # Re-read status after commit to return accurate result
                result = await session.execute(
                    select(EvalRun.status).where(EvalRun.id == eval_run_id)
                )
                return result.scalar_one()
            except Exception as exc:
                await session.rollback()
                # Best-effort: mark the eval run as failed in a fresh transaction
                try:
                    result = await session.execute(
                        select(EvalRun).where(EvalRun.id == eval_run_id)
                    )
                    eval_run = result.scalar_one_or_none()
                    if eval_run and eval_run.status != "failed":
                        eval_run.status = "failed"
                        eval_run.error_message = str(exc)[:2000]
                        eval_run.completed_at = datetime.utcnow()
                        await session.commit()
                except Exception as inner_exc:
                    logger.error(
                        "simulation_task_status_update_failed",
                        eval_run_id=eval_run_id,
                        error=str(inner_exc),
                    )
                raise

    try:
        final_status = asyncio.run(_run())
    except Exception as exc:
        logger.error(
            "simulation_task_failed",
            eval_run_id=eval_run_id,
            error=str(exc),
        )
        return {"status": "failed", "eval_run_id": eval_run_id, "error": str(exc)[:500]}

    logger.info("simulation_task_completed", eval_run_id=eval_run_id, status=final_status)
    return {"status": final_status, "eval_run_id": eval_run_id}
