"""Celery tasks for running agent simulations asynchronously."""

from __future__ import annotations

import asyncio

import structlog

from app.db.session import async_session_factory
from app.services.agent_simulation import AgentSimulationService
from app.workers.celery_app import celery_app

logger = structlog.get_logger()


@celery_app.task(bind=True, name="run_simulation", max_retries=2, default_retry_delay=30)
def run_simulation(self: object, eval_run_id: str) -> dict[str, str]:
    """Execute all conversations for an eval run.

    This is a Celery task that wraps the async simulation service.
    Celery workers are sync, so we use asyncio.run() to bridge.
    """
    logger.info("simulation_task_started", eval_run_id=eval_run_id)

    async def _run() -> None:
        async with async_session_factory() as session:
            service = AgentSimulationService(db=session)
            await service.run_eval(eval_run_id)
            await session.commit()

    asyncio.run(_run())

    logger.info("simulation_task_completed", eval_run_id=eval_run_id)
    return {"status": "completed", "eval_run_id": eval_run_id}
