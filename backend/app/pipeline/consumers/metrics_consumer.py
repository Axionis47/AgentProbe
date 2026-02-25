"""Consumes MetricsAggregated events and marks eval runs as completed."""

from __future__ import annotations

import asyncio

import structlog
from sqlalchemy import select

from app.db.session import async_session_factory
from app.models.eval_run import EvalRun
from app.pipeline.consumers.base import BaseConsumer
from app.pipeline.events import EventEnvelope
from app.pipeline.topics import METRICS_AGGREGATED

logger = structlog.get_logger()


class MetricsAggregatedConsumer(BaseConsumer):
    """Marks eval run as completed when metrics are aggregated."""

    def __init__(self) -> None:
        super().__init__(topic=METRICS_AGGREGATED)

    def handle_event(self, envelope: EventEnvelope) -> None:
        """Update the eval run status to completed."""
        payload = envelope.payload
        eval_run_id = payload.get("eval_run_id")

        if not eval_run_id:
            return

        asyncio.run(self._mark_completed(str(eval_run_id)))

    async def _mark_completed(self, eval_run_id: str) -> None:
        """Set eval run status to completed."""
        async with async_session_factory() as session:
            result = await session.execute(
                select(EvalRun).where(EvalRun.id == eval_run_id)
            )
            eval_run = result.scalar_one_or_none()
            if not eval_run:
                logger.warning("eval_run_not_found", eval_run_id=eval_run_id)
                return

            if eval_run.status != "completed":
                eval_run.status = "completed"
                await session.commit()
                logger.info("eval_run_completed", eval_run_id=eval_run_id)
