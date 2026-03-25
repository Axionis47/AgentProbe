"""Consumes MetricsAggregated events and marks eval runs as completed.

This consumer acts as a secondary completion mechanism. The primary
completion check lives in ``evaluation_tasks._check_eval_run_completion``,
which transitions the eval_run to ``completed`` as soon as all conversations
are evaluated. This consumer provides a fallback that also sets
``completed_at`` after metric aggregation events arrive.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

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
        """Set eval run status to completed with completed_at timestamp."""
        async with async_session_factory() as session:
            result = await session.execute(
                select(EvalRun).where(EvalRun.id == eval_run_id)
            )
            eval_run = result.scalar_one_or_none()
            if not eval_run:
                logger.warning("eval_run_not_found", eval_run_id=eval_run_id)
                return

            if eval_run.status == "completed":
                # Already completed (likely by evaluation_tasks completion check)
                return

            if eval_run.status in ("running_evaluation", "running_simulation"):
                eval_run.status = "completed"
                eval_run.completed_at = datetime.now(timezone.utc)
                await session.commit()
                logger.info("eval_run_completed_by_metrics", eval_run_id=eval_run_id)
