"""Consumes ConversationCompleted events and triggers evaluation."""

from __future__ import annotations

import structlog

from app.pipeline.consumers.base import BaseConsumer
from app.pipeline.events import EventEnvelope
from app.pipeline.topics import CONVERSATION_COMPLETED

logger = structlog.get_logger()


class ConversationCompletedConsumer(BaseConsumer):
    """Listens for completed conversations and dispatches evaluation tasks."""

    def __init__(self) -> None:
        super().__init__(topic=CONVERSATION_COMPLETED)

    def handle_event(self, envelope: EventEnvelope) -> None:
        """Dispatch a Celery evaluation task for the completed conversation."""
        from app.workers.evaluation_tasks import evaluate_conversation

        payload = envelope.payload
        conversation_id = payload.get("conversation_id")
        status = payload.get("status")

        if status != "completed":
            logger.debug(
                "conversation_skipped",
                conversation_id=conversation_id,
                status=status,
            )
            return

        logger.info("conversation_event_received", conversation_id=conversation_id)
        evaluate_conversation.delay(str(conversation_id))
