"""Consumes ConversationCompleted events.

Two side effects per completed conversation:

1. Embed the joined turn text and store it in the Chroma `conversations`
   collection along with metadata. This powers the similarity-search feature
   ("show me other conversations that look like this one").
2. Dispatch the Celery `evaluate_conversation` task so judges can score it.

If the embedding step fails — Chroma down, embedding API hiccup, anything —
we log and continue. The evaluation pipeline is the critical path; similarity
search is a nice-to-have on top.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.chromadb_client import ChromaDBClient
from app.db.session import async_session_factory
from app.engine.llm_client import make_llm_client
from app.models.conversation import Conversation
from app.models.eval_run import EvalRun
from app.pipeline.consumers.base import BaseConsumer
from app.pipeline.events import EventEnvelope
from app.pipeline.topics import CONVERSATION_COMPLETED

logger = structlog.get_logger()


class ConversationCompletedConsumer(BaseConsumer):
    """Embeds the conversation, then dispatches evaluation."""

    def __init__(self) -> None:
        super().__init__(topic=CONVERSATION_COMPLETED)

    def handle_event(self, envelope: EventEnvelope) -> None:
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

        # Embedding is best-effort — failures here must not block evaluation.
        try:
            asyncio.run(_embed_conversation(str(conversation_id)))
        except Exception as exc:  # noqa: BLE001 - we deliberately swallow to protect the pipeline
            logger.warning(
                "conversation_embedding_failed",
                conversation_id=conversation_id,
                error=str(exc),
            )

        evaluate_conversation.delay(str(conversation_id))


async def _embed_conversation(conversation_id: str) -> None:
    """Fetch the conversation, embed its joined text, and write to Chroma."""
    async with async_session_factory() as session:
        conv, run = await _load_conversation_and_run(session, conversation_id)
        if conv is None or run is None:
            logger.debug("embedding_skipped_missing_row", conversation_id=conversation_id)
            return

        text = _join_turns(conv.turns)
        if not text.strip():
            logger.debug("embedding_skipped_empty_text", conversation_id=conversation_id)
            return

    llm = make_llm_client()
    vector = await llm.embed(text)

    collection = ChromaDBClient.get_conversations_collection()
    collection.add(
        ids=[conversation_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{
            "conversation_id": conversation_id,
            "eval_run_id": str(run.id),
            "agent_config_id": str(run.agent_config_id),
            "scenario_id": str(run.scenario_id),
        }],
    )
    logger.info("conversation_embedded", conversation_id=conversation_id, dim=len(vector))


async def _load_conversation_and_run(
    session: AsyncSession, conversation_id: str
) -> tuple[Conversation | None, EvalRun | None]:
    conv_result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = conv_result.scalar_one_or_none()
    if conv is None:
        return None, None

    run_result = await session.execute(select(EvalRun).where(EvalRun.id == conv.eval_run_id))
    run = run_result.scalar_one_or_none()
    return conv, run


def _join_turns(turns: Any) -> str:
    """Flatten turns (stored as a JSONB list of dicts) into a single string."""
    if not isinstance(turns, list):
        return ""
    parts: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "")
        content = turn.get("content", "")
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)
