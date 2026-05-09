"""Tests for Kafka consumers — exercise the side effects without real Kafka.

Each test feeds a synthetic EventEnvelope into the handler and asserts the
right async work happened. DB and external services are mocked; we want to
verify wiring and control flow, not chromadb or postgres internals.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.pipeline.consumers.conversation_consumer import (
    ConversationCompletedConsumer,
    _join_turns,
)
from app.pipeline.events import EventEnvelope


# ---------------------------------------------------------------------------
# _join_turns helper
# ---------------------------------------------------------------------------


def test_join_turns_concatenates_role_and_content():
    turns = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert _join_turns(turns) == "user: hi\nassistant: hello"


def test_join_turns_skips_empty_content():
    turns = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "answer"},
    ]
    assert _join_turns(turns) == "assistant: answer"


def test_join_turns_returns_empty_for_non_list():
    assert _join_turns(None) == ""
    assert _join_turns({}) == ""
    assert _join_turns("not a list") == ""


def test_join_turns_skips_non_dict_entries():
    turns = [{"role": "user", "content": "hi"}, "garbage", 42]
    assert _join_turns(turns) == "user: hi"


# ---------------------------------------------------------------------------
# ConversationCompletedConsumer.handle_event
# ---------------------------------------------------------------------------


def _envelope(conversation_id: str = "conv-1", status: str = "completed") -> EventEnvelope:
    return EventEnvelope(
        version=1,
        event_type="agent.conversation.completed",
        payload={
            "conversation_id": conversation_id,
            "eval_run_id": "run-1",
            "status": status,
        },
    )


def test_skips_non_completed_events(monkeypatch):
    consumer = ConversationCompletedConsumer()

    fake_task = MagicMock()
    monkeypatch.setattr(
        "app.workers.evaluation_tasks.evaluate_conversation",
        fake_task,
    )

    embed_called = MagicMock()
    monkeypatch.setattr(
        "app.pipeline.consumers.conversation_consumer._embed_conversation",
        embed_called,
    )

    consumer.handle_event(_envelope(status="failed"))

    fake_task.delay.assert_not_called()
    embed_called.assert_not_called()


def test_embeds_then_dispatches_evaluation(monkeypatch):
    consumer = ConversationCompletedConsumer()

    fake_embed = AsyncMock()
    monkeypatch.setattr(
        "app.pipeline.consumers.conversation_consumer._embed_conversation",
        fake_embed,
    )

    fake_task = MagicMock()
    monkeypatch.setattr(
        "app.workers.evaluation_tasks.evaluate_conversation",
        fake_task,
    )

    consumer.handle_event(_envelope(conversation_id="abc"))

    fake_embed.assert_awaited_once_with("abc")
    fake_task.delay.assert_called_once_with("abc")


def test_evaluation_dispatched_even_when_embedding_blows_up(monkeypatch):
    """Embedding is best-effort — the evaluation pipeline must keep running."""
    consumer = ConversationCompletedConsumer()

    async def boom(_cid):
        raise RuntimeError("chroma is down")

    monkeypatch.setattr(
        "app.pipeline.consumers.conversation_consumer._embed_conversation",
        boom,
    )

    fake_task = MagicMock()
    monkeypatch.setattr(
        "app.workers.evaluation_tasks.evaluate_conversation",
        fake_task,
    )

    consumer.handle_event(_envelope(conversation_id="abc"))

    fake_task.delay.assert_called_once_with("abc")


# ---------------------------------------------------------------------------
# _embed_conversation — exercises the chroma write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_conversation_writes_to_chroma(monkeypatch):
    from app.pipeline.consumers import conversation_consumer as mod

    fake_conv = SimpleNamespace(
        id="conv-1",
        eval_run_id="run-1",
        turns=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    fake_run = SimpleNamespace(
        id="run-1",
        agent_config_id="agent-1",
        scenario_id="scen-1",
    )

    async def fake_load(session, cid):
        return fake_conv, fake_run

    monkeypatch.setattr(mod, "_load_conversation_and_run", fake_load)

    fake_session = MagicMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(mod, "async_session_factory", lambda: fake_session)

    fake_llm = MagicMock()
    fake_llm.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    monkeypatch.setattr(mod, "LLMClient", lambda: fake_llm)

    fake_collection = MagicMock()
    monkeypatch.setattr(
        mod.ChromaDBClient, "get_conversations_collection", lambda: fake_collection
    )

    await mod._embed_conversation("conv-1")

    fake_llm.embed.assert_awaited_once()
    fake_collection.add.assert_called_once()
    kwargs = fake_collection.add.call_args.kwargs
    assert kwargs["ids"] == ["conv-1"]
    assert kwargs["embeddings"] == [[0.1, 0.2, 0.3]]
    assert kwargs["metadatas"][0] == {
        "conversation_id": "conv-1",
        "eval_run_id": "run-1",
        "agent_config_id": "agent-1",
        "scenario_id": "scen-1",
    }


@pytest.mark.asyncio
async def test_embed_conversation_no_op_when_text_empty(monkeypatch):
    from app.pipeline.consumers import conversation_consumer as mod

    fake_conv = SimpleNamespace(id="conv-1", eval_run_id="run-1", turns=[])
    fake_run = SimpleNamespace(id="run-1", agent_config_id="a", scenario_id="s")

    async def fake_load(session, cid):
        return fake_conv, fake_run

    monkeypatch.setattr(mod, "_load_conversation_and_run", fake_load)

    fake_session = MagicMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(mod, "async_session_factory", lambda: fake_session)

    fake_llm = MagicMock()
    fake_llm.embed = AsyncMock()
    monkeypatch.setattr(mod, "LLMClient", lambda: fake_llm)

    fake_collection = MagicMock()
    monkeypatch.setattr(
        mod.ChromaDBClient, "get_conversations_collection", lambda: fake_collection
    )

    await mod._embed_conversation("conv-1")

    fake_llm.embed.assert_not_awaited()
    fake_collection.add.assert_not_called()


@pytest.mark.asyncio
async def test_embed_conversation_no_op_when_row_missing(monkeypatch):
    from app.pipeline.consumers import conversation_consumer as mod

    async def fake_load(session, cid):
        return None, None

    monkeypatch.setattr(mod, "_load_conversation_and_run", fake_load)

    fake_session = MagicMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(mod, "async_session_factory", lambda: fake_session)

    fake_llm = MagicMock()
    fake_llm.embed = AsyncMock()
    monkeypatch.setattr(mod, "LLMClient", lambda: fake_llm)

    fake_collection = MagicMock()
    monkeypatch.setattr(
        mod.ChromaDBClient, "get_conversations_collection", lambda: fake_collection
    )

    await mod._embed_conversation("conv-1")

    fake_llm.embed.assert_not_awaited()
    fake_collection.add.assert_not_called()
