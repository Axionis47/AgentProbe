"""Tests for the conversations API route handlers.

Database is mocked. The Chroma query is mocked. We're verifying control flow
and the shape of the response, not the integration with real services — that
job belongs to the e2e test added later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.v1.conversations import (
    _shape_chroma_matches,
    get_conversation,
    get_conversation_evaluations,
    get_conversation_metrics,
    get_similar_conversations,
    list_conversations,
)
from app.core.exceptions import NotFoundError


# ---------------------------------------------------------------------------
# _shape_chroma_matches helper
# ---------------------------------------------------------------------------


def test_shape_excludes_source_id():
    result = {
        "ids": [["src", "a", "b"]],
        "distances": [[0.0, 0.1, 0.4]],
        "metadatas": [[{"x": 1}, {"x": 2}, {"x": 3}]],
    }
    out = _shape_chroma_matches(result, exclude_id="src", limit=5)
    assert [m["id"] for m in out] == ["a", "b"]


def test_shape_clamps_similarity_into_zero_one():
    """Cosine distance can drift slightly outside [0, 2]; never let similarity escape [0, 1]."""
    result = {
        "ids": [["a", "b", "c"]],
        "distances": [[-0.05, 0.5, 1.2]],  # impossible-but-defensive values
        "metadatas": [[{}, {}, {}]],
    }
    out = _shape_chroma_matches(result, exclude_id="src", limit=5)
    sims = [m["similarity"] for m in out]
    assert sims[0] == 1.0  # 1 - (-0.05) capped at 1
    assert sims[1] == pytest.approx(0.5)
    assert sims[2] == 0.0  # 1 - 1.2 floored at 0


def test_shape_respects_limit():
    result = {
        "ids": [["src", "a", "b", "c", "d"]],
        "distances": [[0.0, 0.1, 0.2, 0.3, 0.4]],
        "metadatas": [[{}, {}, {}, {}, {}]],
    }
    out = _shape_chroma_matches(result, exclude_id="src", limit=2)
    assert len(out) == 2
    assert [m["id"] for m in out] == ["a", "b"]


def test_shape_handles_empty_result():
    assert _shape_chroma_matches({}, exclude_id="any", limit=5) == []
    assert _shape_chroma_matches({"ids": []}, exclude_id="any", limit=5) == []


# ---------------------------------------------------------------------------
# get_similar_conversations handler
# ---------------------------------------------------------------------------


def _make_conv_row(conv_id: str = "c1"):
    return SimpleNamespace(
        id=conv_id,
        eval_run_id="run-1",
        sequence_num=0,
        turns=[],
        turn_count=0,
        total_tokens=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_latency_ms=0,
        status="completed",
        error_message=None,
        metadata_={},
        started_at=None,
        completed_at=None,
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_returns_404_when_source_conversation_missing(monkeypatch):
    db = MagicMock()
    # First execute returns no row.
    empty_result = MagicMock()
    empty_result.scalar_one_or_none = MagicMock(return_value=None)
    db.execute = AsyncMock(return_value=empty_result)

    with pytest.raises(NotFoundError):
        await get_similar_conversations(
            conv_id="missing",
            limit=5,
            same_scenario=False,
            min_score=None,
            max_score=None,
            score_evaluator="model_judge",
            db=db,
        )


@pytest.mark.asyncio
async def test_returns_empty_when_chroma_unavailable(monkeypatch):
    """If Chroma errors or has no embedding, we degrade to an empty response, not a 500."""
    from app.api.v1 import conversations as mod

    db = MagicMock()
    source_row = _make_conv_row("src")
    source_result = MagicMock()
    source_result.scalar_one_or_none = MagicMock(return_value=source_row)
    db.execute = AsyncMock(return_value=source_result)

    fake_collection = MagicMock()
    fake_collection.get.side_effect = RuntimeError("chroma is down")
    monkeypatch.setattr(
        mod.ChromaDBClient, "get_conversations_collection", lambda: fake_collection
    )

    out = await get_similar_conversations(
        conv_id="src",
        limit=5,
        same_scenario=False,
        min_score=None,
        max_score=None,
        score_evaluator="model_judge",
        db=db,
    )
    assert out.source_conversation_id == "src"
    assert out.items == []


@pytest.mark.asyncio
async def test_returns_hydrated_matches(monkeypatch):
    from app.api.v1 import conversations as mod

    source_row = _make_conv_row("src")
    match_a = _make_conv_row("a")
    match_b = _make_conv_row("b")

    # db.execute is called twice: once to load source, once to hydrate matches.
    source_result = MagicMock()
    source_result.scalar_one_or_none = MagicMock(return_value=source_row)
    hydrate_result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=[match_a, match_b])
    hydrate_result.scalars = MagicMock(return_value=scalars)

    db = MagicMock()
    db.execute = AsyncMock(side_effect=[source_result, hydrate_result])

    fake_collection = MagicMock()
    fake_collection.get.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    fake_collection.query.return_value = {
        "ids": [["src", "a", "b"]],
        "distances": [[0.0, 0.1, 0.2]],
        "metadatas": [[{}, {"score_model_judge": 8.0}, {"score_model_judge": 6.0}]],
    }
    monkeypatch.setattr(
        mod.ChromaDBClient, "get_conversations_collection", lambda: fake_collection
    )

    out = await get_similar_conversations(
        conv_id="src",
        limit=5,
        same_scenario=False,
        min_score=None,
        max_score=None,
        score_evaluator="model_judge",
        db=db,
    )

    assert out.source_conversation_id == "src"
    assert [item.conversation.id for item in out.items] == ["a", "b"]
    assert out.items[0].metadata == {"score_model_judge": 8.0}
    assert out.items[0].similarity == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_skips_matches_whose_db_row_is_gone(monkeypatch):
    """Chroma may know about a conversation that's been deleted from Postgres."""
    from app.api.v1 import conversations as mod

    source_row = _make_conv_row("src")
    match_a = _make_conv_row("a")
    # Notice: 'b' is in chroma but not returned from Postgres hydrate

    source_result = MagicMock()
    source_result.scalar_one_or_none = MagicMock(return_value=source_row)
    hydrate_result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=[match_a])  # only 'a'
    hydrate_result.scalars = MagicMock(return_value=scalars)

    db = MagicMock()
    db.execute = AsyncMock(side_effect=[source_result, hydrate_result])

    fake_collection = MagicMock()
    fake_collection.get.return_value = {"embeddings": [[0.1]]}
    fake_collection.query.return_value = {
        "ids": [["a", "b"]],
        "distances": [[0.1, 0.2]],
        "metadatas": [[{}, {}]],
    }
    monkeypatch.setattr(
        mod.ChromaDBClient, "get_conversations_collection", lambda: fake_collection
    )

    out = await get_similar_conversations(
        conv_id="src",
        limit=5,
        same_scenario=False,
        min_score=None,
        max_score=None,
        score_evaluator="model_judge",
        db=db,
    )
    assert [item.conversation.id for item in out.items] == ["a"]


# ---------------------------------------------------------------------------
# list_conversations
# ---------------------------------------------------------------------------


def _mock_list_db(items: list, total: int | None = None):
    """Build a MagicMock that simulates db.execute for a list+count query."""
    if total is None:
        total = len(items)
    db = MagicMock()
    count_result = MagicMock()
    count_result.scalar_one = MagicMock(return_value=total)

    list_result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=items)
    list_result.scalars = MagicMock(return_value=scalars)

    # The handler calls db.execute(count_query) first, then db.execute(query).
    db.execute = AsyncMock(side_effect=[count_result, list_result])
    return db


@pytest.mark.asyncio
async def test_list_conversations_returns_empty_when_none():
    db = _mock_list_db(items=[], total=0)
    out = await list_conversations(
        eval_run_id=None, status=None, offset=0, limit=20, db=db
    )
    assert out.total == 0
    assert out.items == []


@pytest.mark.asyncio
async def test_list_conversations_returns_paginated_items():
    rows = [_make_conv_row(f"c{i}") for i in range(3)]
    db = _mock_list_db(items=rows, total=10)

    out = await list_conversations(
        eval_run_id="run-1", status="completed", offset=0, limit=3, db=db
    )

    assert out.total == 10
    assert out.offset == 0
    assert out.limit == 3
    assert [c.id for c in out.items] == ["c0", "c1", "c2"]


# ---------------------------------------------------------------------------
# get_conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_conversation_returns_row():
    row = _make_conv_row("c1")
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=row)
    db.execute = AsyncMock(return_value=result)

    out = await get_conversation(conv_id="c1", db=db)
    assert out.id == "c1"


@pytest.mark.asyncio
async def test_get_conversation_raises_when_missing():
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    db.execute = AsyncMock(return_value=result)

    with pytest.raises(NotFoundError):
        await get_conversation(conv_id="missing", db=db)


# ---------------------------------------------------------------------------
# get_conversation_evaluations & get_conversation_metrics
# ---------------------------------------------------------------------------


def _make_eval_row(eval_id: str = "e1"):
    return SimpleNamespace(
        id=eval_id,
        conversation_id="c1",
        evaluator_type="model_judge",
        evaluator_id=None,
        rubric_id=None,
        scores={"helpfulness": 9.0},
        overall_score=9.0,
        reasoning="great",
        per_turn_scores=None,
        metadata_={},
        created_at=datetime.now(timezone.utc),
    )


def _make_metric_row(metric_name: str = "tokens"):
    return SimpleNamespace(
        id="m1",
        conversation_id="c1",
        metric_name=metric_name,
        value=42.0,
        unit=None,
        metadata_={},
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_get_conversation_evaluations_returns_items():
    db = MagicMock()
    result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=[_make_eval_row(), _make_eval_row("e2")])
    result.scalars = MagicMock(return_value=scalars)
    db.execute = AsyncMock(return_value=result)

    out = await get_conversation_evaluations(conv_id="c1", db=db)
    assert out.total == 2


@pytest.mark.asyncio
async def test_get_conversation_evaluations_returns_empty_for_unknown_conv():
    db = MagicMock()
    result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=[])
    result.scalars = MagicMock(return_value=scalars)
    db.execute = AsyncMock(return_value=result)

    out = await get_conversation_evaluations(conv_id="ghost", db=db)
    assert out.total == 0
    assert out.items == []


@pytest.mark.asyncio
async def test_get_conversation_metrics_orders_by_name():
    db = MagicMock()
    result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(
        return_value=[_make_metric_row("a_metric"), _make_metric_row("b_metric")]
    )
    result.scalars = MagicMock(return_value=scalars)
    db.execute = AsyncMock(return_value=result)

    out = await get_conversation_metrics(conv_id="c1", db=db)
    assert out.total == 2
    assert [m.metric_name for m in out.items] == ["a_metric", "b_metric"]
