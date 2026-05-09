"""Tests for the evaluations API route handlers.

DB and the evaluation library functions are mocked. We're verifying
control flow, validation, 404 paths, and basic shaping. The full
end-to-end pairwise behaviour gets exercised in the e2e test.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.evaluations import (
    create_human_evaluation,
    get_calibration,
    get_rankings,
    get_reliability,
)
from app.schemas.evaluation import HumanEvaluationCreate


# ---------------------------------------------------------------------------
# POST /evaluations/human
# ---------------------------------------------------------------------------


def _make_conv_row(conv_id: str = "c1"):
    return SimpleNamespace(id=conv_id, status="completed")


@pytest.mark.asyncio
async def test_create_human_evaluation_persists():
    db = MagicMock()
    conv = _make_conv_row("c1")
    lookup = MagicMock()
    lookup.scalar_one_or_none = MagicMock(return_value=conv)
    db.execute = AsyncMock(return_value=lookup)
    db.add = MagicMock()
    db.flush = AsyncMock()

    async def fake_refresh(ev):
        ev.id = "new-eval"
        ev.created_at = datetime.now(timezone.utc)
        ev.metadata_ = {}

    db.refresh = AsyncMock(side_effect=fake_refresh)

    payload = HumanEvaluationCreate(
        conversation_id="c1",
        scores={"helpfulness": 8.0},
        overall_score=8.0,
        reasoning="solid",
    )
    out = await create_human_evaluation(payload=payload, db=db)

    db.add.assert_called_once()
    persisted = db.add.call_args.args[0]
    assert persisted.conversation_id == "c1"
    assert persisted.evaluator_type == "human"
    assert persisted.overall_score == 8.0
    assert out is persisted


@pytest.mark.asyncio
async def test_create_human_evaluation_404_when_conversation_missing():
    db = MagicMock()
    lookup = MagicMock()
    lookup.scalar_one_or_none = MagicMock(return_value=None)
    db.execute = AsyncMock(return_value=lookup)

    payload = HumanEvaluationCreate(
        conversation_id="ghost",
        scores={"x": 1.0},
        overall_score=5.0,
    )

    with pytest.raises(HTTPException) as exc:
        await create_human_evaluation(payload=payload, db=db)
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# GET /evaluations/rankings — empty path is the easy unit test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_rankings_returns_empty_when_no_pairwise_evals():
    db = MagicMock()
    result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=[])
    result.scalars = MagicMock(return_value=scalars)
    db.execute = AsyncMock(return_value=result)

    out = await get_rankings(scenario_id=None, db=db)
    assert out.total_matches == 0
    assert out.rankings == []
    assert out.scenario_id is None


# ---------------------------------------------------------------------------
# GET /evaluations/reliability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reliability_404_when_no_conversations():
    db = MagicMock()
    conv_result = MagicMock()
    conv_result.all = MagicMock(return_value=[])
    db.execute = AsyncMock(return_value=conv_result)

    with pytest.raises(HTTPException) as exc:
        await get_reliability(eval_run_id="run-x", db=db)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_reliability_404_when_no_human_evaluations():
    db = MagicMock()
    conv_result = MagicMock()
    conv_result.all = MagicMock(return_value=[("c1",), ("c2",)])
    eval_result = MagicMock()
    eval_scalars = MagicMock()
    eval_scalars.all = MagicMock(return_value=[])
    eval_result.scalars = MagicMock(return_value=eval_scalars)
    db.execute = AsyncMock(side_effect=[conv_result, eval_result])

    with pytest.raises(HTTPException) as exc:
        await get_reliability(eval_run_id="run-x", db=db)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_reliability_returns_alpha_when_multiple_raters():
    """At least 2 raters per conversation — alpha is computable."""
    ev1 = SimpleNamespace(conversation_id="c1", scores={"helpfulness": 8.0})
    ev2 = SimpleNamespace(conversation_id="c1", scores={"helpfulness": 7.5})
    ev3 = SimpleNamespace(conversation_id="c2", scores={"helpfulness": 6.0})
    ev4 = SimpleNamespace(conversation_id="c2", scores={"helpfulness": 6.5})

    db = MagicMock()
    conv_result = MagicMock()
    conv_result.all = MagicMock(return_value=[("c1",), ("c2",)])
    eval_result = MagicMock()
    eval_scalars = MagicMock()
    eval_scalars.all = MagicMock(return_value=[ev1, ev2, ev3, ev4])
    eval_result.scalars = MagicMock(return_value=eval_scalars)
    db.execute = AsyncMock(side_effect=[conv_result, eval_result])

    fake_result = SimpleNamespace(
        alpha=0.85,
        num_items=2,
        num_raters=2,
        per_dimension_alpha={"helpfulness": 0.85},
    )
    with patch(
        "app.evaluation.reliability.compute_reliability", return_value=fake_result
    ):
        out = await get_reliability(eval_run_id="run-1", db=db)

    assert out.alpha == 0.85
    assert out.num_items == 2
    assert out.num_raters == 2


# ---------------------------------------------------------------------------
# GET /evaluations/calibration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calibration_404_when_no_conversations():
    db = MagicMock()
    conv_result = MagicMock()
    conv_result.all = MagicMock(return_value=[])
    db.execute = AsyncMock(return_value=conv_result)

    with pytest.raises(HTTPException) as exc:
        await get_calibration(eval_run_id="run-x", db=db)
    assert exc.value.status_code == 404
