"""Tests for the Celery task wrappers.

These call the underlying task functions directly (not through .delay()),
mock the DB session and the service layers, and assert the bridge between
sync Celery and the async services behaves as advertised: success -> happy
dict; exception -> failed dict; eval_run completion check fires when all
conversations are evaluated.

No real DB, no real LLM, no real Celery broker.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — async session factory plumbing
# ---------------------------------------------------------------------------


def _make_fake_session_factory():
    """Build a no-op AsyncSession that supports `async with` and the methods the tasks call."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    factory = MagicMock(return_value=session)
    return factory, session


# ---------------------------------------------------------------------------
# run_simulation
# ---------------------------------------------------------------------------


def test_run_simulation_returns_final_status_on_success(monkeypatch):
    from app.workers import simulation_tasks as mod

    factory, session = _make_fake_session_factory()
    monkeypatch.setattr(mod, "_make_session_factory", lambda: factory)

    # Mock the service so we don't touch the real AgentSimulationService.
    fake_service = MagicMock()
    fake_service.run_eval = AsyncMock()
    fake_service.emit_pending_kafka_events = MagicMock()

    with patch.object(mod, "AgentSimulationService", return_value=fake_service):
        # The status select() result needs to return "completed".
        status_row = MagicMock()
        status_row.scalar_one = MagicMock(return_value="completed")
        session.execute = AsyncMock(return_value=status_row)

        out = mod.run_simulation("run-1")

    assert out == {"status": "completed", "eval_run_id": "run-1"}
    fake_service.run_eval.assert_awaited_once_with("run-1")
    session.commit.assert_awaited()
    fake_service.emit_pending_kafka_events.assert_called_once()


def test_run_simulation_returns_failed_dict_on_exception(monkeypatch):
    from app.workers import simulation_tasks as mod

    factory, session = _make_fake_session_factory()
    monkeypatch.setattr(mod, "_make_session_factory", lambda: factory)

    fake_service = MagicMock()
    fake_service.run_eval = AsyncMock(side_effect=RuntimeError("agent crashed"))

    # The except-branch tries to flip status to failed; let that select find no row
    # so the inner try is harmless.
    fail_lookup = MagicMock()
    fail_lookup.scalar_one_or_none = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=fail_lookup)

    with patch.object(mod, "AgentSimulationService", return_value=fake_service):
        out = mod.run_simulation("run-1")

    assert out["status"] == "failed"
    assert out["eval_run_id"] == "run-1"
    assert "agent crashed" in out["error"]


def test_run_simulation_marks_failed_when_eval_run_exists(monkeypatch):
    """Failure path should flip the eval_run row to 'failed' with the error message."""
    from app.workers import simulation_tasks as mod

    factory, session = _make_fake_session_factory()
    monkeypatch.setattr(mod, "_make_session_factory", lambda: factory)

    fake_service = MagicMock()
    fake_service.run_eval = AsyncMock(side_effect=RuntimeError("boom"))

    fake_eval_run = SimpleNamespace(
        id="run-1", status="running_simulation", error_message=None, completed_at=None
    )
    fail_lookup = MagicMock()
    fail_lookup.scalar_one_or_none = MagicMock(return_value=fake_eval_run)
    session.execute = AsyncMock(return_value=fail_lookup)

    with patch.object(mod, "AgentSimulationService", return_value=fake_service):
        mod.run_simulation("run-1")

    assert fake_eval_run.status == "failed"
    assert fake_eval_run.error_message == "boom"


# ---------------------------------------------------------------------------
# evaluate_conversation
# ---------------------------------------------------------------------------


def test_evaluate_conversation_success_returns_eval_run_id(monkeypatch):
    from app.workers import evaluation_tasks as mod

    factory, session = _make_fake_session_factory()
    monkeypatch.setattr(mod, "_make_session_factory", lambda: factory)

    fake_service = MagicMock()
    fake_service.evaluate_conversation = AsyncMock()

    async def fake_check(s, conv_id):
        return "run-1"

    monkeypatch.setattr(mod, "_check_eval_run_completion", fake_check)

    with patch.object(mod, "EvaluationService", return_value=fake_service):
        out = mod.evaluate_conversation("conv-1", None)

    assert out["status"] == "completed"
    assert out["conversation_id"] == "conv-1"
    assert out["eval_run_id"] == "run-1"
    fake_service.evaluate_conversation.assert_awaited_once_with("conv-1", None)


def test_evaluate_conversation_returns_failed_on_exception(monkeypatch):
    from app.workers import evaluation_tasks as mod

    factory, session = _make_fake_session_factory()
    monkeypatch.setattr(mod, "_make_session_factory", lambda: factory)

    fake_service = MagicMock()
    fake_service.evaluate_conversation = AsyncMock(side_effect=RuntimeError("judge crashed"))

    with patch.object(mod, "EvaluationService", return_value=fake_service):
        out = mod.evaluate_conversation("conv-1")

    assert out["status"] == "failed"
    assert out["conversation_id"] == "conv-1"
    assert "judge crashed" in out["error"]
    session.rollback.assert_awaited()


# ---------------------------------------------------------------------------
# _check_eval_run_completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_completion_marks_run_when_all_evaluated(monkeypatch):
    from app.workers import evaluation_tasks as mod

    fake_run = SimpleNamespace(
        id="run-1", status="running_evaluation", completed_at=None
    )

    # Sequence of execute() calls inside _check_eval_run_completion:
    #   1. select Conversation.eval_run_id WHERE id = conv-1  -> ("run-1",)
    #   2. select count(Conversation) WHERE eval_run_id=run-1, status=completed -> 2
    #   3. select count(distinct Evaluation.conversation_id) ... -> 2
    #   4. select EvalRun WHERE id=run-1 -> fake_run
    eval_run_lookup = MagicMock()
    eval_run_lookup.first = MagicMock(return_value=("run-1",))

    count_total = MagicMock()
    count_total.scalar = MagicMock(return_value=2)

    count_evaluated = MagicMock()
    count_evaluated.scalar = MagicMock(return_value=2)

    run_select = MagicMock()
    run_select.scalar_one_or_none = MagicMock(return_value=fake_run)

    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=[eval_run_lookup, count_total, count_evaluated, run_select]
    )
    session.commit = AsyncMock()

    out = await mod._check_eval_run_completion(session, "conv-1")

    assert out == "run-1"
    assert fake_run.status == "completed"
    assert fake_run.completed_at is not None
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_completion_no_op_when_some_still_pending():
    from app.workers import evaluation_tasks as mod

    fake_run = SimpleNamespace(id="run-1", status="running_evaluation", completed_at=None)

    eval_run_lookup = MagicMock()
    eval_run_lookup.first = MagicMock(return_value=("run-1",))
    count_total = MagicMock()
    count_total.scalar = MagicMock(return_value=5)
    count_evaluated = MagicMock()
    count_evaluated.scalar = MagicMock(return_value=2)  # not all done

    session = MagicMock()
    session.execute = AsyncMock(side_effect=[eval_run_lookup, count_total, count_evaluated])
    session.commit = AsyncMock()

    out = await mod._check_eval_run_completion(session, "conv-1")

    # Returns eval_run_id but does NOT commit completion
    assert out == "run-1"
    assert fake_run.status == "running_evaluation"
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_completion_returns_none_when_conv_unknown():
    from app.workers import evaluation_tasks as mod

    no_row = MagicMock()
    no_row.first = MagicMock(return_value=None)

    session = MagicMock()
    session.execute = AsyncMock(return_value=no_row)

    out = await mod._check_eval_run_completion(session, "ghost")
    assert out is None
