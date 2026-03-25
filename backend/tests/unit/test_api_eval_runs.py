"""Unit tests for the eval_runs API route handlers.

All database operations and Celery tasks are mocked.
Tests verify HTTP status codes, response shapes, and business logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.eval_runs import cancel_eval_run, create_eval_run, get_eval_run, list_eval_runs
from app.core.exceptions import NotFoundError
from app.schemas.eval_run import EvalRunCreate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_eval_run(**overrides):
    """Create a mock EvalRun ORM object with from_attributes support."""
    now = datetime.now(timezone.utc)
    run = MagicMock()
    run.id = overrides.get("id", "run-001")
    run.name = overrides.get("name", "Test Run")
    run.agent_config_id = overrides.get("agent_config_id", "config-001")
    run.scenario_id = overrides.get("scenario_id", "scenario-001")
    run.rubric_id = overrides.get("rubric_id", None)
    run.status = overrides.get("status", "pending")
    run.num_conversations = overrides.get("num_conversations", 5)
    run.config = overrides.get("config", {})
    run.error_message = overrides.get("error_message", None)
    run.started_at = overrides.get("started_at", None)
    run.completed_at = overrides.get("completed_at", None)
    run.created_at = overrides.get("created_at", now)
    return run


def _mock_db_returning(items=None, scalar=MagicMock):
    """Create a mock AsyncSession that returns items from execute().

    Pass scalar=None explicitly to simulate a not-found case.
    The default sentinel (MagicMock class) means 'not specified'.
    """
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()

    mock_result = MagicMock()

    if scalar is not MagicMock:
        mock_result.scalar_one_or_none.return_value = scalar
        mock_result.scalar_one.return_value = scalar if scalar else 0
    if items is not None:
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = items
        mock_result.scalars.return_value = mock_scalars
        mock_result.scalar_one.return_value = len(items)

    db.execute = AsyncMock(return_value=mock_result)
    return db


# ---------------------------------------------------------------------------
# list_eval_runs
# ---------------------------------------------------------------------------


class TestListEvalRuns:
    """Tests for GET /api/v1/eval-runs."""

    @pytest.mark.asyncio
    async def test_list_returns_empty_when_no_runs(self):
        """Listing eval runs with no data returns empty items list."""
        db = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        items_result = MagicMock()
        items_result.scalars.return_value.all.return_value = []

        db.execute = AsyncMock(side_effect=[count_result, items_result])

        result = await list_eval_runs(offset=0, limit=20, status=None, agent_config_id=None, scenario_id=None, db=db)

        assert result.total == 0
        assert result.items == []
        assert result.offset == 0
        assert result.limit == 20

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self):
        """Filtering by status passes the filter to the query."""
        db = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        items_result = MagicMock()
        items_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(side_effect=[count_result, items_result])

        result = await list_eval_runs(offset=0, limit=20, status="running_simulation", agent_config_id=None, scenario_id=None, db=db)

        assert result.total == 0
        # DB should have been called twice (count + items)
        assert db.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_list_with_agent_config_filter(self):
        """Filtering by agent_config_id passes through correctly."""
        db = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        items_result = MagicMock()
        items_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(side_effect=[count_result, items_result])

        result = await list_eval_runs(offset=0, limit=10, status=None, agent_config_id="config-abc", scenario_id=None, db=db)

        assert result.limit == 10

    @pytest.mark.asyncio
    async def test_list_pagination_offset(self):
        """Offset and limit are preserved in the response."""
        db = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 50
        items_result = MagicMock()
        items_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(side_effect=[count_result, items_result])

        result = await list_eval_runs(offset=20, limit=10, status=None, agent_config_id=None, scenario_id=None, db=db)

        assert result.offset == 20
        assert result.limit == 10
        assert result.total == 50


# ---------------------------------------------------------------------------
# create_eval_run
# ---------------------------------------------------------------------------


class TestCreateEvalRun:
    """Tests for POST /api/v1/eval-runs."""

    @pytest.mark.asyncio
    @patch("app.api.v1.eval_runs.run_simulation")
    async def test_create_returns_pending_status(self, mock_task):
        """Creating an eval run returns status=pending and dispatches Celery task."""
        mock_task.delay = MagicMock()

        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        # After refresh, the eval_run object should have an id and created_at
        async def set_attrs(obj):
            obj.id = "run-new-001"
            obj.created_at = datetime.now(timezone.utc)

        db.refresh = AsyncMock(side_effect=set_attrs)

        body = EvalRunCreate(
            name="My Run",
            agent_config_id="config-001",
            scenario_id="scenario-001",
            num_conversations=3,
        )

        result = await create_eval_run(body=body, db=db)

        assert result.status == "pending"
        assert result.num_conversations == 3
        assert db.add.called
        assert db.flush.called
        mock_task.delay.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.api.v1.eval_runs.run_simulation")
    async def test_create_with_default_num_conversations(self, mock_task):
        """Default num_conversations is 5 when not specified."""
        mock_task.delay = MagicMock()

        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        async def set_attrs(obj):
            obj.id = "run-new-002"
            obj.created_at = datetime.now(timezone.utc)

        db.refresh = AsyncMock(side_effect=set_attrs)

        body = EvalRunCreate(
            agent_config_id="config-001",
            scenario_id="scenario-001",
        )

        result = await create_eval_run(body=body, db=db)

        assert result.num_conversations == 5

    @pytest.mark.asyncio
    @patch("app.api.v1.eval_runs.run_simulation")
    async def test_create_with_rubric_id(self, mock_task):
        """Creating an eval run with rubric_id stores it."""
        mock_task.delay = MagicMock()

        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        async def set_attrs(obj):
            obj.id = "run-new-003"
            obj.created_at = datetime.now(timezone.utc)

        db.refresh = AsyncMock(side_effect=set_attrs)

        body = EvalRunCreate(
            agent_config_id="config-001",
            scenario_id="scenario-001",
            rubric_id="rubric-001",
        )

        result = await create_eval_run(body=body, db=db)

        assert result.rubric_id == "rubric-001"


# ---------------------------------------------------------------------------
# get_eval_run
# ---------------------------------------------------------------------------


class TestGetEvalRun:
    """Tests for GET /api/v1/eval-runs/{run_id}."""

    @pytest.mark.asyncio
    async def test_get_returns_existing_run(self):
        """Getting an existing eval run returns its data."""
        mock_run = _make_db_eval_run(id="run-existing")
        db = _mock_db_returning(scalar=mock_run)

        result = await get_eval_run(run_id="run-existing", db=db)

        assert result.id == "run-existing"
        assert result.status == "pending"

    @pytest.mark.asyncio
    async def test_get_nonexistent_raises_not_found(self):
        """Getting a nonexistent eval run raises NotFoundError."""
        db = _mock_db_returning(scalar=None)

        with pytest.raises(NotFoundError):
            await get_eval_run(run_id="nonexistent", db=db)


# ---------------------------------------------------------------------------
# cancel_eval_run
# ---------------------------------------------------------------------------


class TestCancelEvalRun:
    """Tests for POST /api/v1/eval-runs/{run_id}/cancel."""

    @pytest.mark.asyncio
    async def test_cancel_pending_run_transitions_to_cancelled(self):
        """Cancelling a pending run sets status to cancelled."""
        mock_run = _make_db_eval_run(status="pending")
        db = _mock_db_returning(scalar=mock_run)

        result = await cancel_eval_run(run_id="run-001", db=db)

        assert mock_run.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_running_simulation_transitions(self):
        """Cancelling a running_simulation run sets status to cancelled."""
        mock_run = _make_db_eval_run(status="running_simulation")
        db = _mock_db_returning(scalar=mock_run)

        await cancel_eval_run(run_id="run-001", db=db)

        assert mock_run.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_running_evaluation_transitions(self):
        """Cancelling a running_evaluation run sets status to cancelled."""
        mock_run = _make_db_eval_run(status="running_evaluation")
        db = _mock_db_returning(scalar=mock_run)

        await cancel_eval_run(run_id="run-001", db=db)

        assert mock_run.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_completed_run_no_transition(self):
        """Cancelling a completed run does NOT change status."""
        mock_run = _make_db_eval_run(status="completed")
        db = _mock_db_returning(scalar=mock_run)

        await cancel_eval_run(run_id="run-001", db=db)

        assert mock_run.status == "completed"

    @pytest.mark.asyncio
    async def test_cancel_failed_run_no_transition(self):
        """Cancelling a failed run does NOT change status."""
        mock_run = _make_db_eval_run(status="failed")
        db = _mock_db_returning(scalar=mock_run)

        await cancel_eval_run(run_id="run-001", db=db)

        assert mock_run.status == "failed"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_raises_not_found(self):
        """Cancelling a nonexistent run raises NotFoundError."""
        db = _mock_db_returning(scalar=None)

        with pytest.raises(NotFoundError):
            await cancel_eval_run(run_id="nonexistent", db=db)
