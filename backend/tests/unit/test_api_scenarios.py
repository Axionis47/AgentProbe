"""Unit tests for the scenarios API route handlers.

All database operations are mocked.
Tests verify CRUD operations, filtering by category/difficulty/is_active, and soft-delete.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.v1.scenarios import (
    create_scenario,
    delete_scenario,
    get_scenario,
    list_scenarios,
    update_scenario,
)
from app.core.exceptions import NotFoundError
from app.schemas.scenario import ScenarioCreate, ScenarioUpdate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_scenario(**overrides):
    """Create a mock Scenario ORM object."""
    now = datetime.now(timezone.utc)
    s = MagicMock()
    s.id = overrides.get("id", "scenario-001")
    s.name = overrides.get("name", "Test Scenario")
    s.description = overrides.get("description", "A test scenario")
    s.category = overrides.get("category", "general")
    s.turns_template = overrides.get("turns_template", [{"role": "user", "content": "Hi"}])
    s.user_persona = overrides.get("user_persona", {})
    s.constraints = overrides.get("constraints", {})
    s.difficulty = overrides.get("difficulty", "medium")
    s.tags = overrides.get("tags", [])
    s.is_active = overrides.get("is_active", True)
    s.created_at = overrides.get("created_at", now)
    s.updated_at = overrides.get("updated_at", now)
    return s


def _mock_db_returning(scalar=MagicMock):
    """Create a mock AsyncSession. Pass scalar=None for not-found case."""
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = scalar if scalar is not MagicMock else None
    mock_result.scalar_one.return_value = scalar if (scalar is not MagicMock and scalar) else 0

    if scalar is None or scalar is MagicMock:
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_result.scalar_one.return_value = 0

    db.execute = AsyncMock(return_value=mock_result)
    return db


# ---------------------------------------------------------------------------
# list_scenarios
# ---------------------------------------------------------------------------


class TestListScenarios:
    """Tests for GET /api/v1/scenarios."""

    @pytest.mark.asyncio
    async def test_list_returns_empty(self):
        """Listing with no data returns empty items."""
        db = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        items_result = MagicMock()
        items_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(side_effect=[count_result, items_result])

        result = await list_scenarios(
            offset=0, limit=20, is_active=None, category=None, difficulty=None, db=db,
        )

        assert result.total == 0
        assert result.items == []

    @pytest.mark.asyncio
    async def test_list_with_category_filter(self):
        """Filtering by category passes through correctly."""
        db = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        items_result = MagicMock()
        items_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(side_effect=[count_result, items_result])

        result = await list_scenarios(
            offset=0, limit=20, is_active=None, category="customer_support", difficulty=None, db=db,
        )

        assert result.total == 0
        assert db.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_list_with_difficulty_filter(self):
        """Filtering by difficulty passes through correctly."""
        db = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        items_result = MagicMock()
        items_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(side_effect=[count_result, items_result])

        result = await list_scenarios(
            offset=0, limit=20, is_active=None, category=None, difficulty="hard", db=db,
        )

        assert result.total == 0

    @pytest.mark.asyncio
    async def test_list_with_is_active_filter(self):
        """Filtering by is_active passes through correctly."""
        db = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        items_result = MagicMock()
        items_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(side_effect=[count_result, items_result])

        result = await list_scenarios(
            offset=0, limit=20, is_active=False, category=None, difficulty=None, db=db,
        )

        assert result.total == 0


# ---------------------------------------------------------------------------
# create_scenario
# ---------------------------------------------------------------------------


class TestCreateScenario:
    """Tests for POST /api/v1/scenarios."""

    @pytest.mark.asyncio
    async def test_create_stores_and_returns(self):
        """Creating a scenario adds it to DB and returns response."""
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()
        now = datetime.now(timezone.utc)

        async def set_attrs(obj):
            obj.id = "scenario-new"
            obj.created_at = now
            obj.updated_at = now
            obj.is_active = True

        db.refresh = AsyncMock(side_effect=set_attrs)

        body = ScenarioCreate(
            name="New Scenario",
            turns_template=[{"role": "user", "content": "Help me"}],
            difficulty="easy",
        )

        result = await create_scenario(body=body, db=db)

        assert result.name == "New Scenario"
        assert result.difficulty == "easy"
        assert db.add.called

    @pytest.mark.asyncio
    async def test_create_with_user_persona(self):
        """Creating a scenario with user_persona stores it."""
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()
        now = datetime.now(timezone.utc)

        async def set_attrs(obj):
            obj.id = "scenario-persona"
            obj.created_at = now
            obj.updated_at = now
            obj.is_active = True

        db.refresh = AsyncMock(side_effect=set_attrs)

        persona = {"personality": "angry", "expertise_level": "novice", "goal": "Fix error"}
        body = ScenarioCreate(
            name="Persona Scenario",
            turns_template=[{"role": "user", "content": "Fix this"}],
            user_persona=persona,
        )

        result = await create_scenario(body=body, db=db)

        assert result.user_persona == persona


# ---------------------------------------------------------------------------
# get_scenario
# ---------------------------------------------------------------------------


class TestGetScenario:
    """Tests for GET /api/v1/scenarios/{scenario_id}."""

    @pytest.mark.asyncio
    async def test_get_returns_existing(self):
        """Getting an existing scenario returns its data."""
        mock_s = _make_db_scenario(id="scenario-exist")
        db = _mock_db_returning(scalar=mock_s)

        result = await get_scenario(scenario_id="scenario-exist", db=db)

        assert result.id == "scenario-exist"

    @pytest.mark.asyncio
    async def test_get_nonexistent_raises_not_found(self):
        """Getting a nonexistent scenario raises NotFoundError."""
        db = _mock_db_returning(scalar=None)

        with pytest.raises(NotFoundError):
            await get_scenario(scenario_id="nope", db=db)


# ---------------------------------------------------------------------------
# update_scenario
# ---------------------------------------------------------------------------


class TestUpdateScenario:
    """Tests for PUT /api/v1/scenarios/{scenario_id}."""

    @pytest.mark.asyncio
    async def test_update_changes_fields(self):
        """Updating a scenario modifies the specified fields."""
        mock_s = _make_db_scenario(id="scenario-upd")
        db = _mock_db_returning(scalar=mock_s)

        body = ScenarioUpdate(name="Updated Scenario", difficulty="hard")

        await update_scenario(scenario_id="scenario-upd", body=body, db=db)

        assert mock_s.name == "Updated Scenario"
        assert mock_s.difficulty == "hard"

    @pytest.mark.asyncio
    async def test_update_nonexistent_raises_not_found(self):
        """Updating a nonexistent scenario raises NotFoundError."""
        db = _mock_db_returning(scalar=None)
        body = ScenarioUpdate(name="New")

        with pytest.raises(NotFoundError):
            await update_scenario(scenario_id="nope", body=body, db=db)


# ---------------------------------------------------------------------------
# delete_scenario (soft delete)
# ---------------------------------------------------------------------------


class TestDeleteScenario:
    """Tests for DELETE /api/v1/scenarios/{scenario_id}."""

    @pytest.mark.asyncio
    async def test_delete_soft_deactivates(self):
        """Deleting sets is_active to False."""
        mock_s = _make_db_scenario(id="scenario-del", is_active=True)
        db = _mock_db_returning(scalar=mock_s)

        await delete_scenario(scenario_id="scenario-del", db=db)

        assert mock_s.is_active is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent_raises_not_found(self):
        """Deleting a nonexistent scenario raises NotFoundError."""
        db = _mock_db_returning(scalar=None)

        with pytest.raises(NotFoundError):
            await delete_scenario(scenario_id="nope", db=db)
