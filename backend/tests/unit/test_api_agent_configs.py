"""Unit tests for the agent_configs API route handlers.

All database operations are mocked.
Tests verify CRUD operations, filtering, and validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.v1.agent_configs import (
    create_agent_config,
    delete_agent_config,
    get_agent_config,
    list_agent_configs,
    update_agent_config,
)
from app.core.exceptions import NotFoundError
from app.schemas.agent_config import AgentConfigCreate, AgentConfigUpdate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_agent_config(**overrides):
    """Create a mock AgentConfig ORM object."""
    now = datetime.now(timezone.utc)
    cfg = MagicMock()
    cfg.id = overrides.get("id", "config-001")
    cfg.name = overrides.get("name", "Test Agent")
    cfg.description = overrides.get("description", "A test agent")
    cfg.system_prompt = overrides.get("system_prompt", "You are helpful.")
    cfg.model = overrides.get("model", "claude-sonnet-4-20250514")
    cfg.temperature = overrides.get("temperature", 0.7)
    cfg.max_tokens = overrides.get("max_tokens", 4096)
    cfg.tools = overrides.get("tools", [])
    cfg.metadata_ = overrides.get("metadata_", {})
    cfg.metadata = overrides.get("metadata_", {})
    cfg.agent_type = overrides.get("agent_type", "builtin")
    cfg.endpoint_url = overrides.get("endpoint_url", None)
    cfg.is_active = overrides.get("is_active", True)
    cfg.created_at = overrides.get("created_at", now)
    cfg.updated_at = overrides.get("updated_at", now)
    return cfg


def _mock_db_returning(scalar=MagicMock):
    """Create a mock AsyncSession that returns a scalar from execute().

    Pass scalar=None explicitly to simulate a not-found case.
    """
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
# list_agent_configs
# ---------------------------------------------------------------------------


class TestListAgentConfigs:
    """Tests for GET /api/v1/agent-configs."""

    @pytest.mark.asyncio
    async def test_list_returns_empty(self):
        """Listing configs with no data returns an empty list."""
        db = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        items_result = MagicMock()
        items_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(side_effect=[count_result, items_result])

        result = await list_agent_configs(offset=0, limit=20, is_active=None, model=None, db=db)

        assert result.total == 0
        assert result.items == []

    @pytest.mark.asyncio
    async def test_list_with_is_active_filter(self):
        """Filtering by is_active passes through to the query."""
        db = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        items_result = MagicMock()
        items_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(side_effect=[count_result, items_result])

        result = await list_agent_configs(offset=0, limit=20, is_active=True, model=None, db=db)

        assert result.total == 0
        assert db.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_list_with_model_filter(self):
        """Filtering by model passes through to the query."""
        db = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        items_result = MagicMock()
        items_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(side_effect=[count_result, items_result])

        result = await list_agent_configs(offset=0, limit=20, is_active=None, model="gpt-4", db=db)

        assert result.total == 0


# ---------------------------------------------------------------------------
# create_agent_config
# ---------------------------------------------------------------------------


class TestCreateAgentConfig:
    """Tests for POST /api/v1/agent-configs."""

    @pytest.mark.asyncio
    async def test_create_stores_and_returns(self):
        """Creating a config adds it to DB and returns response."""
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        now = datetime.now(timezone.utc)

        async def set_attrs(obj):
            obj.id = "config-new"
            obj.created_at = now
            obj.updated_at = now
            obj.is_active = True
            obj.metadata = {}

        db.refresh = AsyncMock(side_effect=set_attrs)

        body = AgentConfigCreate(
            name="My Agent",
            system_prompt="You are helpful.",
            model="claude-sonnet-4-20250514",
        )

        result = await create_agent_config(body=body, db=db)

        assert result.name == "My Agent"
        assert result.is_active is True
        assert db.add.called

    @pytest.mark.asyncio
    async def test_create_with_tools(self):
        """Creating a config with tools stores them."""
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        now = datetime.now(timezone.utc)

        async def set_attrs(obj):
            obj.id = "config-tools"
            obj.created_at = now
            obj.updated_at = now
            obj.is_active = True
            obj.metadata = {}

        db.refresh = AsyncMock(side_effect=set_attrs)

        tools = [{"name": "search", "description": "Search tool", "parameters": {}}]
        body = AgentConfigCreate(
            name="Tool Agent",
            system_prompt="You have tools.",
            tools=tools,
        )

        result = await create_agent_config(body=body, db=db)

        assert result.tools == tools


# ---------------------------------------------------------------------------
# get_agent_config
# ---------------------------------------------------------------------------


class TestGetAgentConfig:
    """Tests for GET /api/v1/agent-configs/{config_id}."""

    @pytest.mark.asyncio
    async def test_get_returns_existing_config(self):
        """Getting an existing config returns its data."""
        mock_cfg = _make_db_agent_config(id="config-exist")
        db = _mock_db_returning(scalar=mock_cfg)

        result = await get_agent_config(config_id="config-exist", db=db)

        assert result.id == "config-exist"

    @pytest.mark.asyncio
    async def test_get_nonexistent_raises_not_found(self):
        """Getting a nonexistent config raises NotFoundError."""
        db = _mock_db_returning(scalar=None)

        with pytest.raises(NotFoundError):
            await get_agent_config(config_id="nope", db=db)


# ---------------------------------------------------------------------------
# update_agent_config
# ---------------------------------------------------------------------------


class TestUpdateAgentConfig:
    """Tests for PUT /api/v1/agent-configs/{config_id}."""

    @pytest.mark.asyncio
    async def test_update_changes_fields(self):
        """Updating a config modifies the specified fields."""
        mock_cfg = _make_db_agent_config(id="config-upd")
        db = _mock_db_returning(scalar=mock_cfg)

        body = AgentConfigUpdate(name="Updated Name")

        result = await update_agent_config(config_id="config-upd", body=body, db=db)

        assert mock_cfg.name == "Updated Name"

    @pytest.mark.asyncio
    async def test_update_nonexistent_raises_not_found(self):
        """Updating a nonexistent config raises NotFoundError."""
        db = _mock_db_returning(scalar=None)
        body = AgentConfigUpdate(name="New")

        with pytest.raises(NotFoundError):
            await update_agent_config(config_id="nope", body=body, db=db)

    @pytest.mark.asyncio
    async def test_update_metadata_renames_to_metadata_(self):
        """The 'metadata' field in the update body maps to 'metadata_' on the model."""
        mock_cfg = _make_db_agent_config(id="config-meta")
        db = _mock_db_returning(scalar=mock_cfg)

        body = AgentConfigUpdate(metadata={"key": "value"})

        await update_agent_config(config_id="config-meta", body=body, db=db)

        assert mock_cfg.metadata_ == {"key": "value"}


# ---------------------------------------------------------------------------
# delete_agent_config (soft delete)
# ---------------------------------------------------------------------------


class TestDeleteAgentConfig:
    """Tests for DELETE /api/v1/agent-configs/{config_id}."""

    @pytest.mark.asyncio
    async def test_delete_soft_deactivates(self):
        """Deleting a config sets is_active to False (soft delete)."""
        mock_cfg = _make_db_agent_config(id="config-del", is_active=True)
        db = _mock_db_returning(scalar=mock_cfg)

        await delete_agent_config(config_id="config-del", db=db)

        assert mock_cfg.is_active is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent_raises_not_found(self):
        """Deleting a nonexistent config raises NotFoundError."""
        db = _mock_db_returning(scalar=None)

        with pytest.raises(NotFoundError):
            await delete_agent_config(config_id="nope", db=db)
