"""Tests for the rubrics API route handlers.

Database is mocked. Mirrors the pattern in test_api_agent_configs.py.
Rubrics are immutable in this design — PUT creates a new versioned row
with parent_id pointing at the previous one — so the update tests check
that the new row carries an incremented version, not that the old row
mutated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.v1.rubrics import (
    create_rubric,
    delete_rubric,
    get_rubric,
    list_rubric_versions,
    list_rubrics,
    update_rubric,
)
from app.core.exceptions import NotFoundError
from app.schemas.rubric import RubricCreate, RubricUpdate


def _make_rubric_row(
    rubric_id: str = "r1",
    name: str = "default",
    version: int = 1,
    is_active: bool = True,
    parent_id: str | None = None,
) -> SimpleNamespace:
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=rubric_id,
        name=name,
        description=None,
        dimensions=[{"name": "helpfulness", "weight": 1.0}],
        version=version,
        parent_id=parent_id,
        is_active=is_active,
        created_at=now,
    )


def _mock_list_db(items: list, total: int | None = None):
    if total is None:
        total = len(items)
    db = MagicMock()
    count_result = MagicMock()
    count_result.scalar_one = MagicMock(return_value=total)
    list_result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=items)
    list_result.scalars = MagicMock(return_value=scalars)
    db.execute = AsyncMock(side_effect=[count_result, list_result])
    return db


# ---------------------------------------------------------------------------
# list_rubrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_rubrics_returns_empty_when_none():
    db = _mock_list_db([], total=0)
    out = await list_rubrics(offset=0, limit=20, is_active=None, db=db)
    assert out.total == 0
    assert out.items == []


@pytest.mark.asyncio
async def test_list_rubrics_with_is_active_filter():
    rows = [_make_rubric_row("r1"), _make_rubric_row("r2")]
    db = _mock_list_db(rows, total=2)
    out = await list_rubrics(offset=0, limit=20, is_active=True, db=db)
    assert out.total == 2
    assert len(out.items) == 2


# ---------------------------------------------------------------------------
# create_rubric
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_rubric_persists_version_one():
    db = MagicMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    async def fake_refresh(rubric):
        rubric.id = "newly-created"
        rubric.is_active = True
        rubric.parent_id = None
        rubric.created_at = datetime.now(timezone.utc)

    db.refresh = AsyncMock(side_effect=fake_refresh)

    body = RubricCreate(
        name="My Rubric",
        description="for testing",
        dimensions=[{"name": "x", "weight": 1.0}],
    )
    out = await create_rubric(body=body, db=db)

    db.add.assert_called_once()
    persisted = db.add.call_args.args[0]
    assert persisted.version == 1
    assert persisted.name == "My Rubric"
    assert out.version == 1


# ---------------------------------------------------------------------------
# get_rubric
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_rubric_returns_row():
    row = _make_rubric_row("r1")
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=row)
    db.execute = AsyncMock(return_value=result)

    out = await get_rubric(rubric_id="r1", db=db)
    assert out.id == "r1"


@pytest.mark.asyncio
async def test_get_rubric_404_when_missing():
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    db.execute = AsyncMock(return_value=result)

    with pytest.raises(NotFoundError):
        await get_rubric(rubric_id="missing", db=db)


# ---------------------------------------------------------------------------
# update_rubric (creates a new versioned row, doesn't mutate the old one)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_rubric_creates_new_version():
    old = _make_rubric_row("r-old", name="My Rubric", version=2)

    db = MagicMock()
    lookup_result = MagicMock()
    lookup_result.scalar_one_or_none = MagicMock(return_value=old)
    db.execute = AsyncMock(return_value=lookup_result)
    db.add = MagicMock()
    db.flush = AsyncMock()

    async def fake_refresh(new_rubric):
        new_rubric.id = "r-new"
        new_rubric.is_active = True
        new_rubric.created_at = datetime.now(timezone.utc)

    db.refresh = AsyncMock(side_effect=fake_refresh)

    body = RubricUpdate(name="My Rubric v3", dimensions=[{"name": "n", "weight": 1.0}])
    out = await update_rubric(rubric_id="r-old", body=body, db=db)

    persisted = db.add.call_args.args[0]
    assert persisted.version == 3
    assert persisted.parent_id == "r-old"
    assert persisted.name == "My Rubric v3"
    assert out.version == 3


@pytest.mark.asyncio
async def test_update_rubric_404_when_missing():
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    db.execute = AsyncMock(return_value=result)

    with pytest.raises(NotFoundError):
        await update_rubric(rubric_id="ghost", body=RubricUpdate(), db=db)


# ---------------------------------------------------------------------------
# list_rubric_versions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_versions_returns_chain_ordered_by_version_desc():
    head = _make_rubric_row("v3", name="Same", version=3, parent_id="v2")
    versions = [
        _make_rubric_row("v3", name="Same", version=3, parent_id="v2"),
        _make_rubric_row("v2", name="Same", version=2, parent_id="v1"),
        _make_rubric_row("v1", name="Same", version=1, parent_id=None),
    ]

    db = MagicMock()
    head_result = MagicMock()
    head_result.scalar_one_or_none = MagicMock(return_value=head)
    list_result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=versions)
    list_result.scalars = MagicMock(return_value=scalars)
    db.execute = AsyncMock(side_effect=[head_result, list_result])

    out = await list_rubric_versions(rubric_id="v3", db=db)
    assert [r.version for r in out] == [3, 2, 1]


@pytest.mark.asyncio
async def test_list_versions_404_when_unknown():
    db = MagicMock()
    head_result = MagicMock()
    head_result.scalar_one_or_none = MagicMock(return_value=None)
    db.execute = AsyncMock(return_value=head_result)

    with pytest.raises(NotFoundError):
        await list_rubric_versions(rubric_id="ghost", db=db)


# ---------------------------------------------------------------------------
# delete_rubric — soft delete (is_active flip), 204 with no body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_rubric_soft_deactivates():
    row = _make_rubric_row("r1", is_active=True)
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=row)
    db.execute = AsyncMock(return_value=result)
    db.flush = AsyncMock()

    out = await delete_rubric(rubric_id="r1", db=db)
    assert out is None
    assert row.is_active is False


@pytest.mark.asyncio
async def test_delete_rubric_404_when_unknown():
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    db.execute = AsyncMock(return_value=result)

    with pytest.raises(NotFoundError):
        await delete_rubric(rubric_id="ghost", db=db)
