"""Tests for the health API route handlers.

Liveness is trivial — it should never fail. Readiness ought to:
- Return 'ready' when both DB and Redis ping cleanly.
- Return 'degraded' (not 500) when one of them errors out, so a load
  balancer can keep the app in rotation while ops investigate.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.health import liveness, readiness


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_liveness_always_returns_ok():
    out = await liveness()
    assert out == {"status": "ok"}


# ---------------------------------------------------------------------------
# /health/ready
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_readiness_returns_ready_when_all_dependencies_ok():
    db = MagicMock()
    db.execute = AsyncMock(return_value=None)

    fake_redis = MagicMock()
    fake_redis.ping = AsyncMock(return_value=True)

    with patch("app.api.v1.health.get_redis", AsyncMock(return_value=fake_redis)):
        out = await readiness(db=db)

    assert out["status"] == "ready"
    assert out["checks"]["postgres"] == "ok"
    assert out["checks"]["redis"] == "ok"


@pytest.mark.asyncio
async def test_readiness_returns_degraded_when_postgres_unavailable():
    db = MagicMock()
    db.execute = AsyncMock(side_effect=RuntimeError("postgres unreachable"))

    fake_redis = MagicMock()
    fake_redis.ping = AsyncMock(return_value=True)

    with patch("app.api.v1.health.get_redis", AsyncMock(return_value=fake_redis)):
        out = await readiness(db=db)

    assert out["status"] == "degraded"
    assert out["checks"]["postgres"].startswith("error:")
    assert out["checks"]["redis"] == "ok"


@pytest.mark.asyncio
async def test_readiness_returns_degraded_when_redis_unavailable():
    db = MagicMock()
    db.execute = AsyncMock(return_value=None)

    fake_redis = MagicMock()
    fake_redis.ping = AsyncMock(side_effect=ConnectionError("redis down"))

    with patch("app.api.v1.health.get_redis", AsyncMock(return_value=fake_redis)):
        out = await readiness(db=db)

    assert out["status"] == "degraded"
    assert out["checks"]["postgres"] == "ok"
    assert out["checks"]["redis"].startswith("error:")
