import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.redis_client import get_redis
from app.db.session import get_db

router = APIRouter(tags=["health"])
logger = structlog.get_logger()


@router.get("/health")
async def liveness() -> dict[str, str]:
    """Basic liveness check."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Readiness check verifying all dependencies."""
    checks: dict[str, object] = {}

    # PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        logger.error("postgres_health_check_failed", error=str(e))
        checks["postgres"] = f"error: {e}"

    # Redis
    try:
        redis = await get_redis()
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        logger.error("redis_health_check_failed", error=str(e))
        checks["redis"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
    }
