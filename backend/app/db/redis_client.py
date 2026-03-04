import redis.asyncio as redis

from app.config import settings

redis_pool = redis.from_url(
    settings.redis_url,
    decode_responses=True,
    max_connections=20,
)


async def get_redis() -> redis.Redis:  # type: ignore[type-arg]
    return redis_pool
