"""Rate limiting: in-memory (default) and Redis-backed (multi-instance)."""

from __future__ import annotations

import time
from collections import defaultdict

import structlog

logger = structlog.get_logger()


class InMemoryRateLimiter:
    """Per-key sliding-window rate limiter. Resets on restart."""

    def __init__(self) -> None:
        self._windows: dict[str, list[float]] = defaultdict(list)

    async def check(
        self, bucket: str, limit: int, window: int = 60
    ) -> tuple[bool, int, int]:
        """Returns (allowed, remaining, reset_unix)."""
        now = time.time()
        cutoff = now - window
        self._windows[bucket] = [t for t in self._windows[bucket] if t > cutoff]
        current = len(self._windows[bucket])
        reset = int(now) + window

        if current >= limit:
            return False, 0, reset

        self._windows[bucket].append(now)
        remaining = max(0, limit - current - 1)
        return True, remaining, reset


class RedisRateLimiter:
    """Per-key sliding-window rate limiter backed by Redis.

    Survives restarts and works across multiple instances.
    Uses Redis sorted sets with timestamp scores.
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
            )
        return self._redis

    async def check(
        self, bucket: str, limit: int, window: int = 60
    ) -> tuple[bool, int, int]:
        """Returns (allowed, remaining, reset_unix)."""
        now = time.time()
        cutoff = now - window
        reset = int(now) + window
        key = f"engram:rate:{bucket}"

        try:
            r = await self._get_redis()

            pipe = r.pipeline()
            pipe.zremrangebyscore(key, 0, cutoff)
            pipe.zcard(key)
            results = await pipe.execute()
            current = results[1]

            if current >= limit:
                return False, 0, reset

            pipe2 = r.pipeline()
            pipe2.zadd(key, {str(now): now})
            pipe2.expire(key, window + 1)
            await pipe2.execute()

            remaining = max(0, limit - current - 1)
            return True, remaining, reset
        except Exception:
            logger.warning(
                "rate_limiter.redis_fallback",
                bucket=bucket,
                error="Redis unavailable, allowing request",
            )
            return True, limit - 1, reset


async def get_rate_limiter(redis_url: str = "") -> InMemoryRateLimiter | RedisRateLimiter:
    """Factory: returns Redis limiter if redis_url is set, else in-memory."""
    if redis_url:
        try:
            limiter = RedisRateLimiter(redis_url)
            r = await limiter._get_redis()
            await r.ping()
            logger.info("rate_limiter.using_redis")
            return limiter
        except Exception:
            logger.warning("rate_limiter.redis_unavailable_using_memory")
    return InMemoryRateLimiter()
