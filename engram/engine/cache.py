"""Trace cache backed by Redis."""

from __future__ import annotations

import hashlib
import uuid
from typing import List, Optional

import redis.asyncio as redis


class TraceCache:
    """Cache retrieval traces in Redis keyed by namespace + memory IDs."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis = redis.from_url(redis_url, decode_responses=True)

    @staticmethod
    def compute_cache_key(namespace: str, memory_ids: List[str]) -> str:
        """Hash based on namespace + sorted memory IDs."""
        sorted_ids = sorted(memory_ids)
        raw = f"{namespace}:{','.join(sorted_ids)}"
        return f"engram:trace:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

    async def get(self, namespace: str, cache_key: str) -> Optional[str]:
        """Check Redis for cached trace."""
        return await self._redis.get(cache_key)

    async def set(
        self, namespace: str, cache_key: str, trace: str, ttl_seconds: int = 300
    ) -> None:
        """Cache trace with TTL."""
        await self._redis.set(cache_key, trace, ex=ttl_seconds)

    async def close(self) -> None:
        """Close Redis connection."""
        await self._redis.aclose()
