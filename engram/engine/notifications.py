"""Namespace change notifications via Redis pub/sub.

Generic fire-and-forget broadcast — Engram publishes, doesn't know who listens.
Consumers (e.g. Enmotiv) subscribe to ``engram:namespace_changed`` for
dirty-namespace tracking.
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

from engram.config import settings

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None


async def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


async def notify_namespace_changed(namespace: str) -> None:
    """Broadcast that data changed in a namespace. Fire-and-forget."""
    try:
        r = await _get_redis()
        await r.publish("engram:namespace_changed", namespace)
    except Exception:
        logger.debug("namespace_notification_failed", extra={"namespace": namespace})
