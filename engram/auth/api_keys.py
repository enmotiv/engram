"""API key authentication and per-key rate limiting."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from uuid import UUID

import asyncpg
import structlog
from fastapi import Request

from engram.core.errors import EngramError
from engram.core.tracing import set_correlation_id, set_owner_id

logger = structlog.get_logger()

# Redis cache TTLs
_AUTH_CACHE_TTL = 300  # 5 minutes for valid keys
_AUTH_NEGATIVE_TTL = 30  # 30 seconds for invalid keys
_AUTH_CACHE_PREFIX = "engram:auth:"


def hash_api_key(raw_key: str) -> str:
    """SHA-256 hash of a plaintext API key. Never store or log the raw key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# --- Rate Limiter (in-memory sliding window) ---


class RateLimiter:
    """Per-key sliding-window rate limiter. In-memory; resets on restart."""

    def __init__(self) -> None:
        self._windows: dict[str, list[float]] = defaultdict(list)

    def check(
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


_rate_limiter = RateLimiter()


# --- Redis auth cache helpers ---


async def _cache_get(redis, key_hash: str) -> dict | None:
    """Try Redis cache for auth data. Returns None on miss or error."""
    if redis is None:
        return None
    try:
        raw = await redis.get(f"{_AUTH_CACHE_PREFIX}{key_hash}")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


async def _cache_set(redis, key_hash: str, data: dict | None) -> None:
    """Cache auth result in Redis. data=None means negative cache."""
    if redis is None:
        return
    try:
        if data is None:
            await redis.set(
                f"{_AUTH_CACHE_PREFIX}{key_hash}",
                json.dumps({"invalid": True}),
                ex=_AUTH_NEGATIVE_TTL,
            )
        else:
            await redis.set(
                f"{_AUTH_CACHE_PREFIX}{key_hash}",
                json.dumps(data),
                ex=_AUTH_CACHE_TTL,
            )
    except Exception:
        pass  # Cache is optional — DB is always authoritative


async def invalidate_auth_cache(redis, key_hash: str) -> None:
    """Remove a key from the auth cache (e.g., on revocation)."""
    if redis is None:
        return
    try:
        await redis.delete(f"{_AUTH_CACHE_PREFIX}{key_hash}")
    except Exception:
        pass


# --- FastAPI Dependency ---


async def get_owner_id(request: Request) -> UUID:
    """Validate API key, enforce rate limit, return owner_id.

    Checks Redis cache before hitting the DB. Stores rate_info on
    request.state for the response-header middleware.
    """
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise EngramError(
            "UNAUTHORIZED", "Missing or invalid Authorization header", 401
        )

    raw_key = auth_header[7:]
    if not raw_key:
        raise EngramError("UNAUTHORIZED", "Missing API key", 401)

    key_hash = hash_api_key(raw_key)

    # Try Redis cache first
    redis = getattr(request.app.state, "redis", None)
    cached = await _cache_get(redis, key_hash)

    if cached is not None:
        if cached.get("invalid"):
            raise EngramError("UNAUTHORIZED", "Invalid API key", 401)
        owner_id = UUID(cached["owner_id"])
        rate_limit_writes = cached["rate_limit_writes"]
        rate_limit_reads = cached["rate_limit_reads"]
    else:
        # Fall through to DB
        db: asyncpg.Pool = request.app.state.db
        async with db.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT owner_id, rate_limit_writes, rate_limit_reads "
                "FROM api_keys "
                "WHERE key_hash = $1 AND revoked_at IS NULL",
                key_hash,
            )

        if not row:
            await _cache_set(redis, key_hash, None)  # Negative cache
            raise EngramError("UNAUTHORIZED", "Invalid API key", 401)

        owner_id = row["owner_id"]
        rate_limit_writes = row["rate_limit_writes"]
        rate_limit_reads = row["rate_limit_reads"]

        # Populate cache
        await _cache_set(redis, key_hash, {
            "owner_id": str(owner_id),
            "rate_limit_writes": rate_limit_writes,
            "rate_limit_reads": rate_limit_reads,
        })

    # Set ContextVars for structured logging
    cid = getattr(request.state, "correlation_id", "")
    if cid:
        set_correlation_id(cid)
    set_owner_id(str(owner_id))

    # Determine bucket + limit by HTTP method
    method = request.method.upper()
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        limit = rate_limit_writes
        bucket = f"write:{key_hash}"
    else:
        limit = rate_limit_reads
        bucket = f"read:{key_hash}"

    allowed, remaining, reset = _rate_limiter.check(bucket, limit)
    request.state.rate_info = {
        "remaining": remaining,
        "reset": reset,
        "limit": limit,
    }

    if not allowed:
        retry_after = max(1, reset - int(time.time()))
        raise EngramError(
            "RATE_LIMITED", "Too many requests", 429, retry_after=retry_after
        )

    return owner_id
