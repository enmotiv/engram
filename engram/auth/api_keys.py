"""API key authentication and per-key rate limiting."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from uuid import UUID

import asyncpg
import structlog
from fastapi import Request

from engram.core.errors import EngramError
from engram.core.tracing import set_correlation_id, set_owner_id

logger = structlog.get_logger()


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


# --- FastAPI Dependency ---


async def get_owner_id(request: Request) -> UUID:
    """Validate API key, enforce rate limit, return owner_id.

    Stores rate_info on request.state for the response-header middleware.
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

    db: asyncpg.Pool = request.app.state.db
    async with db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT owner_id, rate_limit_writes, rate_limit_reads "
            "FROM api_keys "
            "WHERE key_hash = $1 AND revoked_at IS NULL",
            key_hash,
        )

    if not row:
        raise EngramError("UNAUTHORIZED", "Invalid API key", 401)

    owner_id: UUID = row["owner_id"]

    # Set ContextVars for structured logging
    cid = getattr(request.state, "correlation_id", "")
    if cid:
        set_correlation_id(cid)
    set_owner_id(str(owner_id))

    # Determine bucket + limit by HTTP method
    method = request.method.upper()
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        limit = row["rate_limit_writes"]
        bucket = f"write:{key_hash}"
    else:
        limit = row["rate_limit_reads"]
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
