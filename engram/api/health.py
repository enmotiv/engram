"""Health check endpoint."""

from __future__ import annotations

import redis.asyncio as aioredis
from fastapi import APIRouter, Request
from sqlalchemy import text

from engram.api.schemas import HealthResponse
from engram.config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    db_ok = False
    redis_ok = False
    plugin_name = settings.ENGRAM_PLUGIN

    # Check DB
    try:
        import engram.db.session as sess
        async with sess.async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    # Check Redis
    try:
        r = aioredis.from_url(settings.REDIS_URL)
        await r.ping()
        await r.aclose()
        redis_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if db_ok and redis_ok else "degraded",
        db=db_ok,
        redis=redis_ok,
        plugin=plugin_name,
    )
