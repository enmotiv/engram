"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI

from engram.config import settings
from engram.db.session import async_engine
from engram.engine.cache import TraceCache
from engram.plugins.registry import PluginRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Load plugins (supports multiple comma-separated plugins)
    registry = PluginRegistry.get_instance()
    registry.load_plugins(settings.plugin_paths)
    app.state.registry = registry

    # Redis for trace cache
    app.state.trace_cache = TraceCache(settings.REDIS_URL)

    yield

    # Shutdown
    await app.state.trace_cache.close()
    await async_engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Engram",
        description="Memory system with multi-axis retrieval and spreading activation",
        version="0.1.0",
        lifespan=lifespan,
    )

    from engram.api.edges import router as edges_router
    from engram.api.health import router as health_router
    from engram.api.memories import router as memories_router
    from engram.api.namespaces import router as namespaces_router

    app.include_router(memories_router, prefix="/v1")
    app.include_router(edges_router, prefix="/v1")
    app.include_router(namespaces_router, prefix="/v1")
    app.include_router(health_router, prefix="/v1")

    return app


app = create_app()
