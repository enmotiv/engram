"""FastAPI dependency injection."""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from engram.engine.cache import TraceCache
from engram.engine.edges import EdgeStore
from engram.engine.retriever import Retriever
from engram.engine.store import MemoryStore
from engram.plugins.registry import PluginRegistry


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    # Lazy import to allow test patching of session module
    import engram.db.session as sess
    async with sess.async_session_factory() as session:
        yield session


def get_registry(request: Request) -> PluginRegistry:
    return request.app.state.registry


def get_trace_cache(request: Request) -> TraceCache:
    return request.app.state.trace_cache


async def get_memory_store(
    db: AsyncSession = Depends(get_db),
    registry: PluginRegistry = Depends(get_registry),
) -> MemoryStore:
    return MemoryStore(db, registry)


async def get_edge_store(
    db: AsyncSession = Depends(get_db),
) -> EdgeStore:
    return EdgeStore(db)


async def get_retriever(
    db: AsyncSession = Depends(get_db),
    registry: PluginRegistry = Depends(get_registry),
    cache: TraceCache = Depends(get_trace_cache),
) -> Retriever:
    return Retriever(db, registry, trace_cache=cache)
