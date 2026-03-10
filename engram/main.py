"""FastAPI application entry point."""

import importlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
import structlog
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from engram.config import settings
from engram.db import close_pool, get_pool
from engram.middleware import CorrelationMiddleware, register_error_handlers
from engram.tracing import add_log_level, inject_context

_log_level = getattr(logging, settings.engram_log_level.upper(), logging.INFO)

structlog.configure(
    processors=[
        add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        inject_context,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(_log_level),
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.db = await get_pool(settings.database_url)
    await _check_vector_dimensions(app.state.db)
    await load_extensions(app, app.state.db)
    yield
    await close_pool(app.state.db)


async def _check_vector_dimensions(pool: asyncpg.Pool) -> None:
    """Verify DDL vector dimensions match config. Uses schema metadata."""
    async with pool.acquire() as conn:
        db_dims = await conn.fetchval(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'memory_nodes'::regclass "
            "AND attname = 'vec_temporal'"
        )
    if db_dims is not None and db_dims != settings.engram_embedding_dimensions:
        msg = (
            f"Vector dimension mismatch: database has {db_dims}, "
            f"config has {settings.engram_embedding_dimensions}. "
            "Refusing to start."
        )
        logger.error(
            "startup.dimension_mismatch",
            db=db_dims,
            config=settings.engram_embedding_dimensions,
        )
        raise SystemExit(msg)


app = FastAPI(title="Engram", version="1.0.0", lifespan=lifespan)
app.add_middleware(CorrelationMiddleware)
register_error_handlers(app)
Instrumentator().instrument(app).expose(app)


async def load_extensions(app: FastAPI, db: asyncpg.Pool) -> None:
    ext_str = settings.engram_extensions
    if not ext_str:
        return
    for name in ext_str.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            module = importlib.import_module(name)
            await module.register(app, db)
            logger.info("extension.loaded", name=name)
        except Exception as e:
            logger.error("extension.failed", name=name, error=str(e))
            raise  # fail fast


from engram.routes import edges, health, memories, recall  # noqa: E402

app.include_router(memories.router)
app.include_router(recall.router)
app.include_router(health.router)
app.include_router(edges.router)
