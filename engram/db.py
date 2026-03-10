"""Database connection pool and RLS helper."""

import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg
from asyncpg import Pool


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register pgvector codec on each new pool connection."""
    from pgvector.asyncpg import register_vector

    await register_vector(conn)


async def get_pool(database_url: str) -> Pool:
    """Create connection pool. Called once at startup."""
    return await asyncpg.create_pool(
        database_url,
        min_size=5,
        max_size=20,
        init=_init_connection,
    )


async def close_pool(pool: Pool | None) -> None:
    """Close connection pool. Called at shutdown."""
    if pool is not None:
        await pool.close()


@asynccontextmanager
async def tenant_connection(
    pool: Pool, owner_id: str | UUID
) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a connection with RLS context set. Use this everywhere."""
    # Validate as UUID to prevent injection in SET statement
    safe_id = str(UUID(str(owner_id)))
    async with pool.acquire() as conn:
        await conn.execute(f"SET app.owner_id = '{safe_id}'")
        try:
            yield conn
        finally:
            with contextlib.suppress(Exception):
                await conn.execute("RESET app.owner_id")
