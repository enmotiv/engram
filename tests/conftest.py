"""Shared pytest fixtures for Engram tests."""

import os
from typing import AsyncGenerator

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from engram.db.models import Base

TEST_DATABASE_URL = os.environ.get(
    "ENGRAM_DATABASE_URL",
    "postgresql+asyncpg://engram:engram@localhost:5433/engram",
)

_tables_created = False


@pytest_asyncio.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    global _tables_created
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    if not _tables_created:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        _tables_created = True

    conn = await engine.connect()
    txn = await conn.begin()
    session = AsyncSession(bind=conn, expire_on_commit=False)

    yield session

    await session.close()
    await txn.rollback()
    await conn.close()
    await engine.dispose()
