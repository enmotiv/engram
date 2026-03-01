"""Async SQLAlchemy engine and session factory."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from engram.config import settings

async_engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)
async_session_factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Legacy aliases
engine = async_engine
async_session = async_session_factory


async def get_db():
    """FastAPI dependency — yields a DB session per request."""
    async with async_session_factory() as session:
        yield session
