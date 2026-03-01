"""Integration tests for the Engram client SDK against the API."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from engram.api.app import create_app
from engram.client import EngramClient
from engram.db.models import Base

TEST_DATABASE_URL = "postgresql+asyncpg://engram:engram@localhost:5433/engram"

_tables_created = False


@pytest.fixture
async def app():
    global _tables_created

    import engram.config as cfg
    import engram.db.session as sess

    cfg.settings.DATABASE_URL = TEST_DATABASE_URL
    cfg.settings.REDIS_URL = "redis://localhost:6380/0"

    test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    if not _tables_created:
        async with test_engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        _tables_created = True

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    sess.async_engine = test_engine
    sess.async_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    sess.engine = test_engine
    sess.async_session = sess.async_session_factory

    from engram.engine.cache import TraceCache
    from engram.plugins.registry import PluginRegistry

    application = create_app()
    PluginRegistry.reset()
    registry = PluginRegistry.get_instance()
    registry.load_plugin(cfg.settings.ENGRAM_PLUGIN)
    application.state.registry = registry
    application.state.trace_cache = TraceCache(cfg.settings.REDIS_URL)

    yield application

    await application.state.trace_cache.close()
    PluginRegistry.reset()
    await test_engine.dispose()


@pytest.fixture
async def client(app):
    """Create an EngramClient backed by the ASGI transport (no real HTTP)."""
    transport = ASGITransport(app=app)
    http_client = AsyncClient(transport=transport, base_url="http://test")
    sdk = EngramClient(base_url="http://test")
    # Replace the internal httpx client with the ASGI-backed one
    await sdk._client.aclose()
    sdk._client = http_client
    yield sdk
    await http_client.aclose()


NS = "user:sdk_test"


@pytest.mark.asyncio
async def test_sdk_store_and_get(client):
    """Store a memory via SDK and retrieve by ID."""
    mem = await client.store(NS, "AWS EC2 migration plan for Q1 2025")
    assert mem.id
    assert mem.content == "AWS EC2 migration plan for Q1 2025"
    assert mem.namespace == NS

    fetched = await client.get_memory(mem.id)
    assert fetched.content == mem.content


@pytest.mark.asyncio
async def test_sdk_retrieve_triggered(client):
    """Retrieve with a relevant cue should trigger."""
    for content in [
        "Database replication setup on RDS with read replicas",
        "Load balancer configuration for auto-scaling EC2 fleet",
        "S3 lifecycle policies archive logs after 90 days",
    ]:
        await client.store(NS, content)

    result = await client.retrieve(NS, "What infrastructure decisions were made?")
    assert result.triggered is True
    assert len(result.memories) > 0
    assert result.trace is not None


@pytest.mark.asyncio
async def test_sdk_retrieve_urgency_gate(client):
    """Low-urgency cue should not trigger."""
    result = await client.retrieve(NS, "ok")
    assert result.triggered is False
    assert len(result.memories) == 0


@pytest.mark.asyncio
async def test_sdk_edges(client):
    """Create two memories and an edge via SDK."""
    m1 = await client.store(NS, "Memory A for edge test")
    m2 = await client.store(NS, "Memory B for edge test")

    edge = await client.create_edge(
        source_id=m1.id,
        target_id=m2.id,
        edge_type="excitatory",
        weight=0.8,
        namespace=NS,
    )
    assert edge.edge_type == "excitatory"
    assert edge.weight == 0.8

    edges = await client.get_edges(m1.id)
    assert len(edges) >= 1
    assert any(e.edge_type == "excitatory" for e in edges)


@pytest.mark.asyncio
async def test_sdk_namespace_stats(client):
    """Get namespace stats via SDK."""
    await client.store(NS, "Stats test memory")

    stats = await client.namespace_stats(NS)
    assert stats.namespace == NS
    assert stats.memory_count >= 1


@pytest.mark.asyncio
async def test_sdk_delete_memory(client):
    """Delete a memory via SDK."""
    mem = await client.store(NS, "Temporary memory to delete")
    deleted = await client.delete_memory(mem.id)
    assert deleted is True


@pytest.mark.asyncio
async def test_sdk_health(client):
    """Health check via SDK."""
    health = await client.health()
    assert health.status in ("ok", "degraded")


@pytest.mark.asyncio
async def test_sdk_batch_store(client):
    """Batch store via SDK."""
    memories = await client.batch_store([
        {"namespace": NS, "content": "Batch SDK memory 1"},
        {"namespace": NS, "content": "Batch SDK memory 2"},
        {"namespace": NS, "content": "Batch SDK memory 3"},
    ])
    assert len(memories) == 3
    assert all(m.namespace == NS for m in memories)
