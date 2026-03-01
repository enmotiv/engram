"""Integration tests for the FastAPI API."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from engram.api.app import create_app
from engram.db.models import Base

TEST_DATABASE_URL = "postgresql+asyncpg://engram:engram@localhost:5433/engram"

_tables_created = False


@pytest.fixture
async def app():
    """Create app with test DB and manually initialized state."""
    global _tables_created

    import engram.config as cfg
    import engram.db.session as sess

    # Override to test DB
    cfg.settings.DATABASE_URL = TEST_DATABASE_URL
    cfg.settings.REDIS_URL = "redis://localhost:6380/0"

    test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    if not _tables_created:
        async with test_engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        _tables_created = True

    # Patch session module to use test engine
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    sess.async_engine = test_engine
    sess.async_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    sess.engine = test_engine
    sess.async_session = sess.async_session_factory

    from engram.engine.cache import TraceCache
    from engram.plugins.registry import PluginRegistry

    application = create_app()

    # Manually initialize app state (lifespan doesn't run in httpx tests)
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
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


NS = "user:api_test"


@pytest.mark.asyncio
async def test_create_and_get_memory(client):
    """POST a memory, then GET it by ID."""
    resp = await client.post("/v1/memories", json={
        "namespace": NS,
        "content": "AWS EC2 g5.xlarge migration completed",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["content"] == "AWS EC2 g5.xlarge migration completed"

    # GET by ID
    mem_id = data["id"]
    resp2 = await client.get(f"/v1/memories/{mem_id}")
    assert resp2.status_code == 200
    assert resp2.json()["content"] == data["content"]


@pytest.mark.asyncio
async def test_retrieve_triggered(client):
    """Store memories and retrieve with a relevant cue."""
    # Store 3 memories
    for content in [
        "We migrated production to AWS RDS last Tuesday",
        "EC2 g5.xlarge handles our ML inference",
        "S3 lifecycle policies archive logs after 90 days",
    ]:
        await client.post("/v1/memories", json={"namespace": NS, "content": content})

    # Retrieve
    resp = await client.post("/v1/memories/retrieve", json={
        "namespace": NS,
        "cue": "What AWS infrastructure decisions were made?",
        "max_results": 5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["triggered"] is True
    assert len(data["memories"]) > 0
    assert data["trace"] is not None


@pytest.mark.asyncio
async def test_retrieve_urgency_gate(client):
    """Low-urgency cue should not trigger retrieval."""
    resp = await client.post("/v1/memories/retrieve", json={
        "namespace": NS,
        "cue": "ok",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["triggered"] is False
    assert len(data["memories"]) == 0


@pytest.mark.asyncio
async def test_create_edge(client):
    """Create two memories and an edge between them."""
    r1 = await client.post("/v1/memories", json={"namespace": NS, "content": "Memory Alpha"})
    r2 = await client.post("/v1/memories", json={"namespace": NS, "content": "Memory Beta"})
    id1 = r1.json()["id"]
    id2 = r2.json()["id"]

    resp = await client.post("/v1/edges", json={
        "source_id": id1,
        "target_id": id2,
        "edge_type": "excitatory",
        "weight": 0.7,
        "namespace": NS,
    })
    assert resp.status_code == 200
    edge = resp.json()
    assert edge["edge_type"] == "excitatory"
    assert edge["weight"] == 0.7


@pytest.mark.asyncio
async def test_namespace_stats(client):
    """GET namespace stats should show counts."""
    resp = await client.get(f"/v1/namespaces/{NS}/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["namespace"] == NS
    assert data["memory_count"] >= 0
    assert data["edge_count"] >= 0


@pytest.mark.asyncio
async def test_delete_memory_404(client):
    """DELETE a memory, then GET → 404."""
    r = await client.post("/v1/memories", json={"namespace": NS, "content": "Temporary"})
    mem_id = r.json()["id"]

    # Delete
    resp = await client.delete(f"/v1/memories/{mem_id}")
    assert resp.status_code == 200

    # GET → 404
    resp2 = await client.get(f"/v1/memories/{mem_id}")
    assert resp2.status_code == 404


@pytest.mark.asyncio
async def test_health(client):
    """GET /health should return ok."""
    resp = await client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert data["plugin"] == "engram.plugins.default"


@pytest.mark.asyncio
async def test_batch_create(client):
    """Batch create multiple memories."""
    resp = await client.post("/v1/memories/batch", json={
        "memories": [
            {"namespace": NS, "content": "Batch memory 1"},
            {"namespace": NS, "content": "Batch memory 2"},
            {"namespace": NS, "content": "Batch memory 3"},
        ]
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
