"""Sweep 3 integration verification tests.

Checklist from S3-5:
1. Dimension scores present after memory creation
2. Embedding is 1024 dimensions (BGE-M3)
3. Retrieval works with new embeddings (no regression)
4. Graph store + API works
5-7. Graph content verification (subgraphs, clusters, token budget)
8. Graph history returns snapshots
9. Full test suite no regressions (run separately)
"""

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
    """Create app with test DB."""
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
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


NS = "user:sweep3_verify"


# -- Checklist 1: Dimension scores present --


@pytest.mark.asyncio
async def test_dimension_scores_present(client):
    """POST /v1/memories → response includes dimension scores."""
    resp = await client.post("/v1/memories", json={
        "namespace": NS,
        "content": "I want to be independent and make my own choices",
    })
    assert resp.status_code == 200
    data = resp.json()
    # Default plugin provides: semantic, temporal, importance
    assert "semantic" in data.get("dimension_scores", {})
    assert "temporal" in data.get("dimension_scores", {})
    assert "importance" in data.get("dimension_scores", {})


# -- Checklist 2: 1024-dim BGE-M3 embeddings --


@pytest.mark.asyncio
async def test_embedding_1024_dimensions(app):
    """Stored memory embedding is 1024 dimensions (BGE-M3)."""
    import engram.db.session as sess

    async with sess.async_session_factory() as session:
        from engram.engine.store import MemoryStore
        from engram.plugins.registry import PluginRegistry

        store = MemoryStore(session, PluginRegistry.get_instance())
        mem = await store.create(
            namespace=NS,
            content="Testing 1024-dim BGE-M3 embedding",
        )
        await session.commit()

        assert mem.embedding is not None
        assert len(mem.embedding) == 1024


# -- Checklist 3: Retrieval works (no regression) --


@pytest.mark.asyncio
async def test_retrieval_works_with_1024_embeddings(client):
    """Store memories and retrieve — retrieval works with BGE-M3 embeddings."""
    for content in [
        "We migrated production databases to AWS RDS last month",
        "EC2 instances handle our machine learning inference workload",
        "S3 lifecycle policies archive old logs after 90 days",
    ]:
        await client.post("/v1/memories", json={"namespace": NS, "content": content})

    resp = await client.post("/v1/memories/retrieve", json={
        "namespace": NS,
        "cue": "What AWS infrastructure decisions do you remember?",
        "max_results": 5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["triggered"] is True
    assert len(data["memories"]) > 0


# -- Checklist 4: Graph store + API --


@pytest.mark.asyncio
async def test_graph_store_and_get(client):
    """POST a graph snapshot, then GET it back."""
    mermaid = "graph LR\n  A[Memory 1] --> B[Memory 2]"
    resp = await client.post(f"/v1/graphs/{NS}", json={
        "content": mermaid,
        "node_count": 2,
        "cluster_count": 0,
        "memory_count": 2,
    })
    assert resp.status_code == 200

    resp2 = await client.get(f"/v1/graphs/{NS}")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["content"] == mermaid
    assert data["node_count"] == 2


# -- Checklist 5-7: Graph content verification --


@pytest.mark.asyncio
async def test_graph_content_with_subgraphs(client):
    """Stored graph with subgraphs and clusters round-trips correctly."""
    mermaid = (
        "graph LR\n"
        "  subgraph hippocampus\n"
        "    A[Memory about events]\n"
        "  end\n"
        "  subgraph amygdala\n"
        "    B([Emotional cluster])\n"
        "  end\n"
        "  A --> B\n"
    )
    ns = "user:sweep3_graph_content"
    await client.post(f"/v1/graphs/{ns}", json={
        "content": mermaid,
        "node_count": 2,
        "cluster_count": 1,
        "memory_count": 5,
    })

    resp = await client.get(f"/v1/graphs/{ns}")
    assert resp.status_code == 200
    data = resp.json()
    assert "subgraph" in data["content"]
    assert "([" in data["content"]  # cluster node shape
    # Token budget: mermaid < 500 tokens (rough: words / 0.75)
    word_count = len(data["content"].split())
    token_estimate = word_count / 0.75
    assert token_estimate < 500


# -- Checklist 8: Graph history --


@pytest.mark.asyncio
async def test_graph_history(client):
    """GET /v1/graphs/{ns}/history returns snapshots."""
    ns = "user:sweep3_history"
    for i in range(3):
        await client.post(f"/v1/graphs/{ns}", json={
            "content": f"graph LR\n  A{i}[Node {i}]",
            "node_count": 1,
            "cluster_count": 0,
            "memory_count": 1,
        })

    resp = await client.get(f"/v1/graphs/{ns}/history?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["snapshots"]) >= 3


# -- Checklist: GET nonexistent namespace → 404 --


@pytest.mark.asyncio
async def test_graph_404_for_nonexistent(client):
    """GET graph for nonexistent namespace → 404."""
    resp = await client.get("/v1/graphs/user:nonexistent_ns_42")
    assert resp.status_code == 404
