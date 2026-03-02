"""Tests for S4A-2 (snippets) and S4A-3 (full batch) endpoints."""

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


NS = "user:snippet_full_test"


async def _create_memory(client, content, ns=NS, metadata=None):
    payload = {"namespace": ns, "content": content}
    if metadata:
        payload["metadata"] = metadata
    resp = await client.post("/v1/memories", json=payload)
    assert resp.status_code == 200
    return resp.json()["id"]


async def _create_edge(client, src, tgt, edge_type, weight=0.5):
    resp = await client.post("/v1/edges", json={
        "source_id": src,
        "target_id": tgt,
        "edge_type": edge_type,
        "weight": weight,
        "namespace": NS,
    })
    assert resp.status_code == 200
    return resp.json()


# ──────────────────── S4A-2: Snippets ────────────────────


@pytest.mark.asyncio
async def test_snippets_returns_correct_count(client):
    """POST /v1/memories/snippets with 5 IDs → 5 snippets."""
    ids = []
    for i in range(5):
        mid = await _create_memory(client, f"Snippet test memory number {i}")
        ids.append(mid)

    resp = await client.post("/v1/memories/snippets", json={"ids": ids})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["snippets"]) == 5


@pytest.mark.asyncio
async def test_snippets_content_truncated(client):
    """Content is truncated to 200 chars with '...' suffix."""
    long_content = "A" * 300
    mid = await _create_memory(client, long_content)

    resp = await client.post("/v1/memories/snippets", json={"ids": [mid]})
    data = resp.json()
    snippet = data["snippets"][0]
    assert len(snippet["content"]) == 203  # 200 + "..."
    assert snippet["content"].endswith("...")


@pytest.mark.asyncio
async def test_snippets_short_content_not_truncated(client):
    """Content under 200 chars is NOT truncated."""
    short_content = "Short memory content"
    mid = await _create_memory(client, short_content)

    resp = await client.post("/v1/memories/snippets", json={"ids": [mid]})
    data = resp.json()
    snippet = data["snippets"][0]
    assert snippet["content"] == short_content
    assert "..." not in snippet["content"]


@pytest.mark.asyncio
async def test_snippets_top_5_dimensions_only(client):
    """Only top 5 dimensions are included, sorted by score desc."""
    mid = await _create_memory(client, "Testing dimension filtering for snippet endpoint")

    resp = await client.post("/v1/memories/snippets", json={"ids": [mid]})
    data = resp.json()
    snippet = data["snippets"][0]
    top_dims = snippet["top_dimensions"]
    assert len(top_dims) <= 5
    # Verify sorted descending
    scores = list(top_dims.values())
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_snippets_edge_summary_counts(client):
    """Edge summary has counts, not full edge records."""
    id1 = await _create_memory(client, "Source memory for edge counting")
    id2 = await _create_memory(client, "Target A for edge test")
    id3 = await _create_memory(client, "Target B for edge test")

    await _create_edge(client, id1, id2, "excitatory", 0.8)
    await _create_edge(client, id1, id3, "inhibitory", 0.6)

    resp = await client.post("/v1/memories/snippets", json={"ids": [id1]})
    data = resp.json()
    snippet = data["snippets"][0]
    summary = snippet["edge_summary"]
    assert summary["excitatory_count"] == 1
    assert summary["inhibitory_count"] == 1
    assert summary["associative_count"] == 0


@pytest.mark.asyncio
async def test_snippets_missing_ids_skipped(client):
    """Non-existent IDs are silently skipped."""
    real_id = await _create_memory(client, "Real memory for skip test")
    fake_id = "00000000-0000-0000-0000-000000000000"

    resp = await client.post("/v1/memories/snippets", json={"ids": [real_id, fake_id]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["snippets"]) == 1
    assert data["snippets"][0]["id"] == real_id


@pytest.mark.asyncio
async def test_snippets_texture_summary(client):
    """Texture summary is extracted and truncated to 100 chars."""
    texture_text = "T" * 150
    mid = await _create_memory(
        client,
        "Memory with texture data",
        metadata={"texture": texture_text},
    )

    resp = await client.post("/v1/memories/snippets", json={"ids": [mid]})
    data = resp.json()
    snippet = data["snippets"][0]
    assert snippet["texture_summary"] is not None
    assert len(snippet["texture_summary"]) == 103  # 100 + "..."


# ──────────────────── S4A-3: Full Batch ────────────────────


@pytest.mark.asyncio
async def test_full_batch_returns_complete_records(client):
    """POST /v1/memories/full with 3 IDs → 3 full records."""
    ids = []
    for i in range(3):
        mid = await _create_memory(client, f"Full batch test memory {i}")
        ids.append(mid)

    resp = await client.post("/v1/memories/full", json={"ids": ids})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memories"]) == 3
    for mem in data["memories"]:
        assert "content" in mem
        assert "dimension_scores" in mem
        assert "features" in mem
        assert "edges" in mem


@pytest.mark.asyncio
async def test_full_batch_includes_edges(client):
    """Full records include edges with type and target info."""
    id1 = await _create_memory(client, "Full batch source memory")
    id2 = await _create_memory(client, "Full batch target memory")
    await _create_edge(client, id1, id2, "excitatory", 0.9)

    resp = await client.post("/v1/memories/full", json={"ids": [id1]})
    data = resp.json()
    mem = data["memories"][0]
    assert len(mem["edges"]) >= 1
    edge = mem["edges"][0]
    assert edge["edge_type"] == "excitatory"
    assert edge["target_id"] == id2
    assert edge["weight"] == 0.9


@pytest.mark.asyncio
async def test_full_batch_linked_memory_snippets(client):
    """Linked memories include id + content snippet (first 100 chars)."""
    long_content = "B" * 200
    id1 = await _create_memory(client, "Source memory")
    id2 = await _create_memory(client, long_content)
    await _create_edge(client, id1, id2, "associative")

    resp = await client.post("/v1/memories/full", json={"ids": [id1]})
    data = resp.json()
    mem = data["memories"][0]
    edge = mem["edges"][0]
    linked = edge["linked_memory"]
    assert linked is not None
    assert linked["id"] == id2
    assert len(linked["content"]) == 103  # 100 + "..."
    assert linked["content"].endswith("...")


@pytest.mark.asyncio
async def test_full_batch_missing_ids_skipped(client):
    """Non-existent IDs are silently skipped."""
    real_id = await _create_memory(client, "Real memory for full batch skip")
    fake_id = "00000000-0000-0000-0000-000000000001"

    resp = await client.post("/v1/memories/full", json={"ids": [real_id, fake_id]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memories"]) == 1
    assert data["memories"][0]["id"] == real_id


@pytest.mark.asyncio
async def test_full_batch_content_not_truncated(client):
    """Full batch returns complete content (not truncated)."""
    long_content = "C" * 500
    mid = await _create_memory(client, long_content)

    resp = await client.post("/v1/memories/full", json={"ids": [mid]})
    data = resp.json()
    mem = data["memories"][0]
    assert mem["content"] == long_content
    assert len(mem["content"]) == 500
