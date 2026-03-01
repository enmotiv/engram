"""Test batch resolve endpoint — returns full records with edges."""

import uuid

import pytest

from engram.engine.edges import EdgeStore
from engram.engine.store import MemoryStore
from engram.plugins.registry import PluginRegistry

NS = "user:resolve_test"


@pytest.fixture
def registry():
    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.default")
    yield reg
    PluginRegistry.reset()


@pytest.mark.asyncio
async def test_get_batch_returns_memories(db, registry):
    """get_batch returns multiple memories in a single query."""
    store = MemoryStore(db, registry)
    m1 = await store.create(NS, "First memory about databases")
    m2 = await store.create(NS, "Second memory about cooking")
    m3 = await store.create(NS, "Third memory about fitness")
    await db.flush()

    result = await store.get_batch([m1.id, m2.id, m3.id])
    assert len(result) == 3

    ids = {str(m.id) for m in result}
    assert str(m1.id) in ids
    assert str(m2.id) in ids
    assert str(m3.id) in ids


@pytest.mark.asyncio
async def test_get_batch_with_missing_ids(db, registry):
    """get_batch returns only found memories, ignoring missing IDs."""
    store = MemoryStore(db, registry)
    m1 = await store.create(NS, "Existing memory")
    await db.flush()

    fake_id = uuid.uuid4()
    result = await store.get_batch([m1.id, fake_id])
    assert len(result) == 1
    assert str(result[0].id) == str(m1.id)


@pytest.mark.asyncio
async def test_get_batch_empty_list(db, registry):
    """get_batch with empty list returns empty."""
    store = MemoryStore(db, registry)
    result = await store.get_batch([])
    assert result == []


@pytest.mark.asyncio
async def test_get_edges_batch(db, registry):
    """get_edges_batch returns edges for multiple memories."""
    store = MemoryStore(db, registry)
    edge_store = EdgeStore(db)

    m1 = await store.create(NS, "Memory A")
    m2 = await store.create(NS, "Memory B")
    m3 = await store.create(NS, "Memory C")
    await db.flush()

    await edge_store.create(m1.id, m2.id, "excitatory", namespace=NS)
    await edge_store.create(m2.id, m3.id, "associative", namespace=NS)
    await db.flush()

    edges = await edge_store.get_edges_batch([m1.id, m2.id])
    assert len(edges) >= 2  # at least the two edges we created

    edge_types = {e.edge_type for e in edges}
    assert "excitatory" in edge_types
    assert "associative" in edge_types


@pytest.mark.asyncio
async def test_get_edges_batch_empty(db, registry):
    """get_edges_batch with empty list returns empty."""
    edge_store = EdgeStore(db)
    result = await edge_store.get_edges_batch([])
    assert result == []
