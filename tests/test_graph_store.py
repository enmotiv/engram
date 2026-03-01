"""Tests for GraphStore — CRUD for graph snapshots."""

import pytest

from engram.engine.graph_store import GraphStore


NS = "user:graph_store_test"


@pytest.mark.asyncio
async def test_store_and_get_latest(db):
    """Store a graph snapshot and retrieve it."""
    store = GraphStore(db)

    graph = await store.store(
        namespace=NS,
        content="graph LR\n  A[node a] --> B[node b]",
        node_count=2,
        cluster_count=1,
        memory_count=10,
        metadata={"dimensions": ["emotional_sensitivity"]},
    )
    await db.flush()

    assert graph.id is not None
    assert graph.namespace == NS
    assert graph.content.startswith("graph LR")
    assert graph.node_count == 2
    assert graph.cluster_count == 1
    assert graph.memory_count == 10
    assert graph.version == 1

    latest = await store.get_latest(NS)
    assert latest is not None
    assert latest.id == graph.id


@pytest.mark.asyncio
async def test_get_latest_empty_namespace(db):
    """get_latest returns None for a namespace with no graph."""
    store = GraphStore(db)
    result = await store.get_latest("user:no_graph")
    assert result is None


@pytest.mark.asyncio
async def test_version_increments(db):
    """Subsequent snapshots should increment version."""
    store = GraphStore(db)

    g1 = await store.store(
        namespace=NS,
        content="graph LR\n  A[v1]",
        node_count=1,
        cluster_count=1,
        memory_count=5,
    )
    await db.flush()

    g2 = await store.store(
        namespace=NS,
        content="graph LR\n  A[v2] --> B[new]",
        node_count=2,
        cluster_count=1,
        memory_count=10,
    )
    await db.flush()

    assert g2.version == 2

    latest = await store.get_latest(NS)
    assert latest.id == g2.id
    assert latest.version == 2


@pytest.mark.asyncio
async def test_get_history(db):
    """get_history returns snapshots in reverse chronological order."""
    store = GraphStore(db)

    for i in range(3):
        await store.store(
            namespace=NS,
            content=f"graph LR\n  A[v{i + 1}]",
            node_count=i + 1,
            cluster_count=1,
            memory_count=(i + 1) * 5,
        )
        await db.flush()

    history = await store.get_history(NS, limit=10)
    assert len(history) >= 3
    # Most recent first
    assert history[0].node_count >= history[-1].node_count


@pytest.mark.asyncio
async def test_get_history_pagination(db):
    """get_history respects limit and offset."""
    store = GraphStore(db)

    for i in range(5):
        await store.store(
            namespace=NS,
            content=f"graph LR\n  N{i}[node {i}]",
            node_count=1,
            cluster_count=1,
            memory_count=1,
        )
        await db.flush()

    page1 = await store.get_history(NS, limit=2, offset=0)
    page2 = await store.get_history(NS, limit=2, offset=2)

    assert len(page1) == 2
    assert len(page2) == 2
    assert page1[0].id != page2[0].id


@pytest.mark.asyncio
async def test_metadata_stored(db):
    """Metadata is preserved on the graph record."""
    store = GraphStore(db)

    meta = {"dimensions": ["autonomy", "growth"], "pruned_from": 50}
    graph = await store.store(
        namespace=NS,
        content="graph LR\n  X[meta test]",
        node_count=1,
        cluster_count=1,
        memory_count=1,
        metadata=meta,
    )
    await db.flush()

    latest = await store.get_latest(NS)
    assert latest.metadata_ == meta
