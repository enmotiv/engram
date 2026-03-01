"""Test EdgeStore: typed edges, traversal, decay, pruning."""

from datetime import datetime, timedelta, timezone

import pytest

from engram.db.models import Memory
from engram.engine.edge_ops import EdgeOps
from engram.engine.edges import EdgeStore

NS = "test"


async def _make_memories(db, count=5):
    """Create test memories A through E."""
    labels = ["A", "B", "C", "D", "E"][:count]
    mems = []
    for label in labels:
        m = Memory(namespace=NS, content=f"Memory {label}")
        db.add(m)
        mems.append(m)
    await db.flush()
    return mems


@pytest.mark.asyncio
async def test_create_edge(db):
    mems = await _make_memories(db, 2)
    es = EdgeStore(db)
    edge = await es.create(mems[0].id, mems[1].id, "excitatory", weight=0.8, namespace=NS)
    assert edge.edge_type == "excitatory"
    assert edge.weight == 0.8


@pytest.mark.asyncio
async def test_invalid_edge_type(db):
    mems = await _make_memories(db, 2)
    es = EdgeStore(db)
    with pytest.raises(ValueError, match="Invalid edge type"):
        await es.create(mems[0].id, mems[1].id, "invalid_type")


@pytest.mark.asyncio
async def test_get_outgoing_edges(db):
    a, b, c, d, e = await _make_memories(db, 5)
    es = EdgeStore(db)
    await es.create(a.id, b.id, "excitatory", 0.8, namespace=NS)
    await es.create(a.id, c.id, "inhibitory", 0.6, namespace=NS)
    await es.create(b.id, d.id, "associative", 0.5, namespace=NS)

    edges = await es.get_edges(a.id, "outgoing")
    assert len(edges) == 2
    edge_types = {e.edge_type for e in edges}
    assert "excitatory" in edge_types
    assert "inhibitory" in edge_types


@pytest.mark.asyncio
async def test_neighborhood_traversal(db):
    a, b, c, d, e = await _make_memories(db, 5)
    es = EdgeStore(db)
    await es.create(a.id, b.id, "excitatory", 0.8, namespace=NS)
    await es.create(a.id, c.id, "inhibitory", 0.6, namespace=NS)
    await es.create(b.id, d.id, "associative", 0.5, namespace=NS)
    await es.create(b.id, e.id, "temporal", 0.7, namespace=NS)
    await es.create(c.id, e.id, "modulatory", 0.4, namespace=NS)

    result = await es.get_neighborhood([a.id], hop_depth=2, namespace=NS)
    assert len(result["nodes"]) >= 4  # A, B, C, D, E reachable in 2 hops


@pytest.mark.asyncio
async def test_upsert_updates_weight(db):
    a, b = await _make_memories(db, 2)
    es = EdgeStore(db)
    e1 = await es.create(a.id, b.id, "excitatory", 0.5, namespace=NS)
    a_id = a.id
    e2 = await es.create(a.id, b.id, "excitatory", 0.9, namespace=NS)

    db.expire_all()
    edges = await es.get_edges(a_id, "outgoing", ["excitatory"])
    assert len(edges) == 1
    assert edges[0].weight == 0.9


@pytest.mark.asyncio
async def test_reinforce_traversed(db):
    a, b = await _make_memories(db, 2)
    es = EdgeStore(db)
    edge = await es.create(a.id, b.id, "excitatory", 0.5, namespace=NS)
    original_weight = edge.weight

    ops = EdgeOps(db)
    count = await ops.reinforce_traversed([a.id], [b.id], boost=1.2)
    assert count >= 1

    edges = await es.get_edges(a.id, "outgoing")
    assert edges[0].weight > original_weight


@pytest.mark.asyncio
async def test_apply_decay(db):
    a, b = await _make_memories(db, 2)
    es = EdgeStore(db)
    edge = await es.create(a.id, b.id, "associative", 0.8, namespace=NS)
    # Set last_activated to 60 days ago
    edge.last_activated = datetime.now(timezone.utc) - timedelta(days=60)
    await db.flush()

    ops = EdgeOps(db)
    count = await ops.apply_decay(NS, half_life_days=30)
    assert count >= 1

    edges = await es.get_edges(a.id, "outgoing")
    # 60 days / 30 day half-life = 2 half-lives → weight * 0.25
    assert edges[0].weight < 0.3


@pytest.mark.asyncio
async def test_prune(db):
    a, b = await _make_memories(db, 2)
    es = EdgeStore(db)
    edge = await es.create(a.id, b.id, "temporal", 0.01, namespace=NS)
    edge.last_activated = datetime.now(timezone.utc) - timedelta(days=10)
    await db.flush()

    ops = EdgeOps(db)
    count = await ops.prune(NS, weight_floor=0.05, grace_period_days=7)
    assert count >= 1

    edges = await es.get_edges(a.id, "outgoing")
    assert len(edges) == 0
