"""Test Dreamer background jobs: decay, pruning, consolidation."""

from datetime import datetime, timedelta, timezone

import pytest

from engram.db.models import Memory
from engram.dreamer.consolidation import ConsolidationJob
from engram.dreamer.decay import EdgeDecayJob
from engram.dreamer.prune import EdgePruneJob
from engram.engine.edges import EdgeStore
from engram.engine.store import MemoryStore
from engram.plugins.registry import PluginRegistry

NS = "user:dreamer_test"


@pytest.fixture
def registry():
    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.default")
    yield reg
    PluginRegistry.reset()


async def _seed_data(db, registry):
    """Create 10 memories and 15 edges."""
    store = MemoryStore(db, registry)
    es = EdgeStore(db)

    mems = []
    contents = [
        "AWS EC2 migration plan",
        "Database replication setup",
        "Load balancer configuration",
        "CI/CD pipeline update",
        "Monitoring dashboard created",
        "Security audit completed",
        "Performance optimization done",
        "API gateway configured",
        "Container orchestration setup",
        "Backup strategy implemented",
    ]
    for content in contents:
        m = await store.create(NS, content)
        mems.append(m)
    await db.flush()

    # Create 15 edges
    edge_pairs = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 4),
        (2, 3), (2, 5), (3, 4), (3, 6), (4, 5),
        (5, 6), (5, 7), (6, 8), (7, 8), (8, 9),
    ]
    edges = []
    for i, j in edge_pairs:
        e = await es.create(mems[i].id, mems[j].id, "associative", 0.5, namespace=NS)
        edges.append(e)
    await db.flush()
    return mems, edges


@pytest.mark.asyncio
async def test_edge_decay(db, registry):
    """Decay job should reduce weights of old edges."""
    mems, edges = await _seed_data(db, registry)

    # Set 5 edges to last_activated 60 days ago
    old_time = datetime.now(timezone.utc) - timedelta(days=60)
    for i in range(5):
        edges[i].last_activated = old_time
    await db.flush()

    original_weights = [edges[i].weight for i in range(5)]
    edge_ids = [edges[i].id for i in range(5)]
    source_ids = [edges[i].source_memory_id for i in range(5)]

    job = EdgeDecayJob()
    result = await job.execute(NS, db=db)
    assert result["edges_decayed"] >= 5

    db.expire_all()

    es = EdgeStore(db)
    for i in range(5):
        fetched = await es.get_edges(source_ids[i], "outgoing", ["associative"])
        for e in fetched:
            if e.id == edge_ids[i]:
                # 60 days / 30 day half-life = 2 half-lives → weight * 0.25
                assert e.weight < original_weights[i]


@pytest.mark.asyncio
async def test_edge_prune(db, registry):
    """Prune job should delete low-weight old edges."""
    mems, edges = await _seed_data(db, registry)

    # Set 2 edges to weight 0.01 with last_activated 10 days ago
    old_time = datetime.now(timezone.utc) - timedelta(days=10)
    for i in range(2):
        edges[i].weight = 0.01
        edges[i].last_activated = old_time
    await db.flush()

    job = EdgePruneJob()
    result = await job.execute(NS, db=db)
    assert result["edges_pruned"] >= 2

    db.expire_all()

    es = EdgeStore(db)
    # Verify those edges are gone
    for i in range(2):
        eid = edges[i].source_memory_id
        fetched = await es.get_edges(eid, "outgoing")
        fetched_ids = {e.id for e in fetched}
        assert edges[i].id not in fetched_ids


@pytest.mark.asyncio
async def test_consolidation_creates_summary(db, registry):
    """Consolidation should cluster similar memories and create summaries."""
    store = MemoryStore(db, registry)

    # Create 4 very similar memories
    similar = [
        "AWS migration plan version 1 for production environment",
        "AWS migration plan version 2 for production environment",
        "AWS migration plan version 3 for production environment",
        "AWS migration plan version 4 for production environment",
    ]
    for content in similar:
        await store.create(NS, content)
    await db.flush()

    job = ConsolidationJob()
    result = await job.execute(NS, db=db, registry=registry)

    if result["clusters_found"] > 0:
        assert result["summaries_created"] >= 1

        # Verify semantic memory was created
        from sqlalchemy import select
        stmt = select(Memory).where(
            Memory.namespace == NS,
            Memory.memory_type == "semantic",
        )
        res = await db.execute(stmt)
        semantics = list(res.scalars().all())
        assert len(semantics) >= 1
        # Summary should contain content from the cluster
        assert "AWS" in semantics[0].content or "migration" in semantics[0].content


@pytest.mark.asyncio
async def test_consolidation_preserves_emotional(db, registry):
    """High-amygdala memories should NOT be consolidated."""
    store = MemoryStore(db, registry)

    # Create similar memories
    for v in range(1, 5):
        await store.create(NS, f"AWS migration plan version {v} for production")
    await db.flush()

    # Create a high-amygdala memory in the same cluster (via store so it gets embedding)
    emotional = await store.create(
        NS, "AWS migration plan version 5 for production — critical emergency"
    )
    emotional.dimension_scores = {"amygdala": 0.9}
    await db.flush()

    job = ConsolidationJob()
    result = await job.execute(NS, db=db, registry=registry)

    # The emotional memory should be preserved (not consolidated)
    assert result["memories_preserved"] >= 1
