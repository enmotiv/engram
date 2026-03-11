"""Integration tests for Dreamer: edge classification, decay, dedup, pruning."""


import pytest


async def test_process_new_memories(db_pool, owner_a, seeded_memories):
    """Dreamer should classify edges for unprocessed memories."""
    from engram.dreamer import process_new_memories

    owner_id, _ = owner_a
    stats = await process_new_memories(db_pool, owner_id)

    assert "memories_processed" in stats
    assert "candidates_evaluated" in stats
    assert "edges_created" in stats
    assert stats["memories_processed"] >= 0


async def test_classify_pair():
    """Classification should return valid axis-keyed dict."""
    from engram.services.dreamer import _classify_pair

    result = await _classify_pair(
        "User likes Python", "User prefers dark mode"
    )
    # Mock returns associative edge on semantic axis
    assert isinstance(result, dict)
    if result:
        for _axis, edges in result.items():
            assert isinstance(edges, list)


async def test_store_edges(db_pool, owner_a, seeded_memories):
    """Edges should be stored with correct structure."""

    from engram.db import tenant_connection
    from engram.dreamer import _store_edges

    owner_id, _ = owner_a

    # Get two memory IDs
    async with tenant_connection(db_pool, owner_id) as conn:
        rows = await conn.fetch(
            "SELECT id FROM memory_nodes "
            "WHERE owner_id = $1 AND is_deleted = FALSE LIMIT 2",
            owner_id,
        )

    if len(rows) < 2:
        pytest.skip("Need at least 2 memories")

    source_id = rows[0]["id"]
    target_id = rows[1]["id"]
    classification = {
        "semantic": [{"type": "associative", "weight": 0.8}],
        "temporal": [],
    }

    async with tenant_connection(db_pool, owner_id) as conn:
        count = await _store_edges(
            conn, owner_id, source_id, target_id, classification
        )

    assert count >= 1

    # Verify edge exists
    async with tenant_connection(db_pool, owner_id) as conn:
        src = min(source_id, target_id)
        tgt = max(source_id, target_id)
        edge = await conn.fetchrow(
            "SELECT * FROM edges "
            "WHERE owner_id = $1 AND source_id = $2 AND target_id = $3",
            owner_id,
            src,
            tgt,
        )
    assert edge is not None
    assert edge["edge_type"] == "associative"
    assert edge["axis"] == "semantic"


async def test_store_edges_invalid_type(db_pool, owner_a, seeded_memories):
    """Invalid edge types should be silently skipped."""
    from engram.db import tenant_connection
    from engram.dreamer import _store_edges

    owner_id, _ = owner_a

    async with tenant_connection(db_pool, owner_id) as conn:
        rows = await conn.fetch(
            "SELECT id FROM memory_nodes "
            "WHERE owner_id = $1 AND is_deleted = FALSE LIMIT 2",
            owner_id,
        )

    if len(rows) < 2:
        pytest.skip("Need at least 2 memories")

    classification = {
        "semantic": [{"type": "invalid_type", "weight": 0.8}],
    }

    async with tenant_connection(db_pool, owner_id) as conn:
        count = await _store_edges(
            conn, owner_id, rows[0]["id"], rows[1]["id"], classification
        )
    assert count == 0


async def test_store_edges_low_weight_skipped(db_pool, owner_a, seeded_memories):
    """Edges with weight <= 0.1 should be skipped."""
    from engram.db import tenant_connection
    from engram.dreamer import _store_edges

    owner_id, _ = owner_a

    async with tenant_connection(db_pool, owner_id) as conn:
        rows = await conn.fetch(
            "SELECT id FROM memory_nodes "
            "WHERE owner_id = $1 AND is_deleted = FALSE LIMIT 2",
            owner_id,
        )

    if len(rows) < 2:
        pytest.skip("Need at least 2 memories")

    classification = {
        "semantic": [{"type": "associative", "weight": 0.05}],
    }

    async with tenant_connection(db_pool, owner_id) as conn:
        count = await _store_edges(
            conn, owner_id, rows[0]["id"], rows[1]["id"], classification
        )
    assert count == 0


async def test_decay_activations(db_pool, owner_a, seeded_memories):
    """Decay should run without error. Count may be 0 if all recently accessed."""
    from engram.dreamer import decay_activations

    owner_id, _ = owner_a
    count = await decay_activations(db_pool, owner_id)
    assert isinstance(count, int)
    assert count >= 0


async def test_decay_edge_weights(db_pool, owner_a):
    """Edge weight decay should run without error."""
    from engram.dreamer import decay_edge_weights

    owner_id, _ = owner_a
    count = await decay_edge_weights(db_pool, owner_id)
    assert isinstance(count, int)
    assert count >= 0


async def test_dedup_memories(db_pool, owner_a):
    """Dedup should run without error."""
    from engram.dreamer import dedup_memories

    owner_id, _ = owner_a
    stats = await dedup_memories(db_pool, owner_id)
    assert "pairs_found" in stats
    assert "nodes_deleted" in stats
    assert "edges_transferred" in stats


async def test_prune_edges(db_pool, owner_a):
    """Prune should delete low-weight edges."""
    from engram.dreamer import prune_edges

    owner_id, _ = owner_a
    count = await prune_edges(db_pool, owner_id)
    assert isinstance(count, int)
    assert count >= 0


async def test_run_cycle(db_pool, owner_a, seeded_memories):
    """Full Dreamer cycle should run all sub-jobs."""
    from engram.dreamer import run_cycle

    owner_id, _ = owner_a
    detail = await run_cycle(db_pool, owner_id)

    assert "edge_classification" in detail
    assert "activation_decay" in detail
    assert "edge_decay" in detail
    assert "dedup" in detail
    assert "edge_pruning" in detail


async def test_edge_ordering_consistency(db_pool, owner_a, seeded_memories):
    """Edges should always have source_id < target_id."""
    from engram.db import tenant_connection

    owner_id, _ = owner_a

    async with tenant_connection(db_pool, owner_id) as conn:
        edges = await conn.fetch(
            "SELECT source_id, target_id FROM edges WHERE owner_id = $1",
            owner_id,
        )

    for edge in edges:
        assert edge["source_id"] <= edge["target_id"]
