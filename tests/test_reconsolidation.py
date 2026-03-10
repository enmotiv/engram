"""Integration tests for reconsolidation: boost on read."""

import pytest

from tests.conftest import auth_headers


async def test_reconsolidate_boosts_nodes(client, db_pool, owner_a, seeded_memories):
    """Recalled nodes should have increased access_count and activation."""
    from engram.db import tenant_connection

    owner_id, key = owner_a

    # Get initial state of a node
    async with tenant_connection(db_pool, owner_id) as conn:
        before = await conn.fetchrow(
            "SELECT access_count, activation_level FROM memory_nodes "
            "WHERE owner_id = $1 AND is_deleted = FALSE "
            "ORDER BY created_at ASC LIMIT 1",
            owner_id,
        )

    if not before:
        pytest.skip("No memories to test")

    initial_count = before["access_count"]

    # Trigger recall (which runs reconsolidation internally)
    await client.post(
        "/v1/recall",
        json={"cue": "test reconsolidation"},
        headers=auth_headers(key),
    )

    # Check that at least one node was boosted
    async with tenant_connection(db_pool, owner_id) as conn:
        boosted = await conn.fetchval(
            "SELECT COUNT(*) FROM memory_nodes "
            "WHERE owner_id = $1 AND is_deleted = FALSE "
            "AND access_count > $2",
            owner_id,
            initial_count,
        )

    assert boosted > 0


async def test_reconsolidate_empty_list(db_pool, owner_a):
    """Reconsolidation with no items should be a no-op."""
    from engram.reconsolidation import reconsolidate

    owner_id, _ = owner_a
    stats = await reconsolidate(db_pool, owner_id, [])
    assert stats["nodes_boosted"] == 0
    assert stats["edges_strengthened"] == 0


async def test_reconsolidate_single_item(db_pool, owner_a, seeded_memories):
    """Single item: node boosted, no edges to strengthen (no pairs)."""

    from engram.db import tenant_connection
    from engram.reconsolidation import reconsolidate

    owner_id, _ = owner_a

    async with tenant_connection(db_pool, owner_id) as conn:
        row = await conn.fetchrow(
            "SELECT id FROM memory_nodes "
            "WHERE owner_id = $1 AND is_deleted = FALSE LIMIT 1",
            owner_id,
        )

    if not row:
        pytest.skip("No memories")

    items = [{"id": row["id"], "matched_axes": ["semantic"]}]
    stats = await reconsolidate(db_pool, owner_id, items)
    assert stats["nodes_boosted"] == 1
    assert stats["edges_strengthened"] == 0
