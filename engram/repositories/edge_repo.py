"""Edge repository — all SQL for the edges table."""

from __future__ import annotations

from uuid import UUID

import asyncpg


async def upsert_edge(
    conn: asyncpg.Connection,
    *,
    owner_id: UUID,
    source_id: UUID,
    target_id: UUID,
    edge_type: str,
    axis: str,
    weight: float,
) -> bool:
    """Insert or update an edge. Returns True if successful."""
    # Enforce consistent source < target ordering
    src = min(source_id, target_id)
    tgt = max(source_id, target_id)
    try:
        await conn.execute(
            "INSERT INTO edges "
            "(id, owner_id, source_id, target_id, edge_type, axis, weight) "
            "VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6) "
            "ON CONFLICT (owner_id, source_id, target_id, edge_type, axis) "
            "DO UPDATE SET weight = GREATEST(edges.weight, $6), "
            "updated_at = NOW()",
            owner_id,
            src,
            tgt,
            edge_type,
            axis,
            weight,
        )
        return True
    except asyncpg.CheckViolationError:
        return False


async def count_edges_for_node(
    conn: asyncpg.Connection,
    owner_id: UUID,
    node_id: UUID,
) -> int:
    """Count edges involving a specific node."""
    return await conn.fetchval(
        "SELECT COUNT(*) FROM edges "
        "WHERE owner_id = $1 AND (source_id = $2 OR target_id = $2)",
        owner_id,
        node_id,
    )


async def fetch_edges_for_nodes(
    conn: asyncpg.Connection,
    owner_id: UUID,
    node_ids: list[UUID],
    axis_list: list[str],
) -> list[asyncpg.Record]:
    """Fetch edges connected to any of the given nodes on specified axes."""
    return await conn.fetch(
        "SELECT source_id, target_id, edge_type, weight, axis "
        "FROM edges "
        "WHERE owner_id = $1 AND weight > 0.0 "
        "AND (source_id = ANY($2) OR target_id = ANY($2)) "
        "AND axis = ANY($3)",
        owner_id,
        node_ids,
        axis_list,
    )


async def strengthen_co_retrieval(
    conn: asyncpg.Connection,
    owner_id: UUID,
    sources: list[UUID],
    targets: list[UUID],
    axes: list[str],
) -> int:
    """Strengthen edges between co-retrieved nodes. Returns count."""
    if not sources:
        return 0
    result = await conn.execute(
        "UPDATE edges SET "
        "  weight = LEAST(1.0, weight + 0.05), "
        "  updated_at = NOW() "
        "WHERE owner_id = $1 "
        "  AND edge_type IN ('excitatory', 'associative') "
        "  AND (source_id, target_id, axis) IN ("
        "    SELECT * FROM unnest($2::uuid[], $3::uuid[], $4::text[])"
        "  )",
        owner_id,
        sources,
        targets,
        axes,
    )
    return _parse_count(result)


async def fetch_edges_for_transfer(
    conn: asyncpg.Connection,
    owner_id: UUID,
    node_id: UUID,
) -> list[asyncpg.Record]:
    """Fetch all edges for a node (for dedup transfer)."""
    return await conn.fetch(
        "SELECT source_id, target_id, edge_type, axis, weight "
        "FROM edges "
        "WHERE owner_id = $1 AND (source_id = $2 OR target_id = $2)",
        owner_id,
        node_id,
    )


async def delete_edges_for_node(
    conn: asyncpg.Connection,
    owner_id: UUID,
    node_id: UUID,
) -> None:
    """Delete all edges for a node."""
    await conn.execute(
        "DELETE FROM edges "
        "WHERE owner_id = $1 AND (source_id = $2 OR target_id = $2)",
        owner_id,
        node_id,
    )


async def decay_weights(
    conn: asyncpg.Connection,
    owner_id: UUID,
    stale_days: int = 30,
) -> int:
    """Decay edges not co-activated in N days. Returns count."""
    result = await conn.execute(
        "UPDATE edges SET weight = weight * 0.9 "
        "WHERE owner_id = $1 "
        "  AND updated_at < NOW() - INTERVAL '1 day' * $2 "
        "  AND edge_type != 'structural'",
        owner_id,
        stale_days,
    )
    return _parse_count(result)


async def prune_weak(
    conn: asyncpg.Connection,
    owner_id: UUID,
    min_weight: float = 0.15,
) -> int:
    """Delete edges below weight threshold. Returns count."""
    result = await conn.execute(
        "DELETE FROM edges WHERE owner_id = $1 AND weight < $2 "
        "AND edge_type != 'structural'",
        owner_id,
        min_weight,
    )
    return _parse_count(result)


async def count_all(
    conn: asyncpg.Connection,
    owner_id: UUID,
) -> int:
    """Count all edges for an owner."""
    return await conn.fetchval(
        "SELECT COUNT(*) FROM edges WHERE owner_id = $1",
        owner_id,
    )


def _parse_count(result: str) -> int:
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0
