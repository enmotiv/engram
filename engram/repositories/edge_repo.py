"""Edge repository — all SQL for the edges table."""

from __future__ import annotations

from uuid import UUID

import asyncpg

from engram.config import settings


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
            "(id, owner_id, source_id, target_id, edge_type, axis, weight, "
            " forward_weight, backward_weight) "
            "VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $6, $6) "
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
        "SELECT source_id, target_id, edge_type, weight, axis, "
        "  forward_weight, backward_weight "
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
        "  forward_weight = LEAST(1.0, forward_weight + 0.05), "
        "  backward_weight = LEAST(1.0, backward_weight + 0.05), "
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
        "SELECT source_id, target_id, edge_type, axis, weight, "
        "  forward_weight, backward_weight "
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
        "UPDATE edges SET "
        "  weight = weight * 0.9, "
        "  forward_weight = forward_weight * 0.9, "
        "  backward_weight = backward_weight * 0.9 "
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
    if settings.engram_flag_stdp:
        result = await conn.execute(
            "DELETE FROM edges WHERE owner_id = $1 "
            "AND GREATEST(forward_weight, backward_weight) < $2 "
            "AND edge_type != 'structural'",
            owner_id,
            min_weight,
        )
    else:
        result = await conn.execute(
            "DELETE FROM edges WHERE owner_id = $1 AND weight < $2 "
            "AND edge_type != 'structural'",
            owner_id,
            min_weight,
        )
    return _parse_count(result)


async def apply_stdp_update(
    conn: asyncpg.Connection,
    owner_id: UUID,
    edge_ids: list[UUID],
    forward_deltas: list[float],
    backward_deltas: list[float],
) -> int:
    """Batch update forward/backward weights via STDP deltas. Returns count."""
    if not edge_ids:
        return 0
    result = await conn.execute(
        "UPDATE edges e SET "
        "  forward_weight = LEAST(1.0, GREATEST(0.0, e.forward_weight + d.fd)), "
        "  backward_weight = LEAST(1.0, GREATEST(0.0, e.backward_weight + d.bd)), "
        "  updated_at = NOW() "
        "FROM (SELECT * FROM unnest($2::uuid[], $3::real[], $4::real[]) "
        "  AS t(eid, fd, bd)) d "
        "WHERE e.id = d.eid AND e.owner_id = $1",
        owner_id,
        edge_ids,
        forward_deltas,
        backward_deltas,
    )
    return _parse_count(result)


async def fetch_edge_ids_for_pairs(
    conn: asyncpg.Connection,
    owner_id: UUID,
    sources: list[UUID],
    targets: list[UUID],
    axes: list[str],
) -> list[asyncpg.Record]:
    """Return edge IDs + current weights for co-retrieved pairs."""
    if not sources:
        return []
    return await conn.fetch(
        "SELECT e.id, e.source_id, e.target_id, e.forward_weight, e.backward_weight "
        "FROM edges e "
        "JOIN (SELECT * FROM unnest($2::uuid[], $3::uuid[], $4::text[]) "
        "  AS t(src, tgt, ax)) p "
        "  ON e.source_id = p.src AND e.target_id = p.tgt AND e.axis = p.ax "
        "WHERE e.owner_id = $1 "
        "  AND e.edge_type IN ('excitatory', 'associative')",
        owner_id,
        sources,
        targets,
        axes,
    )


async def fetch_strongest_forward_edges(
    conn: asyncpg.Connection,
    owner_id: UUID,
    node_id: UUID,
    min_weight: float = 0.3,
    limit: int = 5,
) -> list[asyncpg.Record]:
    """Fetch edges with strongest effective forward weight from a node.

    For edges where node_id is the canonical source (min), forward_weight is used.
    For edges where node_id is the canonical target (max), backward_weight is used.
    """
    return await conn.fetch(
        "SELECT source_id, target_id, edge_type, axis, weight, "
        "  forward_weight, backward_weight, "
        "  CASE WHEN source_id = $2 THEN forward_weight "
        "       ELSE backward_weight END AS effective_forward "
        "FROM edges "
        "WHERE owner_id = $1 "
        "  AND (source_id = $2 OR target_id = $2) "
        "  AND edge_type IN ('excitatory', 'associative') "
        "  AND CASE WHEN source_id = $2 THEN forward_weight "
        "           ELSE backward_weight END >= $3 "
        "ORDER BY effective_forward DESC "
        "LIMIT $4",
        owner_id,
        node_id,
        min_weight,
        limit,
    )


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
