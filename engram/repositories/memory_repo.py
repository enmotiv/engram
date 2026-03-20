"""Memory node repository — all SQL for the memory_nodes table."""

from __future__ import annotations

import json
from uuid import UUID

import asyncpg

# Default activation levels by source type — corrections are most important,
# system metadata least. Observation is the fallback for unknown types.
_SOURCE_ACTIVATION: dict[str, float] = {
    "correction": 1.0,
    "observation": 0.7,
    "event": 0.6,
    "conversation": 0.5,
    "system": 0.4,
}


async def insert_memory(
    conn: asyncpg.Connection,
    *,
    node_id: UUID,
    owner_id: UUID,
    content: str,
    content_hash: str,
    source_type: str,
    session_id: UUID | None,
    embedding_model: str,
    embedding_dimensions: int,
    salience: float,
    vectors: dict[str, list[float]],
    metadata: dict | None = None,
    activation_level: float | None = None,
) -> None:
    """Insert a new memory node with all 6 vectors."""
    activation = (
        activation_level
        if activation_level is not None
        else _SOURCE_ACTIVATION.get(source_type, 0.7)
    )
    await conn.execute(
        """INSERT INTO memory_nodes (
            id, owner_id, content, content_hash, source_type,
            session_id, embedding_model, embedding_dimensions,
            salience, activation_level, access_count,
            vec_temporal, vec_emotional, vec_semantic,
            vec_sensory, vec_action, vec_procedural,
            metadata, dreamer_processed
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8,
            $9, $10, 0,
            $11, $12, $13,
            $14, $15, $16,
            $17, FALSE
        )""",
        node_id,
        owner_id,
        content,
        content_hash,
        source_type,
        session_id,
        embedding_model,
        embedding_dimensions,
        salience,
        activation,
        vectors["temporal"],
        vectors["emotional"],
        vectors["semantic"],
        vectors["sensory"],
        vectors["action"],
        vectors["procedural"],
        json.dumps(metadata) if metadata else "{}",
    )


async def find_by_content_hash(
    conn: asyncpg.Connection,
    owner_id: UUID,
    content_hash: str,
) -> UUID | None:
    """Return the ID of an existing memory with this content hash, or None."""
    return await conn.fetchval(
        "SELECT id FROM memory_nodes "
        "WHERE owner_id = $1 AND content_hash = $2",
        owner_id,
        content_hash,
    )


async def upsert_metadata(
    conn: asyncpg.Connection,
    memory_id: UUID,
    owner_id: UUID,
    metadata: dict,
    source_type: str | None = None,
) -> bool:
    """Merge new metadata into an existing memory's JSONB metadata.

    Uses jsonb || jsonb to merge top-level keys. New keys are added,
    existing keys are overwritten. Returns True if a row was updated.
    """
    if source_type:
        result = await conn.execute(
            "UPDATE memory_nodes "
            "SET metadata = metadata || $3::jsonb, "
            "    source_type = $4, "
            "    last_accessed = NOW() "
            "WHERE id = $1 AND owner_id = $2 AND is_deleted = FALSE",
            memory_id,
            owner_id,
            json.dumps(metadata),
            source_type,
        )
    else:
        result = await conn.execute(
            "UPDATE memory_nodes "
            "SET metadata = metadata || $3::jsonb, "
            "    last_accessed = NOW() "
            "WHERE id = $1 AND owner_id = $2 AND is_deleted = FALSE",
            memory_id,
            owner_id,
            json.dumps(metadata),
        )
    return _parse_count(result) > 0


async def find_by_vector_similarity(
    conn: asyncpg.Connection,
    owner_id: UUID,
    vector: list[float],
    axis: str,
    limit: int = 5,
    exclude_id: UUID | None = None,
) -> list[asyncpg.Record]:
    """Find memories by vector similarity on a given axis."""
    col = f"vec_{axis}"
    if exclude_id is not None:
        return await conn.fetch(
            f"SELECT id, content, GREATEST(0.0, 1 - ({col} <=> $1)) AS score "  # noqa: S608
            f"FROM memory_nodes "
            f"WHERE owner_id = $2 AND id != $3 AND is_deleted = FALSE "
            f"ORDER BY {col} <=> $1 LIMIT $4",
            vector,
            owner_id,
            exclude_id,
            limit,
        )
    return await conn.fetch(
        f"SELECT id, content, GREATEST(0.0, 1 - ({col} <=> $1)) AS score "  # noqa: S608
        f"FROM memory_nodes "
        f"WHERE owner_id = $2 AND is_deleted = FALSE "
        f"ORDER BY {col} <=> $1 LIMIT $3",
        vector,
        owner_id,
        limit,
    )


async def fetch_activation_and_content(
    conn: asyncpg.Connection,
    node_ids: list[UUID],
) -> list[asyncpg.Record]:
    """Fetch activation_level, salience, and content for scoring."""
    return await conn.fetch(
        "SELECT id, activation_level, salience, content FROM memory_nodes WHERE id = ANY($1)",
        node_ids,
    )


async def fetch_full_nodes(
    conn: asyncpg.Connection,
    node_ids: list[UUID],
) -> list[asyncpg.Record]:
    """Batch fetch full node data for response building."""
    if not node_ids:
        return []
    return await conn.fetch(
        "SELECT id, owner_id, content, content_hash, created_at, "
        "last_accessed, access_count, activation_level, salience, "
        "source_type, session_id, embedding_model, embedding_dimensions, "
        "metadata, is_deleted, dreamer_processed "
        "FROM memory_nodes WHERE id = ANY($1)",
        node_ids,
    )


async def fetch_unprocessed(
    conn: asyncpg.Connection,
    owner_id: UUID,
) -> list[asyncpg.Record]:
    """Fetch all unprocessed memories with vectors for Dreamer classification."""
    return await conn.fetch(
        "SELECT id, content, "
        "vec_temporal, vec_emotional, vec_semantic, "
        "vec_sensory, vec_action, vec_procedural "
        "FROM memory_nodes "
        "WHERE owner_id = $1 AND dreamer_processed = FALSE "
        "AND is_deleted = FALSE "
        "ORDER BY created_at ASC",
        owner_id,
    )


async def mark_processed(
    conn: asyncpg.Connection,
    memory_id: UUID,
) -> None:
    """Mark a memory as processed by the Dreamer."""
    await conn.execute(
        "UPDATE memory_nodes SET dreamer_processed = TRUE WHERE id = $1",
        memory_id,
    )


async def boost_nodes(
    conn: asyncpg.Connection,
    node_ids: list[UUID],
    owner_id: UUID,
) -> int:
    """Boost activation for retrieved nodes. Returns count updated."""
    result = await conn.execute(
        "UPDATE memory_nodes SET "
        "  last_accessed = NOW(), "
        "  access_count = access_count + 1, "
        "  activation_level = LEAST(1.0, activation_level "
        "    + 0.05 * (1.0 - activation_level) + 0.02) "
        "WHERE id = ANY($1) AND owner_id = $2",
        node_ids,
        owner_id,
    )
    return _parse_count(result)


async def soft_delete(
    conn: asyncpg.Connection,
    memory_id: UUID,
    owner_id: UUID,
) -> int:
    """Soft-delete a memory. Returns count updated."""
    result = await conn.execute(
        "UPDATE memory_nodes SET is_deleted = TRUE "
        "WHERE id = $1 AND owner_id = $2 AND is_deleted = FALSE",
        memory_id,
        owner_id,
    )
    return _parse_count(result)


async def decay_activations(
    conn: asyncpg.Connection,
    owner_id: UUID,
) -> int:
    """Decay activation of stale nodes.

    Decay rate depends on source type and edge density:
    - Corrections barely decay (0.99)
    - Observations use edge-density: hubs (5+ edges) decay at 0.98,
      connected (2+) at 0.95, isolated at 0.92
    - Events decay at 0.94, conversations at 0.90, system at 0.88
    """
    result = await conn.execute(
        """UPDATE memory_nodes mn SET
          activation_level = GREATEST(0.01, mn.activation_level * (
            CASE
              WHEN mn.source_type = 'correction' THEN 0.99
              WHEN mn.source_type = 'observation' THEN
                CASE
                  WHEN sub.edge_ct >= 5 THEN 0.98
                  WHEN sub.edge_ct >= 2 THEN 0.95
                  ELSE 0.92
                END
              WHEN mn.source_type = 'event' THEN 0.94
              WHEN mn.source_type = 'conversation' THEN 0.90
              WHEN mn.source_type = 'system' THEN 0.88
              ELSE 0.95
            END
          ))
        FROM (
          SELECT mn2.id, COUNT(en.id) AS edge_ct
          FROM memory_nodes mn2
          LEFT JOIN edge_nodes en ON (en.source_id = mn2.id OR en.target_id = mn2.id)
            AND en.is_deleted = FALSE
          WHERE mn2.owner_id = $1
            AND mn2.last_accessed < NOW() - INTERVAL '24 hours'
            AND mn2.is_deleted = FALSE
            AND NOT (mn2.metadata->>'pinned' = 'true')
          GROUP BY mn2.id
        ) sub
        WHERE mn.id = sub.id""",
        owner_id,
    )
    return _parse_count(result)


async def find_near_duplicates(
    conn: asyncpg.Connection,
    owner_id: UUID,
    similarity_threshold: float = 0.88,
    sample_size: int = 200,
    max_pairs: int = 50,
) -> list[asyncpg.Record]:
    """Find near-duplicate memory pairs by semantic similarity."""
    return await conn.fetch(
        "WITH sample AS ("
        "  SELECT id, vec_semantic, activation_level"
        "  FROM memory_nodes"
        "  WHERE owner_id = $1 AND is_deleted = FALSE"
        "  ORDER BY RANDOM() LIMIT $2"
        ") "
        "SELECT s.id AS node_a, m.id AS node_b,"
        "  1 - (s.vec_semantic <=> m.vec_semantic) AS sim,"
        "  s.activation_level AS act_a,"
        "  m.activation_level AS act_b "
        "FROM sample s "
        "JOIN memory_nodes m "
        "  ON m.owner_id = $1"
        "  AND m.id > s.id"
        "  AND m.is_deleted = FALSE"
        "  AND 1 - (s.vec_semantic <=> m.vec_semantic) > $3 "
        "LIMIT $4",
        owner_id,
        sample_size,
        similarity_threshold,
        max_pairs,
    )


async def count_active_nodes(
    conn: asyncpg.Connection,
    owner_id: UUID,
    node_ids: list[UUID],
) -> int:
    """Count how many of the given nodes exist and are not deleted."""
    return await conn.fetchval(
        "SELECT COUNT(*) FROM memory_nodes "
        "WHERE owner_id = $1 AND id = ANY($2) AND is_deleted = FALSE",
        owner_id,
        node_ids,
    )


async def fetch_owner_stats(
    conn: asyncpg.Connection,
    owner_id: UUID,
) -> asyncpg.Record:
    """Fetch aggregate stats for an owner's memory nodes."""
    return await conn.fetchrow(
        "SELECT "
        "  COUNT(*) FILTER (WHERE is_deleted = FALSE) AS node_count, "
        "  AVG(activation_level) FILTER (WHERE is_deleted = FALSE) "
        "    AS avg_activation, "
        "  COUNT(*) FILTER "
        "    (WHERE dreamer_processed = FALSE AND is_deleted = FALSE) "
        "    AS unprocessed_nodes "
        "FROM memory_nodes "
        "WHERE owner_id = $1",
        owner_id,
    )


async def fetch_single_memory(
    conn: asyncpg.Connection,
    memory_id: UUID,
    owner_id: UUID,
) -> asyncpg.Record | None:
    """Fetch a single non-deleted memory by ID."""
    return await conn.fetchrow(
        "SELECT id, content, content_hash, created_at, last_accessed, "
        "access_count, activation_level, salience, source_type, "
        "session_id, metadata "
        "FROM memory_nodes "
        "WHERE id = $1 AND owner_id = $2 AND is_deleted = FALSE",
        memory_id,
        owner_id,
    )


async def list_memories_paginated(
    conn: asyncpg.Connection,
    owner_id: UUID,
    *,
    limit: int,
    order_sql: str,
    cursor_dt=None,
    cursor_direction: str = "before",
    source_types: list[str] | None = None,
) -> list[asyncpg.Record]:
    """List memories with cursor pagination and optional filters."""
    conditions = ["owner_id = $1", "is_deleted = FALSE"]
    params: list = [owner_id]
    idx = 2

    if cursor_dt is not None:
        if cursor_direction == "after":
            conditions.append(f"created_at > ${idx}")
        else:
            conditions.append(f"created_at < ${idx}")
        params.append(cursor_dt)
        idx += 1

    if source_types:
        conditions.append(f"source_type = ANY(${idx})")
        params.append(source_types)
        idx += 1

    params.append(limit)

    where = " AND ".join(conditions)
    return await conn.fetch(
        f"SELECT id, content, content_hash, created_at, last_accessed, "  # noqa: S608
        f"access_count, activation_level, salience, source_type, "
        f"session_id, metadata "
        f"FROM memory_nodes "
        f"WHERE {where} "
        f"ORDER BY {order_sql} "
        f"LIMIT ${idx}",
        *params,
    )


async def fetch_content_by_ids(
    conn: asyncpg.Connection,
    node_ids: list[UUID],
) -> list[asyncpg.Record]:
    """Fetch id and content for a list of node IDs."""
    return await conn.fetch(
        "SELECT id, content FROM memory_nodes WHERE id = ANY($1)",
        node_ids,
    )


def _parse_count(result: str) -> int:
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0
