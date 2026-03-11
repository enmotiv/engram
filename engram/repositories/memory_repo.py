"""Memory node repository — all SQL for the memory_nodes table."""

from __future__ import annotations

import json
from uuid import UUID

import asyncpg


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
) -> None:
    """Insert a new memory node with all 6 vectors."""
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
            $9, 1.0, 0,
            $10, $11, $12,
            $13, $14, $15,
            $16, FALSE
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


async def find_by_vector_similarity(
    conn: asyncpg.Connection,
    owner_id: UUID,
    vector: list[float],
    axis: str,
    limit: int = 5,
) -> list[asyncpg.Record]:
    """Find memories by vector similarity on a given axis."""
    col = f"vec_{axis}"
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
    """Fetch activation_level and content for scoring."""
    return await conn.fetch(
        "SELECT id, activation_level, content FROM memory_nodes WHERE id = ANY($1)",
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
        "  activation_level = LEAST(1.0, activation_level + 0.1) "
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
    """Decay activation of stale nodes. Returns count."""
    result = await conn.execute(
        "UPDATE memory_nodes SET "
        "  activation_level = GREATEST(0.01, activation_level * 0.95) "
        "WHERE owner_id = $1 "
        "  AND last_accessed < NOW() - INTERVAL '24 hours' "
        "  AND is_deleted = FALSE "
        "  AND NOT (metadata->>'pinned' = 'true')",
        owner_id,
    )
    return _parse_count(result)


async def find_near_duplicates(
    conn: asyncpg.Connection,
    owner_id: UUID,
    similarity_threshold: float = 0.95,
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


def _parse_count(result: str) -> int:
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0
