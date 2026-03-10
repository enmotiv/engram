"""Write path encoding pipeline. No edges — Dreamer handles those."""

import json
import math
from uuid import UUID

import asyncpg
import structlog

from engram.config import settings
from engram.db import tenant_connection
from engram.embeddings import compute_salience, embed_six_dimensions
from engram.models import AXES, SourceType, compute_content_hash, generate_node_id
from engram.tracing import WRITE_LATENCY, Span

logger = structlog.get_logger()


def _validate_vectors(vectors: dict[str, list[float]]) -> None:
    """Validate all 6 vectors have correct dimensions and finite values."""
    expected = settings.engram_embedding_dimensions
    for axis in AXES:
        vec = vectors[axis]
        if len(vec) != expected:
            msg = f"Vector {axis} has {len(vec)} dims, expected {expected}"
            raise ValueError(msg)
        if not all(math.isfinite(v) for v in vec):
            msg = f"Vector {axis} contains non-finite values"
            raise ValueError(msg)


async def encode_memory(
    db_pool: asyncpg.Pool,
    owner_id: UUID,
    content: str,
    source_type: SourceType,
    session_id: UUID | None = None,
    metadata: dict | None = None,
) -> dict:
    """Full write path. Returns {id, salience} or {duplicate, existing_id}."""
    metadata = metadata or {}
    content_hash = compute_content_hash(content)

    with Span(
        "write_path.total",
        component="write_path",
        expected_ms=600,
        histogram=WRITE_LATENCY,
    ):
        # Phase A: dedup check (fast DB hit, release connection immediately)
        with Span("write_path.dedup", component="write_path", expected_ms=5):
            async with tenant_connection(db_pool, owner_id) as conn:
                existing = await conn.fetchval(
                    "SELECT id FROM memory_nodes "
                    "WHERE owner_id = $1 AND content_hash = $2",
                    owner_id,
                    content_hash,
                )
        if existing:
            return {"duplicate": True, "existing_id": str(existing)}

        # Phase B: embed + salience (no DB connection held)
        with Span("write_path.embed", component="write_path", expected_ms=500):
            vectors = await embed_six_dimensions(content)
            _validate_vectors(vectors)

        with Span("write_path.salience", component="write_path", expected_ms=1):
            salience = await compute_salience(vectors["emotional"])

        # Phase C: insert (fast DB hit)
        node_id = generate_node_id()
        with Span(
            "write_path.insert", component="write_path", expected_ms=20
        ):
            async with tenant_connection(db_pool, owner_id) as conn:
                try:
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
                        source_type.value,
                        session_id,
                        settings.engram_embedding_model,
                        settings.engram_embedding_dimensions,
                        salience,
                        vectors["temporal"],
                        vectors["emotional"],
                        vectors["semantic"],
                        vectors["sensory"],
                        vectors["action"],
                        vectors["procedural"],
                        json.dumps(metadata) if metadata else "{}",
                    )
                except asyncpg.UniqueViolationError:
                    existing = await conn.fetchval(
                        "SELECT id FROM memory_nodes "
                        "WHERE owner_id = $1 AND content_hash = $2",
                        owner_id,
                        content_hash,
                    )
                    return {"duplicate": True, "existing_id": str(existing)}

    logger.info(
        "memory.created",
        component="write_path",
        node_id=str(node_id),
        salience=salience,
    )
    return {"id": str(node_id), "salience": salience}
