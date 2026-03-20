"""Write path encoding pipeline. No edges — Dreamer handles those."""

from __future__ import annotations

import math
from uuid import UUID

import asyncpg
import structlog

from engram.config import settings
from engram.core.db import tenant_connection
from engram.core.tracing import WRITE_LATENCY, Span
from engram.models import AXES, SourceType, compute_content_hash, generate_node_id
from engram.repositories import memory_repo
from engram.services.embedding import compute_salience, embed_six_dimensions

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
    upsert: bool = False,
    initial_activation: float | None = None,
) -> dict:
    """Full write path.

    Returns:
        {id, salience}              — new memory created
        {duplicate, existing_id}    — content exists, upsert=False
        {id, salience, updated}     — content exists, metadata merged (upsert=True)
    """
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
                existing = await memory_repo.find_by_content_hash(
                    conn, owner_id, content_hash
                )
        if existing:
            if upsert and metadata:
                # Merge new metadata into existing memory
                with Span(
                    "write_path.upsert", component="write_path", expected_ms=10
                ):
                    async with tenant_connection(db_pool, owner_id) as conn:
                        await memory_repo.upsert_metadata(
                            conn,
                            existing,
                            owner_id,
                            metadata,
                            source_type=source_type.value,
                        )
                logger.info(
                    "memory.upserted",
                    component="write_path",
                    node_id=str(existing),
                )
                return {
                    "id": str(existing),
                    "salience": 0.0,
                    "updated": True,
                }
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
                    await memory_repo.insert_memory(
                        conn,
                        node_id=node_id,
                        owner_id=owner_id,
                        content=content,
                        content_hash=content_hash,
                        source_type=source_type.value,
                        session_id=session_id,
                        embedding_model=settings.engram_embedding_model,
                        embedding_dimensions=settings.engram_embedding_dimensions,
                        salience=salience,
                        vectors=vectors,
                        metadata=metadata,
                        activation_level=initial_activation,
                    )
                except asyncpg.UniqueViolationError:
                    # Race condition: another request inserted first
                    race_existing = await memory_repo.find_by_content_hash(
                        conn, owner_id, content_hash
                    )
                    if upsert and metadata and race_existing:
                        await memory_repo.upsert_metadata(
                            conn,
                            race_existing,
                            owner_id,
                            metadata,
                            source_type=source_type.value,
                        )
                        return {
                            "id": str(race_existing),
                            "salience": salience,
                            "updated": True,
                        }
                    return {
                        "duplicate": True,
                        "existing_id": str(race_existing),
                    }

    logger.info(
        "memory.created",
        component="write_path",
        node_id=str(node_id),
        salience=salience,
    )

    await _enqueue_classification(str(owner_id))

    return {"id": str(node_id), "salience": salience}


async def _enqueue_classification(owner_id: str) -> None:
    """Enqueue Dreamer classification for this owner. Fire-and-forget."""
    try:
        from arq import create_pool

        from engram.scheduler import _parse_redis_settings

        redis = await create_pool(_parse_redis_settings())
        await redis.enqueue_job(
            "classify_new_memories",
            owner_id,
            _job_id=f"classify:{owner_id}",  # deduplicate per owner
        )
        await redis.close()
    except Exception:
        # Redis down = classification delayed, not lost.
        # Periodic job or manual trigger picks up unprocessed nodes.
        logger.warning(
            "write_path.enqueue_failed",
            component="write_path",
            owner_id=owner_id,
        )
