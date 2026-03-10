"""The Dreamer: per-axis edge classification, decay, dedup, pruning."""

import json
from uuid import UUID

import asyncpg
import structlog

from engram.db import tenant_connection
from engram.embeddings import llm_classify
from engram.models import AXES
from engram.tracing import DREAMER_CYCLE, Span

logger = structlog.get_logger()

_VALID_EDGE_TYPES = {
    "excitatory",
    "inhibitory",
    "associative",
    "temporal",
    "modulatory",
    "structural",
}
_VALID_AXES = set(AXES)

_CLASSIFICATION_PROMPT = """\
Given two memories for the same user, classify their relationship ON EACH AXIS.

MEMORY A (new): "{new_content}"
MEMORY B (existing): "{old_content}"

For each axis, determine what edges exist between A and B.
Only answer YES if there is a clear, obvious relationship. When in doubt, NO.
A pair can have multiple edge types on the same axis.

Edge types:
- EXCITATORY: A reinforces, supports, or provides evidence for B
- INHIBITORY: A contradicts, replaces, or outdates B
- ASSOCIATIVE: A and B are about related topics but independent
- TEMPORAL: A and B are sequential events (A happened after B)
- MODULATORY: A changes how B should be interpreted in certain contexts

TEMPORAL: Does A change B's temporal relevance? Are they sequential?
EMOTIONAL: Does A change the emotional significance of B?
SEMANTIC: Does A reinforce, contradict, or relate to B's core meaning?
SENSORY: Does A update specific facts, names, numbers, or details in B?
ACTION: Does A change what actions or behaviors B implies?
PROCEDURAL: Does A change routines or patterns described in B?

Return JSON only:
{{
  "temporal": [{{"type": "...", "weight": 0.0}}],
  "emotional": [],
  "semantic": [{{"type": "...", "weight": 0.0}}],
  "sensory": [],
  "action": [],
  "procedural": []
}}
Empty array [] means no edges on that axis."""


# --- Candidate Search ---


async def _find_candidates(
    conn: asyncpg.Connection,
    memory_id: UUID,
    owner_id: UUID,
    vectors: dict,
) -> list[dict]:
    """Find candidate pairs using two search strategies."""
    # Strategy 1: Content similarity via semantic vector
    content_rows = await conn.fetch(
        "SELECT id, GREATEST(0.0, 1 - (vec_semantic <=> $1)) AS score "
        "FROM memory_nodes "
        "WHERE owner_id = $2 AND id != $3 AND is_deleted = FALSE "
        "ORDER BY vec_semantic <=> $1 LIMIT 30",
        vectors["semantic"],
        owner_id,
        memory_id,
    )

    # Strategy 2: Per-axis similarity
    axis_candidates: dict[UUID, float] = {}
    for axis in AXES:
        col = f"vec_{axis}"
        rows = await conn.fetch(
            f"SELECT id, GREATEST(0.0, 1 - ({col} <=> $1)) AS score "  # noqa: S608
            f"FROM memory_nodes "
            f"WHERE owner_id = $2 AND id != $3 AND is_deleted = FALSE "
            f"ORDER BY {col} <=> $1 LIMIT 10",
            vectors[axis],
            owner_id,
            memory_id,
        )
        for row in rows:
            nid = row["id"]
            if nid not in axis_candidates:
                axis_candidates[nid] = row["score"]
            else:
                axis_candidates[nid] = max(axis_candidates[nid], row["score"])

    # Union + deduplicate
    all_ids: set[UUID] = set()
    result: list[dict] = []
    for row in content_rows:
        nid = row["id"]
        if nid not in all_ids:
            all_ids.add(nid)
            result.append({"id": nid, "score": row["score"]})
    for nid, score in axis_candidates.items():
        if nid not in all_ids:
            all_ids.add(nid)
            result.append({"id": nid, "score": score})

    return result


# --- LLM Classification ---


async def _classify_pair(
    new_content: str,
    old_content: str,
) -> dict[str, list[dict]]:
    """Classify edges between two memories on all 6 axes. One LLM call."""
    prompt = _CLASSIFICATION_PROMPT.format(
        new_content=new_content,
        old_content=old_content,
    )
    try:
        raw = await llm_classify(prompt)
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        result = json.loads(clean.strip())
        logger.debug(
            "dreamer.classify_pair",
            component="dreamer",
            edges_by_axis={
                k: len(v) for k, v in result.items() if isinstance(v, list)
            },
        )
        return result
    except (json.JSONDecodeError, IndexError, KeyError, AttributeError):
        logger.warning(
            "dreamer.classify_parse_failed",
            component="dreamer",
            raw=raw[:200] if raw else "",
        )
        return {}


# --- Edge Storage ---


async def _store_edges(
    conn: asyncpg.Connection,
    owner_id: UUID,
    source_id: UUID,
    target_id: UUID,
    classification: dict[str, list[dict]],
) -> int:
    """Store classified edges. Returns count of edges created/updated."""
    # Enforce consistent source < target ordering
    src = min(source_id, target_id)
    tgt = max(source_id, target_id)

    count = 0
    for axis, edges in classification.items():
        if axis not in _VALID_AXES:
            continue
        if not isinstance(edges, list):
            continue
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            etype = str(edge.get("type", "")).lower()
            if etype not in _VALID_EDGE_TYPES:
                continue
            raw_weight = edge.get("weight", 0)
            if not isinstance(raw_weight, (int, float)):
                continue
            weight = min(1.0, max(0.0, float(raw_weight)))
            if weight <= 0.1:
                continue

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
                    etype,
                    axis,
                    weight,
                )
                count += 1
            except asyncpg.CheckViolationError:
                pass  # invalid edge_type or axis rejected by DB CHECK
    return count


# --- Edge Classification (Event-Driven) ---


async def process_new_memories(
    db_pool: asyncpg.Pool,
    owner_id: UUID,
) -> dict:
    """Main Dreamer job: classify edges for all unprocessed memories."""
    stats = {
        "memories_processed": 0,
        "candidates_evaluated": 0,
        "edges_created": 0,
    }

    # Fetch unprocessed memories (with vectors for candidate search)
    async with tenant_connection(db_pool, owner_id) as conn:
        unprocessed = await conn.fetch(
            "SELECT id, content, "
            "vec_temporal, vec_emotional, vec_semantic, "
            "vec_sensory, vec_action, vec_procedural "
            "FROM memory_nodes "
            "WHERE owner_id = $1 AND dreamer_processed = FALSE "
            "AND is_deleted = FALSE "
            "ORDER BY created_at ASC",
            owner_id,
        )

    for memory in unprocessed:
        mem_id: UUID = memory["id"]
        vectors = {axis: memory[f"vec_{axis}"] for axis in AXES}

        # Find candidates + fetch their content (fast DB)
        async with tenant_connection(db_pool, owner_id) as conn:
            candidates = await _find_candidates(conn, mem_id, owner_id, vectors)
            if not candidates:
                await conn.execute(
                    "UPDATE memory_nodes SET dreamer_processed = TRUE "
                    "WHERE id = $1",
                    mem_id,
                )
                stats["memories_processed"] += 1
                continue

            cand_ids = [c["id"] for c in candidates]
            content_rows = await conn.fetch(
                "SELECT id, content FROM memory_nodes WHERE id = ANY($1)",
                cand_ids,
            )

        # Classify all pairs (slow LLM calls, no DB connection held)
        cand_contents = {row["id"]: row["content"] for row in content_rows}
        classifications: dict[UUID, dict] = {}
        for cand in candidates:
            old_content = cand_contents.get(cand["id"])
            if not old_content:
                continue
            classification = await _classify_pair(memory["content"], old_content)
            if classification:
                classifications[cand["id"]] = classification

        stats["candidates_evaluated"] += len(candidates)

        # Store edges + mark processed (fast DB)
        async with tenant_connection(db_pool, owner_id) as conn:
            for cand_id, classification in classifications.items():
                edges = await _store_edges(
                    conn, owner_id, mem_id, cand_id, classification
                )
                stats["edges_created"] += edges

            await conn.execute(
                "UPDATE memory_nodes SET dreamer_processed = TRUE "
                "WHERE id = $1",
                mem_id,
            )
        stats["memories_processed"] += 1

    logger.info(
        "dreamer.classification_complete",
        component="dreamer",
        **stats,
    )
    return stats


# --- Activation Decay (Hourly) ---


async def decay_activations(
    db_pool: asyncpg.Pool,
    owner_id: UUID,
) -> int:
    """Decay activation of nodes not accessed in 24 hours. Returns count."""
    async with tenant_connection(db_pool, owner_id) as conn:
        result = await conn.execute(
            "UPDATE memory_nodes SET "
            "  activation_level = GREATEST(0.01, activation_level * 0.95) "
            "WHERE owner_id = $1 "
            "  AND last_accessed < NOW() - INTERVAL '24 hours' "
            "  AND is_deleted = FALSE "
            "  AND NOT (metadata->>'pinned' = 'true')",
            owner_id,
        )
    count = _parse_update_count(result)
    logger.info(
        "dreamer.activation_decay", component="dreamer", decayed=count
    )
    return count


# --- Edge Weight Decay (Weekly) ---


async def decay_edge_weights(
    db_pool: asyncpg.Pool,
    owner_id: UUID,
) -> int:
    """Decay edges not co-activated in 30 days. Returns count."""
    async with tenant_connection(db_pool, owner_id) as conn:
        result = await conn.execute(
            "UPDATE edges SET weight = weight * 0.9 "
            "WHERE owner_id = $1 "
            "  AND updated_at < NOW() - INTERVAL '30 days' "
            "  AND edge_type != 'structural'",
            owner_id,
        )
    count = _parse_update_count(result)
    logger.info(
        "dreamer.edge_decay", component="dreamer", decayed=count
    )
    return count


# --- Dedup (Weekly) ---


async def _transfer_edges(
    conn: asyncpg.Connection,
    owner_id: UUID,
    from_id: UUID,
    to_id: UUID,
) -> int:
    """Transfer edges from deleted node to survivor. Returns count."""
    edges = await conn.fetch(
        "SELECT source_id, target_id, edge_type, axis, weight "
        "FROM edges "
        "WHERE owner_id = $1 AND (source_id = $2 OR target_id = $2)",
        owner_id,
        from_id,
    )

    transferred = 0
    for edge in edges:
        new_source = to_id if edge["source_id"] == from_id else edge["source_id"]
        new_target = to_id if edge["target_id"] == from_id else edge["target_id"]

        # Skip self-loops
        if new_source == new_target:
            continue

        # Enforce min/max ordering
        src = min(new_source, new_target)
        tgt = max(new_source, new_target)

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
            edge["edge_type"],
            edge["axis"],
            edge["weight"],
        )
        transferred += 1

    # Delete original edges pointing to from_id
    await conn.execute(
        "DELETE FROM edges "
        "WHERE owner_id = $1 AND (source_id = $2 OR target_id = $2)",
        owner_id,
        from_id,
    )

    return transferred


async def dedup_memories(
    db_pool: asyncpg.Pool,
    owner_id: UUID,
) -> dict:
    """Find near-identical memories, keep higher activation, soft-delete other."""
    stats = {"pairs_found": 0, "nodes_deleted": 0, "edges_transferred": 0}

    async with tenant_connection(db_pool, owner_id) as conn:
        pairs = await conn.fetch(
            "WITH sample AS ("
            "  SELECT id, vec_semantic, activation_level"
            "  FROM memory_nodes"
            "  WHERE owner_id = $1 AND is_deleted = FALSE"
            "  ORDER BY RANDOM() LIMIT 200"
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
            "  AND 1 - (s.vec_semantic <=> m.vec_semantic) > 0.95 "
            "LIMIT 50",
            owner_id,
        )

        stats["pairs_found"] = len(pairs)

        for pair in pairs:
            # Keep the one with higher activation
            if pair["act_a"] >= pair["act_b"]:
                survivor, deleted = pair["node_a"], pair["node_b"]
            else:
                survivor, deleted = pair["node_b"], pair["node_a"]

            # Transfer edges
            transferred = await _transfer_edges(conn, owner_id, deleted, survivor)
            stats["edges_transferred"] += transferred

            # Soft-delete
            await conn.execute(
                "UPDATE memory_nodes SET is_deleted = TRUE WHERE id = $1",
                deleted,
            )
            stats["nodes_deleted"] += 1

            logger.info(
                "dreamer.dedup_pair",
                survivor=str(survivor),
                deleted=str(deleted),
                similarity=pair["sim"],
                edges_transferred=transferred,
            )

    logger.info("dreamer.dedup_complete", component="dreamer", **stats)
    return stats


# --- Edge Pruning (Monthly) ---


async def prune_edges(
    db_pool: asyncpg.Pool,
    owner_id: UUID,
) -> int:
    """Delete edges with weight < 0.05. Returns count."""
    async with tenant_connection(db_pool, owner_id) as conn:
        result = await conn.execute(
            "DELETE FROM edges WHERE owner_id = $1 AND weight < 0.05 "
            "AND edge_type != 'structural'",
            owner_id,
        )
    count = _parse_delete_count(result)
    logger.info("dreamer.prune_complete", component="dreamer", pruned=count)
    return count


# --- Helpers ---


def _parse_update_count(result: str) -> int:
    """Parse row count from asyncpg result like 'UPDATE 5'."""
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


def _parse_delete_count(result: str) -> int:
    """Parse row count from asyncpg result like 'DELETE 5'."""
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


# --- Full Cycle (Scheduler Entry Point) ---


async def run_cycle(
    db_pool: asyncpg.Pool,
    owner_id: UUID,
) -> dict:
    """Run full Dreamer cycle: classify, decay, dedup, prune. Audit logged."""
    with Span(
        "dreamer.cycle", component="dreamer", histogram=DREAMER_CYCLE
    ):
        with Span("dreamer.classify", component="dreamer"):
            classification = await process_new_memories(db_pool, owner_id)

        with Span("dreamer.decay_activations", component="dreamer"):
            decay_count = await decay_activations(db_pool, owner_id)

        with Span("dreamer.decay_edges", component="dreamer"):
            edge_decay_count = await decay_edge_weights(db_pool, owner_id)

        with Span("dreamer.dedup", component="dreamer"):
            dedup_stats = await dedup_memories(db_pool, owner_id)

        with Span("dreamer.prune", component="dreamer"):
            prune_count = await prune_edges(db_pool, owner_id)

    detail = {
        "edge_classification": classification,
        "activation_decay": {"nodes_decayed": decay_count},
        "edge_decay": {"edges_decayed": edge_decay_count},
        "dedup": dedup_stats,
        "edge_pruning": {"edges_pruned": prune_count},
    }

    logger.info("dreamer.cycle_complete", component="dreamer", detail=detail)
    return detail
