"""The Dreamer: per-axis edge classification, decay, dedup, pruning."""

import asyncio
import json
import re
from uuid import UUID

import asyncpg
import structlog

from engram.config import settings
from engram.core.db import tenant_connection
from engram.core.tracing import DREAMER_CYCLE, Span
from engram.models import AXES
from engram.repositories import edge_repo, memory_repo
from engram.services.embedding import llm_classify

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

STRICT RULES:
- Most memory pairs have NO meaningful relationship. Default to empty arrays.
- Only create an edge if the relationship is specific and non-obvious.
- "Both are about the user's work" is NOT a relationship. That's generic similarity.
- "Both mention a project" is NOT a relationship unless one directly changes, updates, or contradicts the other.
- Weights must reflect strength: 0.3 = weak but real, 0.5 = moderate, 0.7 = strong, 0.9 = direct cause/effect.
- If you're unsure, return an empty array for that axis.
- A good classification produces edges on 1-2 axes at most. If you're creating edges on 4+ axes, you're being too generous.

Edge types:
- EXCITATORY: A directly reinforces or provides specific evidence for B
- INHIBITORY: A contradicts, replaces, or outdates specific claims in B
- ASSOCIATIVE: A and B share a specific entity, event, or decision (not just a topic)
- TEMPORAL: A and B are sequential events in a clear timeline
- MODULATORY: A fundamentally changes how B should be interpreted

TEMPORAL: Are A and B sequential events in a clear timeline?
EMOTIONAL: Does A specifically change the emotional significance of B?
SEMANTIC: Does A reinforce, contradict, or update B's core meaning with specific evidence?
SENSORY: Does A update specific facts, names, numbers, or details in B?
ACTION: Does A directly change what actions or behaviors B implies?
PROCEDURAL: Does A change specific routines or patterns described in B?

Return JSON only. Empty arrays are expected and correct for most axes:
{{
  "temporal": [],
  "emotional": [],
  "semantic": [],
  "sensory": [],
  "action": [],
  "procedural": []
}}"""


# --- Candidate Search ---


async def _find_candidates(
    conn: asyncpg.Connection,
    memory_id: UUID,
    owner_id: UUID,
    vectors: dict,
) -> list[dict]:
    """Find candidate pairs using two search strategies."""
    # Strategy 1: Content similarity via semantic vector
    content_rows = await memory_repo.find_by_vector_similarity(
        conn, owner_id, vectors["semantic"], "semantic",
        limit=15, exclude_id=memory_id,
    )

    # Strategy 2: Per-axis similarity
    axis_candidates: dict[UUID, float] = {}
    for axis in AXES:
        rows = await memory_repo.find_by_vector_similarity(
            conn, owner_id, vectors[axis], axis,
            limit=5, exclude_id=memory_id,
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

    # Filter: only candidates with meaningful similarity
    MIN_CANDIDATE_SIMILARITY = 0.55
    result = [c for c in result if c["score"] >= MIN_CANDIDATE_SIMILARITY]

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
        result = _extract_json(raw)
        logger.debug(
            "dreamer.classify_pair",
            component="dreamer",
            edges_by_axis={
                k: len(v) for k, v in result.items() if isinstance(v, list)
            },
        )
        return result
    except (json.JSONDecodeError, IndexError, KeyError, AttributeError, ValueError):
        logger.warning(
            "dreamer.classify_parse_failed",
            component="dreamer",
            raw=raw[:200] if raw else "",
        )
        return {}


def _extract_json(raw: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and surrounding text."""
    # Try 1: direct parse (ideal case)
    clean = raw.strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Try 2: extract from markdown code fence (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", clean, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try 3: find the first { ... } block (greedy from first { to last })
    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(clean[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in response")


# --- Edge Storage ---


async def _store_edges(
    conn: asyncpg.Connection,
    owner_id: UUID,
    source_id: UUID,
    target_id: UUID,
    classification: dict[str, list[dict]],
) -> int:
    """Store classified edges. Returns count of edges created/updated."""
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
            if weight <= 0.3:
                continue

            success = await edge_repo.upsert_edge(
                conn,
                owner_id=owner_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=etype,
                axis=axis,
                weight=weight,
            )
            if success:
                count += 1
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
        unprocessed = await memory_repo.fetch_unprocessed(conn, owner_id)

    MAX_EDGES_PER_NODE = 20

    for memory in unprocessed:
        mem_id: UUID = memory["id"]
        vectors = {axis: memory[f"vec_{axis}"] for axis in AXES}

        # Skip if this node already has enough edges
        async with tenant_connection(db_pool, owner_id) as conn:
            existing_edge_count = await edge_repo.count_edges_for_node(
                conn, owner_id, mem_id
            )
        if existing_edge_count >= MAX_EDGES_PER_NODE:
            async with tenant_connection(db_pool, owner_id) as conn:
                await memory_repo.mark_processed(conn, mem_id)
            stats["memories_processed"] += 1
            continue

        # Find candidates + fetch their content (fast DB)
        async with tenant_connection(db_pool, owner_id) as conn:
            candidates = await _find_candidates(conn, mem_id, owner_id, vectors)
            if not candidates:
                await memory_repo.mark_processed(conn, mem_id)
                stats["memories_processed"] += 1
                continue

            cand_ids = [c["id"] for c in candidates]
            content_rows = await memory_repo.fetch_content_by_ids(conn, cand_ids)

        # Classify all pairs in parallel (slow LLM calls, no DB connection held)
        cand_contents = {row["id"]: row["content"] for row in content_rows}
        classify_tasks = []
        classify_ids = []
        for cand in candidates:
            old_content = cand_contents.get(cand["id"])
            if not old_content:
                continue
            classify_ids.append(cand["id"])
            classify_tasks.append(
                _classify_pair(memory["content"], old_content)
            )

        # Run up to 5 concurrently to respect OpenRouter rate limits
        _CONCURRENCY = 5
        classifications: dict[UUID, dict] = {}
        for batch_start in range(0, len(classify_tasks), _CONCURRENCY):
            batch = classify_tasks[batch_start:batch_start + _CONCURRENCY]
            batch_ids = classify_ids[batch_start:batch_start + _CONCURRENCY]
            results = await asyncio.gather(*batch, return_exceptions=True)
            for cand_id, result in zip(batch_ids, results):
                if isinstance(result, Exception):
                    logger.warning(
                        "dreamer.classify_failed",
                        component="dreamer",
                        candidate_id=str(cand_id),
                        error=str(result),
                    )
                    continue
                if result:
                    classifications[cand_id] = result

        stats["candidates_evaluated"] += len(candidates)

        # Store edges + mark processed (fast DB)
        async with tenant_connection(db_pool, owner_id) as conn:
            total_edges_for_memory = 0
            for cand_id, classification in classifications.items():
                edges = await _store_edges(
                    conn, owner_id, mem_id, cand_id, classification
                )
                stats["edges_created"] += edges
                total_edges_for_memory += edges

            # Phase 4: Reduce plasticity after edge creation
            if settings.engram_flag_metaplasticity and total_edges_for_memory > 0:
                plasticity_decrement = 0.01 * total_edges_for_memory
                await conn.execute(
                    "UPDATE memory_nodes SET "
                    "  plasticity = GREATEST(0.1, plasticity - $2), "
                    "  modification_count = modification_count + 1 "
                    "WHERE id = $1",
                    mem_id,
                    plasticity_decrement,
                )

            await memory_repo.mark_processed(conn, mem_id)
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
        count = await memory_repo.decay_activations(conn, owner_id)
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
        count = await edge_repo.decay_weights(conn, owner_id)
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
    edges = await edge_repo.fetch_edges_for_transfer(conn, owner_id, from_id)

    transferred = 0
    for edge in edges:
        new_source = to_id if edge["source_id"] == from_id else edge["source_id"]
        new_target = to_id if edge["target_id"] == from_id else edge["target_id"]

        # Skip self-loops
        if new_source == new_target:
            continue

        success = await edge_repo.upsert_edge(
            conn,
            owner_id=owner_id,
            source_id=new_source,
            target_id=new_target,
            edge_type=edge["edge_type"],
            axis=edge["axis"],
            weight=edge["weight"],
        )
        if success:
            transferred += 1

    # Delete original edges pointing to from_id
    await edge_repo.delete_edges_for_node(conn, owner_id, from_id)

    return transferred


async def dedup_memories(
    db_pool: asyncpg.Pool,
    owner_id: UUID,
) -> dict:
    """Find near-identical memories, keep higher activation, soft-delete other."""
    stats = {"pairs_found": 0, "nodes_deleted": 0, "edges_transferred": 0}

    async with tenant_connection(db_pool, owner_id) as conn:
        pairs = await memory_repo.find_near_duplicates(conn, owner_id)

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

            # Soft-delete (using direct execute since we don't need owner_id check
            # within an already-tenant-scoped connection and soft_delete expects
            # the owner_id param for RLS)
            await memory_repo.soft_delete(conn, deleted, owner_id)
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
    """Delete edges with weight < 0.15. Returns count."""
    async with tenant_connection(db_pool, owner_id) as conn:
        count = await edge_repo.prune_weak(conn, owner_id)
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
