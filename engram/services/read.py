"""Read path retrieval pipeline: search, score, spread, filter, respond."""

from __future__ import annotations

import asyncio
import json
from uuid import UUID

import asyncpg
import structlog

from engram.config import settings
from engram.core.db import tenant_connection
from engram.core.tracing import RECALL_LATENCY, RECONSOLIDATION_FAILURES, Span, get_trace
from engram.models import (
    AXES,
    ConfidenceLevel,
    DimensionScores,
    EdgeResponse,
    MemoryNode,
    RecallResponse,
    compute_confidence,
)
from engram.repositories import edge_repo, memory_repo
from engram.services.embedding import embed_six_dimensions
from engram.services.reconsolidation import reconsolidate

logger = structlog.get_logger()

# Column names derived from AXES constant — safe for f-string SQL
_VEC_COLUMNS = {axis: f"vec_{axis}" for axis in AXES}

# Propagation factors by edge type
_EDGE_FACTORS: dict[str, float | None] = {
    "excitatory": 0.5,
    "inhibitory": -0.5,
    "associative": 0.25,
    "temporal": 0.15,
    "structural": 0.4,  # client-managed edges (e.g. entity links)
    "modulatory": None,  # multiplicative, handled separately
}


async def _search_axis(
    db_pool: asyncpg.Pool,
    axis: str,
    vector: list[float],
    owner_id: UUID,
    limit: int,
) -> tuple[str, list[tuple[UUID, float]]]:
    """Search one vector column on its own connection. Returns (axis, results)."""
    async with tenant_connection(db_pool, owner_id) as conn:
        rows = await memory_repo.find_by_vector_similarity(
            conn, owner_id, vector, axis, limit=limit
        )
    return axis, [(row["id"], row["score"]) for row in rows]


async def _multi_axis_search(
    db_pool: asyncpg.Pool,
    cue_vectors: dict[str, list[float]],
    owner_id: UUID,
    per_axis_limit: int = 20,
) -> dict[UUID, dict]:
    """Run 6 parallel searches (one connection per axis).

    Returns {node_id: {scores, matched_axes}}.
    """
    candidates: dict[UUID, dict] = {}
    tc = get_trace()

    with Span(
        "read_path.search_dimensions",
        component="read_path",
        expected_ms=30,
    ):
        tasks = [
            _search_axis(db_pool, axis, cue_vectors[axis], owner_id, per_axis_limit)
            for axis in AXES
        ]
        results = await asyncio.gather(*tasks)

        for axis, axis_results in results:
            if tc:
                tc.per_dimension[axis] = {
                    "candidates": len(axis_results),
                    "top_score": round(axis_results[0][1], 4) if axis_results else 0.0,
                }

            for node_id, score in axis_results:
                if node_id not in candidates:
                    candidates[node_id] = {"scores": {}, "matched_axes": []}
                candidates[node_id]["scores"][axis] = score
                candidates[node_id]["matched_axes"].append(axis)

    if tc:
        tc.unique_candidates = len(candidates)

    return candidates


async def _score_candidates(
    conn: asyncpg.Connection,
    candidates: dict[UUID, dict],
    cue: str = "",
) -> list[dict]:
    """Score and rank candidates by convergence * activation.

    Applies a keyword-match boost when cue tokens appear in content,
    compensating for embedding models that struggle with proper nouns.
    """
    node_ids = list(candidates.keys())
    if not node_ids:
        return []

    with Span(
        "read_path.convergence", component="read_path", expected_ms=1
    ):
        rows = await memory_repo.fetch_activation_and_content(conn, node_ids)
        activations = {row["id"]: row["activation_level"] for row in rows}
        contents = {row["id"]: (row["content"] or "").lower() for row in rows}

        # Tokenize cue for keyword matching (words >= 2 chars)
        cue_tokens = [t.lower() for t in cue.split() if len(t) >= 2]

        # Per-axis similarity threshold — ignore weak matches that add noise
        _AXIS_THRESHOLD = 0.3

        scored = []
        keyword_boost_count = 0
        for node_id, data in candidates.items():
            # Filter out weak per-axis matches before scoring
            strong_scores = {
                axis: score for axis, score in data["scores"].items()
                if score >= _AXIS_THRESHOLD
            }
            dims_matched = len(strong_scores)
            if dims_matched < 2:
                continue  # Need convergence across at least 2 axes
            avg_score = sum(strong_scores.values()) / dims_matched
            convergence = dims_matched * avg_score
            activation = activations.get(node_id, 0.0)
            adjusted = convergence * activation

            # Keyword boost: if cue tokens appear in content, boost score
            # Dampened to prevent minor token overlap from inflating scores
            if cue_tokens:
                content_lower = contents.get(node_id, "")
                matched_tokens = sum(1 for t in cue_tokens if t in content_lower)
                if matched_tokens > 0:
                    # 20% max boost instead of 100% — prevents single-token inflation
                    boost = 1.0 + 0.2 * (matched_tokens / len(cue_tokens))
                    adjusted *= boost
                    keyword_boost_count += 1

            scored.append({
                "id": node_id,
                "dims_matched": dims_matched,
                "convergence_score": convergence,
                "activation_level": activation,
                "adjusted": adjusted,
                "dimension_scores": data["scores"],
                "matched_axes": data["matched_axes"],
            })

        scored.sort(key=lambda x: x["adjusted"], reverse=True)
        scored = scored[:10]  # top 10 seeds for spreading activation

    tc = get_trace()
    if tc:
        tc.convergence_scores = [
            {
                "id": str(s["id"]),
                "dims_matched": s["dims_matched"],
                "avg_score": round(
                    s["convergence_score"] / s["dims_matched"], 2
                )
                if s["dims_matched"]
                else 0,
                "convergence": round(s["convergence_score"], 2),
                "activation": round(s["activation_level"], 2),
                "adjusted": round(s["adjusted"], 2),
            }
            for s in scored
        ]
        tc.keyword_boost_count = keyword_boost_count

    return scored


def _apply_edges(
    rows: list,
    source_set: set[UUID],
    activation_map: dict[UUID, float],
) -> tuple[int, int, int, set[UUID]]:
    """Apply edge activations and return (excitatory, inhibitory, modulatory, new_neighbors)."""
    excitatory_count = 0
    inhibitory_count = 0
    modulatory_count = 0
    new_neighbors: set[UUID] = set()

    for row in rows:
        src: UUID = row["source_id"]
        tgt: UUID = row["target_id"]
        etype: str = row["edge_type"]
        weight: float = row["weight"]

        if src in source_set:
            neighbor = tgt
            src_activation = activation_map.get(src, 0.0)
        else:
            neighbor = src
            src_activation = activation_map.get(tgt, 0.0)

        was_new = neighbor not in activation_map

        if etype == "modulatory":
            current = activation_map.get(neighbor, 0.0)
            activation_map[neighbor] = current * weight
            modulatory_count += 1
        else:
            factor = _EDGE_FACTORS.get(etype, 0.0)
            if factor:
                delta = weight * src_activation * abs(factor)
                if factor < 0:
                    delta = -delta
                current = activation_map.get(neighbor, 0.0)
                activation_map[neighbor] = max(0.0, current + delta)

            if etype == "excitatory":
                excitatory_count += 1
            elif etype == "inhibitory":
                inhibitory_count += 1

        if was_new:
            new_neighbors.add(neighbor)

    return excitatory_count, inhibitory_count, modulatory_count, new_neighbors


async def _spreading_activation(
    conn: asyncpg.Connection,
    seeds: list[dict],
    owner_id: UUID,
) -> tuple[dict[UUID, float], list[asyncpg.Record]]:
    """Propagate activation through axis-matched edges. 2 hops."""
    seed_ids = [s["id"] for s in seeds]
    all_matched_axes: set[str] = set()
    for s in seeds:
        all_matched_axes.update(s["matched_axes"])

    activation_map: dict[UUID, float] = {s["id"]: s["adjusted"] for s in seeds}

    if not seed_ids or not all_matched_axes:
        return activation_map, []

    all_edge_rows: list = []
    total_exc = total_inh = total_mod = 0
    axis_list = list(all_matched_axes)

    with Span(
        "read_path.spreading_activation",
        component="read_path",
        expected_ms=10,
    ):
        # Hop 1: from seeds
        rows = await edge_repo.fetch_edges_for_nodes(
            conn, owner_id, seed_ids, axis_list
        )
        all_edge_rows.extend(rows)

        seed_set = set(seed_ids)
        exc, inh, mod, hop1_neighbors = _apply_edges(
            rows, seed_set, activation_map,
        )
        total_exc += exc
        total_inh += inh
        total_mod += mod

        # Hop 2: from newly discovered neighbors (if any)
        if hop1_neighbors:
            hop2_ids = list(hop1_neighbors)
            rows2 = await edge_repo.fetch_edges_for_nodes(
                conn, owner_id, hop2_ids, axis_list
            )
            all_edge_rows.extend(rows2)

            exc2, inh2, mod2, _ = _apply_edges(
                rows2, hop1_neighbors, activation_map,
            )
            total_exc += exc2
            total_inh += inh2
            total_mod += mod2

    tc = get_trace()
    if tc:
        tc.spreading = {
            "edges_loaded": len(all_edge_rows),
            "excitatory_fired": total_exc,
            "inhibitory_fired": total_inh,
            "modulatory_fired": total_mod,
            "nodes_activated_by_spread": len(activation_map) - len(seeds),
            "hop1_neighbors": len(hop1_neighbors) if hop1_neighbors else 0,
        }

    return activation_map, all_edge_rows


async def _fetch_nodes(
    conn: asyncpg.Connection,
    node_ids: list[UUID],
) -> dict[UUID, MemoryNode]:
    """Batch fetch full node data for response building."""
    if not node_ids:
        return {}

    rows = await memory_repo.fetch_full_nodes(conn, node_ids)

    result = {}
    for row in rows:
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        result[row["id"]] = MemoryNode(
            id=row["id"],
            owner_id=row["owner_id"],
            content=row["content"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            activation_level=row["activation_level"],
            salience=row["salience"],
            source_type=row["source_type"],
            session_id=row["session_id"],
            embedding_model=row["embedding_model"],
            embedding_dimensions=row["embedding_dimensions"],
            metadata=meta,
            is_deleted=row["is_deleted"],
            dreamer_processed=row["dreamer_processed"],
        )

    return result


def _apply_post_filter(
    ranked: list[dict],
    nodes_by_id: dict[UUID, MemoryNode],
    *,
    include_type: str | None = None,
) -> list[dict]:
    """Filter nodes by metadata type.

    When *include_type* is set, ONLY nodes with that metadata.type are kept
    (the global exclude list is ignored).  Otherwise the global
    ENGRAM_RETRIEVAL_EXCLUDE_TAGS exclusion applies as before.
    """
    tc = get_trace()

    if include_type:
        filtered = [
            n
            for n in ranked
            if n["id"] in nodes_by_id
            and nodes_by_id[n["id"]].metadata.get("type") == include_type
        ]
        if tc:
            tc.post_filter = {
                "nodes_excluded": len(ranked) - len(filtered),
                "include_type": include_type,
            }
        return filtered

    exclude = settings.exclude_tags_set

    if not exclude:
        if tc:
            tc.post_filter = {"nodes_excluded": 0, "exclude_tags": []}
        return ranked

    filtered = [
        n
        for n in ranked
        if n["id"] in nodes_by_id
        and nodes_by_id[n["id"]].metadata.get("type") not in exclude
    ]

    if tc:
        tc.post_filter = {
            "nodes_excluded": len(ranked) - len(filtered),
            "exclude_tags": sorted(exclude),
        }

    return filtered


async def recall_memories(
    db_pool: asyncpg.Pool,
    owner_id: UUID,
    cue: str,
    top_k: int = 5,
    min_convergence: float = 0.0,
    include_edges: bool = False,
    metadata_type: str | None = None,
) -> RecallResponse:
    """Full read path. Returns RecallResponse."""
    with Span(
        "read_path.total",
        component="read_path",
        expected_ms=650,
        histogram=RECALL_LATENCY,
    ):
        # Phase A: embed cue (no DB connection held)
        with Span(
            "read_path.embed", component="read_path", expected_ms=500
        ):
            cue_vectors = await embed_six_dimensions(cue)

        tc = get_trace()

        # Phase B1: parallel multi-axis search (6 connections)
        candidates = await _multi_axis_search(
            db_pool, cue_vectors, owner_id
        )

        if not candidates:
            return RecallResponse(
                memories=[], confidence=ConfidenceLevel.LOW, edges=[]
            )

        # Phase B2: score + spread + fetch (one connection)
        async with tenant_connection(db_pool, owner_id) as conn:
            # 2. Convergence scoring
            scored = await _score_candidates(conn, candidates, cue=cue)

            # 3. Spreading activation
            activation_map, edge_rows = await _spreading_activation(
                conn, scored, owner_id
            )

            # 4. Fetch all node data (search results + spread neighbors)
            all_ids = list(
                set(candidates.keys()) | set(activation_map.keys())
            )
            nodes_by_id = await _fetch_nodes(conn, all_ids)

        # Phase C: rank, filter, build response (no DB needed)
        # Same per-axis threshold as scoring phase
        _AXIS_THRESHOLD = 0.3

        ranked = []
        for node_id in all_ids:
            node = nodes_by_id.get(node_id)
            if node is None or node.is_deleted:
                continue

            cand = candidates.get(node_id)
            if cand:
                strong = {a: s for a, s in cand["scores"].items() if s >= _AXIS_THRESHOLD}
                dims_matched = len(strong)
                convergence = (
                    dims_matched * (sum(strong.values()) / dims_matched)
                    if dims_matched > 0 else 0.0
                )
                dim_scores = cand["scores"]
                matched = cand["matched_axes"]
            else:
                convergence = 0.0
                dim_scores = {}
                matched = []

            final_activation = activation_map.get(node_id, 0.0)
            final_score = (
                convergence * final_activation
                if convergence > 0
                else final_activation
            )

            ranked.append({
                "id": node_id,
                "convergence_score": convergence,
                "dims_matched": len(dim_scores),
                "dimension_scores": dim_scores,
                "matched_axes": matched,
                "final_score": final_score,
            })

        ranked.sort(key=lambda x: x["final_score"], reverse=True)

        # Post-filter
        filtered = _apply_post_filter(ranked, nodes_by_id, include_type=metadata_type)

        # Min convergence filter
        if min_convergence > 0.0:
            filtered = [
                n
                for n in filtered
                if n["convergence_score"] >= min_convergence
            ]

        # Trim to top_k
        top = filtered[:top_k]

        # Build response
        memories = []
        for item in top:
            node = nodes_by_id[item["id"]]
            dim_scores_model = DimensionScores(
                **{
                    axis: item["dimension_scores"].get(axis, 0.0)
                    for axis in AXES
                }
            )
            memories.append(
                node.to_response(
                    convergence_score=item["convergence_score"],
                    dimension_scores=dim_scores_model,
                    matched_axes=item["matched_axes"],
                )
            )

        # Confidence from best result
        if top:
            confidence = compute_confidence({
                "dims_matched": top[0]["dims_matched"],
                "convergence_score": top[0]["convergence_score"],
            })
        else:
            confidence = ConfidenceLevel.LOW

        # Edges (only if requested)
        edges = []
        if include_edges and edge_rows:
            for row in edge_rows:
                edges.append(
                    EdgeResponse(
                        source_id=str(row["source_id"]),
                        target_id=str(row["target_id"]),
                        edge_type=row["edge_type"],
                        axis=row["axis"],
                        weight=row["weight"],
                    )
                )

        # Reconsolidation: boost nodes, strengthen co-retrieval edges
        if top:
            with Span(
                "read_path.reconsolidate",
                component="read_path",
                expected_ms=15,
            ):
                try:
                    recon_stats = await reconsolidate(
                        db_pool, owner_id, top
                    )
                    if tc:
                        tc.reconsolidation = recon_stats
                except Exception:
                    RECONSOLIDATION_FAILURES.inc()
                    logger.error(
                        "reconsolidation.failed",
                        component="read_path",
                        exc_info=True,
                    )

    return RecallResponse(
        memories=memories, confidence=confidence, edges=edges
    )
