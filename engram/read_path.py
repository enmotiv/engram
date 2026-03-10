"""Read path retrieval pipeline: search, score, spread, filter, respond."""

import asyncio
import json
from uuid import UUID

import asyncpg
import structlog

from engram.config import settings
from engram.db import tenant_connection
from engram.embeddings import embed_six_dimensions
from engram.models import (
    AXES,
    ConfidenceLevel,
    DimensionScores,
    EdgeResponse,
    MemoryNode,
    RecallResponse,
    compute_confidence,
)
from engram.reconsolidation import reconsolidate
from engram.tracing import RECALL_LATENCY, RECONSOLIDATION_FAILURES, Span, get_trace

logger = structlog.get_logger()

# Column names derived from AXES constant — safe for f-string SQL
_VEC_COLUMNS = {axis: f"vec_{axis}" for axis in AXES}

# Propagation factors by edge type
_EDGE_FACTORS: dict[str, float | None] = {
    "excitatory": 0.5,
    "inhibitory": -0.5,
    "associative": 0.25,
    "temporal": 0.15,
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
    column = _VEC_COLUMNS[axis]
    async with tenant_connection(db_pool, owner_id) as conn:
        rows = await conn.fetch(
            f"SELECT id, GREATEST(0.0, 1 - ({column} <=> $1)) AS score "  # noqa: S608
            f"FROM memory_nodes "
            f"WHERE owner_id = $2 AND is_deleted = FALSE "
            f"ORDER BY {column} <=> $1 "
            f"LIMIT $3",
            vector,
            owner_id,
            limit,
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
) -> list[dict]:
    """Score and rank candidates by convergence * activation."""
    node_ids = list(candidates.keys())
    if not node_ids:
        return []

    with Span(
        "read_path.convergence", component="read_path", expected_ms=1
    ):
        rows = await conn.fetch(
            "SELECT id, activation_level FROM memory_nodes WHERE id = ANY($1)",
            node_ids,
        )
        activations = {row["id"]: row["activation_level"] for row in rows}

        scored = []
        for node_id, data in candidates.items():
            dims_matched = len(data["scores"])
            if dims_matched == 0:
                continue
            avg_score = sum(data["scores"].values()) / dims_matched
            convergence = dims_matched * avg_score
            activation = activations.get(node_id, 0.0)
            adjusted = convergence * activation

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

    return scored


async def _spreading_activation(
    conn: asyncpg.Connection,
    seeds: list[dict],
    owner_id: UUID,
) -> tuple[dict[UUID, float], list[asyncpg.Record]]:
    """Propagate activation through axis-matched edges. 1 hop only."""
    seed_ids = [s["id"] for s in seeds]
    all_matched_axes: set[str] = set()
    for s in seeds:
        all_matched_axes.update(s["matched_axes"])

    activation_map: dict[UUID, float] = {s["id"]: s["adjusted"] for s in seeds}

    if not seed_ids or not all_matched_axes:
        return activation_map, []

    with Span(
        "read_path.spreading_activation",
        component="read_path",
        expected_ms=5,
    ):
        rows = await conn.fetch(
            "SELECT source_id, target_id, edge_type, weight, axis "
            "FROM edges "
            "WHERE owner_id = $1 AND weight > 0.0 "
            "AND (source_id = ANY($2) OR target_id = ANY($2)) "
            "AND axis = ANY($3)",
            owner_id,
            seed_ids,
            list(all_matched_axes),
        )

        seed_set = set(seed_ids)
        excitatory_count = 0
        inhibitory_count = 0
        modulatory_count = 0

        for row in rows:
            src: UUID = row["source_id"]
            tgt: UUID = row["target_id"]
            etype: str = row["edge_type"]
            weight: float = row["weight"]

            if src in seed_set:
                neighbor = tgt
                src_activation = activation_map.get(src, 0.0)
            else:
                neighbor = src
                src_activation = activation_map.get(tgt, 0.0)

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

    tc = get_trace()
    if tc:
        tc.spreading = {
            "edges_loaded": len(rows),
            "excitatory_fired": excitatory_count,
            "inhibitory_fired": inhibitory_count,
            "modulatory_fired": modulatory_count,
            "nodes_activated_by_spread": len(activation_map) - len(seeds),
        }

    return activation_map, list(rows)


async def _fetch_nodes(
    conn: asyncpg.Connection,
    node_ids: list[UUID],
) -> dict[UUID, MemoryNode]:
    """Batch fetch full node data for response building."""
    if not node_ids:
        return {}

    rows = await conn.fetch(
        "SELECT id, owner_id, content, content_hash, created_at, "
        "last_accessed, access_count, activation_level, salience, "
        "source_type, session_id, embedding_model, embedding_dimensions, "
        "metadata, is_deleted, dreamer_processed "
        "FROM memory_nodes WHERE id = ANY($1)",
        node_ids,
    )

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
) -> list[dict]:
    """Remove nodes with excluded metadata tags from final results."""
    exclude = settings.exclude_tags_set
    tc = get_trace()

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
            scored = await _score_candidates(conn, candidates)

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
        ranked = []
        for node_id in all_ids:
            node = nodes_by_id.get(node_id)
            if node is None or node.is_deleted:
                continue

            cand = candidates.get(node_id)
            if cand:
                dims_matched = len(cand["scores"])
                convergence = dims_matched * (
                    sum(cand["scores"].values()) / dims_matched
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
        filtered = _apply_post_filter(ranked, nodes_by_id)

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
