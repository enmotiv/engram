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
    SourceType,
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
    *,
    session_id: UUID | None = None,
    time_window_hours: int | None = None,
    source_types: list[str] | None = None,
) -> tuple[str, list[tuple[UUID, float]]]:
    """Search one vector column on its own connection. Returns (axis, results)."""
    async with tenant_connection(db_pool, owner_id) as conn:
        use_filtered = (
            settings.engram_flag_context_retrieval
            and (session_id is not None or time_window_hours is not None or source_types is not None)
        )
        if use_filtered:
            rows = await memory_repo.find_by_vector_similarity_filtered(
                conn, owner_id, vector, axis, limit=limit,
                session_id=session_id,
                time_window_hours=time_window_hours,
                source_types=source_types,
            )
        else:
            rows = await memory_repo.find_by_vector_similarity(
                conn, owner_id, vector, axis, limit=limit
            )
    return axis, [(row["id"], row["score"]) for row in rows]


async def _multi_axis_search(
    db_pool: asyncpg.Pool,
    cue_vectors: dict[str, list[float]],
    owner_id: UUID,
    per_axis_limit: int = 20,
    *,
    session_id: UUID | None = None,
    time_window_hours: int | None = None,
    source_types: list[str] | None = None,
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
            _search_axis(
                db_pool, axis, cue_vectors[axis], owner_id, per_axis_limit,
                session_id=session_id,
                time_window_hours=time_window_hours,
                source_types=source_types,
            )
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
    *,
    axis_weights: dict[str, float] | None = None,
) -> tuple[list[dict], dict[UUID, float] | None]:
    """Score and rank candidates by convergence * activation.

    Applies a keyword-match boost when cue tokens appear in content,
    compensating for embedding models that struggle with proper nouns.
    When axis_weights is provided and the context retrieval flag is on,
    applies per-axis importance multipliers to the convergence formula.

    Returns (scored_list, plasticity_map). plasticity_map is non-None only
    when the metaplasticity flag is on.
    """
    node_ids = list(candidates.keys())
    if not node_ids:
        return [], None

    use_weights = (
        settings.engram_flag_context_retrieval
        and axis_weights is not None
        and len(axis_weights) > 0
    )

    with Span(
        "read_path.convergence", component="read_path", expected_ms=1
    ):
        rows = await memory_repo.fetch_activation_and_content(conn, node_ids)
        activations = {row["id"]: row["activation_level"] for row in rows}
        saliences = {row["id"]: row["salience"] for row in rows}
        contents = {row["id"]: (row["content"] or "").lower() for row in rows}

        # Build plasticity map when metaplasticity is enabled
        plasticity_map: dict[UUID, float] | None = None
        if settings.engram_flag_metaplasticity:
            plasticity_map = {
                row["id"]: row["plasticity"] for row in rows
            }

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

            if use_weights:
                # Weighted convergence: scale each axis score by its weight
                weighted_sum = sum(
                    score * axis_weights.get(axis, 1.0)
                    for axis, score in strong_scores.items()
                )
                weight_total = sum(
                    axis_weights.get(axis, 1.0) for axis in strong_scores
                )
                avg_score = weighted_sum / weight_total if weight_total > 0 else 0.0
            else:
                avg_score = sum(strong_scores.values()) / dims_matched

            convergence = dims_matched * avg_score
            activation = activations.get(node_id, 0.0)

            # Salience boost: emotionally significant memories get up to 10% boost
            salience = saliences.get(node_id, 0.5)
            salience_boost = 1.0 + 0.2 * (salience - 0.5)  # Range: 0.9 to 1.1
            adjusted = convergence * activation * salience_boost

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

    return scored, plasticity_map


def _attractor_settle(
    candidates: dict[UUID, dict],
    seeds: list[dict],
    edge_rows: list,
    *,
    max_iterations: int = 3,
    damping: float = 0.7,
    delta_threshold: float = 0.01,
) -> list[dict]:
    """Iterative attractor settling for deeper recall (Phase 3).

    Operates entirely on in-memory dicts — no DB, no I/O.
    Uses axis overlap and edge signals to pull related memories
    into the result set via pattern completion.
    """
    if len(seeds) <= 1:
        return seeds

    # Build edge lookup: {(id_a, id_b): weight} for quick access
    edge_lookup: dict[tuple[UUID, UUID], float] = {}
    for row in edge_rows:
        src = row["source_id"]
        tgt = row["target_id"]
        w = row["weight"]
        key = (min(src, tgt), max(src, tgt))
        # Keep max weight if multiple edges between same pair
        edge_lookup[key] = max(edge_lookup.get(key, 0.0), w)

    # Build a lookup for convergence/activation from the original scored seeds
    scored_data: dict[UUID, dict] = {s["id"]: s for s in seeds}

    # Initialize scores: seeds get their adjusted score, others get 0
    current_scores: dict[UUID, float] = {}
    for s in seeds:
        current_scores[s["id"]] = s["adjusted"]
    for cid in candidates:
        if cid not in current_scores:
            current_scores[cid] = 0.0

    current_seeds = list(seeds)
    seed_ids = {s["id"] for s in current_seeds}
    tc = get_trace()
    trace_iterations = []
    delta = 0.0

    for iteration in range(max_iterations):
        new_scores: dict[UUID, float] = {}
        prev_ranking = sorted(current_scores.keys(), key=lambda x: current_scores[x], reverse=True)

        for cid in candidates:
            if cid in seed_ids:
                # Seeds keep their score (they're attractors, not attracted)
                new_scores[cid] = current_scores[cid]
                continue

            cand_data = candidates[cid]
            cand_scores = cand_data["scores"]

            # Compute axis overlap influence from seeds
            axis_influence = 0.0
            for s in current_seeds:
                s_scores = candidates.get(s["id"], {}).get("scores", {})
                all_axes = set(cand_scores) | set(s_scores)
                if not all_axes:
                    continue
                overlap = sum(
                    min(cand_scores.get(a, 0.0), s_scores.get(a, 0.0))
                    for a in all_axes
                ) / len(all_axes)
                axis_influence += current_scores.get(s["id"], 0.0) * overlap

            # Edge signal influence
            edge_influence = 0.0
            for s in current_seeds:
                key = (min(cid, s["id"]), max(cid, s["id"]))
                ew = edge_lookup.get(key, 0.0)
                if ew > 0:
                    edge_influence += ew * current_scores.get(s["id"], 0.0) * 0.3

            total_influence = axis_influence + edge_influence
            new_scores[cid] = damping * current_scores[cid] + (1.0 - damping) * total_influence

        current_scores = new_scores

        # Re-rank and pick new seeds
        all_ranked = sorted(
            current_scores.items(), key=lambda x: x[1], reverse=True
        )
        new_seed_ids = {nid for nid, _ in all_ranked[:10]}
        new_ranking = [nid for nid, _ in all_ranked]

        # Compute delta (sum of absolute rank changes)
        delta = 0.0
        for nid in current_scores:
            old_rank = prev_ranking.index(nid) if nid in prev_ranking else len(prev_ranking)
            new_rank = new_ranking.index(nid) if nid in new_ranking else len(new_ranking)
            delta += abs(old_rank - new_rank)

        trace_iterations.append({
            "iteration": iteration + 1,
            "delta": round(delta, 4),
            "top_3": [str(nid) for nid, _ in all_ranked[:3]],
        })

        # Update seeds for next iteration
        current_seeds = []
        for nid, score in all_ranked[:10]:
            cand = candidates.get(nid, {})
            orig = scored_data.get(nid, {})
            current_seeds.append({
                "id": nid,
                "adjusted": score,
                "dims_matched": len(cand.get("scores", {})),
                "convergence_score": orig.get("convergence_score", 0.0),
                "activation_level": orig.get("activation_level", 0.0),
                "dimension_scores": cand.get("scores", {}),
                "matched_axes": cand.get("matched_axes", []),
            })
        seed_ids = new_seed_ids

        if delta < delta_threshold:
            break

    if tc:
        tc.attractor = {
            "iterations": trace_iterations,
            "settled": delta < delta_threshold if trace_iterations else True,
        }

    # Build result from settled seeds, preserving needed dict keys
    result = []
    for s in current_seeds:
        cand = candidates.get(s["id"])
        if cand:
            orig = scored_data.get(s["id"], {})
            result.append({
                "id": s["id"],
                "dims_matched": len(cand.get("scores", {})),
                "convergence_score": orig.get("convergence_score", 0.0),
                "activation_level": orig.get("activation_level", 0.0),
                "adjusted": s["adjusted"],
                "dimension_scores": cand["scores"],
                "matched_axes": cand.get("matched_axes", []),
            })
    return result


def _apply_edges(
    rows: list,
    source_set: set[UUID],
    activation_map: dict[UUID, float],
    *,
    plasticity_map: dict[UUID, float] | None = None,
) -> tuple[int, int, int, set[UUID]]:
    """Apply edge activations and return (excitatory, inhibitory, modulatory, new_neighbors).

    When plasticity_map is provided (Phase 4), propagation delta is scaled
    by target node's plasticity — established nodes resist spreading activation.
    """
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
                # Phase 4: scale by target plasticity
                if plasticity_map is not None:
                    target_plasticity = plasticity_map.get(neighbor, 0.8)
                    delta *= target_plasticity
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
    *,
    plasticity_map: dict[UUID, float] | None = None,
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
            plasticity_map=plasticity_map,
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
                plasticity_map=plasticity_map,
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
    *,
    session_id: UUID | None = None,
    axis_weights: dict[str, float] | None = None,
    time_window_hours: int | None = None,
    source_types: list[SourceType] | None = None,
    settle: bool = False,
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

        # Convert source_types to strings for SQL
        source_type_strs = (
            [str(s) for s in source_types] if source_types else None
        )

        # Phase B1: parallel multi-axis search (6 connections)
        candidates = await _multi_axis_search(
            db_pool, cue_vectors, owner_id,
            session_id=session_id,
            time_window_hours=time_window_hours,
            source_types=source_type_strs,
        )

        if not candidates:
            return RecallResponse(
                memories=[], confidence=ConfidenceLevel.LOW, edges=[]
            )

        # Phase B2: score + spread + fetch (one connection)
        async with tenant_connection(db_pool, owner_id) as conn:
            # 2. Convergence scoring
            scored, plasticity_map = await _score_candidates(
                conn, candidates, cue=cue,
                axis_weights=axis_weights,
            )

            # 2.5. Attractor settling (Phase 3)
            # Requires both the config flag AND the request flag
            if settle and settings.engram_flag_attractor and scored:
                # We need edge_rows for attractor settling, so do spreading first
                # to get them, then settle, then redo spreading with settled seeds
                activation_map_pre, edge_rows = await _spreading_activation(
                    conn, scored, owner_id,
                    plasticity_map=plasticity_map,
                )
                scored = _attractor_settle(candidates, scored, edge_rows)

                # Re-do spreading with settled seeds
                activation_map, edge_rows = await _spreading_activation(
                    conn, scored, owner_id,
                    plasticity_map=plasticity_map,
                )
            else:
                # 3. Spreading activation
                activation_map, edge_rows = await _spreading_activation(
                    conn, scored, owner_id,
                    plasticity_map=plasticity_map,
                )

            # 4. Fetch all node data (search results + spread neighbors)
            all_ids = list(
                set(candidates.keys()) | set(activation_map.keys())
            )
            nodes_by_id = await _fetch_nodes(conn, all_ids)

        # Phase C: rank, filter, build response (no DB needed)
        # Same per-axis threshold as scoring phase
        _AXIS_THRESHOLD = 0.3

        # Build scored lookup for reuse of weighted convergence from Phase B2
        scored_lookup = {s["id"]: s for s in scored}

        # Check if axis_weights should be applied
        use_weights = (
            settings.engram_flag_context_retrieval
            and axis_weights is not None
            and len(axis_weights) > 0
        )

        ranked = []
        for node_id in all_ids:
            node = nodes_by_id.get(node_id)
            if node is None or node.is_deleted:
                continue

            # Prefer pre-computed convergence from _score_candidates (preserves axis_weights)
            scored_item = scored_lookup.get(node_id)
            cand = candidates.get(node_id)

            if scored_item:
                convergence = scored_item["convergence_score"]
                dim_scores = scored_item["dimension_scores"]
                matched = scored_item["matched_axes"]
            elif cand:
                strong = {a: s for a, s in cand["scores"].items() if s >= _AXIS_THRESHOLD}
                dims_matched = len(strong)
                if dims_matched >= 2:
                    if use_weights:
                        weighted_sum = sum(
                            s * axis_weights.get(a, 1.0) for a, s in strong.items()
                        )
                        weight_total = sum(axis_weights.get(a, 1.0) for a in strong)
                        avg_score = weighted_sum / weight_total if weight_total > 0 else 0.0
                    else:
                        avg_score = sum(strong.values()) / dims_matched
                    convergence = dims_matched * avg_score
                else:
                    convergence = 0.0
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
                        db_pool, owner_id, top,
                        candidate_ids=list(candidates.keys()),
                        candidates=candidates,
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
