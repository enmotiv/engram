"""Update-on-read: boost nodes, strengthen co-retrieval edges, suppress competitors."""

from __future__ import annotations

from itertools import combinations
from uuid import UUID

import asyncpg
import structlog

from engram.config import settings
from engram.core.db import tenant_connection
from engram.core.tracing import Span
from engram.repositories import edge_repo, memory_repo

logger = structlog.get_logger()


def _compute_competitor_penalties(
    returned_items: list[dict],
    candidate_ids: list[UUID],
    candidates: dict[UUID, dict],
    base_penalty: float = 0.03,
) -> tuple[list[UUID], list[float]]:
    """Compute similarity-weighted penalties for competitors.

    Competitors = candidates that were in the search results but didn't
    make the top-k. Near-misses (high axis overlap with winners) get
    penalized more than distant candidates.
    """
    winner_ids = {item["id"] for item in returned_items}
    competitor_ids_out: list[UUID] = []
    penalties_out: list[float] = []

    for comp_id in candidate_ids:
        if comp_id in winner_ids:
            continue
        comp_data = candidates.get(comp_id)
        if not comp_data:
            continue
        comp_scores = comp_data.get("scores", {})
        if not comp_scores:
            continue

        # Find highest axis overlap with any winner
        max_overlap = 0.0
        for winner in returned_items:
            winner_data = candidates.get(winner["id"])
            if not winner_data:
                continue
            winner_scores = winner_data.get("scores", {})
            shared = set(comp_scores) & set(winner_scores)
            if shared:
                overlap = sum(
                    min(comp_scores[a], winner_scores[a]) for a in shared
                ) / len(shared)
                max_overlap = max(max_overlap, overlap)

        scaled_penalty = base_penalty * max_overlap
        if scaled_penalty > 0.001:  # Only penalize if meaningful
            competitor_ids_out.append(comp_id)
            penalties_out.append(scaled_penalty)

    return competitor_ids_out, penalties_out


async def reconsolidate(
    db_pool: asyncpg.Pool,
    owner_id: UUID,
    returned_items: list[dict],
    *,
    candidate_ids: list[UUID] | None = None,
    candidates: dict[UUID, dict] | None = None,
) -> dict:
    """Run reconsolidation after retrieval. Synchronous, not deferred.

    Each item in returned_items must have:
      - "id": UUID
      - "matched_axes": list[str]
    """
    node_ids = [item["id"] for item in returned_items]
    stats: dict = {"nodes_boosted": 0, "edges_strengthened": 0}

    if not node_ids:
        return stats

    with Span(
        "reconsolidation.update",
        component="reconsolidation",
        expected_ms=15,
    ):
        async with tenant_connection(db_pool, owner_id) as conn:
            # Step 0: Capture pre-boost timestamps for STDP
            pre_boost_timestamps: dict[UUID, object] | None = None
            if settings.engram_flag_stdp:
                ts_rows = await conn.fetch(
                    "SELECT id, last_accessed FROM memory_nodes "
                    "WHERE owner_id = $1 AND id = ANY($2)",
                    owner_id,
                    node_ids,
                )
                pre_boost_timestamps = {
                    r["id"]: r["last_accessed"] for r in ts_rows
                }

            # Step 1: Boost returned nodes
            stats["nodes_boosted"] = await memory_repo.boost_nodes(
                conn, node_ids, owner_id
            )

            # Step 2: Strengthen co-retrieval edges on shared axes
            sources: list[UUID] = []
            targets: list[UUID] = []
            axes_list: list[str] = []

            for node_a, node_b in combinations(returned_items, 2):
                shared = set(node_a["matched_axes"]) & set(
                    node_b["matched_axes"]
                )
                if not shared:
                    continue
                src = min(node_a["id"], node_b["id"])
                tgt = max(node_a["id"], node_b["id"])
                for axis in shared:
                    sources.append(src)
                    targets.append(tgt)
                    axes_list.append(axis)

            if sources:
                stats["edges_strengthened"] = (
                    await edge_repo.strengthen_co_retrieval(
                        conn, owner_id, sources, targets, axes_list
                    )
                )

            # Step 2.5: STDP directional weight updates
            if (
                settings.engram_flag_stdp
                and pre_boost_timestamps
                and sources
            ):
                await _apply_stdp(
                    conn,
                    owner_id,
                    sources,
                    targets,
                    axes_list,
                    pre_boost_timestamps,
                    stats,
                )

            # Step 3: Suppress competitors (retrieval-induced forgetting)
            if (
                settings.engram_flag_forgetting
                and candidate_ids
                and candidates
            ):
                comp_ids, penalties = _compute_competitor_penalties(
                    returned_items, candidate_ids, candidates
                )
                if comp_ids:
                    # Phase 2+4 guardrail: established memories resist
                    # suppression — UNLESS a correction is among the winners.
                    # Corrections are authoritative: they override regardless
                    # of how established the competitor is.
                    winner_rows = await conn.fetch(
                        "SELECT source_type FROM memory_nodes "
                        "WHERE id = ANY($1) AND is_deleted = FALSE",
                        node_ids,
                    )
                    winner_has_correction = any(
                        r["source_type"] == "correction" for r in winner_rows
                    )
                    if (
                        settings.engram_flag_metaplasticity
                        and not winner_has_correction
                    ):
                        plast_rows = await memory_repo.fetch_activation_and_content(
                            conn, comp_ids, owner_id=owner_id
                        )
                        plast_map = {
                            r["id"]: r["plasticity"] for r in plast_rows
                        }
                        penalties = [
                            p * plast_map.get(cid, 0.8)
                            for cid, p in zip(comp_ids, penalties)
                        ]

                    stats["competitors_suppressed"] = (
                        await memory_repo.penalize_competitors_scaled(
                            conn, comp_ids, penalties, owner_id
                        )
                    )

    logger.info(
        "reconsolidation.complete",
        component="reconsolidation",
        **stats,
    )
    return stats


async def _apply_stdp(
    conn: asyncpg.Connection,
    owner_id: UUID,
    sources: list[UUID],
    targets: list[UUID],
    axes_list: list[str],
    pre_boost_timestamps: dict[UUID, object],
    stats: dict,
) -> None:
    """Apply STDP directional weight updates based on access order.

    For each co-retrieved pair, compare pre-boost timestamps to determine
    which was accessed first. The edge from the earlier-accessed node
    toward the later-accessed node is strengthened; the reverse weakened.
    """
    from engram.core.tracing import get_trace

    # Fetch edge IDs for the co-retrieved pairs
    edge_rows = await edge_repo.fetch_edge_ids_for_pairs(
        conn, owner_id, sources, targets, axes_list
    )
    if not edge_rows:
        return

    # Build plasticity map if metaplasticity is on
    plasticity_map: dict[UUID, float] | None = None
    if settings.engram_flag_metaplasticity:
        all_node_ids = list(
            {r["source_id"] for r in edge_rows}
            | {r["target_id"] for r in edge_rows}
        )
        plast_rows = await memory_repo.fetch_activation_and_content(
            conn, all_node_ids, owner_id=owner_id
        )
        plasticity_map = {r["id"]: r["plasticity"] for r in plast_rows}

    edge_ids: list[UUID] = []
    forward_deltas: list[float] = []
    backward_deltas: list[float] = []

    _STRENGTHEN = 0.03
    _WEAKEN = -0.01

    for row in edge_rows:
        src_id = row["source_id"]  # canonical min
        tgt_id = row["target_id"]  # canonical max

        ts_src = pre_boost_timestamps.get(src_id)
        ts_tgt = pre_boost_timestamps.get(tgt_id)

        # Skip if either timestamp is None (first-recall edge case)
        if ts_src is None or ts_tgt is None:
            continue

        # Determine direction: which was accessed first (older timestamp)?
        if ts_src == ts_tgt:
            # Simultaneous — no directional signal
            continue

        if ts_src < ts_tgt:
            # Canonical source was accessed first → strengthen forward
            fd, bd = _STRENGTHEN, _WEAKEN
        else:
            # Canonical target was accessed first → strengthen backward
            fd, bd = _WEAKEN, _STRENGTHEN

        # Scale by plasticity if metaplasticity is on
        if plasticity_map is not None:
            p_src = plasticity_map.get(src_id, 0.8)
            p_tgt = plasticity_map.get(tgt_id, 0.8)
            scale = min(p_src, p_tgt)
            fd *= scale
            bd *= scale

        edge_ids.append(row["id"])
        forward_deltas.append(fd)
        backward_deltas.append(bd)

    if edge_ids:
        updated = await edge_repo.apply_stdp_update(
            conn, owner_id, edge_ids, forward_deltas, backward_deltas
        )
        stats["stdp_pairs_updated"] = updated

    tc = get_trace()
    if tc:
        tc.stdp = {
            "pairs_evaluated": len(edge_rows),
            "pairs_updated": len(edge_ids),
            "skipped_no_timestamp": len(edge_rows) - len(edge_ids),
        }
