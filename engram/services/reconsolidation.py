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
