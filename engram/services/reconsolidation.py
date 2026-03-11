"""Update-on-read: boost nodes, strengthen co-retrieval edges."""

from itertools import combinations
from uuid import UUID

import asyncpg
import structlog

from engram.core.db import tenant_connection
from engram.core.tracing import Span
from engram.repositories import edge_repo, memory_repo

logger = structlog.get_logger()


async def reconsolidate(
    db_pool: asyncpg.Pool,
    owner_id: UUID,
    returned_items: list[dict],
) -> dict:
    """Run reconsolidation after retrieval. Synchronous, not deferred.

    Each item in returned_items must have:
      - "id": UUID
      - "matched_axes": list[str]
    """
    node_ids = [item["id"] for item in returned_items]
    stats = {"nodes_boosted": 0, "edges_strengthened": 0}

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

    logger.info(
        "reconsolidation.complete",
        component="reconsolidation",
        **stats,
    )
    return stats
