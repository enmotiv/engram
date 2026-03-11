"""Update-on-read: boost nodes, strengthen co-retrieval edges."""

from itertools import combinations
from uuid import UUID

import asyncpg
import structlog

from engram.core.db import tenant_connection
from engram.core.tracing import Span

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
            # Step 1: Boost returned nodes (one batch UPDATE)
            result = await conn.execute(
                "UPDATE memory_nodes SET "
                "  last_accessed = NOW(), "
                "  access_count = access_count + 1, "
                "  activation_level = LEAST(1.0, activation_level + 0.1) "
                "WHERE id = ANY($1) AND owner_id = $2",
                node_ids,
                owner_id,
            )
            stats["nodes_boosted"] = _parse_update_count(result)

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
                result = await conn.execute(
                    "UPDATE edges SET "
                    "  weight = LEAST(1.0, weight + 0.05), "
                    "  updated_at = NOW() "
                    "WHERE owner_id = $1 "
                    "  AND edge_type IN ('excitatory', 'associative') "
                    "  AND (source_id, target_id, axis) IN ("
                    "    SELECT * FROM unnest($2::uuid[], $3::uuid[], $4::text[])"
                    "  )",
                    owner_id,
                    sources,
                    targets,
                    axes_list,
                )
                stats["edges_strengthened"] = _parse_update_count(result)

    logger.info(
        "reconsolidation.complete",
        component="reconsolidation",
        **stats,
    )
    return stats


def _parse_update_count(result: str) -> int:
    """Parse row count from asyncpg execute result like 'UPDATE 5'."""
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0
