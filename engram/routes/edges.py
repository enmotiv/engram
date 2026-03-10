"""Edge creation endpoint for structural (non-Dreamer) edges.

Clients use this to create decay-exempt edges between memory nodes.
Dreamer-managed edge types (excitatory, inhibitory, etc.) cannot be
created via this endpoint — only ``structural`` edges are allowed.
"""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from engram.auth import get_owner_id
from engram.db import tenant_connection
from engram.errors import EngramError

logger = structlog.get_logger()

router = APIRouter(prefix="/v1")


class CreateEdgeRequest(BaseModel):
    source_id: UUID
    target_id: UUID
    edge_type: str = Field(..., pattern=r"^structural$")
    axis: str = "sensory"
    weight: float = Field(default=0.8, ge=0.0, le=1.0)


@router.post("/edges", status_code=201)
async def create_edge(
    body: CreateEdgeRequest,
    request: Request,
    owner_id: UUID = Depends(get_owner_id),  # noqa: B008
) -> dict:
    """Create a structural edge between two memory nodes.

    Only structural edges can be created via this endpoint.
    Dreamer-managed edge types are rejected.
    """
    # Enforce consistent source < target ordering
    src = min(body.source_id, body.target_id)
    tgt = max(body.source_id, body.target_id)

    if src == tgt:
        raise EngramError("INVALID_INPUT", "Self-loops not allowed", 400)

    db = request.app.state.db
    async with tenant_connection(db, owner_id) as conn:
        # Verify both nodes exist
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM memory_nodes "
            "WHERE owner_id = $1 AND id = ANY($2) AND is_deleted = FALSE",
            owner_id,
            [src, tgt],
        )
        if count < 2:
            raise EngramError(
                "NOT_FOUND",
                "One or both memory nodes not found",
                404,
            )

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
            body.edge_type,
            body.axis,
            body.weight,
        )

    return {
        "data": {
            "source_id": str(src),
            "target_id": str(tgt),
            "edge_type": body.edge_type,
            "axis": body.axis,
            "weight": body.weight,
        }
    }
