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
from engram.core.db import tenant_connection
from engram.core.errors import EngramError
from engram.repositories import edge_repo, memory_repo

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
    src = min(body.source_id, body.target_id)
    tgt = max(body.source_id, body.target_id)

    if src == tgt:
        raise EngramError("INVALID_INPUT", "Self-loops not allowed", 400)

    db = request.app.state.db
    async with tenant_connection(db, owner_id) as conn:
        # Verify both nodes exist
        count = await memory_repo.count_active_nodes(
            conn, owner_id, [src, tgt]
        )
        if count < 2:
            raise EngramError(
                "NOT_FOUND",
                "One or both memory nodes not found",
                404,
            )

        await edge_repo.upsert_edge(
            conn,
            owner_id=owner_id,
            source_id=src,
            target_id=tgt,
            edge_type=body.edge_type,
            axis=body.axis,
            weight=body.weight,
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
