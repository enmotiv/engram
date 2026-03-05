"""Edge endpoints."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from engram.api.deps import get_db, get_edge_store
from engram.api.schemas import CreateEdgeRequest, EdgeResponse, UpdateEdgeRequest
from engram.db.models import Edge as EdgeModel
from engram.engine.edges import EdgeStore

router = APIRouter(tags=["edges"])


def _edge_to_response(edge) -> EdgeResponse:
    return EdgeResponse(
        id=str(edge.id),
        source_memory_id=str(edge.source_memory_id),
        target_memory_id=str(edge.target_memory_id),
        edge_type=edge.edge_type,
        weight=edge.weight,
        namespace=edge.namespace or "default",
        created_at=edge.created_at.isoformat() if edge.created_at else None,
    )


@router.post("/edges", response_model=EdgeResponse)
async def create_edge(
    req: CreateEdgeRequest,
    es: EdgeStore = Depends(get_edge_store),
    db: AsyncSession = Depends(get_db),
):
    edge = await es.create(
        source_id=uuid.UUID(req.source_id),
        target_id=uuid.UUID(req.target_id),
        edge_type=req.edge_type,
        weight=req.weight,
        context=req.context,
        namespace=req.namespace,
    )
    await db.commit()
    return _edge_to_response(edge)


@router.patch("/edges/{edge_id}", response_model=EdgeResponse)
async def update_edge(
    edge_id: str,
    req: UpdateEdgeRequest,
    es: EdgeStore = Depends(get_edge_store),
    db: AsyncSession = Depends(get_db),
):
    edge = await es.update_weight(uuid.UUID(edge_id), req.weight)
    if edge is None:
        raise HTTPException(status_code=404, detail="Edge not found")
    await db.commit()
    return _edge_to_response(edge)


@router.delete("/edges/{edge_id}")
async def delete_edge(
    edge_id: str,
    es: EdgeStore = Depends(get_edge_store),
    db: AsyncSession = Depends(get_db),
):
    deleted = await es.delete(uuid.UUID(edge_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Edge not found")
    await db.commit()
    return {"deleted": True}


@router.get("/memories/{memory_id}/edges")
async def get_edges_for_memory(
    memory_id: str,
    direction: str = "both",
    edge_type: Optional[str] = None,
    es: EdgeStore = Depends(get_edge_store),
):
    edge_types = [edge_type] if edge_type else None
    edges = await es.get_edges(uuid.UUID(memory_id), direction, edge_types=edge_types)
    return [_edge_to_response(e) for e in edges]


@router.get("/edges")
async def list_edges_by_namespace(
    namespace: str,
    edge_type: Optional[str] = None,
    min_weight: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """Query edges by namespace with optional type and weight filters."""
    stmt = select(EdgeModel).where(EdgeModel.namespace == namespace)
    if edge_type:
        stmt = stmt.where(EdgeModel.edge_type == edge_type)
    if min_weight is not None:
        stmt = stmt.where(EdgeModel.weight >= min_weight)
    stmt = stmt.order_by(EdgeModel.weight.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    edges = list(result.scalars().all())
    return [_edge_to_response(e) for e in edges]
