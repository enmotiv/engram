"""Graph snapshot endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from engram.api.deps import get_db
from engram.engine.graph_store import GraphStore

router = APIRouter(tags=["graphs"])


class GraphSnapshotResponse(BaseModel):
    id: str
    namespace: str
    content: str
    node_count: int
    cluster_count: int
    memory_count: int
    version: int
    created_at: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class StoreGraphRequest(BaseModel):
    content: str
    node_count: int
    cluster_count: int = 0
    memory_count: int
    metadata: dict = Field(default_factory=dict)


def _graph_to_response(graph) -> GraphSnapshotResponse:
    return GraphSnapshotResponse(
        id=str(graph.id),
        namespace=graph.namespace,
        content=graph.content,
        node_count=graph.node_count,
        cluster_count=graph.cluster_count,
        memory_count=graph.memory_count,
        version=graph.version,
        created_at=graph.created_at.isoformat() if graph.created_at else None,
        metadata=graph.metadata_ or {},
    )


@router.get("/graphs/{namespace:path}", response_model=GraphSnapshotResponse)
async def get_latest_graph(
    namespace: str,
    db: AsyncSession = Depends(get_db),
):
    """Get the latest graph snapshot for a namespace."""
    store = GraphStore(db)
    graph = await store.get_latest(namespace)
    if graph is None:
        raise HTTPException(status_code=404, detail="No graph snapshot found")
    return _graph_to_response(graph)


@router.get("/graphs/{namespace:path}/history")
async def get_graph_history(
    namespace: str,
    limit: int = Query(default=10, le=50),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Get paginated graph snapshot history."""
    store = GraphStore(db)
    graphs = await store.get_history(namespace, limit=limit, offset=offset)
    return {"snapshots": [_graph_to_response(g) for g in graphs]}


@router.post("/graphs/{namespace:path}", response_model=GraphSnapshotResponse)
async def store_graph(
    namespace: str,
    req: StoreGraphRequest,
    db: AsyncSession = Depends(get_db),
):
    """Store a new graph snapshot."""
    store = GraphStore(db)
    graph = await store.store(
        namespace=namespace,
        content=req.content,
        node_count=req.node_count,
        cluster_count=req.cluster_count,
        memory_count=req.memory_count,
        metadata=req.metadata,
    )
    await db.commit()
    return _graph_to_response(graph)
