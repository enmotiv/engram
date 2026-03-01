"""Namespace endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from engram.api.deps import get_db
from engram.api.schemas import NamespaceStats
from engram.db.models import Edge, Memory

router = APIRouter(tags=["namespaces"])


@router.get("/namespaces/{namespace}/stats", response_model=NamespaceStats)
async def get_namespace_stats(
    namespace: str,
    db: AsyncSession = Depends(get_db),
):
    # Memory count
    mem_result = await db.execute(
        select(func.count()).where(Memory.namespace == namespace)
    )
    memory_count = mem_result.scalar() or 0

    # Edge count
    edge_result = await db.execute(
        select(func.count()).where(Edge.namespace == namespace)
    )
    edge_count = edge_result.scalar() or 0

    # Last activity
    last_result = await db.execute(
        select(func.max(Memory.updated_at)).where(Memory.namespace == namespace)
    )
    last_activity = last_result.scalar()

    return NamespaceStats(
        namespace=namespace,
        memory_count=memory_count,
        edge_count=edge_count,
        last_activity=last_activity.isoformat() if last_activity else None,
    )


@router.delete("/namespaces/{namespace}")
async def delete_namespace(
    namespace: str,
    db: AsyncSession = Depends(get_db),
):
    # Delete edges first (FK constraint)
    await db.execute(delete(Edge).where(Edge.namespace == namespace))
    # Delete memories
    result = await db.execute(delete(Memory).where(Memory.namespace == namespace))
    await db.commit()
    return {"deleted": True, "memories_removed": result.rowcount}
