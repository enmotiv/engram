"""Memory endpoints."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from engram.api.deps import get_db, get_memory_store, get_retriever
from engram.api.schemas import (
    BatchCreateRequest,
    CreateMemoryRequest,
    MemoryResponse,
    RetrieveRequest,
    UpdateMemoryRequest,
)
from engram.engine.models import RetrievalOptions
from engram.engine.retriever import Retriever
from engram.engine.store import MemoryStore

router = APIRouter(tags=["memories"])


def _memory_to_response(mem) -> MemoryResponse:
    return MemoryResponse(
        id=str(mem.id),
        namespace=mem.namespace,
        content=mem.content,
        memory_type=mem.memory_type or "episodic",
        dimension_scores=mem.dimension_scores or {},
        features=mem.features or {},
        metadata=mem.features or {},  # features IS the metadata store in Engram
        activation=mem.activation or 0.0,
        salience=mem.salience or 0.5,
        access_count=mem.access_count or 0,
        created_at=mem.created_at.isoformat() if mem.created_at else None,
    )


@router.post("/memories", response_model=MemoryResponse)
async def create_memory(
    req: CreateMemoryRequest,
    store: MemoryStore = Depends(get_memory_store),
    db: AsyncSession = Depends(get_db),
):
    mem = await store.create(
        namespace=req.namespace,
        content=req.content,
        memory_type=req.memory_type,
        metadata=req.metadata,
        memory_id=req.id,
    )
    await db.commit()
    return _memory_to_response(mem)


@router.post("/memories/batch")
async def batch_create(
    req: BatchCreateRequest,
    store: MemoryStore = Depends(get_memory_store),
    db: AsyncSession = Depends(get_db),
):
    results = []
    for m in req.memories:
        mem = await store.create(
            namespace=m.namespace,
            content=m.content,
            memory_type=m.memory_type,
            metadata=m.metadata,
            memory_id=m.id,
        )
        results.append(_memory_to_response(mem))
    await db.commit()
    return {"created": results}


@router.get("/memories")
async def list_memories(
    namespace: str,
    memory_type: Optional[str] = None,
    limit: int = 50,
    store: MemoryStore = Depends(get_memory_store),
):
    """List/filter memories by namespace and optional metadata filters."""
    from sqlalchemy import select
    from engram.db.models import Memory

    stmt = (
        select(Memory)
        .where(Memory.namespace == namespace)
        .order_by(Memory.created_at.desc())
        .limit(limit)
    )
    if memory_type:
        # memory_type is stored inside features JSONB as enmotiv_type
        stmt = stmt.where(
            Memory.features["enmotiv_type"].astext == memory_type
        )
    result = await store.db.execute(stmt)
    memories = list(result.scalars().all())
    return {"memories": [_memory_to_response(m) for m in memories]}


@router.post("/memories/retrieve")
async def retrieve_memories(
    req: RetrieveRequest,
    retriever: Retriever = Depends(get_retriever),
):
    opts = RetrievalOptions(
        max_results=req.max_results,
        urgency_threshold=req.urgency_threshold,
        hop_depth=req.hop_depth,
    )
    result = await retriever.retrieve(
        namespace=req.namespace,
        cue=req.cue,
        context=req.context,
        options=opts,
    )
    return result.model_dump()


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    store: MemoryStore = Depends(get_memory_store),
):
    mem = await store.get(uuid.UUID(memory_id))
    if mem is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _memory_to_response(mem)


@router.patch("/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    req: UpdateMemoryRequest,
    store: MemoryStore = Depends(get_memory_store),
    db: AsyncSession = Depends(get_db),
):
    mem = await store.update(
        uuid.UUID(memory_id),
        metadata=req.metadata,
        features=req.features,
    )
    if mem is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    await db.commit()
    return _memory_to_response(mem)


@router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    store: MemoryStore = Depends(get_memory_store),
    db: AsyncSession = Depends(get_db),
):
    deleted = await store.delete(uuid.UUID(memory_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    await db.commit()
    return {"deleted": True}
