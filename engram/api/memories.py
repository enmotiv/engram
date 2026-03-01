"""Memory endpoints."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from engram.api.deps import get_db, get_edge_store, get_memory_store, get_retriever
from engram.api.schemas import (
    BatchCreateRequest,
    BatchEnrichRequest,
    BatchEnrichResponse,
    CreateMemoryRequest,
    HeaderSearchRequest,
    MemoryResponse,
    ResolvedMemory,
    ResolveRequest,
    ResolveResponse,
    RetrieveRequest,
    UpdateMemoryRequest,
)
from engram.engine.edges import EdgeStore
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


# --- Progressive resolution endpoints ---


@router.post("/memories/search", response_model=dict)
async def search_headers(
    req: HeaderSearchRequest,
    retriever: Retriever = Depends(get_retriever),
):
    """Layer 0 header scan — returns metadata without content."""
    result = await retriever.search_headers(
        namespace=req.namespace,
        cue=req.cue,
        context=req.context,
        max_results=req.max_results,
        urgency_threshold=req.urgency_threshold,
    )
    return result.model_dump()


@router.post("/memories/resolve", response_model=ResolveResponse)
async def resolve_memories(
    req: ResolveRequest,
    store: MemoryStore = Depends(get_memory_store),
    edge_store: EdgeStore = Depends(get_edge_store),
    db: AsyncSession = Depends(get_db),
):
    """Batch ID fetch — returns full records with edges for promoted IDs."""
    memory_uuids = []
    for mid in req.memory_ids:
        try:
            memory_uuids.append(uuid.UUID(mid))
        except ValueError:
            continue

    memories = await store.get_batch(memory_uuids)

    # Build edges map if requested
    edges_by_memory: dict[str, list] = {}
    if req.include_edges and memories:
        found_ids = [m.id for m in memories]
        edges = await edge_store.get_edges_batch(found_ids, direction="both")
        for edge in edges:
            src = str(edge.source_memory_id)
            tgt = str(edge.target_memory_id)
            edge_dict = {
                "id": str(edge.id),
                "source_memory_id": src,
                "target_memory_id": tgt,
                "edge_type": edge.edge_type,
                "weight": edge.weight,
            }
            edges_by_memory.setdefault(src, []).append(edge_dict)
            if tgt != src:
                edges_by_memory.setdefault(tgt, []).append(edge_dict)

    found_ids_set = {str(m.id) for m in memories}
    not_found = [mid for mid in req.memory_ids if mid not in found_ids_set]

    resolved = []
    for mem in memories:
        mid = str(mem.id)
        resolved.append(
            ResolvedMemory(
                id=mid,
                namespace=mem.namespace,
                content=mem.content,
                memory_type=mem.memory_type or "episodic",
                dimension_scores=mem.dimension_scores or {},
                features=mem.features or {},
                activation=mem.activation or 0.0,
                salience=mem.salience or 0.5,
                access_count=mem.access_count or 0,
                created_at=mem.created_at.isoformat() if mem.created_at else None,
                last_accessed=mem.last_accessed.isoformat() if mem.last_accessed else None,
                edges=edges_by_memory.get(mid, []),
            )
        )

    return ResolveResponse(memories=resolved, not_found=not_found)


@router.put("/memories/batch-enrich", response_model=BatchEnrichResponse)
async def batch_enrich(
    req: BatchEnrichRequest,
    store: MemoryStore = Depends(get_memory_store),
    db: AsyncSession = Depends(get_db),
):
    """Batch enrichment updates — update metadata/features for multiple memories."""
    updated = []
    failed = []
    for item in req.updates:
        try:
            mem = await store.update(
                uuid.UUID(item.memory_id),
                metadata=item.metadata,
                features=item.features,
            )
            if mem is None:
                failed.append(item.memory_id)
            else:
                updated.append(item.memory_id)
        except Exception:
            failed.append(item.memory_id)
    await db.commit()
    return BatchEnrichResponse(updated=updated, failed=failed)
