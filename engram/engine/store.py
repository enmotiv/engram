"""MemoryStore — CRUD operations for memories with automatic embedding."""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Memory
from engram.engine.embedding import EmbeddingService
from engram.engine.notifications import notify_namespace_changed
from engram.plugins.registry import PluginRegistry


class MemoryStore:
    """Creates, reads, updates, and deletes memories with automatic embedding."""

    def __init__(self, db: AsyncSession, registry: Optional[PluginRegistry] = None):
        self.db = db
        self._registry = registry or PluginRegistry.get_instance()
        self._embedding = EmbeddingService(self._registry)

    async def create(
        self,
        namespace: str,
        content: str,
        memory_type: str = "episodic",
        metadata: Optional[Dict] = None,
        memory_id: Optional[str] = None,
    ) -> Memory:
        embedding = await self._embedding.get_embedding(content)
        dimension_scores = await self._embedding.get_dimension_scores(content)

        kwargs: Dict = dict(
            namespace=namespace,
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            dimension_scores=dimension_scores,
            features=metadata or {},
        )
        if memory_id:
            import uuid as _uuid
            kwargs["id"] = _uuid.UUID(memory_id)

        # Populate per-region embeddings for multi-axis retrieval
        region_embeddings = await self._embedding.get_region_embeddings(content)
        for region, vec in region_embeddings.items():
            col_name = f"{region}_embedding"
            kwargs[col_name] = vec

        mem = Memory(**kwargs)
        self.db.add(mem)
        await self.db.flush()
        await notify_namespace_changed(namespace)
        return mem

    async def get(self, memory_id: uuid.UUID) -> Optional[Memory]:
        return await self.db.get(Memory, memory_id)

    async def list(
        self,
        namespace: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Memory]:
        stmt = (
            select(Memory)
            .where(Memory.namespace == namespace)
            .order_by(Memory.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update(
        self,
        memory_id: uuid.UUID,
        metadata: Optional[Dict] = None,
        features: Optional[Dict] = None,
    ) -> Optional[Memory]:
        """Merge metadata and/or features into an existing memory's features column."""
        mem = await self.db.get(Memory, memory_id)
        if mem is None:
            return None
        existing = mem.features or {}
        if metadata:
            existing.update(metadata)
        if features:
            existing.update(features)
        mem.features = existing
        await self.db.flush()
        await notify_namespace_changed(mem.namespace)
        return mem

    async def delete(self, memory_id: uuid.UUID) -> bool:
        mem = await self.db.get(Memory, memory_id)
        if mem is None:
            return False
        namespace = mem.namespace
        await self.db.delete(mem)
        await self.db.flush()
        await notify_namespace_changed(namespace)
        return True

    async def get_batch(self, memory_ids: List[uuid.UUID]) -> List[Memory]:
        """Fetch multiple memories by ID in a single query."""
        if not memory_ids:
            return []
        stmt = select(Memory).where(Memory.id.in_(memory_ids))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def batch_create(self, memories: List[Dict]) -> List[Memory]:
        results = []
        namespaces: set[str] = set()
        for item in memories:
            mem = await self.create(
                namespace=item["namespace"],
                content=item["content"],
                memory_type=item.get("memory_type", "episodic"),
                metadata=item.get("metadata"),
            )
            results.append(mem)
            namespaces.add(item["namespace"])
        # Deduplicated notifications (create already notifies per-item,
        # but batch_create delegates to create which handles it)
        return results
