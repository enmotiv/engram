"""MemoryStore — CRUD operations for memories with automatic embedding."""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Memory
from engram.engine.embedding import EmbeddingService
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
    ) -> Memory:
        embedding = await self._embedding.get_embedding(content)
        dimension_scores = await self._embedding.get_dimension_scores(content)

        mem = Memory(
            namespace=namespace,
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            dimension_scores=dimension_scores,
            features=metadata or {},
        )
        self.db.add(mem)
        await self.db.flush()
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

    async def delete(self, memory_id: uuid.UUID) -> bool:
        mem = await self.db.get(Memory, memory_id)
        if mem is None:
            return False
        await self.db.delete(mem)
        await self.db.flush()
        return True

    async def batch_create(self, memories: List[Dict]) -> List[Memory]:
        results = []
        for item in memories:
            mem = await self.create(
                namespace=item["namespace"],
                content=item["content"],
                memory_type=item.get("memory_type", "episodic"),
                metadata=item.get("metadata"),
            )
            results.append(mem)
        return results
