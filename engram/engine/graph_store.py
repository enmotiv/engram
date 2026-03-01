"""GraphStore — CRUD operations for graph snapshots."""

from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Graph


class GraphStore:
    """Creates, reads, and queries graph snapshots."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_latest(self, namespace: str) -> Optional[Graph]:
        """Get the most recent graph snapshot for a namespace."""
        stmt = (
            select(Graph)
            .where(Graph.namespace == namespace)
            .order_by(Graph.created_at.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_history(
        self, namespace: str, limit: int = 10, offset: int = 0
    ) -> List[Graph]:
        """Get paginated snapshot history for a namespace."""
        stmt = (
            select(Graph)
            .where(Graph.namespace == namespace)
            .order_by(Graph.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def store(
        self,
        namespace: str,
        content: str,
        node_count: int,
        cluster_count: int,
        memory_count: int,
        metadata: Optional[Dict] = None,
    ) -> Graph:
        """Create a new graph snapshot."""
        graph = Graph(
            namespace=namespace,
            content=content,
            node_count=node_count,
            cluster_count=cluster_count,
            memory_count=memory_count,
            metadata_=metadata or {},
        )
        self.db.add(graph)
        await self.db.flush()
        return graph
