"""EdgeStore — typed edge CRUD + neighborhood traversal."""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import VALID_EDGE_TYPES, Edge
from engram.db.queries import NEIGHBORHOOD_CTE


class EdgeStore:
    """Creates, queries, and traverses typed edges between memories."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        edge_type: str,
        weight: float = 0.5,
        context: Optional[dict] = None,
        namespace: str = "default",
    ) -> Edge:
        if edge_type not in VALID_EDGE_TYPES:
            raise ValueError(f"Invalid edge type '{edge_type}'. Must be one of: {VALID_EDGE_TYPES}")

        stmt = pg_insert(Edge).values(
            source_memory_id=source_id,
            target_memory_id=target_id,
            edge_type=edge_type,
            weight=weight,
            context=context or {},
            namespace=namespace,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_edge_source_target_type",
            set_={"weight": stmt.excluded.weight, "context": stmt.excluded.context},
        )
        result = await self.db.execute(stmt.returning(Edge))
        await self.db.flush()
        row = result.scalars().first()
        if row is None:
            # Fallback: fetch the edge that was upserted
            stmt2 = select(Edge).where(
                Edge.source_memory_id == source_id,
                Edge.target_memory_id == target_id,
                Edge.edge_type == edge_type,
            )
            result2 = await self.db.execute(stmt2)
            row = result2.scalars().first()
        return row

    async def get_edges(
        self,
        memory_id: uuid.UUID,
        direction: str = "outgoing",
        edge_types: Optional[List[str]] = None,
    ) -> List[Edge]:
        if direction == "outgoing":
            stmt = select(Edge).where(Edge.source_memory_id == memory_id)
        elif direction == "incoming":
            stmt = select(Edge).where(Edge.target_memory_id == memory_id)
        else:  # both
            stmt = select(Edge).where(
                (Edge.source_memory_id == memory_id) | (Edge.target_memory_id == memory_id)
            )

        if edge_types:
            stmt = stmt.where(Edge.edge_type.in_(edge_types))

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_neighborhood(
        self,
        memory_ids: List[uuid.UUID],
        hop_depth: int = 1,
        edge_types: Optional[List[str]] = None,
        namespace: str = "default",
    ) -> Dict:
        result = await self.db.execute(
            text(NEIGHBORHOOD_CTE),
            {"seed_ids": memory_ids, "namespace": namespace, "max_depth": hop_depth},
        )
        rows = result.fetchall()

        node_ids = set()
        edges = []
        for r in rows:
            node_ids.add(r.source_memory_id)
            node_ids.add(r.target_memory_id)
            edge_data = {
                "source_memory_id": str(r.source_memory_id),
                "target_memory_id": str(r.target_memory_id),
                "edge_type": r.edge_type,
                "weight": r.weight,
                "depth": r.depth,
            }
            if edge_types is None or r.edge_type in edge_types:
                edges.append(edge_data)

        return {"nodes": [str(n) for n in node_ids], "edges": edges}

    async def batch_create(self, edges: List[Dict]) -> List[Edge]:
        results = []
        for e in edges:
            edge = await self.create(
                source_id=e["source_id"],
                target_id=e["target_id"],
                edge_type=e["edge_type"],
                weight=e.get("weight", 0.5),
                context=e.get("context"),
                namespace=e.get("namespace", "default"),
            )
            results.append(edge)
        return results

    async def delete(self, edge_id: uuid.UUID) -> bool:
        edge = await self.db.get(Edge, edge_id)
        if edge is None:
            return False
        await self.db.delete(edge)
        await self.db.flush()
        return True

    async def update_weight(self, edge_id: uuid.UUID, new_weight: float) -> Optional[Edge]:
        edge = await self.db.get(Edge, edge_id)
        if edge is None:
            return None
        edge.weight = new_weight
        await self.db.flush()
        return edge
