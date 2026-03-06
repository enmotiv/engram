"""Edge decay job — exponential weight decay on inactive edges."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Edge, Memory
from engram.engine.edge_ops import EdgeOps
from engram.engine.interfaces import WorkerJob

logger = logging.getLogger(__name__)


class EdgeDecayJob(WorkerJob):
    def name(self) -> str:
        return "edge_decay"

    async def should_run(self, namespace: str, **kwargs) -> bool:
        return True  # Always eligible; decay is idempotent

    async def execute(self, namespace: str, **kwargs) -> dict:
        db: AsyncSession = kwargs["db"]
        ops = EdgeOps(db)
        count = await ops.apply_decay(namespace, half_life_days=30)
        await db.flush()

        # Recompute node activation after edge decay
        nodes_updated = 0
        try:
            nodes_updated = await self._update_node_activation(db, namespace)
            await db.flush()
        except Exception:
            logger.warning("node activation update failed (non-fatal)", exc_info=True)

        return {"edges_decayed": count, "nodes_updated": nodes_updated}

    async def _update_node_activation(
        self, db: AsyncSession, namespace: str, half_life_days: int = 30
    ) -> int:
        """Recompute activation for all memory nodes after edge decay.

        activation = edge_strength × recency_decay, where:
        - edge_strength = mean weight of all connected edges (in + out)
        - recency_decay = exponential decay from last_accessed
        """
        now = datetime.now(timezone.utc)
        decay_constant = math.log(2) / half_life_days

        # Fetch all memories in namespace
        mem_stmt = select(Memory).where(Memory.namespace == namespace)
        mem_result = await db.execute(mem_stmt)
        memories = mem_result.scalars().all()

        # Fetch all edges in namespace
        edge_stmt = select(Edge).where(Edge.namespace == namespace)
        edge_result = await db.execute(edge_stmt)
        edges = edge_result.scalars().all()

        # Build per-node edge weight index
        node_edges: dict[str, list[float]] = {}
        for edge in edges:
            sid = str(edge.source_memory_id)
            tid = str(edge.target_memory_id)
            node_edges.setdefault(sid, []).append(edge.weight)
            node_edges.setdefault(tid, []).append(edge.weight)

        updated = 0
        for mem in memories:
            # Edge strength: mean of all connected edge weights (0.0 if isolated)
            weights = node_edges.get(str(mem.id), [])
            edge_strength = sum(weights) / len(weights) if weights else 0.0

            # Recency decay: falloff from last_accessed
            if mem.last_accessed:
                days_since = (now - mem.last_accessed).total_seconds() / 86400
                recency = math.exp(-decay_constant * days_since)
            else:
                recency = 0.1  # never accessed — low but not zero

            mem.activation = round(edge_strength * recency, 4)

            # Enforce decay floor if set (e.g. entity nodes protected by plugins)
            floor = None
            if mem.features and isinstance(mem.features, dict):
                floor = mem.features.get("decay_floor")
            if floor is not None:
                try:
                    mem.activation = max(mem.activation, float(floor))
                except (TypeError, ValueError):
                    pass  # malformed floor — ignore

            updated += 1

        return updated
