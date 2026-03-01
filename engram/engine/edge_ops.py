"""Higher-level edge operations: reinforcement, decay, pruning."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import List

from sqlalchemy import select, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Edge


class EdgeOps:
    """Hebbian reinforcement, exponential decay, and pruning."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def reinforce_traversed(
        self,
        seed_ids: List[uuid.UUID],
        included_ids: List[uuid.UUID],
        boost: float = 1.1,
    ) -> int:
        """Strengthen edges between seed and included memories."""
        all_ids = set(seed_ids) | set(included_ids)
        stmt = select(Edge).where(
            Edge.source_memory_id.in_(all_ids),
            Edge.target_memory_id.in_(all_ids),
        )
        result = await self.db.execute(stmt)
        edges = result.scalars().all()

        count = 0
        now = datetime.now(timezone.utc)
        for edge in edges:
            edge.weight = min(edge.weight * boost, 1.0)
            edge.last_activated = now
            count += 1
        await self.db.flush()
        return count

    async def apply_decay(
        self,
        namespace: str,
        half_life_days: int = 30,
    ) -> int:
        """Exponential decay on edge weights based on time since last activation."""
        stmt = select(Edge).where(
            Edge.namespace == namespace,
            Edge.last_activated.is_not(None),
        )
        result = await self.db.execute(stmt)
        edges = result.scalars().all()

        now = datetime.now(timezone.utc)
        count = 0
        for edge in edges:
            days_since = (now - edge.last_activated).total_seconds() / 86400
            if days_since > 0:
                decay_factor = 0.5 ** (days_since / half_life_days)
                edge.weight = edge.weight * decay_factor
                count += 1
        await self.db.flush()
        return count

    async def prune(
        self,
        namespace: str,
        weight_floor: float = 0.05,
        grace_period_days: int = 7,
    ) -> int:
        """Delete edges below weight_floor that haven't been activated recently."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=grace_period_days)
        stmt = select(Edge).where(
            Edge.namespace == namespace,
            Edge.weight < weight_floor,
            (Edge.last_activated < cutoff) | (Edge.last_activated.is_(None)),
        )
        result = await self.db.execute(stmt)
        edges = result.scalars().all()

        count = 0
        for edge in edges:
            await self.db.delete(edge)
            count += 1
        await self.db.flush()
        return count
