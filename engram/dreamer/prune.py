"""Edge pruning job — delete low-weight inactive edges."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from engram.engine.edge_ops import EdgeOps
from engram.engine.interfaces import WorkerJob


class EdgePruneJob(WorkerJob):
    def name(self) -> str:
        return "edge_prune"

    async def should_run(self, namespace: str, **kwargs) -> bool:
        return True

    async def execute(self, namespace: str, **kwargs) -> dict:
        db: AsyncSession = kwargs["db"]
        ops = EdgeOps(db)
        count = await ops.prune(namespace, weight_floor=0.05, grace_period_days=7)
        await db.flush()
        return {"edges_pruned": count}
