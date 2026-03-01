"""Edge decay job — exponential weight decay on inactive edges."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from engram.engine.edge_ops import EdgeOps
from engram.engine.interfaces import WorkerJob


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
        return {"edges_decayed": count}
