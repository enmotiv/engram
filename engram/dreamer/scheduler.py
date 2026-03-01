"""arq worker settings for the Dreamer."""

from __future__ import annotations

from arq.cron import cron
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from engram.config import settings
from engram.dreamer.consolidation import ConsolidationJob
from engram.dreamer.decay import EdgeDecayJob
from engram.dreamer.prune import EdgePruneJob
from engram.plugins.registry import PluginRegistry

_decay_job = EdgeDecayJob()
_prune_job = EdgePruneJob()
_consolidation_job = ConsolidationJob()


async def startup(ctx):
    """Initialize DB and plugin on worker startup."""
    engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)
    ctx["engine"] = engine
    ctx["session_factory"] = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    registry = PluginRegistry.get_instance()
    registry.load_plugin(settings.ENGRAM_PLUGIN)
    ctx["registry"] = registry


async def shutdown(ctx):
    """Cleanup on worker shutdown."""
    engine = ctx.get("engine")
    if engine:
        await engine.dispose()


async def run_decay(ctx):
    """Run edge decay on all namespaces."""
    async with ctx["session_factory"]() as db:
        from sqlalchemy import distinct, select
        from engram.db.models import Memory
        result = await db.execute(select(distinct(Memory.namespace)))
        namespaces = [r[0] for r in result.fetchall()]

        for ns in namespaces:
            await _decay_job.execute(ns, db=db)
        await db.commit()


async def run_prune(ctx):
    """Run edge pruning on all namespaces."""
    async with ctx["session_factory"]() as db:
        from sqlalchemy import distinct, select
        from engram.db.models import Memory
        result = await db.execute(select(distinct(Memory.namespace)))
        namespaces = [r[0] for r in result.fetchall()]

        for ns in namespaces:
            await _prune_job.execute(ns, db=db)
        await db.commit()


async def run_consolidation(ctx):
    """Run memory consolidation on all namespaces."""
    async with ctx["session_factory"]() as db:
        from sqlalchemy import distinct, select
        from engram.db.models import Memory
        result = await db.execute(select(distinct(Memory.namespace)))
        namespaces = [r[0] for r in result.fetchall()]

        for ns in namespaces:
            await _consolidation_job.execute(ns, db=db, registry=ctx["registry"])
        await db.commit()


class WorkerSettings:
    functions = [run_decay, run_prune, run_consolidation]
    cron_jobs = [
        cron(run_decay, hour=3, minute=0),        # Daily at 3:00 UTC
        cron(run_prune, hour=3, minute=30),        # Daily at 3:30 UTC
        cron(run_consolidation, minute=45),         # Hourly at :45
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = None  # Set from ENGRAM_REDIS_URL at runtime
