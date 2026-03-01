"""arq worker settings for the Dreamer."""

from __future__ import annotations

import logging

from arq.cron import cron
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from engram.config import settings
from engram.dreamer.consolidation import ConsolidationJob
from engram.dreamer.decay import EdgeDecayJob
from engram.dreamer.graph_generation import GraphGenerationJob
from engram.dreamer.modulatory import ModulatoryDiscoveryJob
from engram.dreamer.prune import EdgePruneJob
from engram.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)

_decay_job = EdgeDecayJob()
_prune_job = EdgePruneJob()
_consolidation_job = ConsolidationJob()
_modulatory_job = ModulatoryDiscoveryJob()
_graph_job = GraphGenerationJob()


async def startup(ctx):
    """Initialize DB and plugins on worker startup."""
    engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)
    ctx["engine"] = engine
    ctx["session_factory"] = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    registry = PluginRegistry.get_instance()
    registry.load_plugins(settings.plugin_paths)
    ctx["registry"] = registry


async def shutdown(ctx):
    """Cleanup on worker shutdown."""
    engine = ctx.get("engine")
    if engine:
        await engine.dispose()


async def _get_namespaces(db):
    """Fetch all unique namespaces from the memories table."""
    from sqlalchemy import distinct, select
    from engram.db.models import Memory
    result = await db.execute(select(distinct(Memory.namespace)))
    return [r[0] for r in result.fetchall()]


async def run_decay(ctx):
    """Run edge decay on all namespaces."""
    async with ctx["session_factory"]() as db:
        for ns in await _get_namespaces(db):
            await _decay_job.execute(ns, db=db)
        await db.commit()


async def run_prune(ctx):
    """Run edge pruning on all namespaces."""
    async with ctx["session_factory"]() as db:
        for ns in await _get_namespaces(db):
            await _prune_job.execute(ns, db=db)
        await db.commit()


async def run_consolidation(ctx):
    """Run memory consolidation on all namespaces."""
    async with ctx["session_factory"]() as db:
        for ns in await _get_namespaces(db):
            await _consolidation_job.execute(ns, db=db, registry=ctx["registry"])
        await db.commit()


async def run_modulatory(ctx):
    """Run modulatory edge discovery on all namespaces."""
    async with ctx["session_factory"]() as db:
        for ns in await _get_namespaces(db):
            await _modulatory_job.execute(ns, db=db)
        await db.commit()


async def run_graph_generation(ctx):
    """Run graph generation on all namespaces that have enough new memories."""
    registry: PluginRegistry = ctx["registry"]
    # Use plugin-registered generator if available, otherwise default
    generator = registry.graph_generator
    if generator:
        job = GraphGenerationJob(generator)
    else:
        job = _graph_job

    async with ctx["session_factory"]() as db:
        for ns in await _get_namespaces(db):
            try:
                if await job.should_run(ns, db=db):
                    result = await job.execute(ns, db=db)
                    logger.info("Graph generation on %s: %s", ns, result)
            except Exception:
                logger.warning("Graph generation failed on %s", ns, exc_info=True)
        await db.commit()


async def run_plugin_jobs(ctx):
    """Run all plugin-registered WorkerJobs on each namespace."""
    registry: PluginRegistry = ctx["registry"]
    if not registry.jobs:
        return

    async with ctx["session_factory"]() as db:
        for ns in await _get_namespaces(db):
            for job in registry.jobs:
                try:
                    if await job.should_run(ns, db=db):
                        result = await job.execute(ns, db=db)
                        logger.info("Plugin job %s on %s: %s", job.name(), ns, result)
                except Exception:
                    logger.warning(
                        "Plugin job %s failed on %s", job.name(), ns, exc_info=True
                    )
        await db.commit()


class WorkerSettings:
    functions = [run_decay, run_prune, run_consolidation, run_modulatory, run_graph_generation, run_plugin_jobs]
    cron_jobs = [
        cron(run_decay, hour=3, minute=0),        # Daily at 3:00 UTC
        cron(run_prune, hour=3, minute=30),        # Daily at 3:30 UTC
        cron(run_consolidation, minute=45),         # Hourly at :45
        cron(run_modulatory, weekday=0, hour=4),    # Weekly: Monday 4:00 UTC
        cron(run_graph_generation, hour={1, 7, 13, 19}),  # Every 6 hours
        cron(run_plugin_jobs, hour={2, 8, 14, 20}),  # Every 6 hours
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = None  # Set from ENGRAM_REDIS_URL at runtime
