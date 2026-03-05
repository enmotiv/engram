"""arq worker settings for the Dreamer."""

from __future__ import annotations

import logging

from arq.cron import cron
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from engram.config import settings
from engram.dreamer.backfill_region_embeddings import BackfillRegionEmbeddingsJob
from engram.dreamer.consolidation import ConsolidationJob
from engram.dreamer.decay import EdgeDecayJob
from engram.dreamer.dimension_rescoring import DimensionRescoringJob
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
_rescoring_job = DimensionRescoringJob()
_backfill_regions_job = BackfillRegionEmbeddingsJob()


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
        try:
            namespaces = await _get_namespaces(db)
        except Exception:
            logger.warning("run_decay: failed to fetch namespaces", exc_info=True)
            return
        for ns in namespaces:
            try:
                async with db.begin_nested():
                    await _decay_job.execute(ns, db=db)
            except Exception:
                logger.warning("Edge decay failed on %s", ns, exc_info=True)
        await db.commit()


async def run_prune(ctx):
    """Run edge pruning on all namespaces."""
    async with ctx["session_factory"]() as db:
        try:
            namespaces = await _get_namespaces(db)
        except Exception:
            logger.warning("run_prune: failed to fetch namespaces", exc_info=True)
            return
        for ns in namespaces:
            try:
                async with db.begin_nested():
                    await _prune_job.execute(ns, db=db)
            except Exception:
                logger.warning("Edge pruning failed on %s", ns, exc_info=True)
        await db.commit()


async def run_consolidation(ctx):
    """Run memory consolidation on all namespaces."""
    async with ctx["session_factory"]() as db:
        try:
            namespaces = await _get_namespaces(db)
        except Exception:
            logger.warning("run_consolidation: failed to fetch namespaces", exc_info=True)
            return
        for ns in namespaces:
            try:
                async with db.begin_nested():
                    await _consolidation_job.execute(ns, db=db, registry=ctx["registry"])
            except Exception:
                logger.warning("Consolidation failed on %s", ns, exc_info=True)
        await db.commit()


async def run_modulatory(ctx):
    """Run modulatory edge discovery on all namespaces."""
    async with ctx["session_factory"]() as db:
        try:
            namespaces = await _get_namespaces(db)
        except Exception:
            logger.warning("run_modulatory: failed to fetch namespaces", exc_info=True)
            return
        for ns in namespaces:
            try:
                async with db.begin_nested():
                    await _modulatory_job.execute(ns, db=db)
            except Exception:
                logger.warning("Modulatory discovery failed on %s", ns, exc_info=True)
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
        try:
            namespaces = await _get_namespaces(db)
        except Exception:
            logger.warning("run_graph_generation: failed to fetch namespaces", exc_info=True)
            return
        for ns in namespaces:
            try:
                async with db.begin_nested():
                    if await job.should_run(ns, db=db):
                        result = await job.execute(ns, db=db)
                        logger.info("Graph generation on %s: %s", ns, result)
            except Exception:
                logger.warning("Graph generation failed on %s", ns, exc_info=True)
        await db.commit()


async def run_dimension_rescoring(ctx):
    """Backfill LLM dimension scores for memories with stale/heuristic scores."""
    async with ctx["session_factory"]() as db:
        try:
            namespaces = await _get_namespaces(db)
        except Exception:
            logger.warning("dimension_rescoring: failed to fetch namespaces", exc_info=True)
            return
        for ns in namespaces:
            try:
                async with db.begin_nested():
                    if await _rescoring_job.should_run(ns, db=db):
                        result = await _rescoring_job.execute(ns, db=db)
                        if result.get("rescored", 0) > 0:
                            logger.info("Dimension rescoring on %s: %s", ns, result)
            except Exception:
                logger.warning("Dimension rescoring failed on %s", ns, exc_info=True)
        await db.commit()


async def run_backfill_region_embeddings(ctx):
    """Backfill per-region embeddings for existing memories."""
    async with ctx["session_factory"]() as db:
        try:
            namespaces = await _get_namespaces(db)
        except Exception:
            logger.warning("backfill_region_embeddings: failed to fetch namespaces", exc_info=True)
            return
        for ns in namespaces:
            try:
                async with db.begin_nested():
                    if await _backfill_regions_job.should_run(ns, db=db):
                        result = await _backfill_regions_job.execute(ns, db=db)
                        if result.get("backfilled", 0) > 0:
                            logger.info("Region embedding backfill on %s: %s", ns, result)
            except Exception:
                logger.warning("Region embedding backfill failed on %s", ns, exc_info=True)
        await db.commit()


async def run_plugin_jobs(ctx):
    """Run all plugin-registered WorkerJobs on each namespace."""
    registry: PluginRegistry = ctx["registry"]
    if not registry.jobs:
        return

    async with ctx["session_factory"]() as db:
        try:
            namespaces = await _get_namespaces(db)
        except Exception:
            logger.warning("run_plugin_jobs: failed to fetch namespaces", exc_info=True)
            return
        for ns in namespaces:
            for job in registry.jobs:
                try:
                    async with db.begin_nested():
                        if await job.should_run(ns, db=db):
                            result = await job.execute(ns, db=db)
                            logger.info("Plugin job %s on %s: %s", job.name(), ns, result)
                except Exception:
                    logger.warning(
                        "Plugin job %s failed on %s", job.name(), ns, exc_info=True
                    )
        await db.commit()


def _parse_redis_settings():
    """Parse ENGRAM_REDIS_URL into arq RedisSettings."""
    from arq.connections import RedisSettings
    from urllib.parse import urlparse

    url = settings.REDIS_URL
    parsed = urlparse(url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password,
        database=int(parsed.path.lstrip("/") or 0),
    )


class WorkerSettings:
    functions = [run_decay, run_prune, run_consolidation, run_modulatory, run_graph_generation, run_dimension_rescoring, run_backfill_region_embeddings, run_plugin_jobs]
    max_tries = 3
    cron_jobs = [
        cron(run_decay, hour=3, minute=0),        # Daily at 3:00 UTC
        cron(run_prune, hour=3, minute=30),        # Daily at 3:30 UTC
        cron(run_consolidation, minute=45),         # Hourly at :45
        cron(run_modulatory, weekday=0, hour=4),    # Weekly: Monday 4:00 UTC
        cron(run_graph_generation, hour={1, 7, 13, 19}),  # Every 6 hours
        cron(run_dimension_rescoring, hour={0, 6, 12, 18}),  # Every 6 hours
        cron(run_backfill_region_embeddings, hour={1, 7, 13, 19}, minute=30),  # Every 6h, offset from graphs
        cron(run_plugin_jobs, hour={2, 8, 14, 20}),  # Every 6 hours
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _parse_redis_settings()
