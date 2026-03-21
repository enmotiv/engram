"""Dreamer scheduler — arq worker settings + job definitions."""

import logging
from urllib.parse import urlparse
from uuid import UUID

from arq import cron
from arq.connections import RedisSettings

from engram.config import settings

logger = logging.getLogger(__name__)


async def startup(ctx: dict) -> None:
    """Called once when the worker starts. Run migrations + initialize DB pool."""
    from engram.core.db import get_pool
    from engram.migrations.runner import run_migrations

    await run_migrations()
    ctx["db_pool"] = await get_pool(settings.database_url)
    logger.info("dreamer.worker_started")


async def shutdown(ctx: dict) -> None:
    """Called once when the worker stops. Close DB pool."""
    from engram.core.db import close_pool

    pool = ctx.get("db_pool")
    if pool:
        await close_pool(pool)
    logger.info("dreamer.worker_stopped")


# -- Event-Driven Jobs --------------------------------------------------------


async def classify_new_memories(ctx: dict, owner_id: str) -> dict:
    """Process all unclassified memories for one owner.

    Enqueued by the write path after every successful INSERT.
    Idempotent — safe to enqueue multiple times for the same owner.
    """
    from engram.services.dreamer import process_new_memories

    pool = ctx["db_pool"]
    return await process_new_memories(pool, UUID(owner_id))


# -- Periodic Jobs -------------------------------------------------------------


async def classify_all_unprocessed(ctx: dict) -> dict:
    """Periodic: classify unprocessed memories for all owners.

    Catches memories that weren't classified event-driven (e.g. bulk imports,
    failed enqueue, or duplicates that were later re-inserted).
    """
    from engram.services.dreamer import process_new_memories

    pool = ctx["db_pool"]
    owners = await _get_all_owner_ids(pool)
    total_classified = 0
    total_edges = 0
    for oid in owners:
        result = await process_new_memories(pool, oid)
        total_classified += result.get("memories_processed", 0)
        total_edges += result.get("edges_created", 0)
    return {
        "owners": len(owners),
        "total_classified": total_classified,
        "total_edges": total_edges,
    }


async def decay_all_activations(ctx: dict) -> dict:
    """Hourly: decay activation on all owners."""
    from engram.services.dreamer import decay_activations

    pool = ctx["db_pool"]
    owners = await _get_all_owner_ids(pool)
    total = 0
    for oid in owners:
        count = await decay_activations(pool, oid)
        total += count
    return {"owners": len(owners), "total_decayed": total}


async def decay_all_edge_weights(ctx: dict) -> dict:
    """Weekly: decay stale edge weights on all owners."""
    from engram.services.dreamer import decay_edge_weights

    pool = ctx["db_pool"]
    owners = await _get_all_owner_ids(pool)
    total = 0
    for oid in owners:
        count = await decay_edge_weights(pool, oid)
        total += count
    return {"owners": len(owners), "total_decayed": total}


async def dedup_all_memories(ctx: dict) -> dict:
    """Weekly: find and merge near-duplicate memories on all owners."""
    from engram.services.dreamer import dedup_memories

    pool = ctx["db_pool"]
    owners = await _get_all_owner_ids(pool)
    total_deleted = 0
    total_edges = 0
    for oid in owners:
        stats = await dedup_memories(pool, oid)
        total_deleted += stats["nodes_deleted"]
        total_edges += stats["edges_transferred"]
    return {
        "owners": len(owners),
        "total_deleted": total_deleted,
        "total_edges_transferred": total_edges,
    }


async def prune_all_edges(ctx: dict) -> dict:
    """Monthly: remove edges with weight < 0.15 on all owners."""
    from engram.services.dreamer import prune_edges

    pool = ctx["db_pool"]
    owners = await _get_all_owner_ids(pool)
    total = 0
    for oid in owners:
        count = await prune_edges(pool, oid)
        total += count
    return {"owners": len(owners), "total_pruned": total}


# -- Helpers -------------------------------------------------------------------


async def _get_all_owner_ids(pool) -> list[UUID]:
    """Fetch all owner IDs. Used by periodic jobs to iterate tenants."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id FROM owners")
    return [row["id"] for row in rows]


# -- Worker Settings -----------------------------------------------------------


def _parse_redis_settings() -> RedisSettings:
    """Parse REDIS_URL into arq RedisSettings."""
    parsed = urlparse(settings.redis_url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password,
        database=int(parsed.path.lstrip("/") or "0"),
    )


class WorkerSettings:
    """arq worker configuration. Run with: arq engram.scheduler.WorkerSettings"""

    functions = [
        classify_new_memories,
    ]

    cron_jobs = [
        cron(
            classify_all_unprocessed,
            hour={0, 6, 12, 18},
            minute={15},
        ),  # Every 6 hours
        cron(
            decay_all_activations,
            hour=set(range(24)),
        ),  # Every hour
        cron(
            decay_all_edge_weights,
            weekday={0},
            hour={3},
            minute={0},
        ),  # Monday 3am
        cron(
            dedup_all_memories,
            weekday={0},
            hour={4},
            minute={0},
        ),  # Monday 4am
        cron(
            prune_all_edges,
            day={1},
            hour={5},
            minute={0},
        ),  # 1st of month 5am
    ]

    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _parse_redis_settings()

    max_jobs = 10
    job_timeout = 300
    keep_result = 3600
    retry_jobs = True
    max_tries = 3
