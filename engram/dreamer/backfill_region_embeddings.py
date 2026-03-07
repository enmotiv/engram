"""Backfill per-region embeddings for existing memories.

One-time dreamer job that iterates memories with NULL regional embeddings,
decomposes content via LLM, embeds each region independently, and populates
the 6 region embedding columns.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Memory
from engram.engine.interfaces import WorkerJob
from engram.engine.llm import get_llm_service

logger = logging.getLogger(__name__)

# Max memories to backfill per run (rate limit embedding API)
BATCH_SIZE = 50

REGION_COLUMNS = [
    "hippo_embedding",
    "amyg_embedding",
    "pfc_embedding",
    "sensory_embedding",
    "striatum_embedding",
    "cerebellum_embedding",
]


def _needs_backfill(mem: Memory) -> bool:
    """Check if a memory needs region embedding backfill."""
    return mem.hippo_embedding is None


class BackfillRegionEmbeddingsJob(WorkerJob):
    """Backfill per-region embeddings for memories with NULL regional columns.

    Runs in batches via the dreamer scheduler. Rate-limited to avoid
    hammering the embedding API.
    """

    def name(self) -> str:
        return "backfill_region_embeddings"

    async def should_run(self, namespace: str, **kwargs) -> bool:
        llm = get_llm_service()
        return llm.is_available()

    async def execute(self, namespace: str, **kwargs) -> dict:
        db: AsyncSession = kwargs["db"]

        # Find memories without region embeddings
        stmt = (
            select(Memory)
            .where(
                Memory.namespace == namespace,
                Memory.content.is_not(None),
                Memory.hippo_embedding.is_(None),
            )
            .limit(BATCH_SIZE)
        )
        result = await db.execute(stmt)
        memories = list(result.scalars().all())

        if not memories:
            return {"backfilled": 0, "skipped": 0, "total_checked": 0}

        logger.info(
            "BackfillRegionEmbeddingsJob: found %d memories needing region backfill "
            "in namespace=%s",
            len(memories), namespace,
        )

        from engram.plugins.brain_regions.encoder import BrainRegionEncoder
        encoder = BrainRegionEncoder()

        backfilled = 0
        skipped = 0

        for mem in memories:
            try:
                # Decompose content into region-specific text
                region_texts = await encoder.decompose(mem.content)
                if not region_texts:
                    skipped += 1
                    continue

                # Embed each region independently
                for region, region_text in region_texts.items():
                    vec = await encoder.embed(region_text)
                    if vec is not None:
                        setattr(mem, f"{region}_embedding", vec)

                # Mark as multi-axis encoded
                if mem.features is None:
                    mem.features = {}
                existing = dict(mem.features)
                existing["multi_axis_encoded"] = True
                mem.features = existing

                backfilled += 1
            except Exception:
                logger.debug("Region embedding backfill failed for memory %s", mem.id, exc_info=True)
                if mem.features is None:
                    mem.features = {}
                existing = dict(mem.features)
                if "multi_axis_encoded" not in existing:
                    existing["multi_axis_encoded"] = False
                    mem.features = existing
                skipped += 1

        await db.flush()
        return {
            "backfilled": backfilled,
            "skipped": skipped,
            "total_checked": len(memories),
        }
