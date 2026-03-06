"""Axis rescoring job — backfill LLM brain-region scores for existing memories."""

from __future__ import annotations

import logging
from typing import Dict, List

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Memory
from engram.engine.interfaces import WorkerJob
from engram.engine.llm import get_llm_service

logger = logging.getLogger(__name__)

# Current scoring version — bump when encoder model/prompt changes
SCORING_VERSION = "llm-v1"

# Max memories to rescore per run (rate limit LLM usage)
BATCH_SIZE = 50

# Per-dimension threshold: skip update if all dims within this range
DIFF_THRESHOLD = 0.15


def _needs_rescoring(mem: Memory) -> bool:
    """Check if a memory needs LLM rescoring."""
    features = mem.features or {}
    return features.get("_scoring_version") != SCORING_VERSION


def _scores_diverge(old: Dict[str, float], new: Dict[str, float]) -> bool:
    """Return True if any dimension differs by more than DIFF_THRESHOLD."""
    for key in set(old) | set(new):
        old_val = old.get(key, 0.0)
        new_val = new.get(key, 0.0)
        if abs(old_val - new_val) > DIFF_THRESHOLD:
            return True
    return False


class AxisRescoringJob(WorkerJob):
    """Backfill LLM-based axis scores for memories with stale/heuristic scores.

    Runs in batches, compares old vs new, only updates when scores meaningfully
    diverge. Tracks scoring version in features metadata.
    """

    def name(self) -> str:
        return "axis_rescoring"

    async def should_run(self, namespace: str, **kwargs) -> bool:
        llm = get_llm_service()
        return llm.is_available()

    async def execute(self, namespace: str, **kwargs) -> dict:
        db: AsyncSession = kwargs["db"]

        # Find memories without current scoring version
        stmt = (
            select(Memory)
            .where(
                Memory.namespace == namespace,
                Memory.content.is_not(None),
            )
            .limit(BATCH_SIZE * 2)  # Oversample — some may already be current
        )
        result = await db.execute(stmt)
        memories = [m for m in result.scalars().all() if _needs_rescoring(m)]

        if not memories:
            return {"rescored": 0, "skipped": 0, "total_checked": 0}

        # Cap to batch size
        batch = memories[:BATCH_SIZE]

        from engram.plugins.brain_regions.encoder import BrainRegionEncoder
        encoder = BrainRegionEncoder()

        rescored = 0
        skipped = 0

        for mem in batch:
            try:
                new_scores = await encoder.llm_encode(mem.content)
            except Exception:
                logger.debug("LLM rescoring failed for memory %s, skipping", mem.id)
                skipped += 1
                continue

            old_scores = mem.dimension_scores or {}

            if _scores_diverge(old_scores, new_scores):
                mem.dimension_scores = new_scores
                # Recalculate salience from updated scores
                new_score_vals = list(new_scores.values())
                mem.salience = sum(new_score_vals) / len(new_score_vals) if new_score_vals else 0.5
                features = dict(mem.features or {})
                features["_scoring_version"] = SCORING_VERSION
                features["_prev_dimension_scores"] = old_scores
                mem.features = features
                rescored += 1
            else:
                # Scores are close enough — just stamp the version
                features = dict(mem.features or {})
                features["_scoring_version"] = SCORING_VERSION
                mem.features = features
                skipped += 1

        await db.flush()
        return {
            "rescored": rescored,
            "skipped": skipped,
            "total_checked": len(batch),
        }
