"""Edge classification job — find nearest neighbors and create typed edges."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Memory
from engram.engine.edges import EdgeStore
from engram.engine.interfaces import WorkerJob

# Contradiction / superseding keywords → inhibitory
_INHIBITORY_KEYWORDS = {
    "but", "however", "instead", "no longer", "migrated from", "replaced",
    "deprecated", "removed", "switched from", "abandoned", "cancelled",
    "rejected", "not", "never", "wrong", "incorrect", "false",
}

# Temporal keywords → temporal edge
_TEMPORAL_PATTERNS = re.compile(
    r"\b(yesterday|today|tomorrow|last\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday)|"
    r"next\s+(?:week|month)|january|february|march|april|may|june|july|august|"
    r"september|october|november|december|\d{4}[-/]\d{2}[-/]\d{2}|"
    r"q[1-4]\s+\d{4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2})\b",
    re.IGNORECASE,
)


def _has_temporal_tokens(text_content: str) -> bool:
    return bool(_TEMPORAL_PATTERNS.search(text_content))


def _has_inhibitory_signal(content_a: str, content_b: str) -> bool:
    """Check if the pair of contents suggests contradiction/superseding."""
    combined = (content_a + " " + content_b).lower()
    return any(kw in combined for kw in _INHIBITORY_KEYWORDS)


def classify_edge_heuristic(
    content_a: str,
    content_b: str,
    similarity: float,
) -> Optional[Tuple[str, float]]:
    """Classify edge type using heuristics. Returns (edge_type, confidence) or None."""
    # Check for inhibitory signal first
    if _has_inhibitory_signal(content_a, content_b) and similarity > 0.3:
        return ("inhibitory", min(0.7, similarity))

    # Temporal if both have date tokens
    if _has_temporal_tokens(content_a) and _has_temporal_tokens(content_b):
        return ("temporal", 0.6)

    # High similarity → excitatory (reinforcing)
    if similarity > 0.7:
        return ("excitatory", similarity)

    # Moderate similarity → associative
    if similarity > 0.4:
        return ("associative", similarity * 0.8)

    # Too dissimilar — no edge
    return None


class EdgeClassificationJob(WorkerJob):
    def name(self) -> str:
        return "edge_classification"

    async def should_run(self, namespace: str, **kwargs) -> bool:
        return True

    async def execute(self, namespace: str, **kwargs) -> dict:
        db: AsyncSession = kwargs["db"]
        memory_id = kwargs.get("memory_id")

        if not memory_id:
            return {"edges_created": 0}

        mem = await db.get(Memory, memory_id)
        if mem is None or mem.embedding is None:
            return {"edges_created": 0}

        # Find 10 nearest neighbors by cosine similarity
        neighbors = await self._find_neighbors(db, namespace, mem, limit=10)

        edge_store = EdgeStore(db)
        edges_created = 0

        for neighbor, similarity in neighbors:
            result = classify_edge_heuristic(mem.content, neighbor.content, similarity)
            if result is None:
                continue

            edge_type, confidence = result
            if confidence < 0.4:
                continue

            await edge_store.create(
                source_id=mem.id,
                target_id=neighbor.id,
                edge_type=edge_type,
                weight=confidence,
                context={"auto_classified": True, "similarity": similarity},
                namespace=namespace,
            )
            edges_created += 1

        await db.flush()
        return {"edges_created": edges_created}

    async def _find_neighbors(
        self,
        db: AsyncSession,
        namespace: str,
        memory: Memory,
        limit: int = 10,
    ) -> List[Tuple[Memory, float]]:
        """Find nearest neighbors by cosine similarity."""
        vec_literal = "[" + ",".join(str(v) for v in memory.embedding) + "]"
        stmt = text(
            f"SELECT id, 1 - (embedding <=> '{vec_literal}'::vector) AS similarity "
            f"FROM memories "
            f"WHERE namespace = :ns AND id != :mem_id AND embedding IS NOT NULL "
            f"ORDER BY embedding <=> '{vec_literal}'::vector "
            f"LIMIT :lim"
        )
        result = await db.execute(stmt, {"ns": namespace, "mem_id": memory.id, "lim": limit})
        rows = result.fetchall()

        neighbors = []
        for row in rows:
            neighbor = await db.get(Memory, row.id)
            if neighbor:
                neighbors.append((neighbor, float(row.similarity)))

        return neighbors
