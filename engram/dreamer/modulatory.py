"""Modulatory edge discovery — find deep structural pattern matches across topics."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Memory
from engram.engine.edges import EdgeStore
from engram.engine.interfaces import WorkerJob


class ModulatoryDiscoveryJob(WorkerJob):
    """Discover modulatory edges between memories with matching structural patterns
    but different surface content. These are the 'aha moment' connections."""

    def name(self) -> str:
        return "modulatory_discovery"

    async def should_run(self, namespace: str, **kwargs) -> bool:
        return True

    async def execute(self, namespace: str, **kwargs) -> dict:
        db: AsyncSession = kwargs["db"]

        # Find memories with populated feature_vectors
        stmt = select(Memory).where(
            Memory.namespace == namespace,
            Memory.feature_vector.is_not(None),
            Memory.embedding.is_not(None),
        )
        result = await db.execute(stmt)
        memories = list(result.scalars().all())

        if len(memories) < 2:
            return {"pairs_scanned": 0, "modulatory_edges_created": 0}

        edge_store = EdgeStore(db)
        pairs_scanned = 0
        edges_created = 0

        # Check existing modulatory edges to avoid duplicates
        existing_pairs: Set[Tuple[str, str]] = set()

        for i, m1 in enumerate(memories):
            for m2 in memories[i + 1:]:
                pairs_scanned += 1

                pair_key = (str(m1.id), str(m2.id))
                reverse_key = (str(m2.id), str(m1.id))
                if pair_key in existing_pairs or reverse_key in existing_pairs:
                    continue

                # Compute feature vector similarity (high = structurally similar)
                feature_sim = await self._cosine_similarity(
                    db, list(m1.feature_vector), list(m2.feature_vector), "vector(32)"
                )

                # Compute content embedding similarity (low = topically different)
                content_sim = await self._cosine_similarity(
                    db, list(m1.embedding), list(m2.embedding), "vector(1024)"
                )

                if feature_sim is None or content_sim is None:
                    continue

                # Core formula: high feature similarity + low content similarity
                if feature_sim <= 0.8 or content_sim >= 0.4:
                    continue

                recency = 1.0  # simplified — could factor in created_at
                salience = max(m1.salience or 0.5, m2.salience or 0.5)
                score = feature_sim * (1 - content_sim) * recency * salience

                if score <= 0.5:
                    continue

                # Determine which structural features matched
                matched_features = self._find_matched_features(m1, m2)

                # Average the feature vectors for context_embedding
                avg_feature_vec = [
                    (a + b) / 2 for a, b in zip(m1.feature_vector, m2.feature_vector)
                ]

                await edge_store.create(
                    source_id=m1.id,
                    target_id=m2.id,
                    edge_type="modulatory",
                    weight=round(score, 4),
                    context={
                        "matched_features": matched_features,
                        "feature_similarity": round(feature_sim, 4),
                        "content_similarity": round(content_sim, 4),
                        "score": round(score, 4),
                    },
                    namespace=namespace,
                )

                existing_pairs.add(pair_key)
                edges_created += 1

        await db.flush()
        return {
            "pairs_scanned": pairs_scanned,
            "modulatory_edges_created": edges_created,
        }

    async def _cosine_similarity(
        self,
        db: AsyncSession,
        vec_a: List[float],
        vec_b: List[float],
        vec_type: str,
    ) -> float | None:
        """Compute cosine similarity between two vectors using pgvector."""
        lit_a = "[" + ",".join(str(v) for v in vec_a) + "]"
        lit_b = "[" + ",".join(str(v) for v in vec_b) + "]"
        stmt = text(
            f"SELECT 1 - ('{lit_a}'::{vec_type} <=> '{lit_b}'::{vec_type}) AS sim"
        )
        result = await db.execute(stmt)
        val = result.scalar()
        return float(val) if val is not None else None

    def _find_matched_features(self, m1: Memory, m2: Memory) -> Dict[str, str]:
        """Find which structural feature categories match between two memories."""
        f1 = m1.features or {}
        f2 = m2.features or {}
        matched = {}
        for key in ("domain", "action_type", "dynamic", "scope", "causality", "valence", "abstraction"):
            if key in f1 and key in f2 and f1[key] == f2[key]:
                matched[key] = f1[key]
        return matched
