"""Consolidation job — cluster similar episodic memories into semantic summaries."""

from __future__ import annotations

import logging
from typing import Dict, List, Set

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Memory
from engram.engine.edges import EdgeStore
from engram.engine.interfaces import WorkerJob
from engram.engine.llm import get_llm_service
from engram.engine.store import MemoryStore
from engram.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


class ConsolidationJob(WorkerJob):
    def name(self) -> str:
        return "consolidation"

    async def should_run(self, namespace: str, **kwargs) -> bool:
        return True

    async def execute(self, namespace: str, **kwargs) -> dict:
        db: AsyncSession = kwargs["db"]
        registry: PluginRegistry = kwargs.get("registry") or PluginRegistry.get_instance()

        store = MemoryStore(db, registry)
        edge_store = EdgeStore(db)

        # Find episodic memories with embeddings
        stmt = select(Memory).where(
            Memory.namespace == namespace,
            Memory.memory_type == "episodic",
            Memory.embedding.is_not(None),
        )
        result = await db.execute(stmt)
        memories = list(result.scalars().all())

        if len(memories) < 3:
            return {"clusters_found": 0, "summaries_created": 0}

        consolidatable = memories

        # Find clusters via pairwise cosine similarity > 0.85
        clusters = await self._find_clusters(db, namespace, consolidatable)

        summaries_created = 0
        for cluster_mems in clusters:
            if len(cluster_mems) < 3:
                continue

            # Generate summary — LLM abstractive or heuristic concatenation
            summary_content = await self._summarize_cluster(cluster_mems)

            # Create semantic summary memory
            summary = await store.create(
                namespace=namespace,
                content=summary_content,
                memory_type="semantic",
            )

            # Create temporal edges from source memories to summary
            for source in cluster_mems:
                await edge_store.create(
                    source_id=source.id,
                    target_id=summary.id,
                    edge_type="temporal",
                    weight=0.6,
                    namespace=namespace,
                )

            summaries_created += 1

        await db.flush()
        return {
            "clusters_found": len(clusters),
            "summaries_created": summaries_created,
        }

    async def _find_clusters(
        self,
        db: AsyncSession,
        namespace: str,
        memories: List[Memory],
    ) -> List[List[Memory]]:
        """Find clusters of memories with cosine similarity > 0.85."""
        if not memories:
            return []

        # Build adjacency via pairwise similarity
        mem_by_id = {str(m.id): m for m in memories}
        adjacency: Dict[str, Set[str]] = {str(m.id): set() for m in memories}

        # For each pair, check cosine similarity
        for i, m1 in enumerate(memories):
            vec1 = "[" + ",".join(str(v) for v in m1.embedding) + "]"
            for m2 in memories[i + 1:]:
                vec2 = "[" + ",".join(str(v) for v in m2.embedding) + "]"
                stmt = text(
                    f"SELECT 1 - ('{vec1}'::vector <=> '{vec2}'::vector) AS sim"
                )
                result = await db.execute(stmt)
                sim = result.scalar()
                if sim and sim > 0.85:
                    adjacency[str(m1.id)].add(str(m2.id))
                    adjacency[str(m2.id)].add(str(m1.id))

        # Simple greedy clustering: connected components
        visited: Set[str] = set()
        clusters: List[List[Memory]] = []

        for mid in adjacency:
            if mid in visited:
                continue
            cluster_ids: Set[str] = set()
            stack = [mid]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                cluster_ids.add(current)
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        stack.append(neighbor)

            if len(cluster_ids) >= 3:
                clusters.append([mem_by_id[cid] for cid in cluster_ids])

        return clusters

    async def _summarize_cluster(self, cluster_mems: List[Memory]) -> str:
        """Summarize a cluster of memories. Uses LLM if available, else concatenation."""
        llm = get_llm_service()
        if not llm.is_available():
            combined = " | ".join(m.content for m in cluster_mems)
            return combined[:500]

        combined = "\n- ".join(m.content for m in cluster_mems)
        prompt = (
            "Summarize these related memories into a single concise observation "
            "(1-2 sentences). Preserve key details and specifics.\n\n"
            f"Memories:\n- {combined}"
        )
        try:
            return await llm.complete(prompt, max_tokens=200, temperature=0.3)
        except Exception:
            logger.debug("LLM summarization failed, using concatenation fallback")
            combined = " | ".join(m.content for m in cluster_mems)
            return combined[:500]
