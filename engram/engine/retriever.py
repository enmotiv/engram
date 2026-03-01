"""Multi-axis retrieval with convergence scoring."""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Memory
from engram.engine.embedding import EmbeddingService
from engram.engine.models import MemoryResult, RetrievalOptions, RetrievalResult
from engram.engine.urgency import score_urgency
from engram.plugins.registry import PluginRegistry


class Retriever:
    """Multi-axis retrieval with convergence scoring and spreading activation."""

    def __init__(
        self,
        db: AsyncSession,
        registry: Optional[PluginRegistry] = None,
    ):
        self.db = db
        self._registry = registry or PluginRegistry.get_instance()
        self._embedding = EmbeddingService(self._registry)

    async def retrieve(
        self,
        namespace: str,
        cue: str,
        context: Optional[dict] = None,
        options: Optional[RetrievalOptions] = None,
    ) -> RetrievalResult:
        opts = options or RetrievalOptions()
        start = time.monotonic()

        # Step 0: Urgency gate
        urgency = score_urgency(cue, context)
        if urgency < opts.urgency_threshold:
            return RetrievalResult(
                triggered=False,
                urgency=urgency,
                retrieval_ms=round((time.monotonic() - start) * 1000, 2),
            )

        # Step 1: Encode the cue
        cue_embedding = await self._embedding.get_embedding(cue)
        cue_scores = await self._embedding.get_dimension_scores(cue)

        if cue_embedding is None:
            return RetrievalResult(
                triggered=True,
                urgency=urgency,
                retrieval_ms=round((time.monotonic() - start) * 1000, 2),
            )

        # Step 2: Vector search — pgvector cosine similarity
        top_k = opts.max_results * 4  # oversample for convergence filtering
        candidates = await self._vector_search(namespace, cue_embedding, top_k)

        # Step 3: Convergence scoring
        scored = self._convergence_score(candidates, cue_scores)

        # Step 4: Sort and limit
        scored.sort(key=lambda m: m.convergence_score, reverse=True)
        results = scored[: opts.max_results]

        elapsed = round((time.monotonic() - start) * 1000, 2)

        return RetrievalResult(
            triggered=True,
            urgency=urgency,
            memories=results,
            retrieval_ms=elapsed,
        )

    async def _vector_search(
        self, namespace: str, cue_embedding: List[float], top_k: int
    ) -> List[dict]:
        """Cosine similarity search via pgvector."""
        vec_literal = "[" + ",".join(str(v) for v in cue_embedding) + "]"
        # asyncpg doesn't support :: casts in parameterized queries,
        # so we embed the vector literal directly in the SQL string.
        stmt = text(
            f"SELECT id, content, memory_type, dimension_scores, activation, salience, "
            f"1 - (embedding <=> '{vec_literal}'::vector) AS cosine_sim "
            f"FROM memories "
            f"WHERE namespace = :ns AND embedding IS NOT NULL "
            f"ORDER BY embedding <=> '{vec_literal}'::vector "
            f"LIMIT :k"
        )
        result = await self.db.execute(
            stmt, {"ns": namespace, "k": top_k}
        )
        rows = result.fetchall()
        return [
            {
                "id": str(r.id),
                "content": r.content,
                "memory_type": r.memory_type,
                "dimension_scores": r.dimension_scores or {},
                "activation": r.activation or 0.0,
                "cosine_sim": r.cosine_sim,
            }
            for r in rows
        ]

    def _convergence_score(
        self, candidates: List[dict], cue_scores: Dict[str, float]
    ) -> List[MemoryResult]:
        """Score each candidate on multi-dimension convergence."""
        results = []
        for c in candidates:
            mem_scores = c["dimension_scores"]
            dims_matched = []
            dim_score_sum = 0.0
            dim_count = 0

            for dim, cue_val in cue_scores.items():
                mem_val = mem_scores.get(dim, 0.0)
                score = 1.0 - abs(mem_val - cue_val)
                if score > 0.3:
                    dims_matched.append(dim)
                    dim_score_sum += score
                    dim_count += 1

            avg_dim_score = dim_score_sum / dim_count if dim_count > 0 else 0.0
            cosine_sim = c.get("cosine_sim", 0.0)

            # Convergence: dims_matched * avg_dimension_score * cosine_similarity
            convergence = len(dims_matched) * avg_dim_score * cosine_sim

            results.append(
                MemoryResult(
                    id=c["id"],
                    content=c["content"],
                    activation=c["activation"],
                    dimensions_matched=dims_matched,
                    convergence_score=round(convergence, 4),
                    retrieval_path="direct",
                )
            )
        return results
