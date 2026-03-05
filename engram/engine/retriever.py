"""Multi-axis retrieval with convergence scoring and spreading activation."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Edge, Memory
from engram.engine.cache import TraceCache
from engram.engine.edge_ops import EdgeOps
from engram.engine.edges import EdgeStore
from engram.engine.embedding import EmbeddingService
from engram.engine.models import (
    HeaderSearchResult,
    MemoryHeaderResult,
    MemoryResult,
    RetrievalOptions,
    RetrievalResult,
)
from engram.engine.tracer import TraceGenerator
from engram.engine.urgency import score_urgency
from engram.config import settings
from engram.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)

# Edge-type propagation multipliers
_EDGE_MULTIPLIERS = {
    "excitatory": 1.0,
    "inhibitory": -1.0,
    "associative": 0.5,
    "temporal": 0.5,
    "modulatory": 0.3,
}

# Brain regions for multi-axis search
_REGIONS = ["hippo", "amyg", "pfc", "sensory", "striatum", "cerebellum"]

# Convergence formula blending parameter.
# alpha=1.0 → pure breadth*depth (dims_matched * mean_score)
# alpha=0.0 → pure max_region_score (best single-axis match)
# Default 0.7 balances breadth of match with peak match quality.
_CONVERGENCE_ALPHA = 0.7


class Retriever:
    """Multi-axis retrieval with convergence scoring and spreading activation."""

    def __init__(
        self,
        db: AsyncSession,
        registry: Optional[PluginRegistry] = None,
        trace_cache: Optional[TraceCache] = None,
    ):
        self.db = db
        self._registry = registry or PluginRegistry.get_instance()
        self._embedding = EmbeddingService(self._registry)
        self._edge_store = EdgeStore(db)
        self._edge_ops = EdgeOps(db)
        self._tracer = TraceGenerator(self._registry)
        self._cache = trace_cache

    async def retrieve(
        self,
        namespace: str,
        cue: str,
        context: Optional[dict] = None,
        options: Optional[RetrievalOptions] = None,
        dimensional_cues: Optional[Dict[str, str]] = None,
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

        # Step 1: Encode the cue — semantic embedding + per-region decomposition
        cue_embedding = await self._embedding.get_embedding(cue)
        region_vectors = await self._embedding.get_region_embeddings(cue)

        # Merge caller-provided dimensional cues (override LLM decomposition)
        if dimensional_cues:
            for region, text in dimensional_cues.items():
                if region in _REGIONS and text.strip():
                    vec = await self._embedding.embed_text(text)
                    if vec is not None:
                        region_vectors[region] = vec

        if cue_embedding is None and not region_vectors:
            return RetrievalResult(
                triggered=True,
                urgency=urgency,
                retrieval_ms=round((time.monotonic() - start) * 1000, 2),
            )

        # Step 2: Multi-axis search (6 independent pgvector queries)
        top_k = opts.max_results * 4
        if region_vectors:
            scored = await self._multi_axis_search(
                namespace, region_vectors, cue_embedding, top_k
            )
        else:
            # Fallback: single-embedding cosine search when decomposition fails
            scored = await self._single_axis_fallback(namespace, cue_embedding, top_k)

        # Step 3: Spreading activation through typed edges
        activated, suppressed, traversed_edges = await self._spread_activation(
            scored, namespace, opts.hop_depth, context
        )

        # Step 4: Sort by activation and limit
        activated.sort(key=lambda m: m.activation, reverse=True)
        results = activated[: opts.max_results]

        # Step 5: Generate trace
        trace = None
        if results and opts.trace_format == "graph":
            memory_ids = [m.id for m in results]
            if suppressed:
                memory_ids += [s["id"] for s in suppressed]

            cache_key = None
            if self._cache:
                cache_key = TraceCache.compute_cache_key(namespace, memory_ids)
                cached = await self._cache.get(namespace, cache_key)
                if cached:
                    trace = cached

            if trace is None:
                trace = self._tracer.generate(results, traversed_edges, suppressed)
                if self._cache and cache_key:
                    await self._cache.set(namespace, cache_key, trace)

        elapsed = round((time.monotonic() - start) * 1000, 2)

        result = RetrievalResult(
            triggered=True,
            urgency=urgency,
            memories=results,
            suppressed=suppressed,
            trace=trace,
            retrieval_ms=elapsed,
        )

        # Reconsolidation (Hebbian learning)
        if results and opts.reconsolidate:
            await self._reconsolidate(result, namespace)

        return result

    async def search_headers(
        self,
        namespace: str,
        cue: str,
        context: Optional[dict] = None,
        max_results: int = 50,
        urgency_threshold: float = 0.3,
    ) -> HeaderSearchResult:
        """Layer 0 header search — returns metadata without content.

        Same pipeline as retrieve() (urgency -> encode -> vector search ->
        convergence -> spreading activation) but the SQL selects NO content
        column. Returns MemoryHeaderResult objects for lightweight scanning.
        """
        start = time.monotonic()

        urgency = score_urgency(cue, context)
        if urgency < urgency_threshold:
            return HeaderSearchResult(
                triggered=False,
                urgency=urgency,
                retrieval_ms=round((time.monotonic() - start) * 1000, 2),
            )

        cue_embedding = await self._embedding.get_embedding(cue)
        cue_scores = await self._embedding.get_dimension_scores(cue)

        if cue_embedding is None:
            return HeaderSearchResult(
                triggered=True,
                urgency=urgency,
                retrieval_ms=round((time.monotonic() - start) * 1000, 2),
            )

        # Vector search — header columns only (no content)
        top_k = max_results * 2
        candidates = await self._header_vector_search(namespace, cue_embedding, top_k)

        # Convergence scoring on headers
        scored = self._score_headers(candidates, cue_scores)

        # Sort by activation (convergence score) and limit
        scored.sort(key=lambda h: h.activation, reverse=True)
        headers = scored[:max_results]

        # Pre-Layer 0: read latest graph snapshot for top-down hints
        graph_content = None
        try:
            from engram.engine.graph_store import GraphStore
            graph_store = GraphStore(self.db)
            graph = await graph_store.get_latest(namespace)
            if graph:
                graph_content = graph.content
        except Exception:
            pass  # graph read is non-fatal

        elapsed = round((time.monotonic() - start) * 1000, 2)
        return HeaderSearchResult(
            triggered=True,
            urgency=urgency,
            headers=headers,
            retrieval_ms=elapsed,
            graph_content=graph_content,
        )

    async def _header_vector_search(
        self, namespace: str, cue_embedding: List[float], top_k: int
    ) -> List[dict]:
        """Cosine similarity search returning header columns only (no content)."""
        vec_literal = "[" + ",".join(str(v) for v in cue_embedding) + "]"
        stmt = text(
            f"SELECT id, memory_type, dimension_scores, features, activation, "
            f"salience, access_count, created_at, last_accessed, "
            f"1 - (embedding <=> '{vec_literal}'::vector) AS cosine_sim "
            f"FROM memories "
            f"WHERE namespace = :ns AND embedding IS NOT NULL "
            f"ORDER BY embedding <=> '{vec_literal}'::vector "
            f"LIMIT :k"
        )
        result = await self.db.execute(stmt, {"ns": namespace, "k": top_k})
        rows = result.fetchall()
        return [
            {
                "id": str(r.id),
                "memory_type": r.memory_type,
                "dimension_scores": r.dimension_scores or {},
                "features": r.features or {},
                "activation": r.activation or 0.0,
                "salience": r.salience or 0.5,
                "access_count": r.access_count or 0,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "last_accessed": r.last_accessed.isoformat() if r.last_accessed else None,
                "cosine_sim": r.cosine_sim,
            }
            for r in rows
        ]

    def _score_headers(
        self, candidates: List[dict], cue_scores: Dict[str, float]
    ) -> List[MemoryHeaderResult]:
        """Score header candidates on multi-dimension convergence.

        Formula: convergence = dims_matched * avg_dim_score
        Cosine similarity removed — with multi-axis search, per-region scores
        replace the single cosine proxy.
        """
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
            convergence = len(dims_matched) * avg_dim_score

            # Extract features JSONB fields
            features = c.get("features", {})

            results.append(
                MemoryHeaderResult(
                    id=c["id"],
                    memory_type=c.get("memory_type", "episodic"),
                    activation=convergence,
                    convergence_score=round(convergence, 4),
                    retrieval_path="direct",
                    dimensions_matched=dims_matched,
                    enrichment_status=features.get("enrichment_status", "raw"),
                    vad_summary=features.get("vad_summary"),
                    topic_tags=features.get("topic_tags", []),
                    entity_ids=features.get("entity_ids", []),
                    dimension_confidence=features.get("dimension_confidence", {}),
                    salience=c.get("salience", 0.5),
                    created_at=c.get("created_at"),
                    last_accessed=c.get("last_accessed"),
                    access_count=c.get("access_count", 0),
                )
            )
        return results

    # -- Multi-axis search (6 independent pgvector queries) --

    async def _multi_axis_search(
        self,
        namespace: str,
        region_vectors: Dict[str, List[float]],
        cue_embedding: Optional[List[float]],
        top_k: int,
    ) -> List[MemoryResult]:
        """Run 6 independent pgvector queries and aggregate per-region scores.

        Each region query returns the top-k most similar memories for that axis.
        Candidates are merged. Convergence blends breadth*depth with peak score:

            convergence = alpha * dims_matched * mean_score
                        + (1 - alpha) * max_region_score

        This prevents memories matching many regions weakly from outranking
        memories matching fewer regions strongly.
        """
        candidates: Dict[str, dict] = {}

        for region, vec in region_vectors.items():
            matches = await self._region_search(namespace, vec, region, top_k=20)
            for match in matches:
                node_id = match["id"]
                if node_id not in candidates:
                    candidates[node_id] = {
                        "id": node_id,
                        "content": match["content"],
                        "memory_type": match.get("memory_type", "episodic"),
                        "region_scores": {},
                        "dimension_scores": match.get("dimension_scores", {}),
                    }
                candidates[node_id]["region_scores"][region] = match["cosine_sim"]

        # If no multi-axis results (all region columns NULL), fall back
        if not candidates and cue_embedding is not None:
            return await self._single_axis_fallback(namespace, cue_embedding, top_k)

        # Compute convergence with alpha blending
        results = []
        for cand in candidates.values():
            region_scores = cand["region_scores"]
            dims_matched = list(region_scores.keys())
            scores = list(region_scores.values())
            mean_score = sum(scores) / len(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0

            breadth_depth = len(dims_matched) * mean_score
            convergence = (
                _CONVERGENCE_ALPHA * breadth_depth
                + (1 - _CONVERGENCE_ALPHA) * max_score
            )

            # Amygdala asymmetry: high-amygdala memories get a convergence boost
            if settings.INTEGRATION_ENABLED and settings.AMYGDALA_ASYMMETRY_ENABLED:
                dim_scores = cand.get("dimension_scores", {})
                amyg_score = dim_scores.get("amygdala", dim_scores.get("amyg", 0.0))
                if amyg_score >= settings.AMYGDALA_ASYMMETRY_THRESHOLD:
                    convergence *= settings.AMYGDALA_ASYMMETRY_MULTIPLIER

            results.append(
                MemoryResult(
                    id=cand["id"],
                    content=cand["content"],
                    activation=convergence,
                    dimensions_matched=dims_matched,
                    convergence_score=round(convergence, 4),
                    retrieval_path="direct",
                )
            )
        return results

    async def _region_search(
        self,
        namespace: str,
        region_vec: List[float],
        region: str,
        top_k: int = 20,
    ) -> List[dict]:
        """Per-region pgvector cosine similarity search."""
        col_name = f"{region}_embedding"
        vec_literal = "[" + ",".join(str(v) for v in region_vec) + "]"
        stmt = text(
            f"SELECT id, content, memory_type, dimension_scores, "
            f"1 - ({col_name} <=> '{vec_literal}'::vector) AS cosine_sim "
            f"FROM memories "
            f"WHERE namespace = :ns AND {col_name} IS NOT NULL "
            f"ORDER BY {col_name} <=> '{vec_literal}'::vector "
            f"LIMIT :k"
        )
        result = await self.db.execute(stmt, {"ns": namespace, "k": top_k})
        rows = result.fetchall()
        return [
            {
                "id": str(r.id),
                "content": r.content,
                "memory_type": r.memory_type,
                "dimension_scores": (
                    r.dimension_scores if isinstance(r.dimension_scores, dict)
                    else {}
                ),
                "cosine_sim": r.cosine_sim,
            }
            for r in rows
        ]

    # -- Fallback: single-embedding search --

    async def _single_axis_fallback(
        self, namespace: str, cue_embedding: List[float], top_k: int
    ) -> List[MemoryResult]:
        """Cosine-only fallback when region decomposition is unavailable.

        Uses the single semantic embedding column. Convergence = cosine_sim only
        (no dimension scoring — we don't trust heuristic scores).
        """
        vec_literal = "[" + ",".join(str(v) for v in cue_embedding) + "]"
        stmt = text(
            f"SELECT id, content, memory_type, "
            f"1 - (embedding <=> '{vec_literal}'::vector) AS cosine_sim "
            f"FROM memories "
            f"WHERE namespace = :ns AND embedding IS NOT NULL "
            f"ORDER BY embedding <=> '{vec_literal}'::vector "
            f"LIMIT :k"
        )
        result = await self.db.execute(stmt, {"ns": namespace, "k": top_k})
        rows = result.fetchall()
        return [
            MemoryResult(
                id=str(r.id),
                content=r.content,
                activation=r.cosine_sim,
                dimensions_matched=[],
                convergence_score=round(r.cosine_sim, 4),
                retrieval_path="direct",
            )
            for r in rows
        ]

    async def _spread_activation(
        self,
        seeds: List[MemoryResult],
        namespace: str,
        hop_depth: int,
        context: Optional[dict] = None,
    ) -> tuple[List[MemoryResult], List[dict], List[dict]]:
        """Spread activation through typed edges from seed memories.

        Returns (activated_memories, suppressed_memories, traversed_edges).
        """
        if not seeds:
            return [], [], []

        # Map memory_id -> MemoryResult for all active nodes
        active: Dict[str, MemoryResult] = {m.id: m for m in seeds}
        suppressed: List[dict] = []
        traversed_edges: List[dict] = []

        # Track which nodes are direct results vs discovered via spreading
        direct_ids = {m.id for m in seeds}

        for _hop in range(hop_depth):
            current_ids = [uuid.UUID(mid) for mid in active.keys()]

            # Get all outgoing edges from current active nodes
            all_edges: List[Edge] = []
            for mid in current_ids:
                edges = await self._edge_store.get_edges(mid, "outgoing")
                all_edges.extend(edges)

            if not all_edges:
                break

            # Collect neighbor IDs we need to fetch
            neighbor_ids_to_fetch = set()
            for edge in all_edges:
                tid = str(edge.target_memory_id)
                if tid not in active:
                    neighbor_ids_to_fetch.add(edge.target_memory_id)

            # Batch-fetch neighbor memories
            neighbor_map: Dict[str, Memory] = {}
            if neighbor_ids_to_fetch:
                stmt = select(Memory).where(
                    Memory.id.in_(list(neighbor_ids_to_fetch)),
                    Memory.namespace == namespace,
                )
                result = await self.db.execute(stmt)
                for mem in result.scalars().all():
                    neighbor_map[str(mem.id)] = mem

            # Propagate activation
            for edge in all_edges:
                source_id = str(edge.source_memory_id)
                target_id = str(edge.target_memory_id)

                if source_id not in active:
                    continue

                source_activation = active[source_id].activation

                # Self-loop edges: flat additive boost, not recursive
                if source_id == target_id:
                    active[source_id].activation += edge.weight
                    traversed_edges.append({
                        "source_memory_id": source_id,
                        "target_memory_id": target_id,
                        "edge_type": edge.edge_type,
                        "weight": edge.weight,
                        "context": edge.context or {},
                    })
                    continue

                # Modulatory edges only fire if context overlaps
                if edge.edge_type == "modulatory":
                    if not self._context_overlaps(edge.context_features, context):
                        continue

                # Record edge for trace generation
                traversed_edges.append({
                    "source_memory_id": source_id,
                    "target_memory_id": target_id,
                    "edge_type": edge.edge_type,
                    "weight": edge.weight,
                    "context": edge.context or {},
                })

                # Inhibitory edges: multiplicative suppression on target
                if edge.edge_type == "inhibitory":
                    if target_id in active:
                        active[target_id].activation *= (1 - edge.weight)
                    # Don't discover new nodes via inhibition
                    continue

                # Excitatory / associative / temporal / modulatory: additive delta
                multiplier = _EDGE_MULTIPLIERS.get(edge.edge_type, 0.5)
                delta = source_activation * edge.weight * multiplier

                if target_id in active:
                    # Update existing node's activation
                    active[target_id].activation += delta
                else:
                    # Discover new node via spreading
                    mem = neighbor_map.get(target_id)
                    if mem is None:
                        continue
                    active[target_id] = MemoryResult(
                        id=target_id,
                        content=mem.content,
                        activation=delta,
                        dimensions_matched=[],
                        convergence_score=0.0,
                        retrieval_path="spreading",
                    )

        # Filter: remove memories with activation <= 0 (inhibited out)
        result_list = []
        for mid, mem_result in active.items():
            if mem_result.activation <= 0:
                suppressed.append({
                    "id": mid,
                    "content": mem_result.content,
                    "reason": "inhibited",
                    "final_activation": round(mem_result.activation, 4),
                })
            else:
                # Mark retrieval_path correctly
                if mid not in direct_ids:
                    mem_result.retrieval_path = "spreading"
                result_list.append(mem_result)

        return result_list, suppressed, traversed_edges

    def _context_overlaps(
        self, edge_context: Optional[dict], retrieval_context: Optional[dict]
    ) -> bool:
        """Check if edge context features overlap with retrieval context.

        For modulatory edges created by ModulatoryDiscoveryJob, edge_context
        contains 'matched_features' — a dict of structural feature categories
        that matched (e.g. {"action_type": "migration", "dynamic": "transition"}).
        The retrieval context can include 'features' with the cue's structural
        features. If any feature category+value matches, the gate opens.
        """
        if not edge_context or not retrieval_context:
            return False

        # Check matched_features against retrieval context features
        matched = edge_context.get("matched_features")
        ctx_features = retrieval_context.get("features")
        if matched and ctx_features:
            for key, value in matched.items():
                if ctx_features.get(key) == value:
                    return True

        # Fallback: simple key overlap check
        edge_keys = set(edge_context.keys())
        ctx_keys = set(retrieval_context.keys())
        return bool(edge_keys & ctx_keys)

    async def _reconsolidate(self, result: RetrievalResult, namespace: str) -> None:
        """Hebbian learning: strengthen edges between co-retrieved memories."""
        if len(result.memories) < 2:
            return
        try:
            memory_ids = [uuid.UUID(m.id) for m in result.memories]
            await self._edge_ops.reinforce_traversed(
                seed_ids=memory_ids[:1],
                included_ids=memory_ids[1:],
                boost=1.1,
            )
        except Exception:
            pass  # fire-and-forget — don't crash on reconsolidation failure
