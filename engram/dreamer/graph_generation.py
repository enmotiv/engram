"""Default OSS graph generator — produces mermaid snapshots from memories and edges.

Runs as a Dreamer WorkerJob triggered every N new memories per namespace.
No LLM calls — pure graph operations:
1. Load memories (id, content[:50], dimension_scores, created_at) + edges
2. Score relevance: edge connectivity x recency x dimension strength
3. Cluster: groups of 3+ with mutual excitatory edges + same top dimension
4. Prune: remove below-threshold + orphans (unless recent). Target 30-80 nodes
5. Assign subgraphs by top-scoring dimension
6. Render mermaid (target < 500 tokens)
7. Store via GraphStore
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from engram.db.models import Edge, Graph, Memory
from engram.engine.graph_store import GraphStore
from engram.engine.interfaces import GraphGenerator, WorkerJob

logger = logging.getLogger(__name__)

# Trigger: run when N new memories exist since last graph snapshot
NEW_MEMORY_THRESHOLD = 15
TARGET_NODE_RANGE = (30, 80)
MAX_MERMAID_CHARS = 2000  # rough proxy for 500 tokens


class DefaultGraphGenerator(GraphGenerator):
    """Generates mermaid graph snapshots from memories and edges."""

    async def generate(
        self, namespace: str, memories: List, edges: List
    ) -> dict:
        """Generate a graph snapshot from memory/edge data."""
        if not memories:
            return {
                "content": "graph LR\n  empty[no memories]",
                "node_count": 0,
                "cluster_count": 0,
                "memory_count": 0,
                "metadata": {},
            }

        # Score nodes
        edge_counts = defaultdict(int)
        edge_map: Dict[str, set] = defaultdict(set)
        for edge in edges:
            src = edge.get("source_id") or str(getattr(edge, "source_memory_id", ""))
            tgt = edge.get("target_id") or str(getattr(edge, "target_memory_id", ""))
            etype = edge.get("edge_type") or getattr(edge, "edge_type", "associative")
            if etype == "excitatory":
                edge_counts[src] += 1
                edge_counts[tgt] += 1
                edge_map[src].add(tgt)
                edge_map[tgt].add(src)

        # Build node list with scores
        nodes: List[Dict] = []
        for mem in memories:
            mid = mem.get("id") or str(getattr(mem, "id", ""))
            dim_scores = mem.get("dimension_scores") or getattr(mem, "dimension_scores", {}) or {}
            content = (mem.get("content") or getattr(mem, "content", ""))[:50]
            created = mem.get("created_at") or getattr(mem, "created_at", None)

            # Score = connectivity x dimension strength x recency
            connectivity = edge_counts.get(mid, 0)
            max_dim = max(dim_scores.values()) if dim_scores else 0.0
            top_dim = max(dim_scores, key=dim_scores.get) if dim_scores else "unknown"

            recency = 0.5
            if created:
                now = datetime.now(timezone.utc)
                if isinstance(created, str):
                    try:
                        created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        created = None
                if created:
                    if hasattr(created, "tzinfo") and created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    age_days = (now - created).days
                    recency = max(0.1, 1.0 - age_days / 365.0)

            score = (connectivity + 1) * max_dim * recency
            label = self._to_label(content)

            nodes.append({
                "id": mid,
                "label": label,
                "score": score,
                "top_dim": top_dim,
                "connectivity": connectivity,
            })

        # Sort by score, take top N
        nodes.sort(key=lambda n: n["score"], reverse=True)
        target_max = TARGET_NODE_RANGE[1]
        nodes = nodes[:target_max]

        # Cluster by top dimension
        clusters: Dict[str, List[Dict]] = defaultdict(list)
        for node in nodes:
            clusters[node["top_dim"]].append(node)

        # Render mermaid
        mermaid = self._render_mermaid(clusters, edge_map)

        # Trim if too long
        if len(mermaid) > MAX_MERMAID_CHARS:
            # Re-render with fewer nodes
            nodes = nodes[:TARGET_NODE_RANGE[0]]
            clusters = defaultdict(list)
            for node in nodes:
                clusters[node["top_dim"]].append(node)
            mermaid = self._render_mermaid(clusters, edge_map)

        return {
            "content": mermaid,
            "node_count": sum(len(c) for c in clusters.values()),
            "cluster_count": len(clusters),
            "memory_count": len(memories),
            "metadata": {
                "dimensions": list(clusters.keys()),
                "pruned_from": len(memories),
            },
        }

    def _render_mermaid(
        self,
        clusters: Dict[str, List[Dict]],
        edge_map: Dict[str, set],
    ) -> str:
        """Render mermaid graph from clusters."""
        lines = ["graph LR"]
        node_ids = set()

        for dim, nodes in clusters.items():
            dim_label = dim.replace("_", " ").title()
            lines.append(f"  subgraph {dim_label}")
            for node in nodes:
                short_id = node["id"][:8]
                lines.append(f"    {short_id}[{node['label']}]")
                node_ids.add(node["id"])
            lines.append("  end")

        # Add edges between rendered nodes
        rendered_edges = set()
        for node_id in node_ids:
            short_src = node_id[:8]
            for tgt_id in edge_map.get(node_id, set()):
                if tgt_id in node_ids:
                    short_tgt = tgt_id[:8]
                    edge_key = tuple(sorted([short_src, short_tgt]))
                    if edge_key not in rendered_edges:
                        rendered_edges.add(edge_key)
                        lines.append(f"  {short_src} --> {short_tgt}")

        return "\n".join(lines)

    @staticmethod
    def _to_label(content: str) -> str:
        """Convert content snippet to a short label."""
        words = content.split()[:5]
        label = " ".join(words)
        # Escape mermaid special chars
        label = label.replace("[", "(").replace("]", ")").replace('"', "'")
        return label


class GraphGenerationJob(WorkerJob):
    """Dreamer job that generates graph snapshots when new memories accumulate."""

    def __init__(self, generator: Optional[GraphGenerator] = None):
        self._generator = generator or DefaultGraphGenerator()

    def name(self) -> str:
        return "graph_generation"

    async def should_run(self, namespace: str, **kwargs) -> bool:
        """Run when N new memories exist since last graph snapshot."""
        db: AsyncSession = kwargs.get("db")
        if db is None:
            return False

        graph_store = GraphStore(db)
        latest = await graph_store.get_latest(namespace)

        # Count memories since last snapshot
        stmt = select(func.count()).select_from(Memory).where(
            Memory.namespace == namespace
        )
        if latest:
            stmt = stmt.where(Memory.created_at > latest.created_at)

        result = await db.execute(stmt)
        new_count = result.scalar() or 0

        return new_count >= NEW_MEMORY_THRESHOLD

    async def execute(self, namespace: str, **kwargs) -> dict:
        """Generate and store a graph snapshot."""
        db: AsyncSession = kwargs.get("db")
        if db is None:
            return {"status": "skipped", "reason": "no db session"}

        # Load memories
        stmt = (
            select(Memory)
            .where(Memory.namespace == namespace)
            .where(Memory.embedding.isnot(None))
            .order_by(Memory.created_at.desc())
            .limit(200)
        )
        result = await db.execute(stmt)
        memories = result.scalars().all()

        if not memories:
            return {"status": "skipped", "reason": "no memories"}

        memory_dicts = [m.to_dict() for m in memories]

        # Load edges
        memory_ids = [m.id for m in memories]
        edge_stmt = select(Edge).where(
            (Edge.source_memory_id.in_(memory_ids))
            | (Edge.target_memory_id.in_(memory_ids))
        )
        edge_result = await db.execute(edge_stmt)
        edges = [e.to_dict() for e in edge_result.scalars().all()]

        # Generate
        snapshot_data = await self._generator.generate(namespace, memory_dicts, edges)

        # Store
        graph_store = GraphStore(db)
        graph = await graph_store.store(
            namespace=namespace,
            content=snapshot_data["content"],
            node_count=snapshot_data["node_count"],
            cluster_count=snapshot_data["cluster_count"],
            memory_count=snapshot_data["memory_count"],
            metadata=snapshot_data.get("metadata"),
        )

        return {
            "status": "completed",
            "graph_id": str(graph.id),
            "node_count": snapshot_data["node_count"],
            "cluster_count": snapshot_data["cluster_count"],
        }
