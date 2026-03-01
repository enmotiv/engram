"""Graph trace generation — compact notation for LLM system prompt injection."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from engram.engine.models import MemoryResult
from engram.plugins.registry import PluginRegistry

# Arrow notation per edge type
_EDGE_ARROWS = {
    "excitatory": "-->",
    "inhibitory": "--x",
    "associative": "---",
    "temporal": "==>",
    "modulatory": "~~>",
}

NOTATION_KEY = "--> reinforces | --x supersedes | ~~> pattern link | --- related | ==> sequential"

# Common stopwords to skip when generating labels
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "was", "were", "are", "be", "been",
    "has", "had", "have", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can",
    "for", "and", "but", "or", "nor", "not", "no",
    "in", "on", "at", "to", "of", "by", "from", "with",
    "that", "this", "it", "its", "our", "we", "us",
    "very", "just", "also", "so", "than", "then",
})


class TraceGenerator:
    """Generate compact graph notation from retrieved memories and edges."""

    def __init__(self, registry: Optional[PluginRegistry] = None):
        self._registry = registry or PluginRegistry.get_instance()

    def generate(
        self,
        memories: List[MemoryResult],
        edges: List[dict],
        suppressed: Optional[List[dict]] = None,
    ) -> str:
        if not memories:
            return ""

        # Build memory_id → label mapping
        labels: Dict[str, str] = {}
        for mem in memories:
            labels[mem.id] = self._make_label(mem)

        # Include suppressed memories in labels
        if suppressed:
            for s in suppressed:
                sid = s["id"]
                if sid not in labels:
                    labels[sid] = self._make_suppressed_label(s)

        # Build edge lines
        lines = [NOTATION_KEY]

        # Add node declarations
        for mem in memories:
            label = labels[mem.id]
            annotation = self._annotate(mem)
            lines.append(f"{label}{annotation}")

        # Add suppressed nodes
        if suppressed:
            for s in suppressed:
                label = labels[s["id"]]
                reason = s.get("reason", "inhibited")
                lines.append(f"{label}[{reason}]")

        # Add edges between retrieved/suppressed memories
        mem_ids = {m.id for m in memories}
        if suppressed:
            mem_ids |= {s["id"] for s in suppressed}

        for edge in edges:
            src = str(edge.get("source_memory_id", ""))
            tgt = str(edge.get("target_memory_id", ""))
            etype = edge.get("edge_type", "associative")

            if src in labels and tgt in labels:
                arrow = _EDGE_ARROWS.get(etype, "---")
                src_label = labels[src]
                tgt_label = labels[tgt]

                if etype == "modulatory":
                    ctx = edge.get("context", {})
                    ctx_str = ",".join(ctx.keys()) if ctx else ""
                    lines.append(f"{src_label} {arrow}({ctx_str}) {tgt_label}")
                else:
                    lines.append(f"{src_label} {arrow} {tgt_label}")

        trace = "\n".join(lines)

        # Apply plugin enrichment if available
        enricher = self._registry.get_trace_enricher()
        if enricher:
            trace = enricher.enrich(trace, memories, edges)

        return trace

    def _make_label(self, mem: MemoryResult) -> str:
        """Create a 2-3 word snake_case label from content."""
        words = re.findall(r"[A-Za-z0-9]+", mem.content)
        significant = [w.lower() for w in words if w.lower() not in _STOPWORDS]
        label_words = significant[:3] if significant else ["memory"]
        return "_".join(label_words)

    def _make_suppressed_label(self, suppressed: dict) -> str:
        """Create label for a suppressed memory."""
        content = suppressed.get("content", "unknown")
        words = re.findall(r"[A-Za-z0-9]+", content)
        significant = [w.lower() for w in words if w.lower() not in _STOPWORDS]
        label_words = significant[:3] if significant else ["suppressed"]
        return "_".join(label_words)

    def _annotate(self, mem: MemoryResult) -> str:
        """Add bracket annotations to a memory label."""
        parts = []
        if mem.retrieval_path != "direct":
            parts.append(mem.retrieval_path)
        if mem.dimensions_matched:
            parts.append(mem.dimensions_matched[0])
        if not parts:
            parts.append("direct")
        return "[" + ", ".join(parts) + "]"
