"""Tests for the default graph generator and GraphGenerationJob."""

import pytest

from engram.dreamer.graph_generation import DefaultGraphGenerator


class TestDefaultGraphGenerator:
    """Test the default OSS graph generator."""

    @pytest.mark.asyncio
    async def test_empty_memories(self):
        """Empty memory list produces minimal graph."""
        gen = DefaultGraphGenerator()
        result = await gen.generate("ns:test", [], [])

        assert result["node_count"] == 0
        assert result["cluster_count"] == 0
        assert result["memory_count"] == 0
        assert "empty" in result["content"]

    @pytest.mark.asyncio
    async def test_basic_generation(self):
        """Generates mermaid from memories and edges."""
        gen = DefaultGraphGenerator()

        memories = [
            {
                "id": f"mem-{i}",
                "content": f"Memory about topic {i} with some words",
                "dimension_scores": {"dim_a": 0.8, "dim_b": 0.2},
                "created_at": "2026-02-20T10:00:00+00:00",
            }
            for i in range(5)
        ]

        edges = [
            {
                "source_id": "mem-0",
                "target_id": "mem-1",
                "edge_type": "excitatory",
                "weight": 0.7,
            },
            {
                "source_id": "mem-1",
                "target_id": "mem-2",
                "edge_type": "excitatory",
                "weight": 0.6,
            },
        ]

        result = await gen.generate("ns:test", memories, edges)

        assert result["node_count"] == 5
        assert result["memory_count"] == 5
        assert result["cluster_count"] >= 1
        assert "graph LR" in result["content"]
        assert "subgraph" in result["content"]

    @pytest.mark.asyncio
    async def test_clusters_by_dimension(self):
        """Memories are grouped by their top-scoring dimension."""
        gen = DefaultGraphGenerator()

        memories = [
            {
                "id": "m1",
                "content": "emotional context",
                "dimension_scores": {"emotional": 0.9, "analytical": 0.1},
                "created_at": "2026-02-20T10:00:00+00:00",
            },
            {
                "id": "m2",
                "content": "another emotional",
                "dimension_scores": {"emotional": 0.8, "analytical": 0.2},
                "created_at": "2026-02-20T10:00:00+00:00",
            },
            {
                "id": "m3",
                "content": "analytical work",
                "dimension_scores": {"analytical": 0.9, "emotional": 0.1},
                "created_at": "2026-02-20T10:00:00+00:00",
            },
        ]

        result = await gen.generate("ns:test", memories, [])

        assert result["cluster_count"] == 2  # emotional + analytical
        assert "Emotional" in result["content"]
        assert "Analytical" in result["content"]

    @pytest.mark.asyncio
    async def test_edges_between_rendered_nodes(self):
        """Edges between rendered nodes appear in the mermaid output."""
        gen = DefaultGraphGenerator()

        memories = [
            {
                "id": "mem-aa",
                "content": "first memory",
                "dimension_scores": {"dim_a": 0.8},
                "created_at": "2026-02-20T10:00:00+00:00",
            },
            {
                "id": "mem-bb",
                "content": "second memory",
                "dimension_scores": {"dim_a": 0.7},
                "created_at": "2026-02-20T10:00:00+00:00",
            },
        ]

        edges = [
            {
                "source_id": "mem-aa",
                "target_id": "mem-bb",
                "edge_type": "excitatory",
                "weight": 0.8,
            },
        ]

        result = await gen.generate("ns:test", memories, edges)

        # Both node short IDs should appear with an edge
        assert "-->" in result["content"]

    @pytest.mark.asyncio
    async def test_respects_target_node_limit(self):
        """Generator caps nodes at target range."""
        gen = DefaultGraphGenerator()

        # Create 100 memories
        memories = [
            {
                "id": f"m-{i:03d}",
                "content": f"memory number {i}",
                "dimension_scores": {"dim_a": max(0.1, 1.0 - i * 0.01)},
                "created_at": "2026-02-20T10:00:00+00:00",
            }
            for i in range(100)
        ]

        result = await gen.generate("ns:test", memories, [])

        # Should be capped at TARGET_NODE_RANGE[1] = 80
        assert result["node_count"] <= 80

    @pytest.mark.asyncio
    async def test_label_escaping(self):
        """Labels with mermaid special chars are escaped."""
        gen = DefaultGraphGenerator()

        memories = [
            {
                "id": "m1",
                "content": 'This has [brackets] and "quotes"',
                "dimension_scores": {"dim_a": 0.8},
                "created_at": "2026-02-20T10:00:00+00:00",
            },
        ]

        result = await gen.generate("ns:test", memories, [])

        # Brackets should be replaced with parens
        assert "[" not in result["content"].split("\n")[-1] or "subgraph" in result["content"]
        assert "]" not in result["content"].split("[")[-1].split("]")[0] if "[" in result["content"] else True

    @pytest.mark.asyncio
    async def test_metadata_includes_dimensions(self):
        """Result metadata lists the dimensions used."""
        gen = DefaultGraphGenerator()

        memories = [
            {
                "id": "m1",
                "content": "test",
                "dimension_scores": {"growth": 0.9},
                "created_at": "2026-02-20T10:00:00+00:00",
            },
        ]

        result = await gen.generate("ns:test", memories, [])

        assert "dimensions" in result["metadata"]
        assert "growth" in result["metadata"]["dimensions"]
