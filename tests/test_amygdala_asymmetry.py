"""Tests for amygdala asymmetry boost in multi-axis retrieval."""

import pytest
from unittest.mock import AsyncMock, patch

from engram.config import settings
from engram.engine.models import MemoryResult
from engram.engine.retriever import Retriever


@pytest.mark.asyncio
async def test_amygdala_high_score_gets_boost(db):
    """Memory with amygdala=0.8 should outrank memory with amygdala=0.3."""
    retriever = Retriever(db)

    # Mock _region_search to return two candidates with different amygdala scores
    async def mock_region_search(namespace, vec, region, top_k=20):
        return [
            {
                "id": "mem-high-amyg",
                "content": "emotionally significant memory",
                "memory_type": "episodic",
                "dimension_scores": {"amygdala": 0.8, "hippo": 0.5},
                "cosine_sim": 0.7,
            },
            {
                "id": "mem-low-amyg",
                "content": "neutral factual memory",
                "memory_type": "episodic",
                "dimension_scores": {"amygdala": 0.3, "hippo": 0.5},
                "cosine_sim": 0.7,
            },
        ]

    retriever._region_search = mock_region_search

    results = await retriever._multi_axis_search(
        "test:ns", {"hippo": [0.1] * 1024}, None, top_k=10
    )

    by_id = {r.id: r for r in results}
    assert by_id["mem-high-amyg"].activation > by_id["mem-low-amyg"].activation


@pytest.mark.asyncio
async def test_amygdala_disabled_no_boost(db):
    """With AMYGDALA_ASYMMETRY_ENABLED=false, both memories should have equal activation."""
    retriever = Retriever(db)

    async def mock_region_search(namespace, vec, region, top_k=20):
        return [
            {
                "id": "mem-high-amyg",
                "content": "emotionally significant memory",
                "memory_type": "episodic",
                "dimension_scores": {"amygdala": 0.8, "hippo": 0.5},
                "cosine_sim": 0.7,
            },
            {
                "id": "mem-low-amyg",
                "content": "neutral factual memory",
                "memory_type": "episodic",
                "dimension_scores": {"amygdala": 0.3, "hippo": 0.5},
                "cosine_sim": 0.7,
            },
        ]

    retriever._region_search = mock_region_search

    with patch.object(settings, "AMYGDALA_ASYMMETRY_ENABLED", False):
        results = await retriever._multi_axis_search(
            "test:ns", {"hippo": [0.1] * 1024}, None, top_k=10
        )

    by_id = {r.id: r for r in results}
    # Without boost, both should have the same activation (same cosine_sim, same region match)
    assert by_id["mem-high-amyg"].activation == by_id["mem-low-amyg"].activation


@pytest.mark.asyncio
async def test_integration_disabled_no_boost(db):
    """With INTEGRATION_ENABLED=false, amygdala boost should not apply."""
    retriever = Retriever(db)

    async def mock_region_search(namespace, vec, region, top_k=20):
        return [
            {
                "id": "mem-high-amyg",
                "content": "emotionally significant memory",
                "memory_type": "episodic",
                "dimension_scores": {"amygdala": 0.8},
                "cosine_sim": 0.7,
            },
            {
                "id": "mem-low-amyg",
                "content": "neutral factual memory",
                "memory_type": "episodic",
                "dimension_scores": {"amygdala": 0.3},
                "cosine_sim": 0.7,
            },
        ]

    retriever._region_search = mock_region_search

    with patch.object(settings, "INTEGRATION_ENABLED", False):
        results = await retriever._multi_axis_search(
            "test:ns", {"hippo": [0.1] * 1024}, None, top_k=10
        )

    by_id = {r.id: r for r in results}
    assert by_id["mem-high-amyg"].activation == by_id["mem-low-amyg"].activation
