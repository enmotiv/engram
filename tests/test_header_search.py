"""Test Layer 0 header search — returns headers without content."""

import pytest

from engram.engine.models import HeaderSearchResult
from engram.engine.retriever import Retriever
from engram.engine.store import MemoryStore
from engram.plugins.registry import PluginRegistry

NS = "user:header_search_test"

MEMORIES = [
    "We migrated our production database to AWS RDS last Tuesday",
    "The pasta carbonara recipe uses guanciale not bacon for authentic flavor",
    "Sprint retrospective revealed we need better estimation practices",
    "Morning runs at 6am have improved my focus throughout the day",
    "Progressive overload with 5% weekly weight increase for squat gains",
]


@pytest.fixture
def registry():
    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.default")
    yield reg
    PluginRegistry.reset()


async def _seed(db, registry):
    store = MemoryStore(db, registry)
    for content in MEMORIES:
        await store.create(NS, content, metadata={"enrichment_status": "enriched"})
    await db.flush()


@pytest.mark.asyncio
async def test_header_search_returns_headers(db, registry):
    """search_headers returns MemoryHeaderResult objects without content."""
    await _seed(db, registry)
    retriever = Retriever(db, registry)

    result = await retriever.search_headers(
        namespace=NS,
        cue="database migration",
        max_results=10,
        urgency_threshold=0.0,  # bypass urgency gate
    )

    assert isinstance(result, HeaderSearchResult)
    assert result.triggered is True
    assert len(result.headers) > 0

    # Headers should NOT have content
    for header in result.headers:
        assert header.id  # has ID
        assert not hasattr(header, "content") or "content" not in header.model_fields
        assert header.memory_type  # has type
        assert isinstance(header.convergence_score, float)
        assert isinstance(header.salience, float)


@pytest.mark.asyncio
async def test_header_search_extracts_features(db, registry):
    """search_headers extracts enrichment_status from features JSONB."""
    await _seed(db, registry)
    retriever = Retriever(db, registry)

    result = await retriever.search_headers(
        namespace=NS,
        cue="exercise fitness",
        max_results=5,
        urgency_threshold=0.0,
    )

    assert result.triggered
    for header in result.headers:
        assert header.enrichment_status == "enriched"


@pytest.mark.asyncio
async def test_header_search_urgency_gating(db, registry):
    """Low urgency cue should not trigger search."""
    await _seed(db, registry)
    retriever = Retriever(db, registry)

    result = await retriever.search_headers(
        namespace=NS,
        cue="hi",
        urgency_threshold=0.9,  # very high threshold
    )

    assert result.triggered is False
    assert len(result.headers) == 0


@pytest.mark.asyncio
async def test_header_search_empty_namespace(db, registry):
    """Search on empty namespace returns empty results."""
    retriever = Retriever(db, registry)

    result = await retriever.search_headers(
        namespace="user:empty_ns",
        cue="anything",
        urgency_threshold=0.0,
    )

    assert result.triggered is True
    assert len(result.headers) == 0
