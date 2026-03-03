"""Test batch enrichment endpoint — updates features for multiple memories."""

import uuid

import pytest

from engram.engine.store import MemoryStore
from engram.plugins.registry import PluginRegistry

NS = "user:batch_enrich_test"


@pytest.fixture
def registry():
    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.default")
    yield reg
    PluginRegistry.reset()


@pytest.mark.asyncio
async def test_batch_update_features(db, registry):
    """Batch update merges features into multiple memories."""
    store = MemoryStore(db, registry)
    m1 = await store.create(NS, "Memory one", metadata={"enrichment_status": "raw"})
    m2 = await store.create(NS, "Memory two", metadata={"enrichment_status": "raw"})
    await db.flush()

    # Update both memories with enrichment data
    result1 = await store.update(
        m1.id,
        features={"enrichment_status": "enriched", "topic_tags": ["cooking"]},
    )
    result2 = await store.update(
        m2.id,
        features={"enrichment_status": "enriched", "topic_tags": ["fitness"]},
    )
    await db.flush()

    assert result1 is not None
    assert result1.features["enrichment_status"] == "enriched"
    assert result1.features["topic_tags"] == ["cooking"]

    assert result2 is not None
    assert result2.features["enrichment_status"] == "enriched"
    assert result2.features["topic_tags"] == ["fitness"]


@pytest.mark.asyncio
async def test_batch_update_partial_failure(db, registry):
    """Update with non-existent ID returns None."""
    store = MemoryStore(db, registry)
    m1 = await store.create(NS, "Existing memory")
    await db.flush()

    # Update existing memory
    result = await store.update(
        m1.id,
        features={"enrichment_status": "enriched"},
    )
    assert result is not None

    # Update non-existent memory
    result = await store.update(
        uuid.uuid4(),
        features={"enrichment_status": "enriched"},
    )
    assert result is None


@pytest.mark.asyncio
async def test_batch_update_preserves_existing(db, registry):
    """Update merges into existing features without overwriting."""
    store = MemoryStore(db, registry)
    m1 = await store.create(
        NS, "Memory with data",
        metadata={"enrichment_status": "raw", "source_type": "observation"},
    )
    await db.flush()

    result = await store.update(
        m1.id,
        features={"enrichment_status": "enriched", "new_field": "value"},
    )
    await db.flush()

    assert result is not None
    # New fields added
    assert result.features["enrichment_status"] == "enriched"
    assert result.features["new_field"] == "value"
    # Existing fields preserved
    assert result.features["source_type"] == "observation"
