"""Test MemoryStore CRUD with automatic embedding."""

import pytest

from engram.db.models import Memory
from engram.engine.store import MemoryStore
from engram.plugins.registry import PluginRegistry

NAMESPACE = "user:test_store"


@pytest.fixture
def registry():
    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.default")
    yield reg
    PluginRegistry.reset()


@pytest.fixture
def store(db, registry):
    return MemoryStore(db, registry)


SAMPLE_CONTENTS = [
    "We had a team meeting about Q3 roadmap priorities",
    "Decided to migrate from MongoDB to PostgreSQL for better consistency",
    "Feeling anxious about the upcoming product launch deadline",
    "The server latency improved from 200ms to 45ms after the cache layer",
    "Need to exercise more — signed up for a morning run club",
]


@pytest.mark.asyncio
async def test_create_memory_with_embedding(store):
    mem = await store.create(NAMESPACE, SAMPLE_CONTENTS[0])
    assert mem.id is not None
    assert mem.namespace == NAMESPACE
    assert mem.content == SAMPLE_CONTENTS[0]
    assert mem.embedding is not None
    assert len(list(mem.embedding)) == 384
    assert "semantic" in mem.dimension_scores


@pytest.mark.asyncio
async def test_create_five_memories(store):
    mems = []
    for content in SAMPLE_CONTENTS:
        mem = await store.create(NAMESPACE, content)
        mems.append(mem)

    assert len(mems) == 5
    for mem in mems:
        assert mem.embedding is not None
        assert len(list(mem.embedding)) == 384
        assert set(mem.dimension_scores.keys()) == {"semantic", "temporal", "importance"}


@pytest.mark.asyncio
async def test_list_memories(store):
    for content in SAMPLE_CONTENTS:
        await store.create(NAMESPACE, content)

    results = await store.list(NAMESPACE)
    assert len(results) == 5


@pytest.mark.asyncio
async def test_get_by_id(store):
    mem = await store.create(NAMESPACE, "specific memory for get test")
    fetched = await store.get(mem.id)
    assert fetched is not None
    assert fetched.content == "specific memory for get test"


@pytest.mark.asyncio
async def test_delete(store):
    mem = await store.create(NAMESPACE, "to be deleted")
    assert await store.delete(mem.id) is True
    assert await store.get(mem.id) is None


@pytest.mark.asyncio
async def test_delete_nonexistent(store):
    import uuid
    assert await store.delete(uuid.uuid4()) is False


@pytest.mark.asyncio
async def test_batch_create(store):
    items = [
        {"namespace": NAMESPACE, "content": c}
        for c in SAMPLE_CONTENTS
    ]
    mems = await store.batch_create(items)
    assert len(mems) == 5
    for mem in mems:
        assert mem.embedding is not None


@pytest.mark.asyncio
async def test_list_respects_namespace(store):
    await store.create("user:other_ns", "wrong namespace")
    await store.create(NAMESPACE, "right namespace")
    results = await store.list(NAMESPACE)
    assert all(m.namespace == NAMESPACE for m in results)
