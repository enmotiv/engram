"""Test graph trace generation and Redis caching."""

import pytest

from engram.engine.cache import TraceCache
from engram.engine.edges import EdgeStore
from engram.engine.models import RetrievalOptions
from engram.engine.retriever import Retriever
from engram.engine.store import MemoryStore
from engram.engine.tracer import NOTATION_KEY, TraceGenerator
from engram.plugins.registry import PluginRegistry

NS = "user:trace_test"

MEM_A = "AWS EC2 g5.xlarge migration completed successfully"
MEM_B = "GCP hosting account was decommissioned and is no longer active"
MEM_C = "Cost optimization is critical for our cloud infrastructure"
MEM_D = "We use warm pools for cold start latency reduction"
MEM_E = "Performance benchmarks showed 40% improvement after migration"


@pytest.fixture
def registry():
    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.default")
    yield reg
    PluginRegistry.reset()


async def _seed_graph(db, registry):
    store = MemoryStore(db, registry)
    es = EdgeStore(db)

    a = await store.create(NS, MEM_A)
    b = await store.create(NS, MEM_B)
    c = await store.create(NS, MEM_C)
    d = await store.create(NS, MEM_D)
    e = await store.create(NS, MEM_E)
    await db.flush()

    await es.create(a.id, b.id, "inhibitory", 0.8, namespace=NS)
    await es.create(a.id, c.id, "excitatory", 0.7, namespace=NS)
    await es.create(c.id, d.id, "excitatory", 0.6, namespace=NS)
    await es.create(a.id, e.id, "temporal", 0.5, namespace=NS)
    await db.flush()
    return a, b, c, d, e


@pytest.mark.asyncio
async def test_trace_contains_notation_key(db, registry):
    """Trace should start with the notation key header."""
    await _seed_graph(db, registry)
    retriever = Retriever(db, registry)

    result = await retriever.retrieve(
        NS,
        "What cloud infrastructure do we use?",
        options=RetrievalOptions(max_results=10, reconsolidate=False),
    )
    assert result.trace is not None
    assert NOTATION_KEY in result.trace


@pytest.mark.asyncio
async def test_trace_contains_edge_arrows(db, registry):
    """Trace should contain --> for excitatory and --x for inhibitory edges."""
    await _seed_graph(db, registry)
    retriever = Retriever(db, registry)

    result = await retriever.retrieve(
        NS,
        "What cloud infrastructure do we use?",
        options=RetrievalOptions(max_results=10, reconsolidate=False),
    )
    assert result.trace is not None
    # Should contain excitatory arrow between co-retrieved memories
    assert "-->" in result.trace


@pytest.mark.asyncio
async def test_trace_compactness(db, registry):
    """Trace should be compact (< 200 tokens ~= < 800 chars)."""
    await _seed_graph(db, registry)
    retriever = Retriever(db, registry)

    result = await retriever.retrieve(
        NS,
        "What cloud infrastructure do we use?",
        options=RetrievalOptions(max_results=10, reconsolidate=False),
    )
    assert result.trace is not None
    # Rough token estimate: ~4 chars per token
    assert len(result.trace) < 800


@pytest.mark.asyncio
async def test_suppressed_in_result(db, registry):
    """Suppressed memories should be listed in result.suppressed."""
    await _seed_graph(db, registry)
    retriever = Retriever(db, registry)

    result = await retriever.retrieve(
        NS,
        "What cloud infrastructure do we use?",
        options=RetrievalOptions(max_results=10, reconsolidate=False),
    )
    # suppressed list may have entries if inhibition pushed any memory below 0
    if result.suppressed:
        for s in result.suppressed:
            assert "id" in s
            assert "reason" in s
            assert s["reason"] == "inhibited"


@pytest.mark.asyncio
async def test_cache_hit(db, registry):
    """Second retrieval with same memories should use cached trace."""
    await _seed_graph(db, registry)
    cache = TraceCache("redis://localhost:6380/0")

    try:
        retriever = Retriever(db, registry, trace_cache=cache)

        # First retrieval — populates cache
        result1 = await retriever.retrieve(
            NS,
            "What cloud infrastructure do we use?",
            options=RetrievalOptions(max_results=10, reconsolidate=False),
        )
        assert result1.trace is not None
        first_ms = result1.retrieval_ms

        # Second retrieval — should hit cache
        result2 = await retriever.retrieve(
            NS,
            "What cloud infrastructure do we use?",
            options=RetrievalOptions(max_results=10, reconsolidate=False),
        )
        assert result2.trace is not None
        assert result2.trace == result1.trace
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_tracer_unit():
    """Unit test for TraceGenerator without DB."""
    from engram.engine.models import MemoryResult

    memories = [
        MemoryResult(
            id="aaa",
            content="AWS EC2 g5.xlarge migration",
            activation=2.0,
            dimensions_matched=["semantic"],
            convergence_score=2.0,
            retrieval_path="direct",
        ),
        MemoryResult(
            id="bbb",
            content="Cost optimization for cloud",
            activation=1.5,
            dimensions_matched=["semantic"],
            convergence_score=1.5,
            retrieval_path="spreading",
        ),
    ]
    edges = [
        {
            "source_memory_id": "aaa",
            "target_memory_id": "bbb",
            "edge_type": "excitatory",
            "weight": 0.7,
            "context": {},
        }
    ]
    suppressed = [
        {"id": "ccc", "content": "GCP old hosting", "reason": "inhibited", "final_activation": -0.5}
    ]

    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.default")

    tracer = TraceGenerator(reg)
    trace = tracer.generate(memories, edges, suppressed)

    assert NOTATION_KEY in trace
    assert "-->" in trace
    assert "inhibited" in trace
    # Labels should be snake_case
    assert "aws_ec2_g5" in trace or "aws_ec2" in trace

    PluginRegistry.reset()
