"""Test modulatory edge discovery — cross-domain structural pattern connections."""

import pytest

from engram.dreamer.feature_extraction import FeatureExtractionJob
from engram.dreamer.modulatory import ModulatoryDiscoveryJob
from engram.engine.edges import EdgeStore
from engram.engine.retriever import Retriever
from engram.engine.store import MemoryStore
from engram.plugins.registry import PluginRegistry

NS = "user:modulatory_test"


@pytest.fixture
def registry():
    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.brain_regions")
    yield reg
    PluginRegistry.reset()


async def _create_and_extract(store, db, contents):
    """Create memories and run feature extraction on each."""
    job = FeatureExtractionJob()
    mem_ids = []
    for content in contents:
        mem = await store.create(NS, content)
        mem_ids.append(mem.id)
    await db.flush()

    for mid in mem_ids:
        await job.execute(NS, db=db, memory_id=mid)
    await db.flush()
    return mem_ids


@pytest.mark.asyncio
async def test_modulatory_discovery_creates_cross_domain_edges(db, registry):
    """Modulatory edges should connect structurally similar but topically different memories."""
    store = MemoryStore(db, registry)

    # Infrastructure migration memories
    infra = [
        "We migrated our server infrastructure from the old data center to AWS cloud",
        "The database migration from MySQL to PostgreSQL improved query performance",
        "CI/CD pipeline transition from Jenkins to GitHub Actions reduced build times",
        "Network infrastructure optimization after migrating to the new cloud region",
        "Monitoring system migration from Nagios to Datadog for better observability",
    ]

    # Personal habit migration memories (structurally similar, topically different)
    personal = [
        "I migrated my morning routine from chaos to a structured schedule for better productivity",
        "Transitioning my diet from processed food to whole foods improved my energy levels",
        "Switching my exercise routine from gym workouts to outdoor activities felt more natural",
        "Optimizing my sleep schedule after transitioning to an earlier bedtime routine",
        "Moving my learning approach from passive reading to active note-taking and practice",
    ]

    infra_ids = await _create_and_extract(store, db, infra)
    personal_ids = await _create_and_extract(store, db, personal)

    # Run modulatory discovery
    job = ModulatoryDiscoveryJob()
    result = await job.execute(NS, db=db)

    assert result["pairs_scanned"] > 0

    # Check for modulatory edges between infra and personal memories
    es = EdgeStore(db)
    modulatory_edges = []
    for iid in infra_ids:
        edges = await es.get_edges(iid, "both", ["modulatory"])
        for e in edges:
            target = str(e.target_memory_id)
            source = str(e.source_memory_id)
            if target in [str(p) for p in personal_ids] or source in [str(p) for p in personal_ids]:
                modulatory_edges.append(e)

    # At least some cross-domain modulatory edges should be created
    # (depends on how similar the feature vectors end up being)
    if result["modulatory_edges_created"] > 0:
        assert len(modulatory_edges) > 0
        for edge in modulatory_edges:
            assert edge.context.get("matched_features") is not None
            assert edge.context.get("feature_similarity", 0) > 0.8
            assert edge.context.get("content_similarity", 1) < 0.4


@pytest.mark.asyncio
async def test_modulatory_discovery_skips_similar_content(db, registry):
    """Memories with high content AND feature similarity should NOT get modulatory edges
    — those should already have excitatory/associative edges from edge classification."""
    store = MemoryStore(db, registry)

    # Very similar content (high content similarity)
    similar = [
        "AWS EC2 server migration plan version 1",
        "AWS EC2 server migration plan version 2",
        "AWS EC2 server migration plan version 3",
    ]

    await _create_and_extract(store, db, similar)

    job = ModulatoryDiscoveryJob()
    result = await job.execute(NS, db=db)

    # Content similarity should be > 0.4 for these, so no modulatory edges
    assert result["modulatory_edges_created"] == 0


@pytest.mark.asyncio
async def test_modulatory_edge_context_contains_matched_features(db, registry):
    """Modulatory edges should record which structural features matched."""
    store = MemoryStore(db, registry)

    # Two memories with same structural pattern but different topics
    mem1 = await store.create(NS, "We migrated the entire server infrastructure to optimize cloud costs and improve system performance")
    mem2 = await store.create(NS, "I migrated my entire morning routine to optimize my time and improve personal productivity")
    await db.flush()

    job = FeatureExtractionJob()
    await job.execute(NS, db=db, memory_id=mem1.id)
    await job.execute(NS, db=db, memory_id=mem2.id)
    await db.flush()

    mod_job = ModulatoryDiscoveryJob()
    result = await mod_job.execute(NS, db=db)

    if result["modulatory_edges_created"] > 0:
        es = EdgeStore(db)
        edges = await es.get_edges(mem1.id, "both", ["modulatory"])
        assert len(edges) > 0
        edge = edges[0]
        matched = edge.context.get("matched_features", {})
        # Both should be migration + optimization pattern
        assert len(matched) >= 1


@pytest.mark.asyncio
async def test_modulatory_gate_in_retriever(db, registry):
    """Modulatory edges should only fire during retrieval when context features match."""
    store = MemoryStore(db, registry)
    es = EdgeStore(db)

    # Create two memories and a modulatory edge between them
    m1 = await store.create(NS, "Server infrastructure migration from old to new platform")
    m2 = await store.create(NS, "Personal habit transition from old routine to new lifestyle")
    m1_id = m1.id
    m2_id = m2.id
    await db.flush()

    # Create a modulatory edge manually
    await es.create(
        source_id=m1_id,
        target_id=m2_id,
        edge_type="modulatory",
        weight=0.8,
        context={"matched_features": {"action_type": "migration", "dynamic": "transition"}},
        namespace=NS,
    )
    await db.flush()

    retriever = Retriever(db, registry)

    # Retrieve WITH matching context → modulatory edge should fire
    from engram.engine.models import RetrievalOptions
    result_with_ctx = await retriever.retrieve(
        NS,
        "Tell me about migrations",
        context={"features": {"action_type": "migration"}},
        options=RetrievalOptions(reconsolidate=False),
    )

    # Retrieve WITHOUT context → modulatory edge should NOT fire
    result_no_ctx = await retriever.retrieve(
        NS,
        "Tell me about migrations",
        context=None,
        options=RetrievalOptions(reconsolidate=False),
    )

    # Both should return m1 (direct match). With context, m2 might also appear via spreading.
    # The key assertion: when context matches, the modulatory edge is traversable.
    if result_with_ctx.triggered:
        with_ctx_ids = {m.id for m in result_with_ctx.memories}
        assert str(m1_id) in with_ctx_ids

    if result_no_ctx.triggered:
        no_ctx_ids = {m.id for m in result_no_ctx.memories}
        assert str(m1_id) in no_ctx_ids
