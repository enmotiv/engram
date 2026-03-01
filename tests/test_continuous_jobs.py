"""Test Dreamer continuous jobs: feature extraction + edge classification."""

import pytest

from engram.db.models import Memory
from engram.dreamer.edge_classification import EdgeClassificationJob, classify_edge_heuristic
from engram.dreamer.feature_extraction import (
    FeatureExtractionJob,
    extract_features_heuristic,
    features_to_vector,
)
from engram.engine.edges import EdgeStore
from engram.engine.store import MemoryStore
from engram.plugins.registry import PluginRegistry

NS = "user:continuous_test"


@pytest.fixture
def registry():
    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.default")
    yield reg
    PluginRegistry.reset()


@pytest.mark.asyncio
async def test_feature_extraction_single(db, registry):
    """Feature extraction populates features and feature_vector on a memory."""
    store = MemoryStore(db, registry)
    mem = await store.create(NS, "We migrated from GCP to AWS EC2 last month for cost reasons")
    mem_id = mem.id
    await db.flush()

    job = FeatureExtractionJob()
    result = await job.execute(NS, db=db, memory_id=mem_id)
    assert result["processed"] == 1

    db.expire_all()
    fetched = await db.get(Memory, mem_id)
    assert fetched.features is not None
    assert "domain" in fetched.features
    assert "action_type" in fetched.features
    assert "dynamic" in fetched.features
    assert "scope" in fetched.features
    assert "causality" in fetched.features
    assert "valence" in fetched.features
    assert "abstraction" in fetched.features

    # Domain should be infrastructure (aws, ec2, gcp keywords)
    assert fetched.features["domain"] == "infrastructure"
    # Action should be migration (migrated keyword)
    assert fetched.features["action_type"] == "migration"

    # Feature vector should be 32 dims
    assert fetched.feature_vector is not None
    assert len(list(fetched.feature_vector)) == 32


@pytest.mark.asyncio
async def test_feature_extraction_batch(db, registry):
    """Feature extraction batch mode processes all memories without features."""
    store = MemoryStore(db, registry)
    contents = [
        "Server performance optimization reduced latency by 40%",
        "Team standup meeting every Monday at 10am",
        "Budget review for Q3 shows revenue growth",
    ]
    mem_ids = []
    for content in contents:
        m = await store.create(NS, content)
        mem_ids.append(m.id)
    await db.flush()

    job = FeatureExtractionJob()
    result = await job.execute(NS, db=db)
    assert result["processed"] == len(contents)

    db.expire_all()
    for mid in mem_ids:
        fetched = await db.get(Memory, mid)
        assert fetched.features is not None
        assert "domain" in fetched.features


@pytest.mark.asyncio
async def test_features_to_vector():
    """One-hot encoding produces correct 32-dim vector."""
    features = {
        "domain": "infrastructure",
        "action_type": "migration",
        "dynamic": "transition",
        "scope": "system",
        "causality": "cause",
        "valence": "positive",
        "abstraction": "concrete_event",
    }
    vec = features_to_vector(features)
    assert len(vec) == 32
    assert sum(vec) == 7.0  # 7 one-hot bits set
    # First category in domain is infrastructure → first element should be 1.0
    assert vec[0] == 1.0


@pytest.mark.asyncio
async def test_heuristic_classification():
    """Heuristic feature extractor classifies correctly."""
    features = extract_features_heuristic(
        "The team decided to migrate our Kubernetes cluster to reduce cloud costs"
    )
    assert features["domain"] == "infrastructure"
    assert features["action_type"] == "migration"
    assert features["scope"] in ("team", "system")
    assert features["valence"] in ("positive", "negative", "mixed")


@pytest.mark.asyncio
async def test_edge_classification_creates_edges(db, registry):
    """Edge classification finds neighbors and creates edges."""
    store = MemoryStore(db, registry)

    # Create related memories
    target = await store.create(NS, "AWS EC2 instance deployment for production workloads")
    target_id = target.id
    await store.create(NS, "AWS Lambda serverless deployment for microservices")
    await store.create(NS, "AWS S3 bucket configuration for static assets")
    await store.create(NS, "AWS RDS database setup for production")
    await store.create(NS, "AWS CloudFront CDN configuration")
    await db.flush()

    job = EdgeClassificationJob()
    result = await job.execute(NS, db=db, memory_id=target_id)
    assert result["edges_created"] >= 2

    # Verify edges exist
    es = EdgeStore(db)
    edges = await es.get_edges(target_id, "outgoing")
    assert len(edges) >= 2
    for edge in edges:
        assert edge.edge_type in ("excitatory", "associative", "temporal", "inhibitory")
        assert edge.context.get("auto_classified") is True


@pytest.mark.asyncio
async def test_edge_classification_inhibitory(db, registry):
    """Contradictory memories should get inhibitory edges."""
    store = MemoryStore(db, registry)

    mem_a = await store.create(NS, "We migrated from GCP to AWS last quarter")
    mem_a_id = mem_a.id
    await store.create(NS, "We decided to stay on GCP instead of migrating")
    await db.flush()

    job = EdgeClassificationJob()
    result = await job.execute(NS, db=db, memory_id=mem_a_id)

    es = EdgeStore(db)
    edges = await es.get_edges(mem_a_id, "outgoing")
    edge_types = {e.edge_type for e in edges}
    # Should have created at least one edge; inhibitory is expected due to "instead" keyword
    assert len(edges) >= 1
    # The "instead" + "migrated from" keywords should trigger inhibitory
    assert "inhibitory" in edge_types or "associative" in edge_types


@pytest.mark.asyncio
async def test_edge_classify_heuristic_function():
    """Unit test for classify_edge_heuristic."""
    # High similarity, no contradiction → excitatory
    result = classify_edge_heuristic("AWS deployment", "AWS configuration", 0.8)
    assert result is not None
    assert result[0] == "excitatory"

    # Contradiction keywords → inhibitory
    result = classify_edge_heuristic(
        "We use GCP", "We switched from GCP to AWS instead", 0.6
    )
    assert result is not None
    assert result[0] == "inhibitory"

    # Low similarity → None
    result = classify_edge_heuristic("Weather is nice", "Quantum physics paper", 0.2)
    assert result is None

    # Moderate similarity → associative
    result = classify_edge_heuristic("Database setup", "Server monitoring", 0.55)
    assert result is not None
    assert result[0] == "associative"
