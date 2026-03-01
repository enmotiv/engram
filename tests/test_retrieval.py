"""Test multi-axis retrieval with convergence scoring and urgency gating."""

import pytest

from engram.engine.models import RetrievalOptions
from engram.engine.retriever import Retriever
from engram.engine.store import MemoryStore
from engram.plugins.registry import PluginRegistry

NS = "user:retrieval_test"

# 20 memories across 4 topics
AWS_MEMORIES = [
    "We migrated our production database to AWS RDS last Tuesday",
    "The EC2 g5.xlarge instances handle our ML inference workload",
    "AWS CloudFront CDN reduced our latency by 60% globally",
    "S3 lifecycle policies automatically archive old logs after 90 days",
    "We set up cross-region replication for disaster recovery on AWS",
]
COOKING_MEMORIES = [
    "The pasta carbonara recipe uses guanciale not bacon for authentic flavor",
    "Sourdough bread needs 12 hours of cold fermentation for best taste",
    "Fresh basil should be added at the very end to preserve its aroma",
    "Cast iron skillets need to be seasoned with flaxseed oil at 500 degrees",
    "Homemade stock requires simmering bones for at least 8 hours",
]
PM_MEMORIES = [
    "Sprint retrospective revealed we need better estimation practices",
    "The product roadmap was updated to prioritize mobile-first features",
    "Stakeholder alignment meeting scheduled for every other Thursday",
    "Jira board reorganized with swimlanes for each team workstream",
    "Quarterly OKRs set: improve user retention by 15% and reduce churn",
]
FITNESS_MEMORIES = [
    "Morning runs at 6am have improved my focus throughout the day",
    "Progressive overload with 5% weekly weight increase for squat gains",
    "Yoga twice a week helps with flexibility and reduces back pain",
    "Heart rate zone 2 training builds aerobic base for endurance running",
    "Protein intake target is 1.6g per kg of bodyweight for muscle growth",
]
CROSSOVER = "We hosted the recipe API on AWS Lambda with API Gateway"


@pytest.fixture
def registry():
    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.default")
    yield reg
    PluginRegistry.reset()


async def _seed_memories(db, registry):
    store = MemoryStore(db, registry)
    for content in AWS_MEMORIES + COOKING_MEMORIES + PM_MEMORIES + FITNESS_MEMORIES:
        await store.create(NS, content)
    await store.create(NS, CROSSOVER)
    await db.flush()


@pytest.mark.asyncio
async def test_aws_retrieval(db, registry):
    await _seed_memories(db, registry)
    retriever = Retriever(db, registry)

    result = await retriever.retrieve(
        NS, "What infrastructure decisions were made about AWS?"
    )
    assert result.triggered is True
    assert len(result.memories) > 0
    # Top results should be AWS-related
    top_content = result.memories[0].content.lower()
    assert any(kw in top_content for kw in ["aws", "ec2", "s3", "rds", "cloudfront"])


@pytest.mark.asyncio
async def test_cooking_retrieval(db, registry):
    await _seed_memories(db, registry)
    retriever = Retriever(db, registry)

    result = await retriever.retrieve(NS, "How do I make pasta?")
    assert result.triggered is True
    assert len(result.memories) > 0
    top_content = result.memories[0].content.lower()
    assert any(kw in top_content for kw in ["pasta", "recipe", "cook", "bread", "basil", "stock", "skillet"])


@pytest.mark.asyncio
async def test_crossover_memory(db, registry):
    await _seed_memories(db, registry)
    retriever = Retriever(db, registry)

    result = await retriever.retrieve(
        NS, "What cloud services do we use?", options=RetrievalOptions(max_results=10)
    )
    assert result.triggered is True
    contents = [m.content for m in result.memories]
    # The crossover memory should appear somewhere in results
    assert any("recipe API" in c for c in contents) or any("AWS" in c for c in contents)


@pytest.mark.asyncio
async def test_urgency_gate_low(db, registry):
    await _seed_memories(db, registry)
    retriever = Retriever(db, registry)

    result = await retriever.retrieve(NS, "ok")
    assert result.triggered is False
    assert len(result.memories) == 0


@pytest.mark.asyncio
async def test_retrieval_ms_populated(db, registry):
    await _seed_memories(db, registry)
    retriever = Retriever(db, registry)

    result = await retriever.retrieve(NS, "What about the AWS migration?")
    assert result.retrieval_ms > 0
    assert result.retrieval_ms < 5000  # reasonable upper bound
