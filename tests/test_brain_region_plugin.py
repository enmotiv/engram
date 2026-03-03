"""Test the brain-region plugin: 5 dimensions, LLM fallback, heuristic scoring."""

import pytest

from engram.plugins.brain_regions.encoder import BrainRegionEncoder
from engram.plugins.registry import PluginRegistry

SAMPLE_TEXT = (
    "I was terrified when the server crashed at 3am last Tuesday — the red error "
    "logs scrolling by reminded me of the time we lost the production database"
)


@pytest.fixture
def registry():
    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.brain_regions")
    yield reg
    PluginRegistry.reset()


def test_brain_region_dimensions_loaded(registry):
    names = registry.dimensions.names()
    assert len(names) == 5
    assert "hippocampus" in names
    assert "amygdala" in names
    assert "prefrontal_cortex" in names
    assert "sensory_cortices" in names
    assert "striatum" in names


def test_brain_region_weights(registry):
    weights = registry.dimensions.default_weights()
    assert weights["prefrontal_cortex"] == 1.0
    assert weights["hippocampus"] == 0.9


@pytest.mark.asyncio
async def test_encode_returns_5_scores(registry):
    scores = await registry.encoder.encode(SAMPLE_TEXT)
    assert set(scores.keys()) == {
        "hippocampus", "amygdala", "prefrontal_cortex", "sensory_cortices", "striatum"
    }
    for v in scores.values():
        assert 0.0 <= v <= 1.0


@pytest.mark.asyncio
async def test_heuristic_detects_temporal(registry):
    scores = await registry.encoder.encode(SAMPLE_TEXT)
    # "last Tuesday", "3am", "time" — should detect temporal content
    assert scores["hippocampus"] > 0.0


@pytest.mark.asyncio
async def test_heuristic_detects_emotion(registry):
    scores = await registry.encoder.encode(SAMPLE_TEXT)
    # "terrified" — should detect emotional content
    assert scores["amygdala"] > 0.0


@pytest.mark.asyncio
async def test_heuristic_detects_sensory(registry):
    scores = await registry.encoder.encode(SAMPLE_TEXT)
    # "red" — should detect sensory content
    assert scores["sensory_cortices"] > 0.0


@pytest.mark.asyncio
async def test_embed_returns_1024_dim(registry):
    vec = await registry.encoder.embed(SAMPLE_TEXT)
    assert isinstance(vec, list)
    assert len(vec) == 1024


def test_heuristic_fallback_directly():
    encoder = BrainRegionEncoder()
    scores = encoder.heuristic_encode(
        "I was terrified and angry when the deployment failed yesterday"
    )
    assert scores["amygdala"] > 0.0
    assert scores["hippocampus"] > 0.0
    assert all(0.0 <= v <= 1.0 for v in scores.values())
