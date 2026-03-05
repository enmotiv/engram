"""Test the plugin system: registry, brain_regions plugin, encode + embed pipeline."""

import pytest

from engram.plugins.registry import PluginRegistry

BRAIN_REGION_DIMS = {"hippocampus", "amygdala", "prefrontal_cortex", "sensory_cortices", "striatum", "cerebellum"}
BRAIN_REGION_WEIGHTS = {
    "hippocampus": 0.9,
    "amygdala": 0.8,
    "prefrontal_cortex": 1.0,
    "sensory_cortices": 0.6,
    "striatum": 0.7,
    "cerebellum": 0.5,
}


@pytest.fixture
def registry():
    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.brain_regions")
    yield reg
    PluginRegistry.reset()


def test_singleton():
    PluginRegistry.reset()
    a = PluginRegistry.get_instance()
    b = PluginRegistry.get_instance()
    assert a is b
    PluginRegistry.reset()


def test_brain_region_dimensions_loaded(registry):
    assert registry.dimensions is not None
    assert set(registry.dimensions.names()) == BRAIN_REGION_DIMS


def test_brain_region_weights(registry):
    weights = registry.dimensions.default_weights()
    assert weights == BRAIN_REGION_WEIGHTS


@pytest.mark.asyncio
async def test_encode_returns_6_brain_region_scores(registry):
    scores = await registry.encoder.encode(
        "We migrated the database to AWS last Tuesday"
    )
    assert set(scores.keys()) == BRAIN_REGION_DIMS
    for v in scores.values():
        assert 0.0 <= v <= 1.0


@pytest.mark.asyncio
async def test_encode_hippocampus_temporal_detection(registry):
    scores = await registry.encoder.encode(
        "We migrated the database to AWS last Tuesday"
    )
    assert scores["hippocampus"] > 0.0, "Should detect 'last Tuesday' as hippocampus-relevant"


@pytest.mark.asyncio
async def test_encode_amygdala_emotion_detection(registry):
    scores = await registry.encoder.encode(
        "I felt terrified and overwhelmed by the situation"
    )
    assert scores["amygdala"] > 0.0, "Should detect emotion words as amygdala-relevant"


@pytest.mark.asyncio
async def test_embed_returns_1024_dim_vector(registry):
    vec = await registry.encoder.embed(
        "We migrated the database to AWS last Tuesday"
    )
    assert isinstance(vec, list)
    assert len(vec) == 1024
    assert all(isinstance(x, float) for x in vec)


def test_encoder_loaded(registry):
    assert registry.encoder is not None
