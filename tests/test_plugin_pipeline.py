"""Test the plugin system: registry, default plugin, encode + embed pipeline."""

import pytest

from engram.plugins.registry import PluginRegistry


@pytest.fixture
def registry():
    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.default")
    yield reg
    PluginRegistry.reset()


def test_singleton():
    PluginRegistry.reset()
    a = PluginRegistry.get_instance()
    b = PluginRegistry.get_instance()
    assert a is b
    PluginRegistry.reset()


def test_default_dimensions_loaded(registry):
    assert registry.dimensions is not None
    assert registry.dimensions.names() == ["semantic", "temporal", "importance"]


def test_default_weights(registry):
    weights = registry.dimensions.default_weights()
    assert weights == {"semantic": 1.0, "temporal": 0.7, "importance": 0.8}


@pytest.mark.asyncio
async def test_encode_returns_dimension_scores(registry):
    scores = await registry.encoder.encode(
        "We migrated the database to AWS last Tuesday"
    )
    assert set(scores.keys()) == {"semantic", "temporal", "importance"}
    for v in scores.values():
        assert 0.0 <= v <= 1.0


@pytest.mark.asyncio
async def test_encode_temporal_detection(registry):
    scores = await registry.encoder.encode(
        "We migrated the database to AWS last Tuesday"
    )
    assert scores["temporal"] > 0.0, "Should detect 'last Tuesday' as temporal"


@pytest.mark.asyncio
async def test_embed_returns_384_dim_vector(registry):
    vec = await registry.encoder.embed(
        "We migrated the database to AWS last Tuesday"
    )
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)


def test_encoder_loaded(registry):
    assert registry.encoder is not None
