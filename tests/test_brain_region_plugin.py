"""Test the brain-region plugin: 6 dimensions, LLM decomposition, uniform fallback."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.plugins.brain_regions.encoder import (
    BrainRegionEncoder,
    REQUIRED_REGIONS,
    _SHORT_TO_LONG,
)
from engram.plugins.registry import PluginRegistry

SAMPLE_TEXT = (
    "I was terrified when the server crashed at 3am last Tuesday — the red error "
    "logs scrolling by reminded me of the time we lost the production database"
)

# Mock LLM response matching the decompose schema
MOCK_LLM_RESPONSE = json.dumps({
    "hippo": {"text": "3am last Tuesday, reminded of past event", "score": 0.7},
    "amyg": {"text": "terrified, lost production database", "score": 0.8},
    "pfc": {"text": "server crash analysis, database recovery", "score": 0.5},
    "sensory": {"text": "red error logs scrolling", "score": 0.6},
    "striatum": {"text": "monitoring server, checking logs", "score": 0.4},
    "cerebellum": {"text": "familiar crash pattern from past incident", "score": 0.5},
})


@pytest.fixture
def registry():
    PluginRegistry.reset()
    reg = PluginRegistry.get_instance()
    reg.load_plugin("engram.plugins.brain_regions")
    yield reg
    PluginRegistry.reset()


def test_brain_region_dimensions_loaded(registry):
    names = registry.dimensions.names()
    assert len(names) == 6
    assert "hippocampus" in names
    assert "amygdala" in names
    assert "prefrontal_cortex" in names
    assert "sensory_cortices" in names
    assert "striatum" in names
    assert "cerebellum" in names


def test_brain_region_weights(registry):
    weights = registry.dimensions.default_weights()
    assert weights["prefrontal_cortex"] == 1.0
    assert weights["hippocampus"] == 0.9
    assert weights["cerebellum"] == 0.5


@pytest.mark.asyncio
async def test_encode_returns_6_scores_with_llm(registry):
    """When LLM is available, encode() returns 6 dimension scores."""
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.complete = AsyncMock(return_value=MOCK_LLM_RESPONSE)

    with patch("engram.plugins.brain_regions.encoder.get_llm_service", return_value=mock_llm):
        scores = await registry.encoder.encode(SAMPLE_TEXT)

    expected_keys = set(_SHORT_TO_LONG.values())
    assert set(scores.keys()) == expected_keys
    for v in scores.values():
        assert 0.0 <= v <= 1.0


@pytest.mark.asyncio
async def test_encode_uniform_fallback_when_llm_unavailable(registry):
    """When LLM fails, encode() returns uniform 0.5 scores."""
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

    with patch("engram.plugins.brain_regions.encoder.get_llm_service", return_value=mock_llm):
        scores = await registry.encoder.encode(SAMPLE_TEXT)

    expected_keys = set(_SHORT_TO_LONG.values())
    assert set(scores.keys()) == expected_keys
    for v in scores.values():
        assert v == 0.5


@pytest.mark.asyncio
async def test_llm_encode_returns_scores(registry):
    """llm_encode() returns {long_key: float} for AxisRescoringJob."""
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.complete = AsyncMock(return_value=MOCK_LLM_RESPONSE)

    encoder = BrainRegionEncoder()
    with patch("engram.plugins.brain_regions.encoder.get_llm_service", return_value=mock_llm):
        scores = await encoder.llm_encode(SAMPLE_TEXT)

    expected_keys = set(_SHORT_TO_LONG.values())
    assert set(scores.keys()) == expected_keys
    assert scores["amygdala"] == 0.8
    assert scores["hippocampus"] == 0.7


@pytest.mark.asyncio
async def test_llm_encode_raises_when_unavailable():
    """llm_encode() raises RuntimeError when LLM is unavailable."""
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = False
    mock_llm.complete = AsyncMock()

    encoder = BrainRegionEncoder()
    with patch("engram.plugins.brain_regions.encoder.get_llm_service", return_value=mock_llm):
        with pytest.raises(RuntimeError, match="LLM not available"):
            await encoder.llm_encode(SAMPLE_TEXT)


@pytest.mark.asyncio
async def test_decompose_returns_6_strings():
    """decompose() returns {short_key: text_string} for multi-axis embedding."""
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.complete = AsyncMock(return_value=MOCK_LLM_RESPONSE)

    encoder = BrainRegionEncoder()
    with patch("engram.plugins.brain_regions.encoder.get_llm_service", return_value=mock_llm):
        region_texts = await encoder.decompose(SAMPLE_TEXT)

    assert set(region_texts.keys()) == REQUIRED_REGIONS
    for text in region_texts.values():
        assert isinstance(text, str)
        assert len(text) > 0


@pytest.mark.asyncio
async def test_decompose_returns_empty_on_failure():
    """decompose() returns {} when LLM fails — caller falls back to single-embedding."""
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

    encoder = BrainRegionEncoder()
    with patch("engram.plugins.brain_regions.encoder.get_llm_service", return_value=mock_llm):
        region_texts = await encoder.decompose(SAMPLE_TEXT)

    assert region_texts == {}


@pytest.mark.asyncio
async def test_decompose_rejects_partial_response():
    """decompose() returns {} when LLM returns incomplete regions."""
    partial_response = json.dumps({
        "hippo": {"text": "some text", "score": 0.5},
        "amyg": {"text": "some text", "score": 0.5},
        # Missing pfc, sensory, striatum, cerebellum
    })

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.complete = AsyncMock(return_value=partial_response)

    encoder = BrainRegionEncoder()
    with patch("engram.plugins.brain_regions.encoder.get_llm_service", return_value=mock_llm):
        region_texts = await encoder.decompose(SAMPLE_TEXT)

    assert region_texts == {}


@pytest.mark.asyncio
async def test_llm_decompose_clamps_scores():
    """_llm_decompose() clamps scores to [0.0, 1.0]."""
    response = json.dumps({
        "hippo": {"text": "time ref", "score": 1.5},
        "amyg": {"text": "emotion", "score": -0.3},
        "pfc": {"text": "concept", "score": 0.8},
        "sensory": {"text": "detail", "score": 0.6},
        "striatum": {"text": "action", "score": 0.4},
        "cerebellum": {"text": "routine", "score": 0.3},
    })

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.complete = AsyncMock(return_value=response)

    encoder = BrainRegionEncoder()
    with patch("engram.plugins.brain_regions.encoder.get_llm_service", return_value=mock_llm):
        full = await encoder._llm_decompose(SAMPLE_TEXT)

    assert full["hippo"]["score"] == 1.0
    assert full["amyg"]["score"] == 0.0
