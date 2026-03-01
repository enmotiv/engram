"""Test multi-plugin support in PluginRegistry (Sweep 3).

Verifies:
- Multiple plugins load additively (dimensions + encoders merge)
- Composite dimension set returns all dimension names
- Composite encoder merges scores from all encoders
- Composite encoder delegates embedding to first encoder
- Single plugin still works (backward compat)
- Plugin jobs are accumulated across plugins
"""

from __future__ import annotations

from typing import Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from engram.engine.interfaces import DimensionSet, Encoder, WorkerJob
from engram.plugins.registry import PluginRegistry


# -- Fixtures: mock plugins ------------------------------------------------


class _DimSetA(DimensionSet):
    def names(self) -> List[str]:
        return ["alpha", "beta"]

    def default_weights(self) -> Dict[str, float]:
        return {"alpha": 0.6, "beta": 0.4}


class _DimSetB(DimensionSet):
    def names(self) -> List[str]:
        return ["gamma", "delta"]

    def default_weights(self) -> Dict[str, float]:
        return {"gamma": 0.5, "delta": 0.5}


class _EncoderA(Encoder):
    async def encode(self, text: str) -> Dict[str, float]:
        return {"alpha": 0.8, "beta": 0.2}

    async def embed(self, text: str) -> Optional[List[float]]:
        return [0.1, 0.2, 0.3]


class _EncoderB(Encoder):
    async def encode(self, text: str) -> Dict[str, float]:
        return {"gamma": 0.7, "delta": 0.3}

    async def embed(self, text: str) -> Optional[List[float]]:
        return None  # Secondary encoder defers embedding


class _MockJob(WorkerJob):
    def __init__(self, job_name: str):
        self._name = job_name

    def name(self) -> str:
        return self._name

    async def should_run(self, namespace: str, **kwargs) -> bool:
        return True

    async def execute(self, namespace: str, **kwargs) -> dict:
        return {"ran": True}


@pytest.fixture
def registry():
    PluginRegistry.reset()
    yield PluginRegistry.get_instance()
    PluginRegistry.reset()


# -- Tests -----------------------------------------------------------------


def test_single_plugin_backward_compat(registry):
    """Single plugin works exactly as before."""
    registry.register_dimensions(_DimSetA())
    registry.register_encoder(_EncoderA())

    assert registry.dimensions.names() == ["alpha", "beta"]
    assert registry.dimensions.default_weights() == {"alpha": 0.6, "beta": 0.4}


def test_multi_plugin_dimensions_merge(registry):
    """Multiple dimension sets are merged."""
    registry.register_dimensions(_DimSetA())
    registry.register_dimensions(_DimSetB())

    names = registry.dimensions.names()
    assert names == ["alpha", "beta", "gamma", "delta"]

    weights = registry.dimensions.default_weights()
    assert weights == {"alpha": 0.6, "beta": 0.4, "gamma": 0.5, "delta": 0.5}


def test_duplicate_dimension_names_deduplicated(registry):
    """If two plugins declare the same dimension, it appears once."""

    class _DimSetOverlap(DimensionSet):
        def names(self):
            return ["alpha", "gamma"]

        def default_weights(self):
            return {"alpha": 0.9, "gamma": 0.1}

    registry.register_dimensions(_DimSetA())
    registry.register_dimensions(_DimSetOverlap())

    names = registry.dimensions.names()
    assert names == ["alpha", "beta", "gamma"]
    # Second plugin's weight for 'alpha' overrides first
    assert registry.dimensions.default_weights()["alpha"] == 0.9


@pytest.mark.asyncio
async def test_multi_plugin_encoder_merge(registry):
    """Composite encoder merges scores from all encoders."""
    registry.register_encoder(_EncoderA())
    registry.register_encoder(_EncoderB())

    scores = await registry.encoder.encode("test text")
    assert scores == {"alpha": 0.8, "beta": 0.2, "gamma": 0.7, "delta": 0.3}


@pytest.mark.asyncio
async def test_composite_encoder_embedding_first_wins(registry):
    """Embedding comes from first encoder that returns non-None."""
    registry.register_encoder(_EncoderA())
    registry.register_encoder(_EncoderB())

    vec = await registry.encoder.embed("test text")
    assert vec == [0.1, 0.2, 0.3]  # From _EncoderA


@pytest.mark.asyncio
async def test_composite_encoder_skips_failed(registry):
    """If first encoder fails, second still contributes."""

    class _FailEncoder(Encoder):
        async def encode(self, text):
            raise RuntimeError("boom")

        async def embed(self, text):
            raise RuntimeError("boom")

    registry.register_encoder(_FailEncoder())
    registry.register_encoder(_EncoderB())

    scores = await registry.encoder.encode("test")
    assert scores == {"gamma": 0.7, "delta": 0.3}

    vec = await registry.encoder.embed("test")
    assert vec is None  # _EncoderB returns None for embed


def test_jobs_accumulate(registry):
    """Jobs from multiple plugins accumulate."""
    registry.register_jobs([_MockJob("job_a")])
    registry.register_jobs([_MockJob("job_b"), _MockJob("job_c")])

    assert len(registry.jobs) == 3
    assert [j.name() for j in registry.jobs] == ["job_a", "job_b", "job_c"]


def test_load_plugins_comma_separated(registry):
    """load_plugins() handles comma-separated module paths."""
    registry.load_plugins("engram.plugins.default")

    assert registry.dimensions is not None
    assert "semantic" in registry.dimensions.names()


def test_no_dimensions_returns_none(registry):
    """Empty registry returns None for dimensions/encoder."""
    assert registry.dimensions is None
    assert registry.encoder is None


def test_dimensions_setter_backward_compat(registry):
    """Direct assignment via .dimensions = still works."""
    ds = _DimSetA()
    registry.dimensions = ds
    assert registry.dimensions is ds

    registry.dimensions = None
    assert registry.dimensions is None
