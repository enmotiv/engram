"""Plugin registry — singleton that holds the active plugin components.

Supports multiple plugins loaded additively. Dimension sets and encoders
from all plugins are merged so convergence scoring operates across all
registered dimensions simultaneously.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, List, Optional

from engram.engine.interfaces import (
    DimensionSet,
    Encoder,
    RetrievalPostProcessor,
    TraceEnricher,
    WorkerJob,
)

logger = logging.getLogger(__name__)


class _CompositeDimensionSet(DimensionSet):
    """Merges names and weights from multiple DimensionSets."""

    def __init__(self, sets: List[DimensionSet]):
        self._sets = sets

    def names(self) -> List[str]:
        seen = set()
        result = []
        for ds in self._sets:
            for n in ds.names():
                if n not in seen:
                    seen.add(n)
                    result.append(n)
        return result

    def default_weights(self) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for ds in self._sets:
            merged.update(ds.default_weights())
        return merged


class _CompositeEncoder(Encoder):
    """Runs multiple encoders and merges dimension scores.

    Embedding is delegated to the first encoder (primary).
    """

    def __init__(self, encoders: List[Encoder]):
        self._encoders = encoders

    async def encode(self, text: str) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for enc in self._encoders:
            try:
                scores = await enc.encode(text)
                merged.update(scores)
            except Exception:
                logger.warning("Encoder %s failed, skipping", type(enc).__name__, exc_info=True)
        return merged

    async def embed(self, text: str) -> Optional[List[float]]:
        for enc in self._encoders:
            try:
                vec = await enc.embed(text)
                if vec is not None:
                    return vec
            except Exception:
                continue
        return None


class PluginRegistry:
    _instance: Optional[PluginRegistry] = None

    def __init__(self):
        self._dimension_sets: List[DimensionSet] = []
        self._encoders: List[Encoder] = []
        self.jobs: List[WorkerJob] = []
        self.trace_enricher: Optional[TraceEnricher] = None
        self.post_processor: Optional[RetrievalPostProcessor] = None

    @classmethod
    def get_instance(cls) -> PluginRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton — mainly for testing."""
        cls._instance = None

    # -- Dimension sets -------------------------------------------------------

    def register_dimensions(self, dim_set: DimensionSet) -> None:
        """Add a dimension set. Multiple plugins can register dimensions additively."""
        self._dimension_sets.append(dim_set)

    @property
    def dimensions(self) -> Optional[DimensionSet]:
        """Composite view of all registered dimension sets."""
        if not self._dimension_sets:
            return None
        if len(self._dimension_sets) == 1:
            return self._dimension_sets[0]
        return _CompositeDimensionSet(self._dimension_sets)

    @dimensions.setter
    def dimensions(self, value: Optional[DimensionSet]) -> None:
        """Backward compat: direct assignment replaces all dimension sets."""
        self._dimension_sets = [value] if value else []

    # -- Encoders -------------------------------------------------------------

    def register_encoder(self, encoder: Encoder) -> None:
        """Add an encoder. Multiple plugins can register encoders additively."""
        self._encoders.append(encoder)

    @property
    def encoder(self) -> Optional[Encoder]:
        """Composite view of all registered encoders."""
        if not self._encoders:
            return None
        if len(self._encoders) == 1:
            return self._encoders[0]
        return _CompositeEncoder(self._encoders)

    @encoder.setter
    def encoder(self, value: Optional[Encoder]) -> None:
        """Backward compat: direct assignment replaces all encoders."""
        self._encoders = [value] if value else []

    # -- Jobs -----------------------------------------------------------------

    def register_jobs(self, jobs: List[WorkerJob]) -> None:
        self.jobs.extend(jobs)

    # -- Trace / post-processing ----------------------------------------------

    def register_trace_enricher(self, enricher: TraceEnricher) -> None:
        self.trace_enricher = enricher

    def register_post_processor(self, processor: RetrievalPostProcessor) -> None:
        self.post_processor = processor

    def get_trace_enricher(self) -> Optional[TraceEnricher]:
        return self.trace_enricher

    # -- Plugin loading -------------------------------------------------------

    def load_plugin(self, module_path: str) -> None:
        """Import a plugin module and call its register() function."""
        mod = importlib.import_module(module_path)
        if not hasattr(mod, "register"):
            raise AttributeError(f"Plugin module {module_path} has no register() function")
        mod.register(self)

    def load_plugins(self, plugin_paths: str) -> None:
        """Load one or more plugins from a comma-separated string."""
        for path in plugin_paths.split(","):
            path = path.strip()
            if path:
                self.load_plugin(path)
