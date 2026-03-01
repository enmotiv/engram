"""Plugin registry — singleton that holds the active plugin components."""

from __future__ import annotations

import importlib
from typing import List, Optional

from engram.engine.interfaces import (
    DimensionSet,
    Encoder,
    RetrievalPostProcessor,
    TraceEnricher,
    WorkerJob,
)


class PluginRegistry:
    _instance: Optional[PluginRegistry] = None

    def __init__(self):
        self.dimensions: Optional[DimensionSet] = None
        self.encoder: Optional[Encoder] = None
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

    def register_dimensions(self, dim_set: DimensionSet) -> None:
        self.dimensions = dim_set

    def register_encoder(self, encoder: Encoder) -> None:
        self.encoder = encoder

    def register_jobs(self, jobs: List[WorkerJob]) -> None:
        self.jobs.extend(jobs)

    def register_trace_enricher(self, enricher: TraceEnricher) -> None:
        self.trace_enricher = enricher

    def register_post_processor(self, processor: RetrievalPostProcessor) -> None:
        self.post_processor = processor

    def get_trace_enricher(self) -> Optional[TraceEnricher]:
        return self.trace_enricher

    def load_plugin(self, module_path: str) -> None:
        """Import a plugin module and call its register() function."""
        mod = importlib.import_module(module_path)
        if not hasattr(mod, "register"):
            raise AttributeError(f"Plugin module {module_path} has no register() function")
        mod.register(self)
