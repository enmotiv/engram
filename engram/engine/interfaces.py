"""Abstract base classes defining the Engram plugin interface."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class DimensionSet(ABC):
    """Defines the dimensions a plugin scores memories on."""

    @abstractmethod
    def names(self) -> List[str]:
        ...

    @abstractmethod
    def default_weights(self) -> Dict[str, float]:
        ...


class Encoder(ABC):
    """Encodes text into dimension scores and embeddings."""

    @abstractmethod
    async def encode(self, text: str) -> Dict[str, float]:
        """Score text on each dimension. Returns {dim_name: 0.0-1.0}."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> Optional[List[float]]:
        """Produce a dense embedding vector for the text."""
        ...


class WorkerJob(ABC):
    """A background job the Dreamer worker can execute."""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def should_run(self, namespace: str, **kwargs) -> bool:
        ...

    @abstractmethod
    async def execute(self, namespace: str, **kwargs) -> dict:
        ...


class TraceEnricher(ABC):
    """Adds domain-specific annotations to graph traces."""

    @abstractmethod
    def enrich(self, trace: str, memories: list, edges: list) -> str:
        ...


class RetrievalPostProcessor(ABC):
    """Post-processes retrieval results before returning them."""

    @abstractmethod
    async def process(self, results):
        ...
