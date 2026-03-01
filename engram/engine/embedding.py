"""Embedding service with LRU cache."""

import hashlib
from functools import lru_cache
from typing import Dict, List, Optional

from engram.plugins.registry import PluginRegistry


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class EmbeddingService:
    """Wraps the active encoder with content-hash caching."""

    def __init__(self, registry: Optional[PluginRegistry] = None):
        self._registry = registry or PluginRegistry.get_instance()
        self._embed_cache: dict = {}
        self._score_cache: dict = {}
        self._max_cache = 1000

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        key = _content_hash(text)
        if key in self._embed_cache:
            return self._embed_cache[key]
        vec = await self._registry.encoder.embed(text)
        if len(self._embed_cache) < self._max_cache:
            self._embed_cache[key] = vec
        return vec

    async def get_dimension_scores(self, text: str) -> Dict[str, float]:
        key = _content_hash(text)
        if key in self._score_cache:
            return self._score_cache[key]
        scores = await self._registry.encoder.encode(text)
        if len(self._score_cache) < self._max_cache:
            self._score_cache[key] = scores
        return scores
