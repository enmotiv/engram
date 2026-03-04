"""Default encoder — embedding via API or local + heuristic dimension scoring."""

import logging
import re
from typing import Dict, List, Optional


from engram.engine.interfaces import Encoder

logger = logging.getLogger(__name__)

# Date-like patterns: YYYY-MM-DD, "last Monday", "yesterday", month names, etc.
_DATE_PATTERNS = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"yesterday|today|tomorrow|last\s+\w+day|next\s+\w+day|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


class DefaultEncoder(Encoder):
    """Produces 1024-dim embeddings via API or local BGE-M3, plus heuristic dimension scores."""

    def __init__(self):
        self._embed_model = None

    async def embed(self, text: str) -> Optional[List[float]]:
        from engram.config import settings

        if settings.EMBEDDING_PROVIDER == "api" and settings.EMBEDDING_API_URL:
            return await self._embed_via_api(text)
        return await self._embed_local(text)

    async def _embed_via_api(self, text: str) -> Optional[List[float]]:
        """Call an OpenAI-compatible embedding API (OpenRouter, etc.)."""
        import httpx
        from engram.config import settings

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    settings.EMBEDDING_API_URL,
                    headers={"Authorization": f"Bearer {settings.EMBEDDING_API_KEY}"},
                    json={"model": settings.EMBEDDING_MODEL, "input": text},
                )
                resp.raise_for_status()
                data = resp.json()
                return data["data"][0]["embedding"]
        except Exception:
            logger.warning("API embedding failed, falling back to local", exc_info=True)
            return await self._embed_local(text)

    async def _embed_local(self, text: str) -> Optional[List[float]]:
        """Local sentence-transformers embedding (requires [ml] extra)."""
        model = self._get_local_model()
        if model is None:
            return None
        vec = model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def _get_local_model(self):
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer("BAAI/bge-m3")
            except ImportError:
                logger.warning("sentence-transformers not installed — local embedding unavailable")
                return None
        return self._embed_model

    async def encode(self, text: str) -> Dict[str, float]:
        # semantic: placeholder — 0.5 (would need a reference vector for real scoring)
        semantic = 0.5

        # temporal: proportion of date-like tokens
        words = text.split()
        date_matches = len(_DATE_PATTERNS.findall(text))
        temporal = min(date_matches / max(len(words), 1) * 5.0, 1.0)

        # importance: text length heuristic (longer = more important, normalized)
        importance = min(len(text) / 500.0, 1.0)

        return {
            "semantic": round(semantic, 4),
            "temporal": round(temporal, 4),
            "importance": round(importance, 4),
        }
