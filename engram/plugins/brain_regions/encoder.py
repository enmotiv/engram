"""Brain-region encoder — LLM-based scoring with uniform fallback."""

import json
import logging
from typing import Any, Dict, List, Optional

from engram.engine.interfaces import Encoder
from engram.engine.llm import get_llm_service
from engram.plugins.brain_regions.prompts import BRAIN_REGION_DECOMPOSE_PROMPT

logger = logging.getLogger(__name__)

# Canonical short region keys used by decompose() and multi-axis retrieval
REQUIRED_REGIONS = frozenset({
    "hippo", "amyg", "pfc", "sensory", "striatum", "cerebellum"
})

# Map short region keys → long dimension names for backward compat with
# dimension_scores stored in DB and convergence scoring
_SHORT_TO_LONG = {
    "hippo": "hippocampus",
    "amyg": "amygdala",
    "pfc": "prefrontal_cortex",
    "sensory": "sensory_cortices",
    "striatum": "striatum",
    "cerebellum": "cerebellum",
}

# Uniform fallback scores (long keys) when LLM is unavailable
_UNIFORM_SCORES = {long_name: 0.5 for long_name in _SHORT_TO_LONG.values()}


class BrainRegionEncoder(Encoder):
    """Scores text on 6 brain regions via LLM. Uniform 0.5 fallback on failure."""

    def __init__(self):
        self._embed_model = None  # lazy local model (only if EMBEDDING_PROVIDER=local)

    async def embed(self, text: str) -> Optional[List[float]]:
        from engram.config import settings

        if settings.EMBEDDING_PROVIDER == "api" and settings.EMBEDDING_API_URL:
            return await self._embed_via_api(text)
        return await self._embed_local(text)

    async def _embed_via_api(self, text: str) -> Optional[List[float]]:
        """Call an OpenAI-compatible embedding API (OpenRouter, etc.)."""
        import httpx
        from engram.config import settings

        if not settings.EMBEDDING_API_KEY:
            logger.warning("EMBEDDING_API_KEY is empty — cannot call embedding API")
            return await self._embed_local(text)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    settings.EMBEDDING_API_URL,
                    headers={"Authorization": f"Bearer {settings.EMBEDDING_API_KEY}"},
                    json={"model": settings.EMBEDDING_MODEL, "input": text},
                )
                resp.raise_for_status()
                data = resp.json()
                if "data" not in data:
                    logger.warning("Embedding API unexpected response: %s", str(data)[:300])
                    return await self._embed_local(text)
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
        """Score text on each dimension. Returns {long_dim_name: 0.0-1.0}.

        Uses LLM decomposition. Falls back to uniform 0.5 on failure.
        """
        try:
            full = await self._llm_decompose(text)
            return {_SHORT_TO_LONG[r]: full[r]["score"] for r in REQUIRED_REGIONS}
        except Exception:
            logger.warning("LLM encoding failed, returning uniform scores", exc_info=True)
            return dict(_UNIFORM_SCORES)

    async def llm_encode(self, text: str) -> Dict[str, float]:
        """LLM-based scoring — called by dreamer's DimensionRescoringJob.

        Returns {long_dim_name: 0.0-1.0}. Raises on failure.
        """
        llm = get_llm_service()
        if not llm.is_available():
            raise RuntimeError("LLM not available for dimension rescoring")
        full = await self._llm_decompose(text)
        return {_SHORT_TO_LONG[r]: full[r]["score"] for r in REQUIRED_REGIONS}

    async def decompose(self, text: str) -> Dict[str, str]:
        """Returns {short_region: text_string} for retrieval embedding.

        Each string captures region-specific content from the input,
        suitable for independent embedding and per-region pgvector search.
        Returns empty dict on failure — caller falls back to single-embedding.
        """
        try:
            full = await self._llm_decompose(text)
            return {region: full[region]["text"] for region in REQUIRED_REGIONS}
        except Exception:
            logger.warning("Cue decomposition failed, returning empty", exc_info=True)
            return {}

    async def _llm_decompose(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Single LLM call returning both text strings and scores per region.

        Raises ValueError on malformed output — callers catch and fall back.
        Never returns partial results.
        """
        llm = get_llm_service()
        prompt = BRAIN_REGION_DECOMPOSE_PROMPT.format(text=text)
        raw = await llm.complete(prompt, max_tokens=400, temperature=0.1, json_mode=True)

        result = json.loads(raw)  # Raises JSONDecodeError if not valid JSON

        # Validate all 6 regions present
        missing = REQUIRED_REGIONS - set(result.keys())
        if missing:
            raise ValueError(f"Missing regions in LLM response: {missing}")

        # Validate each region has both text and score
        for region in REQUIRED_REGIONS:
            entry = result[region]
            if not isinstance(entry, dict):
                raise ValueError(f"Region '{region}' is not a dict: {type(entry)}")
            if "text" not in entry or "score" not in entry:
                raise ValueError(f"Region '{region}' missing text or score: {entry.keys()}")
            if not isinstance(entry["text"], str) or not entry["text"].strip():
                raise ValueError(f"Region '{region}' has empty/non-string text")
            entry["score"] = max(0.0, min(1.0, float(entry["score"])))

        return result
