"""Brain-region encoder — LLM-based scoring with heuristic fallback."""

import json
import logging
import os
import re
from typing import Dict, List, Optional

import httpx

from engram.engine.interfaces import Encoder
from engram.plugins.brain_regions.prompts import BRAIN_REGION_PROMPT

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

_EMOTION_WORDS = frozenset({
    "happy", "sad", "angry", "afraid", "terrified", "anxious", "excited", "frustrated",
    "disappointed", "grateful", "nervous", "worried", "thrilled", "furious", "devastated",
    "joyful", "miserable", "ecstatic", "overwhelmed", "relieved", "scared", "panic",
    "love", "hate", "rage", "grief", "shame", "guilt", "pride", "jealous",
})

_SENSORY_WORDS = frozenset({
    "red", "blue", "green", "bright", "dark", "loud", "quiet", "smooth", "rough",
    "cold", "hot", "warm", "sharp", "soft", "sweet", "bitter", "sour", "light",
    "glowing", "flickering", "buzzing", "humming", "scrolling", "flashing", "blinking",
})

_ACTION_VERBS = frozenset({
    "run", "walk", "build", "fix", "deploy", "migrate", "push", "pull", "click",
    "drag", "type", "write", "code", "test", "debug", "ship", "launch", "start",
    "stop", "restart", "install", "configure", "execute", "trigger", "process",
})

_ABSTRACT_WORDS = frozenset({
    "strategy", "architecture", "design", "pattern", "concept", "theory", "framework",
    "principle", "goal", "objective", "plan", "decision", "analysis", "evaluation",
    "hypothesis", "abstraction", "system", "process", "methodology", "approach",
})

_TEMPORAL_WORDS = frozenset({
    "yesterday", "today", "tomorrow", "last", "next", "before", "after", "when",
    "then", "first", "second", "finally", "recently", "previously", "initially",
    "morning", "evening", "night", "week", "month", "year", "ago", "later",
})

BRAIN_REGION_KEYS = frozenset({
    "hippocampus", "amygdala", "prefrontal_cortex", "sensory_cortices", "striatum"
})


class BrainRegionEncoder(Encoder):
    """Scores text on 5 brain regions via LLM or heuristic fallback."""

    def __init__(self):
        self._embed_model = None

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embed_model

    async def embed(self, text: str) -> Optional[List[float]]:
        model = self._get_embed_model()
        vec = model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    async def encode(self, text: str) -> Dict[str, float]:
        try:
            return await self._llm_encode(text)
        except Exception:
            logger.debug("Ollama unavailable, using heuristic fallback")
            return self.heuristic_encode(text)

    async def _llm_encode(self, text: str) -> Dict[str, float]:
        prompt = BRAIN_REGION_PROMPT.format(text=text)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": False, "format": "json"},
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")

        scores = json.loads(raw)
        if not BRAIN_REGION_KEYS.issubset(scores.keys()):
            raise ValueError(f"Missing keys in LLM response: {scores.keys()}")
        return {k: max(0.0, min(1.0, float(scores[k]))) for k in BRAIN_REGION_KEYS}

    def heuristic_encode(self, text: str) -> Dict[str, float]:
        """Keyword-count heuristic fallback for each brain region."""
        words = set(re.findall(r"\b\w+\b", text.lower()))
        total = max(len(words), 1)

        hippocampus = min(len(words & _TEMPORAL_WORDS) / total * 8.0, 1.0)
        amygdala = min(len(words & _EMOTION_WORDS) / total * 8.0, 1.0)
        prefrontal = min(len(words & _ABSTRACT_WORDS) / total * 8.0, 1.0)
        sensory = min(len(words & _SENSORY_WORDS) / total * 8.0, 1.0)
        striatum = min(len(words & _ACTION_VERBS) / total * 8.0, 1.0)

        return {
            "hippocampus": round(hippocampus, 4),
            "amygdala": round(amygdala, 4),
            "prefrontal_cortex": round(prefrontal, 4),
            "sensory_cortices": round(sensory, 4),
            "striatum": round(striatum, 4),
        }
