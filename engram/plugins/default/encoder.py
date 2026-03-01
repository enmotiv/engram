"""Default encoder — sentence-transformers embedding + heuristic dimension scoring."""

import re
from typing import Dict, List, Optional


from engram.engine.interfaces import Encoder

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
    """Produces 384-dim embeddings via sentence-transformers and heuristic dimension scores."""

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    async def embed(self, text: str) -> Optional[List[float]]:
        model = self._get_model()
        vec = model.encode(text, convert_to_numpy=True)
        return vec.tolist()

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
