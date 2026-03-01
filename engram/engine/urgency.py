"""Urgency scoring — heuristic-based, no LLM calls."""

import re


_RETRIEVAL_KEYWORDS = frozenset({
    "remember", "recall", "last time", "before", "previously",
    "earlier", "mentioned", "talked about", "discussed", "said",
})

_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def score_urgency(cue: str, context: dict = None) -> float:
    """Score 0-1 based on how likely the cue needs memory retrieval."""
    score = 0.0

    # Question mark
    if "?" in cue:
        score += 0.3

    # Retrieval keywords (max 0.4)
    cue_lower = cue.lower()
    keyword_hits = sum(1 for kw in _RETRIEVAL_KEYWORDS if kw in cue_lower)
    score += min(keyword_hits * 0.2, 0.4)

    # Cue length
    if len(cue) > 20:
        score += 0.1

    # Proper nouns
    if _PROPER_NOUN_RE.search(cue):
        score += 0.2

    return min(score, 1.0)
