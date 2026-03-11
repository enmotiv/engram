"""Backward compatibility — use engram.services.embedding instead."""

from engram.services.embedding import (
    compute_salience,
    embed_six_dimensions,
    get_client,
    get_salience_anchor,
    llm_classify,
)
from engram.utilities.vectors import cosine_similarity, normalize_l2

__all__ = [
    "compute_salience",
    "cosine_similarity",
    "embed_six_dimensions",
    "get_client",
    "get_salience_anchor",
    "llm_classify",
    "normalize_l2",
]
