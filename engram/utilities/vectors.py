"""Vector math utilities."""

from __future__ import annotations

import math


def normalize_l2(vec: list[float]) -> list[float]:
    """L2-normalize a vector. Returns zero vector if norm is zero."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))
