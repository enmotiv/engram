"""Shared enums, helpers, and base types."""

from __future__ import annotations

import hashlib
import secrets
from enum import StrEnum
from uuid import UUID

from uuid_extensions import uuid7


class SourceType(StrEnum):
    CONVERSATION = "conversation"
    EVENT = "event"
    OBSERVATION = "observation"
    CORRECTION = "correction"
    SYSTEM = "system"


class AxisName(StrEnum):
    TEMPORAL = "temporal"
    EMOTIONAL = "emotional"
    SEMANTIC = "semantic"
    SENSORY = "sensory"
    ACTION = "action"
    PROCEDURAL = "procedural"


class EdgeType(StrEnum):
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    ASSOCIATIVE = "associative"
    TEMPORAL = "temporal"
    MODULATORY = "modulatory"
    STRUCTURAL = "structural"


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ErrorResponse:
    code: str
    message: str
    status: int
    correlation_id: str
    existing_id: str | None = None


def generate_node_id() -> UUID:
    """Generate a time-sortable UUID v7 for memory node IDs."""
    return uuid7()


def generate_correlation_id() -> str:
    """Generate a correlation ID for request tracing: req_ + 12 hex chars."""
    return f"req_{secrets.token_hex(6)}"


def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_confidence(
    top_result: dict[str, float],
) -> ConfidenceLevel:
    dims = top_result.get("dims_matched", 0)
    conv = top_result.get("convergence_score", 0.0)
    if dims >= 4 and conv >= 3.0:
        return ConfidenceLevel.HIGH
    elif dims >= 2 or conv >= 1.5:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW
