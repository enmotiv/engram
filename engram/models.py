"""Pydantic models for Engram API contracts."""

import hashlib
import secrets
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
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


DIMENSION_PREFIXES: dict[str, str] = {
    "temporal": ("Encode the temporal and sequential context of this experience: "),
    "emotional": (
        "Encode the emotional tone, urgency, and significance of this experience: "
    ),
    "semantic": ("Encode the semantic meaning and core concepts of this experience: "),
    "sensory": (
        "Encode the specific factual details, names,"
        " numbers, and identifiers in this experience: "
    ),
    "action": (
        "Encode the actions, goals, and behavioral context of this experience: "
    ),
    "procedural": (
        "Encode any repeated patterns, routines, or"
        " procedural sequences in this experience: "
    ),
}

AXES = list(DIMENSION_PREFIXES.keys())


# --- Request Models ---


class CreateMemoryRequest(BaseModel):
    content: str = Field(..., max_length=4096, min_length=1)
    source_type: SourceType
    session_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallRequest(BaseModel):
    cue: str = Field(..., max_length=4096, min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    min_convergence: float = Field(default=0.0, ge=0.0)
    include_edges: bool = False


# --- Internal Models ---


class MemoryNode(BaseModel):
    """Database row representation excluding vector columns."""

    id: UUID
    owner_id: UUID
    content: str = Field(..., min_length=1, max_length=4096)
    content_hash: str = Field(..., min_length=64, max_length=64)
    created_at: datetime
    last_accessed: datetime
    access_count: int = Field(default=0, ge=0)
    activation_level: float = Field(default=1.0, ge=0.0, le=1.0)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    source_type: SourceType
    session_id: UUID | None = None
    embedding_model: str = Field(..., max_length=100)
    embedding_dimensions: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_deleted: bool = False
    dreamer_processed: bool = False

    def to_response(
        self,
        convergence_score: float,
        dimension_scores: "DimensionScores",
        matched_axes: list[str],
    ) -> "MemoryResponse":
        """Convert to API response, adding retrieval-time fields."""
        return MemoryResponse(
            id=str(self.id),
            content=self.content,
            content_hash=self.content_hash,
            convergence_score=convergence_score,
            activation_level=self.activation_level,
            dimension_scores=dimension_scores,
            matched_axes=matched_axes,
            metadata=self.metadata,
            salience=self.salience,
            created_at=self.created_at,
            last_accessed=self.last_accessed,
            access_count=self.access_count,
            session_id=(str(self.session_id) if self.session_id is not None else None),
            source_type=self.source_type,
        )


# --- Response Models ---


class DimensionScores(BaseModel):
    temporal: float = 0.0
    emotional: float = 0.0
    semantic: float = 0.0
    sensory: float = 0.0
    action: float = 0.0
    procedural: float = 0.0


class MemoryResponse(BaseModel):
    id: str
    content: str
    content_hash: str
    convergence_score: float
    activation_level: float
    dimension_scores: DimensionScores
    matched_axes: list[str]
    metadata: dict[str, Any]
    salience: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    session_id: str | None = None
    source_type: SourceType


class EdgeResponse(BaseModel):
    source_id: str
    target_id: str
    edge_type: EdgeType
    axis: AxisName
    weight: float = Field(..., ge=0.0, le=1.0)


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecallResponse(BaseModel):
    memories: list[MemoryResponse]
    confidence: ConfidenceLevel
    edges: list[EdgeResponse] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    code: str
    message: str
    status: int
    correlation_id: str
    existing_id: str | None = None


# --- Helpers ---


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
