"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from engram.db.models import VALID_EDGE_TYPES


# --- Memory schemas ---

class CreateMemoryRequest(BaseModel):
    id: Optional[str] = None  # Client-provided UUID; auto-generated if omitted
    namespace: str
    content: str
    memory_type: str = "episodic"
    metadata: Optional[dict] = None


class BatchCreateRequest(BaseModel):
    memories: List[CreateMemoryRequest]


class MemoryResponse(BaseModel):
    id: str
    namespace: str
    content: str
    memory_type: str
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    features: Optional[Dict] = Field(default_factory=dict)
    metadata: Optional[Dict] = Field(default_factory=dict)
    activation: float = 0.0
    salience: float = 0.5
    access_count: int = 0
    created_at: Optional[str] = None


class UpdateMemoryRequest(BaseModel):
    metadata: Optional[dict] = None
    features: Optional[dict] = None


class RetrieveRequest(BaseModel):
    namespace: str
    cue: str
    context: Optional[dict] = None
    max_results: int = 5
    urgency_threshold: float = 0.3
    hop_depth: int = 2


# --- Edge schemas ---

class CreateEdgeRequest(BaseModel):
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 0.5
    context: Optional[dict] = None
    namespace: str = "default"


class UpdateEdgeRequest(BaseModel):
    weight: float


class EdgeResponse(BaseModel):
    id: str
    source_memory_id: str
    target_memory_id: str
    edge_type: str
    weight: float
    namespace: str
    created_at: Optional[str] = None


# --- Namespace schemas ---

class NamespaceStats(BaseModel):
    namespace: str
    memory_count: int
    edge_count: int
    last_activity: Optional[str] = None
    dimension_coverage: Dict[str, int] = Field(default_factory=dict)


# --- Health ---

class HealthResponse(BaseModel):
    status: str = "ok"
    db: bool = False
    redis: bool = False
    plugin: str = ""
