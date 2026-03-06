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


class EnhancedMemoryResponse(BaseModel):
    """Extended memory response with optional edges and embedding."""

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
    embedding: Optional[List[float]] = None
    edges: Optional[List[dict]] = None
    edge_count: Optional[int] = None


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
    axis_cues: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Axis-specific query vector override. "
            "Driven by the perceiver's current emotional or cognitive state signal. "
            "NOT derived from relational dimension scores (impact, autonomy, etc.)."
        ),
    )
    exclude_types: Optional[List[str]] = Field(
        default=None,
        description=(
            "Memory types to exclude from results. "
            "Applied after collection, before scoring. "
            "Default: None (include all types)."
        ),
    )
    axis_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Per-axis convergence weight multipliers. "
            "Keys: hippo, amyg, pfc, sensory, striatum, cerebellum. "
            "Missing keys default to 1.0. Range: [0.1, 3.0]. "
            "Driven by capacity state. NOT derived from relational dimension scores."
        ),
    )


# --- Header search / resolve / batch-enrich schemas ---

class HeaderSearchRequest(BaseModel):
    namespace: str
    cue: str
    context: Optional[dict] = None
    max_results: int = 50
    urgency_threshold: float = 0.3


class MemoryHeader(BaseModel):
    id: str
    memory_type: str = "episodic"
    activation: float = 0.0
    convergence_score: float = 0.0
    retrieval_path: str = "direct"
    dimensions_matched: List[str] = Field(default_factory=list)
    enrichment_status: str = "raw"
    vad_summary: Optional[Dict[str, float]] = None
    topic_tags: List[str] = Field(default_factory=list)
    entity_ids: List[str] = Field(default_factory=list)
    dimension_confidence: Dict[str, float] = Field(default_factory=dict)
    salience: float = 0.5
    created_at: Optional[str] = None
    last_accessed: Optional[str] = None
    access_count: int = 0


class HeaderSearchResponse(BaseModel):
    triggered: bool
    urgency: float
    headers: List[MemoryHeader] = Field(default_factory=list)
    suppressed: List[dict] = Field(default_factory=list)
    retrieval_ms: float = 0.0


class ResolveRequest(BaseModel):
    memory_ids: List[str]
    include_edges: bool = True


class ResolvedMemory(BaseModel):
    id: str
    namespace: str
    content: str
    memory_type: str
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    features: Optional[Dict] = Field(default_factory=dict)
    activation: float = 0.0
    salience: float = 0.5
    access_count: int = 0
    created_at: Optional[str] = None
    last_accessed: Optional[str] = None
    edges: List[dict] = Field(default_factory=list)


class ResolveResponse(BaseModel):
    memories: List[ResolvedMemory] = Field(default_factory=list)
    not_found: List[str] = Field(default_factory=list)


class BatchEnrichItem(BaseModel):
    memory_id: str
    metadata: Optional[Dict] = None
    features: Optional[Dict] = None


class BatchEnrichRequest(BaseModel):
    updates: List[BatchEnrichItem]


class BatchEnrichResponse(BaseModel):
    updated: List[str] = Field(default_factory=list)
    failed: List[str] = Field(default_factory=list)


# --- Snippet (Layer 1) schemas ---

class SnippetRequest(BaseModel):
    ids: List[str]


class EdgeSummary(BaseModel):
    excitatory_count: int = 0
    inhibitory_count: int = 0
    associative_count: int = 0


class SnippetMemoryResponse(BaseModel):
    id: str
    created_at: Optional[str] = None
    namespace: str
    memory_type: str = "episodic"
    content: str  # first 200 chars, truncated with "..."
    top_dimensions: Dict[str, float] = Field(default_factory=dict)  # top 5 by score
    entity_ids: List[str] = Field(default_factory=list)
    edge_summary: EdgeSummary = Field(default_factory=EdgeSummary)
    texture_summary: Optional[str] = None
    enrichment_status: str = "raw"


class SnippetResponse(BaseModel):
    snippets: List[SnippetMemoryResponse] = Field(default_factory=list)


# --- Full batch (Layer 2) schemas ---

class FullBatchRequest(BaseModel):
    ids: List[str]


class LinkedMemorySnippet(BaseModel):
    id: str
    content: str  # first 100 chars


class FullEdge(BaseModel):
    id: str
    edge_type: str
    target_id: str
    weight: float
    linked_memory: Optional[LinkedMemorySnippet] = None


class FullMemoryResponse(BaseModel):
    id: str
    namespace: str
    content: str
    memory_type: str = "episodic"
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    features: Optional[Dict] = Field(default_factory=dict)
    activation: float = 0.0
    salience: float = 0.5
    access_count: int = 0
    created_at: Optional[str] = None
    last_accessed: Optional[str] = None
    edges: List[FullEdge] = Field(default_factory=list)


class FullBatchResponse(BaseModel):
    memories: List[FullMemoryResponse] = Field(default_factory=list)


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
