"""Pydantic models for retrieval pipeline."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RetrievalOptions(BaseModel):
    max_results: int = 5
    urgency_threshold: float = 0.3
    hop_depth: int = 2
    trace_format: str = "graph"
    reconsolidate: bool = True


class MemoryResult(BaseModel):
    id: str
    content: str
    activation: float = 0.0
    dimensions_matched: List[str] = Field(default_factory=list)
    convergence_score: float = 0.0
    retrieval_path: str = "direct"  # "direct" or "spreading"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dimension_scores: Dict[str, float] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    triggered: bool
    urgency: float
    memories: List[MemoryResult] = Field(default_factory=list)
    suppressed: List[dict] = Field(default_factory=list)
    trace: Optional[str] = None
    retrieval_ms: float = 0.0


class MemoryHeaderResult(BaseModel):
    """Layer 0 header — metadata only, no content."""

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


class HeaderSearchResult(BaseModel):
    """Result of a header-only search (Layer 0)."""

    triggered: bool
    urgency: float
    headers: List[MemoryHeaderResult] = Field(default_factory=list)
    suppressed: List[dict] = Field(default_factory=list)
    retrieval_ms: float = 0.0
    graph_content: Optional[str] = None  # mermaid from latest graph snapshot
