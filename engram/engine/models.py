"""Pydantic models for retrieval pipeline."""

from typing import Dict, List, Optional

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


class RetrievalResult(BaseModel):
    triggered: bool
    urgency: float
    memories: List[MemoryResult] = Field(default_factory=list)
    suppressed: List[dict] = Field(default_factory=list)
    trace: Optional[str] = None
    retrieval_ms: float = 0.0
