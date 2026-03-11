"""Recall request/response models."""

from pydantic import BaseModel, Field

from engram.models.common import ConfidenceLevel
from engram.models.edge import EdgeResponse
from engram.models.memory import MemoryResponse


class RecallRequest(BaseModel):
    cue: str = Field(..., max_length=4096, min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    min_convergence: float = Field(default=0.0, ge=0.0)
    include_edges: bool = False
    metadata_type: str | None = Field(
        default=None,
        description="Filter to only return nodes with this metadata.type value. "
        "Overrides ENGRAM_RETRIEVAL_EXCLUDE_TAGS when set.",
    )


class RecallResponse(BaseModel):
    memories: list[MemoryResponse]
    confidence: ConfidenceLevel
    edges: list[EdgeResponse] = Field(default_factory=list)
