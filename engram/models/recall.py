"""Recall request/response models."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from engram.models.common import ConfidenceLevel, SourceType
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
    # Phase 1: Context-scaffolded retrieval
    session_id: UUID | None = Field(
        default=None, description="Filter to memories from this session"
    )
    axis_weights: dict[str, float] | None = Field(
        default=None,
        description="Per-axis importance multipliers, e.g. {'emotional': 2.0, 'semantic': 0.5}",
    )
    time_window_hours: int | None = Field(
        default=None, ge=1, description="Only memories from last N hours"
    )
    source_types: list[SourceType] | None = Field(
        default=None, description="Filter by source types"
    )
    # Phase 3: Attractor dynamics
    settle: bool = Field(
        default=False, description="Enable iterative attractor settling for deeper recall"
    )
    # Phase 5: STDP sequence mode
    sequence_mode: bool = Field(
        default=False, description="Follow directional chains from top seed (requires STDP flag)"
    )


class RecallResponse(BaseModel):
    memories: list[MemoryResponse]
    confidence: ConfidenceLevel
    edges: list[EdgeResponse] = Field(default_factory=list)
    chain_confidence: float | None = Field(
        default=None,
        description="Minimum forward weight in the sequence chain (only set in sequence_mode)",
    )
