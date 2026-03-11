"""Pydantic models for Engram API contracts.

Re-exports everything for backward compatibility:
  from engram.models import CreateMemoryRequest, AXES, etc.
"""

from engram.models.common import (
    AxisName,
    ConfidenceLevel,
    EdgeType,
    ErrorResponse,
    SourceType,
    compute_confidence,
    compute_content_hash,
    generate_correlation_id,
    generate_node_id,
)
from engram.models.edge import EdgeResponse
from engram.models.memory import (
    AXES,
    DIMENSION_PREFIXES,
    CreateMemoryRequest,
    DimensionScores,
    MemoryNode,
    MemoryResponse,
)
from engram.models.recall import RecallRequest, RecallResponse

__all__ = [
    "AXES",
    "AxisName",
    "ConfidenceLevel",
    "CreateMemoryRequest",
    "DIMENSION_PREFIXES",
    "DimensionScores",
    "EdgeResponse",
    "EdgeType",
    "ErrorResponse",
    "MemoryNode",
    "MemoryResponse",
    "RecallRequest",
    "RecallResponse",
    "SourceType",
    "compute_confidence",
    "compute_content_hash",
    "generate_correlation_id",
    "generate_node_id",
]
