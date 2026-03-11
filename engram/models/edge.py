"""Edge models."""

from pydantic import BaseModel, Field

from engram.models.common import AxisName, EdgeType


class EdgeResponse(BaseModel):
    source_id: str
    target_id: str
    edge_type: EdgeType
    axis: AxisName
    weight: float = Field(..., ge=0.0, le=1.0)
