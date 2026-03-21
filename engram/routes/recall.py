"""POST /v1/recall endpoint."""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Query, Request

from engram.auth import get_owner_id
from engram.models import RecallRequest
from engram.services.read import recall_memories
from engram.core.tracing import TraceCollector, set_trace

logger = structlog.get_logger()

router = APIRouter(prefix="/v1")


@router.post("/recall")
async def recall(
    body: RecallRequest,
    request: Request,
    owner_id: UUID = Depends(get_owner_id),  # noqa: B008
    trace: bool = Query(default=False),
) -> dict:
    tc = None
    if trace:
        tc = TraceCollector(body.cue)
        set_trace(tc)

    try:
        result = await recall_memories(
            db_pool=request.app.state.db,
            owner_id=owner_id,
            cue=body.cue,
            top_k=body.top_k,
            min_convergence=body.min_convergence,
            include_edges=body.include_edges,
            metadata_type=body.metadata_type,
            session_id=body.session_id,
            axis_weights=body.axis_weights,
            time_window_hours=body.time_window_hours,
            source_types=body.source_types,
            settle=body.settle,
        )
    finally:
        set_trace(None)

    if trace and tc:
        data = result.model_dump(mode="json")
        data["trace"] = tc.finish()
        return {"data": data}

    return {"data": result}
