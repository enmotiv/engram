"""Health check and per-owner stats endpoints."""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from engram.auth import get_owner_id
from engram.core.db import tenant_connection
from engram.core.tracing import AVG_ACTIVATION, EDGES_TOTAL, NODES_TOTAL
from engram.repositories import edge_repo, memory_repo
from engram.services.embedding import get_client

logger = structlog.get_logger()

router = APIRouter(prefix="/v1")


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """DB + OpenRouter connectivity check. No auth required."""
    db_status = "connected"
    openrouter_status = "reachable"
    overall = "healthy"

    # Check DB
    try:
        async with request.app.state.db.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception:
        db_status = "disconnected"
        overall = "unhealthy"

    # Check OpenRouter (lightweight models list)
    try:
        client = get_client()
        await client.models.list()
    except Exception:
        openrouter_status = "unreachable"
        if overall == "healthy":
            overall = "degraded"

    status_code = 200 if overall == "healthy" else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall,
            "db": db_status,
            "openrouter": openrouter_status,
        },
    )


@router.get("/stats")
async def stats(
    request: Request,
    owner_id: UUID = Depends(get_owner_id),  # noqa: B008
) -> dict:
    """Per-owner node/edge statistics."""
    db = request.app.state.db
    async with tenant_connection(db, owner_id) as conn:
        row = await memory_repo.fetch_owner_stats(conn, owner_id)
        edge_count = await edge_repo.count_all(conn, owner_id)

    node_count = row["node_count"]
    avg_act = float(row["avg_activation"] or 0)
    oid = str(owner_id)

    # Update Prometheus gauges
    NODES_TOTAL.labels(owner_id=oid).set(node_count)
    EDGES_TOTAL.labels(owner_id=oid).set(edge_count)
    AVG_ACTIVATION.labels(owner_id=oid).set(avg_act)

    return {
        "data": {
            "node_count": node_count,
            "edge_count": edge_count,
            "avg_activation": round(avg_act, 4),
            "unprocessed_nodes": row["unprocessed_nodes"],
        }
    }


@router.post("/admin/dreamer/run")
async def trigger_dreamer(
    request: Request,
    owner_id: UUID = Depends(get_owner_id),  # noqa: B008
) -> dict:
    """Manually trigger a full Dreamer cycle for this owner."""
    from engram.services.dreamer import run_cycle

    result = await run_cycle(request.app.state.db, owner_id)
    return {"data": result}


@router.post("/admin/dreamer/classify")
async def trigger_classification(
    request: Request,
    owner_id: UUID = Depends(get_owner_id),  # noqa: B008
) -> dict:
    """Manually trigger edge classification for unprocessed memories."""
    from engram.services.dreamer import process_new_memories

    result = await process_new_memories(request.app.state.db, owner_id)
    return {"data": result}
