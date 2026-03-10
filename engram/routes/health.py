"""Health check and per-owner stats endpoints."""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from engram.auth import get_owner_id
from engram.db import tenant_connection
from engram.embeddings import get_client
from engram.tracing import AVG_ACTIVATION, EDGES_TOTAL, NODES_TOTAL

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
        row = await conn.fetchrow(
            "SELECT "
            "  COUNT(*) FILTER (WHERE is_deleted = FALSE) AS node_count, "
            "  AVG(activation_level) FILTER (WHERE is_deleted = FALSE) "
            "    AS avg_activation, "
            "  COUNT(*) FILTER "
            "    (WHERE dreamer_processed = FALSE AND is_deleted = FALSE) "
            "    AS unprocessed_nodes "
            "FROM memory_nodes "
            "WHERE owner_id = $1",
            owner_id,
        )
        edge_count = await conn.fetchval(
            "SELECT COUNT(*) FROM edges WHERE owner_id = $1",
            owner_id,
        )

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
