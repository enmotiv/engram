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


@router.get("/stats/detailed")
async def stats_detailed(
    request: Request,
    owner_id: UUID = Depends(get_owner_id),  # noqa: B008
) -> dict:
    """Detailed per-owner diagnostics: recall activity, dreamer state, plasticity distribution."""
    db = request.app.state.db
    async with tenant_connection(db, owner_id) as conn:
        row = await conn.fetchrow(
            """SELECT
              COUNT(*) FILTER (WHERE is_deleted = FALSE) AS total_nodes,
              COUNT(*) FILTER (WHERE access_count > 0 AND is_deleted = FALSE) AS recalled_nodes,
              MAX(access_count) AS max_access_count,
              ROUND(AVG(access_count) FILTER (WHERE access_count > 0), 1) AS avg_access_count,
              COUNT(*) FILTER (WHERE dreamer_processed = TRUE AND is_deleted = FALSE) AS dreamer_processed,
              COUNT(*) FILTER (WHERE dreamer_processed = FALSE AND is_deleted = FALSE) AS dreamer_pending
            FROM memory_nodes WHERE owner_id = $1""",
            owner_id,
        )
        edge_row = await conn.fetchrow(
            "SELECT COUNT(*) AS edge_count FROM edges WHERE owner_id = $1",
            owner_id,
        )

        # Plasticity distribution (only if column exists)
        plast_row = None
        try:
            plast_row = await conn.fetchrow(
                """SELECT
                  COUNT(*) FILTER (WHERE plasticity <= 0.2) AS hardened,
                  COUNT(*) FILTER (WHERE plasticity > 0.2 AND plasticity <= 0.5) AS moderate,
                  COUNT(*) FILTER (WHERE plasticity > 0.5 AND plasticity <= 0.8) AS flexible,
                  COUNT(*) FILTER (WHERE plasticity > 0.8) AS fresh,
                  ROUND(AVG(plasticity)::numeric, 3) AS avg_plasticity,
                  ROUND(AVG(modification_count)::numeric, 1) AS avg_modification_count
                FROM memory_nodes WHERE owner_id = $1 AND is_deleted = FALSE""",
                owner_id,
            )
        except Exception:
            pass  # Column doesn't exist yet (pre-migration)

        # Trust level breakdown
        trust_row = await conn.fetch(
            """SELECT
              metadata->>'trust_level' AS trust,
              COUNT(*) AS ct
            FROM memory_nodes
            WHERE owner_id = $1 AND is_deleted = FALSE
            GROUP BY metadata->>'trust_level'
            ORDER BY ct DESC""",
            owner_id,
        )

    result = {
        "recall_activity": {
            "total_nodes": row["total_nodes"],
            "nodes_ever_recalled": row["recalled_nodes"],
            "max_access_count": row["max_access_count"],
            "avg_access_count": float(row["avg_access_count"] or 0),
        },
        "dreamer": {
            "processed": row["dreamer_processed"],
            "pending": row["dreamer_pending"],
            "edge_count": edge_row["edge_count"],
        },
        "trust_levels": {
            (r["trust"] or "none"): r["ct"] for r in trust_row
        },
    }

    if plast_row:
        result["plasticity"] = {
            "hardened_le_0.2": plast_row["hardened"],
            "moderate_0.2_0.5": plast_row["moderate"],
            "flexible_0.5_0.8": plast_row["flexible"],
            "fresh_gt_0.8": plast_row["fresh"],
            "avg_plasticity": float(plast_row["avg_plasticity"] or 0),
            "avg_modification_count": float(plast_row["avg_modification_count"] or 0),
        }

    return {"data": result}


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
