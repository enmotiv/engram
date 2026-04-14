"""Memory CRUD endpoints: create, get, list, delete."""

import json
from datetime import datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from engram.auth import get_owner_id
from engram.core.db import tenant_connection
from engram.core.errors import EngramError
from engram.models import CreateMemoryRequest, SourceType
from engram.repositories import memory_repo
from engram.services.write import encode_memory

logger = structlog.get_logger()

router = APIRouter(prefix="/v1")

# Cursor pagination is designed for created_at sorts. When using
# activation_level:desc, cursor behavior is approximate — the cursor
# still filters on created_at, so pages may overlap or skip rows
# whose activation changed between requests.
_VALID_SORTS = {
    "created_at:desc": "created_at DESC",
    "created_at:asc": "created_at ASC",
    "activation_level:desc": "activation_level DESC, created_at DESC",
}


# --- POST /v1/memories ---


@router.post("/memories", status_code=201)
async def create_memory(
    body: CreateMemoryRequest,
    request: Request,
    owner_id: UUID = Depends(get_owner_id),  # noqa: B008
    upsert: bool = Query(default=False),
) -> dict:
    try:
        result = await encode_memory(
            db_pool=request.app.state.db,
            owner_id=owner_id,
            content=body.content,
            source_type=body.source_type,
            session_id=body.session_id,
            metadata=body.metadata,
            metadata_type=body.metadata_type,
            upsert=upsert,
            initial_activation=body.initial_activation,
        )
    except ValueError:
        raise EngramError(
            "INVALID_INPUT", "Embedding validation failed", 400
        ) from None

    if result.get("duplicate"):
        raise EngramError(
            "DUPLICATE_CONTENT",
            "Content already exists for this owner.",
            409,
            existing_id=result["existing_id"],
        )

    if result.get("updated"):
        return JSONResponse(content={"data": result}, status_code=200)

    return {"data": result}


# --- GET /v1/memories/:id ---


@router.get("/memories/{memory_id}")
async def get_memory(
    memory_id: UUID,
    request: Request,
    owner_id: UUID = Depends(get_owner_id),  # noqa: B008
) -> dict:
    db = request.app.state.db
    async with tenant_connection(db, owner_id) as conn:
        row = await memory_repo.fetch_single_memory(conn, memory_id, owner_id)

    if not row:
        raise EngramError("NOT_FOUND", "Memory not found", 404)

    return {"data": _row_to_dict(row)}


# --- GET /v1/memories ---


@router.get("/memories")
async def list_memories(
    request: Request,
    owner_id: UUID = Depends(get_owner_id),  # noqa: B008
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    source_type: str | None = Query(default=None),
    metadata_type: str | None = Query(default=None, max_length=50),
    sort: str = Query(default="created_at:desc"),
) -> dict:
    if sort not in _VALID_SORTS:
        raise EngramError("INVALID_INPUT", f"Invalid sort: {sort}", 400)

    # Parse cursor
    cursor_dt: datetime | None = None
    if cursor:
        try:
            cursor_dt = datetime.fromisoformat(cursor)
        except ValueError:
            raise EngramError(
                "INVALID_INPUT", "Invalid cursor format (ISO 8601 required)", 400
            ) from None

    # Parse source_type filter
    source_types: list[str] | None = None
    if source_type:
        types = [t.strip() for t in source_type.split(",")]
        valid = {st.value for st in SourceType}
        invalid = set(types) - valid
        if invalid:
            raise EngramError(
                "INVALID_INPUT",
                f"Invalid source_type: {', '.join(sorted(invalid))}",
                400,
            )
        source_types = types

    cursor_direction = "after" if sort == "created_at:asc" else "before"

    db = request.app.state.db
    async with tenant_connection(db, owner_id) as conn:
        rows = await memory_repo.list_memories_paginated(
            conn,
            owner_id,
            limit=limit + 1,  # fetch one extra to detect next page
            order_sql=_VALID_SORTS[sort],
            cursor_dt=cursor_dt,
            cursor_direction=cursor_direction,
            source_types=source_types,
            metadata_type=metadata_type,
        )

    has_more = len(rows) > limit
    rows = rows[:limit]

    memories = [_row_to_dict(row) for row in rows]

    next_cursor = None
    if has_more and rows:
        next_cursor = rows[-1]["created_at"].isoformat()

    return {"data": {"memories": memories, "next_cursor": next_cursor}}


# --- PATCH /v1/memories/:id ---


@router.patch("/memories/{memory_id}")
async def patch_memory(
    memory_id: UUID,
    request: Request,
    owner_id: UUID = Depends(get_owner_id),  # noqa: B008
) -> dict:
    """Merge metadata into an existing memory. Top-level keys are overwritten."""
    body = await request.json()
    metadata = body.get("metadata")
    source_type = body.get("source_type")

    if not metadata and not source_type:
        raise EngramError("INVALID_INPUT", "Nothing to update", 400)

    db = request.app.state.db
    async with tenant_connection(db, owner_id) as conn:
        updated = await memory_repo.upsert_metadata(
            conn, memory_id, owner_id,
            metadata or {},
            source_type=source_type,
        )

    if not updated:
        raise EngramError("NOT_FOUND", "Memory not found", 404)

    return {"data": {"id": str(memory_id), "updated": True}}


@router.patch("/memories/by-meta/{meta_key}/{meta_value}")
async def patch_memory_by_meta(
    meta_key: str,
    meta_value: str,
    request: Request,
    owner_id: UUID = Depends(get_owner_id),  # noqa: B008
) -> dict:
    """Find memory by metadata key/value, then merge metadata updates.

    Used by Enmotiv to sync trust_level and enrichment by enmotiv_id
    without needing to track Engram's internal UUIDs.
    """
    body = await request.json()
    metadata = body.get("metadata")
    source_type = body.get("source_type")

    if not metadata and not source_type:
        raise EngramError("INVALID_INPUT", "Nothing to update", 400)

    if meta_key not in ("enmotiv_id", "enmotiv_type"):
        raise EngramError(
            "INVALID_INPUT",
            f"Lookup by metadata.{meta_key} not supported",
            400,
        )

    db = request.app.state.db
    async with tenant_connection(db, owner_id) as conn:
        memory_id = await memory_repo.find_by_metadata_value(
            conn, owner_id, meta_key, meta_value
        )
        if not memory_id:
            raise EngramError("NOT_FOUND", f"No memory with {meta_key}={meta_value}", 404)

        await memory_repo.upsert_metadata(
            conn, memory_id, owner_id,
            metadata or {},
            source_type=source_type,
        )

    return {"data": {"id": str(memory_id), "updated": True}}


# --- DELETE /v1/memories/:id ---


@router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: UUID,
    request: Request,
    owner_id: UUID = Depends(get_owner_id),  # noqa: B008
) -> dict:
    db = request.app.state.db
    async with tenant_connection(db, owner_id) as conn:
        count = await memory_repo.soft_delete(conn, memory_id, owner_id)

    if count == 0:
        raise EngramError("NOT_FOUND", "Memory not found", 404)

    return {"data": {"id": str(memory_id), "deleted": True}}


# --- Helpers ---


def _row_to_dict(row) -> dict:  # noqa: ANN001
    """Convert an asyncpg Record to a serializable dict."""
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return {
        "id": str(row["id"]),
        "content": row["content"],
        "content_hash": row["content_hash"],
        "activation_level": row["activation_level"],
        "salience": row["salience"],
        "source_type": row["source_type"],
        "session_id": str(row["session_id"]) if row["session_id"] else None,
        "metadata": meta,
        "metadata_type": row.get("metadata_type"),
        "created_at": row["created_at"].isoformat(),
        "last_accessed": row["last_accessed"].isoformat(),
        "access_count": row["access_count"],
    }
