"""Memory CRUD endpoints: create, get, list, delete."""

import json
from datetime import datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Query, Request

from engram.auth import get_owner_id
from engram.db import tenant_connection
from engram.errors import EngramError
from engram.models import CreateMemoryRequest, SourceType
from engram.write_path import encode_memory

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
) -> dict:
    try:
        result = await encode_memory(
            db_pool=request.app.state.db,
            owner_id=owner_id,
            content=body.content,
            source_type=body.source_type,
            session_id=body.session_id,
            metadata=body.metadata,
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
        row = await conn.fetchrow(
            "SELECT id, content, content_hash, created_at, last_accessed, "
            "access_count, activation_level, salience, source_type, "
            "session_id, metadata "
            "FROM memory_nodes "
            "WHERE id = $1 AND owner_id = $2 AND is_deleted = FALSE",
            memory_id,
            owner_id,
        )

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

    # Build query dynamically
    conditions = ["owner_id = $1", "is_deleted = FALSE"]
    params: list = [owner_id]
    idx = 2

    if cursor_dt:
        if sort == "created_at:asc":
            conditions.append(f"created_at > ${idx}")
        else:
            conditions.append(f"created_at < ${idx}")
        params.append(cursor_dt)
        idx += 1

    if source_types:
        conditions.append(f"source_type = ANY(${idx})")
        params.append(source_types)
        idx += 1

    # Fetch limit + 1 to detect next page
    params.append(limit + 1)

    where = " AND ".join(conditions)
    order = _VALID_SORTS[sort]

    db = request.app.state.db
    async with tenant_connection(db, owner_id) as conn:
        rows = await conn.fetch(
            f"SELECT id, content, content_hash, created_at, last_accessed, "  # noqa: S608
            f"access_count, activation_level, salience, source_type, "
            f"session_id, metadata "
            f"FROM memory_nodes "
            f"WHERE {where} "
            f"ORDER BY {order} "
            f"LIMIT ${idx}",
            *params,
        )

    has_more = len(rows) > limit
    rows = rows[:limit]

    memories = [_row_to_dict(row) for row in rows]

    next_cursor = None
    if has_more and rows:
        next_cursor = rows[-1]["created_at"].isoformat()

    return {"data": {"memories": memories, "next_cursor": next_cursor}}


# --- DELETE /v1/memories/:id ---


@router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: UUID,
    request: Request,
    owner_id: UUID = Depends(get_owner_id),  # noqa: B008
) -> dict:
    db = request.app.state.db
    async with tenant_connection(db, owner_id) as conn:
        result = await conn.execute(
            "UPDATE memory_nodes SET is_deleted = TRUE "
            "WHERE id = $1 AND owner_id = $2 AND is_deleted = FALSE",
            memory_id,
            owner_id,
        )

    count = _parse_update_count(result)
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
        "created_at": row["created_at"].isoformat(),
        "last_accessed": row["last_accessed"].isoformat(),
        "access_count": row["access_count"],
    }


def _parse_update_count(result: str) -> int:
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0
