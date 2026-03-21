"""Admin endpoints: API key management, settings health, Dreamer triggers."""

from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field

from engram.config import settings
from engram.core.errors import EngramError

logger = structlog.get_logger()

router = APIRouter(prefix="/v1/admin")


def _verify_admin(secret: str) -> None:
    """Verify admin secret. Raises EngramError if invalid."""
    if not settings.engram_admin_secret:
        raise EngramError(
            "UNAUTHORIZED",
            "Admin secret not configured. Set ENGRAM_ADMIN_SECRET.",
            401,
        )
    if secret != settings.engram_admin_secret:
        raise EngramError("UNAUTHORIZED", "Invalid admin secret", 401)


# --- Provision (owner + key in one call) ---


class ProvisionRequest(BaseModel):
    owner_id: UUID
    label: str = ""


@router.post("/provision", status_code=201)
async def provision_owner(
    body: ProvisionRequest,
    request: Request,
    x_admin_secret: str = Header(...),
) -> dict:
    """Create owner + API key in one call. Idempotent for owner (ON CONFLICT)."""
    _verify_admin(x_admin_secret)

    from engram.repositories.owner_repo import create_api_key, create_owner

    db = request.app.state.db
    async with db.acquire() as conn:
        await create_owner(conn, owner_id=body.owner_id, label=body.label)
        raw_key, key_id = await create_api_key(
            conn, owner_id=body.owner_id, label=body.label,
        )

    return {
        "data": {
            "owner_id": str(body.owner_id),
            "api_key": raw_key,
            "key_id": str(key_id),
        }
    }


# --- API Key Management ---


class CreateKeyRequest(BaseModel):
    owner_id: UUID
    label: str = ""
    rate_limit_reads: int = Field(default=100, ge=1)
    rate_limit_writes: int = Field(default=50, ge=1)


@router.post("/keys", status_code=201)
async def create_api_key(
    body: CreateKeyRequest,
    request: Request,
    x_admin_secret: str = Header(...),
) -> dict:
    """Create a new API key for an owner. Returns the raw key ONCE."""
    _verify_admin(x_admin_secret)

    from engram.repositories.owner_repo import create_api_key

    db = request.app.state.db
    async with db.acquire() as conn:
        raw_key, key_id = await create_api_key(
            conn,
            owner_id=body.owner_id,
            label=body.label,
            rate_limit_reads=body.rate_limit_reads,
            rate_limit_writes=body.rate_limit_writes,
        )

    return {
        "data": {
            "api_key": raw_key,
            "key_id": str(key_id),
            "owner_id": str(body.owner_id),
            "label": body.label,
        }
    }


@router.get("/keys/{owner_id}")
async def list_api_keys(
    owner_id: UUID,
    request: Request,
    x_admin_secret: str = Header(...),
) -> dict:
    """List API keys for an owner (metadata only, never the raw key)."""
    _verify_admin(x_admin_secret)

    from engram.repositories.owner_repo import list_api_keys

    db = request.app.state.db
    async with db.acquire() as conn:
        rows = await list_api_keys(conn, owner_id)

    keys = [
        {
            "id": str(row["id"]),
            "label": row["label"],
            "rate_limit_reads": row["rate_limit_reads"],
            "rate_limit_writes": row["rate_limit_writes"],
            "created_at": row["created_at"].isoformat(),
            "revoked": row["revoked_at"] is not None,
        }
        for row in rows
    ]
    return {"data": {"keys": keys}}


@router.delete("/keys/{owner_id}/{key_id}")
async def revoke_api_key(
    owner_id: UUID,
    key_id: UUID,
    request: Request,
    x_admin_secret: str = Header(...),
) -> dict:
    """Revoke an API key."""
    _verify_admin(x_admin_secret)

    from engram.repositories.owner_repo import revoke_api_key

    db = request.app.state.db
    async with db.acquire() as conn:
        revoked = await revoke_api_key(conn, key_id, owner_id)

    if not revoked:
        raise EngramError("NOT_FOUND", "API key not found or already revoked", 404)

    return {"data": {"key_id": str(key_id), "revoked": True}}


# --- Settings Health ---


@router.get("/settings")
async def settings_health(
    request: Request,
    x_admin_secret: str = Header(...),
) -> dict:
    """Show which settings are configured (without revealing values)."""
    _verify_admin(x_admin_secret)

    return {
        "data": {
            "openrouter_api_key": "configured" if settings.openrouter_api_key else "missing",
            "openrouter_base_url": settings.openrouter_base_url,
            "database_url": "configured" if settings.database_url else "missing",
            "redis_url": "configured" if settings.redis_url else "missing",
            "engram_admin_secret": "configured" if settings.engram_admin_secret else "not set",
            "engram_embedding_model": settings.engram_embedding_model,
            "engram_embedding_dimensions": settings.engram_embedding_dimensions,
            "engram_llm_model": settings.engram_llm_model,
            "engram_log_level": settings.engram_log_level,
            "engram_extensions": settings.engram_extensions or "(none)",
            "engram_retrieval_exclude_tags": settings.engram_retrieval_exclude_tags or "(none)",
        }
    }
