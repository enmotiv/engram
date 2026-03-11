"""Owner and API key repository."""

from __future__ import annotations

import secrets
from uuid import UUID

import asyncpg

from engram.utilities.hashing import hash_api_key


async def create_api_key(
    conn: asyncpg.Connection,
    owner_id: UUID,
    label: str = "",
    rate_limit_reads: int = 100,
    rate_limit_writes: int = 50,
) -> tuple[str, UUID]:
    """Create a new API key for an owner. Returns (raw_key, key_id).

    The raw key is returned ONCE — it's never stored or retrievable after this.
    """
    raw_key = f"engram-v2-{secrets.token_hex(24)}"
    key_hash = hash_api_key(raw_key)

    key_id = await conn.fetchval(
        "INSERT INTO api_keys (owner_id, key_hash, label, rate_limit_reads, rate_limit_writes) "
        "VALUES ($1, $2, $3, $4, $5) RETURNING id",
        owner_id,
        key_hash,
        label,
        rate_limit_reads,
        rate_limit_writes,
    )

    return raw_key, key_id


async def revoke_api_key(
    conn: asyncpg.Connection,
    key_id: UUID,
    owner_id: UUID,
) -> bool:
    """Revoke an API key. Returns True if found and revoked."""
    result = await conn.execute(
        "UPDATE api_keys SET revoked_at = NOW() "
        "WHERE id = $1 AND owner_id = $2 AND revoked_at IS NULL",
        key_id,
        owner_id,
    )
    return _parse_count(result) > 0


async def list_api_keys(
    conn: asyncpg.Connection,
    owner_id: UUID,
) -> list[asyncpg.Record]:
    """List API keys for an owner (metadata only, never the hash)."""
    return await conn.fetch(
        "SELECT id, label, rate_limit_reads, rate_limit_writes, "
        "created_at, revoked_at "
        "FROM api_keys WHERE owner_id = $1 ORDER BY created_at DESC",
        owner_id,
    )


async def get_all_owner_ids(
    conn: asyncpg.Connection,
) -> list[UUID]:
    """Fetch all owner IDs."""
    rows = await conn.fetch("SELECT id FROM owners")
    return [row["id"] for row in rows]


def _parse_count(result: str) -> int:
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0
