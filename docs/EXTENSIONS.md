# Extensions

Engram supports a plugin system for adding routes, background jobs, and custom functionality.

## Extension Contract

An extension is a Python module with a `register` async function:

```python
async def register(app: FastAPI, db: asyncpg.Pool) -> None:
    """Called once at startup. Add routes, state, or background tasks."""
```

## Configuration

Extensions are loaded via the `ENGRAM_EXTENSIONS` environment variable:

```bash
ENGRAM_EXTENSIONS=ext_dashboard,ext_analytics
```

Comma-separated list of importable Python module names. Each module must have a `register(app, db)` function.

## Rules

1. **Route namespace**: Extensions must use `/ext/` prefix for all routes to avoid conflicts with core `/v1/` endpoints.
2. **No core table writes**: Extensions should not write to `memory_nodes` or `edges` directly. Use the write path and Dreamer APIs.
3. **Auth**: Extensions should use `Depends(get_owner_id)` for authenticated endpoints.
4. **Fail fast**: If an extension fails to load at startup, the application refuses to start.
5. **No blocking**: Extension registration must be async and should complete quickly.

## Example: Dashboard Extension

```python
# ext_dashboard.py
"""Dashboard extension providing read-only analytics."""

from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, FastAPI

from engram.auth import get_owner_id
from engram.db import tenant_connection

router = APIRouter(prefix="/ext/dashboard")


@router.get("/activity")
async def activity_summary(
    owner_id: UUID = Depends(get_owner_id),
    db: asyncpg.Pool = None,
) -> dict:
    async with tenant_connection(db, owner_id) as conn:
        recent = await conn.fetchval(
            "SELECT COUNT(*) FROM memory_nodes "
            "WHERE owner_id = $1 AND is_deleted = FALSE "
            "AND created_at > NOW() - INTERVAL '7 days'",
            owner_id,
        )
    return {"data": {"memories_last_7_days": recent}}


async def register(app: FastAPI, db: asyncpg.Pool) -> None:
    # Inject db pool into route closures
    for route in router.routes:
        route.endpoint.__defaults__ = (
            *route.endpoint.__defaults__[:-1],
            db,
        )
    app.include_router(router)
```

## Loading Order

Extensions are loaded after the database pool is created and vector dimensions are verified, but before the application starts accepting requests. This means extensions can safely query the database during registration.

## Error Handling

If any extension raises an exception during `register()`, the application logs the error and exits. This prevents partial initialization.

```
ERROR extension.failed name=ext_broken error="ModuleNotFoundError: No module named 'ext_broken'"
```
