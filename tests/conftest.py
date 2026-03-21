"""Shared fixtures for Engram test suite."""

import asyncio
import hashlib
import math
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"
TEST_DB_URL = os.environ.get("TEST_DATABASE_URL")


# --- Feature flag helpers ---


@pytest.fixture
def enable_flag(monkeypatch):
    """Helper to enable a feature flag for a single test.

    Patches the settings object referenced by all service modules,
    even if engram.config was reloaded by a prior test.
    """
    def _enable(flag_name: str):
        import engram.config
        monkeypatch.setattr(engram.config.settings, flag_name, True)
        # Also patch settings in service modules that may have imported
        # the old settings object before a reload
        for mod_path in [
            "engram.services.read",
            "engram.services.reconsolidation",
            "engram.services.dreamer",
            "engram.repositories.memory_repo",
        ]:
            import sys
            mod = sys.modules.get(mod_path)
            if mod and hasattr(mod, "settings"):
                monkeypatch.setattr(mod.settings, flag_name, True)
    return _enable


@pytest.fixture
def disable_flag(monkeypatch):
    """Helper to explicitly disable a feature flag for a single test."""
    def _disable(flag_name: str):
        import engram.config
        monkeypatch.setattr(engram.config.settings, flag_name, False)
        for mod_path in [
            "engram.services.read",
            "engram.services.reconsolidation",
            "engram.services.dreamer",
            "engram.repositories.memory_repo",
        ]:
            import sys
            mod = sys.modules.get(mod_path)
            if mod and hasattr(mod, "settings"):
                monkeypatch.setattr(mod.settings, flag_name, False)
    return _disable


# --- Helpers ---


def _make_vector(seed: float, dims: int = 1024) -> list[float]:
    """Generate a deterministic L2-normalized vector from a seed value."""
    vec = [math.sin(seed * (i + 1) + i * 0.1) for i in range(dims)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def auth_headers(raw_key: str) -> dict[str, str]:
    """Build Authorization header from a raw API key."""
    return {"Authorization": f"Bearer {raw_key}"}


# --- Session-scoped event loop (required for session-scoped async fixtures) ---


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# --- Database fixtures (session-scoped, skip if no TEST_DATABASE_URL) ---


@pytest_asyncio.fixture(scope="session")
async def db_pool():
    """Create a test database pool and apply migrations."""
    if not TEST_DB_URL:
        pytest.skip("TEST_DATABASE_URL not set")

    import asyncpg
    from pgvector.asyncpg import register_vector

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(
        TEST_DB_URL, min_size=2, max_size=5, init=_init
    )

    # Clean slate + apply migrations
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS api_keys CASCADE")
        await conn.execute("DROP TABLE IF EXISTS edges CASCADE")
        await conn.execute("DROP TABLE IF EXISTS memory_nodes CASCADE")
        await conn.execute("DROP TABLE IF EXISTS owners CASCADE")
        for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
            await conn.execute(sql_file.read_text())

    yield pool

    # Teardown
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS api_keys CASCADE")
        await conn.execute("DROP TABLE IF EXISTS edges CASCADE")
        await conn.execute("DROP TABLE IF EXISTS memory_nodes CASCADE")
        await conn.execute("DROP TABLE IF EXISTS owners CASCADE")
    await pool.close()


@pytest_asyncio.fixture(scope="session")
async def owner_a(db_pool):
    """Create test owner A with an API key. Returns (owner_id, raw_api_key)."""
    owner_id = uuid4()
    raw_key = f"test_key_a_{uuid4().hex[:16]}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO owners (id, label) VALUES ($1, $2)",
            owner_id,
            "Test Owner A",
        )
        await conn.execute(
            "INSERT INTO api_keys (owner_id, key_hash, label) "
            "VALUES ($1, $2, $3)",
            owner_id,
            key_hash,
            "test-a",
        )

    return owner_id, raw_key


@pytest_asyncio.fixture(scope="session")
async def owner_b(db_pool):
    """Create test owner B with an API key. Returns (owner_id, raw_api_key)."""
    owner_id = uuid4()
    raw_key = f"test_key_b_{uuid4().hex[:16]}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO owners (id, label) VALUES ($1, $2)",
            owner_id,
            "Test Owner B",
        )
        await conn.execute(
            "INSERT INTO api_keys (owner_id, key_hash, label) "
            "VALUES ($1, $2, $3)",
            owner_id,
            key_hash,
            "test-b",
        )

    return owner_id, raw_key


@pytest_asyncio.fixture(scope="session")
async def owner_c(db_pool):
    """Create test owner C with an API key. Never seeded — used for empty-DB tests."""
    owner_id = uuid4()
    raw_key = f"test_key_c_{uuid4().hex[:16]}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO owners (id, label) VALUES ($1, $2)",
            owner_id,
            "Test Owner C",
        )
        await conn.execute(
            "INSERT INTO api_keys (owner_id, key_hash, label) "
            "VALUES ($1, $2, $3)",
            owner_id,
            key_hash,
            "test-c",
        )

    return owner_id, raw_key


# --- Mock fixtures ---


def _mock_embed_fn():
    """Create a mock embed_six_dimensions that returns deterministic vectors."""
    from engram.models import AXES

    async def _embed(content: str) -> dict[str, list[float]]:
        h = hash(content) % 10000
        return {
            axis: _make_vector(h + i * 100) for i, axis in enumerate(AXES)
        }

    return _embed


@pytest.fixture(autouse=True)
def mock_embeddings():
    """Mock all OpenRouter calls for every test."""
    mock_embed = _mock_embed_fn()

    mock_client = MagicMock()
    mock_client.models.list = AsyncMock(return_value=MagicMock())

    classify_json = (
        '{"temporal": [], "emotional": [], '
        '"semantic": [{"type": "associative", "weight": 0.7}], '
        '"sensory": [], "action": [], "procedural": []}'
    )

    with (
        patch(
            "engram.services.write.embed_six_dimensions",
            side_effect=mock_embed,
        ),
        patch(
            "engram.services.write.compute_salience",
            AsyncMock(return_value=0.5),
        ),
        patch(
            "engram.services.read.embed_six_dimensions",
            side_effect=mock_embed,
        ),
        patch(
            "engram.services.dreamer.llm_classify",
            AsyncMock(return_value=classify_json),
        ),
        patch(
            "engram.routes.health.get_client",
            return_value=mock_client,
        ),
    ):
        yield


# --- HTTP Client ---


@pytest_asyncio.fixture()
async def client(db_pool):
    """httpx AsyncClient pointed at the FastAPI app with test DB pool."""
    from engram.main import app

    app.state.db = db_pool
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as c:
        yield c


# --- Seeded data ---


@pytest_asyncio.fixture(scope="session")
async def seeded_memories(db_pool, owner_a):
    """Seed 5 memories for owner A. Returns list of node ID strings."""
    from engram.write_path import encode_memory

    contents = [
        "User prefers dark mode for all interfaces",
        "Meeting with team at 3pm on Fridays",
        "Favorite programming language is Python",
        "Lives in San Francisco, California",
        "Allergic to shellfish, avoid seafood restaurants",
    ]

    owner_id, _ = owner_a
    ids = []
    for content in contents:
        result = await encode_memory(
            db_pool, owner_id, content, "conversation"
        )
        if "id" in result:
            ids.append(result["id"])
    return ids
