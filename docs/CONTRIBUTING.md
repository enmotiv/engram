# Contributing

## Dev Setup

### Prerequisites

- Python 3.12+
- Docker (for PostgreSQL + pgvector)

### Quick Start

```bash
# Start the database
docker compose up -d db

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env: set OPENROUTER_API_KEY and DATABASE_URL

# Run the server
uvicorn engram.main:app --reload --host 0.0.0.0 --port 8000
```

The database migrations are applied automatically via Docker's `initdb.d` volume mount.

## Running Tests

### Unit Tests (no database required)

```bash
pytest tests/test_unit.py -v
```

### Integration Tests (requires test database)

```bash
# Start the test database (same Docker container works)
export TEST_DATABASE_URL="postgresql://engram:engram_dev@localhost:5434/engram"

# Run all tests
pytest -v

# Run a specific test file
pytest tests/test_write_path.py -v

# Run a specific test
pytest tests/test_security.py::test_sql_injection_in_content -v
```

Integration tests are automatically skipped if `TEST_DATABASE_URL` is not set.

### Test Categories

| File | What It Tests |
|------|-------------|
| `test_unit.py` | Pure math: normalization, scoring, hashing, validation |
| `test_write_path.py` | Create, duplicate detection, CRUD operations |
| `test_read_path.py` | Recall, trace mode, filtering, response structure |
| `test_tenant_isolation.py` | Cross-owner access prevention |
| `test_dreamer.py` | Edge classification, decay, dedup, pruning |
| `test_reconsolidation.py` | Activation boost on read |
| `test_security.py` | Auth, injection, rate limiting, error safety |

## Code Style

Engram uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for issues
ruff check engram/ tests/

# Auto-fix
ruff check --fix engram/ tests/

# Format
ruff format engram/ tests/
```

Configuration is in `pyproject.toml`. Key rules: pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, bugbear, simplify, annotations.

## Security Audit

Run before every release:

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

## PR Process

1. Create a feature branch from `main`
2. Make your changes
3. Run `ruff check engram/ tests/` — must pass
4. Run `pytest` — all tests must pass
5. Open a PR with a clear description of what and why
6. One approval required before merge

## Project Structure

```
engram/
  main.py          # FastAPI app, lifespan, structlog config
  config.py        # Pydantic settings from env vars
  db.py            # Connection pool, tenant_connection (RLS)
  auth.py          # API key auth, rate limiter
  models.py        # Pydantic models, enums, helpers
  errors.py        # EngramError exception
  middleware.py     # Correlation ID, error handlers
  tracing.py       # ContextVars, Span, TraceCollector, Prometheus
  embeddings.py    # OpenRouter client, embed, classify
  write_path.py    # Encode pipeline
  read_path.py     # Retrieval pipeline
  reconsolidation.py  # Update-on-read
  dreamer.py       # Edge classification + maintenance
  routes/
    memories.py    # CRUD endpoints
    recall.py      # Recall endpoint
    health.py      # Health + stats
tests/
  conftest.py      # Fixtures, mocks, test DB setup
  test_*.py        # Test files
migrations/
  001_initial.sql  # Core schema
  002_api_keys.sql # API key table
docs/
  ARCHITECTURE.md
  API.md
  SECURITY.md
  EXTENSIONS.md
  CONTRIBUTING.md
  CHANGELOG.md
```
