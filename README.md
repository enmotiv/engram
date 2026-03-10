# Engram

Memory infrastructure that takes, classifies, and connects memories.

Engram does **not** merge, summarize, consolidate, create new content, interpret behavior, or infer preferences. It stores memories as-is, embeds them across six perceptual dimensions, and builds a typed edge graph between them.

## Quick Start

```bash
# 1. Start PostgreSQL with pgvector
docker compose up -d db

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure and run
cp .env.example .env  # set OPENROUTER_API_KEY and DATABASE_URL
uvicorn engram.main:app --host 0.0.0.0 --port 8000
```

## First Memory

```bash
# Create a memory
curl -X POST http://localhost:8000/v1/memories \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "source_type": "conversation"}'

# Recall it
curl -X POST http://localhost:8000/v1/recall \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"cue": "interface preferences"}'
```

## Architecture

Engram embeds each memory into six dimensions (temporal, emotional, semantic, sensory, action, procedural), then retrieves by convergence across those dimensions. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## API Reference

Full endpoint documentation with request/response schemas and curl examples. See [docs/API.md](docs/API.md).

## Security

Tenant isolation via Row-Level Security, API key authentication, vector-as-PII policy. See [docs/SECURITY.md](docs/SECURITY.md).

## Extensions

Plugin system for adding routes and background jobs. See [docs/EXTENSIONS.md](docs/EXTENSIONS.md).

## Contributing

Dev setup, test commands, code style. See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).
