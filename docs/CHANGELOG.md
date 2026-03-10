# Changelog

All notable changes to Engram are documented here. This project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-03-08

### Added

- **Core schema**: Memory nodes with 6 vector dimensions (temporal, emotional, semantic, sensory, action, procedural), typed edges with per-axis classification
- **Write path**: Content dedup via SHA-256, 6-axis embedding via OpenRouter, salience scoring from emotional vector
- **Read path**: Multi-axis vector search, convergence scoring, spreading activation (1-hop with excitatory/inhibitory/associative/temporal/modulatory edges), post-filtering, reconsolidation
- **Dreamer**: LLM-based edge classification across all 6 axes, activation decay, edge weight decay, near-duplicate detection with edge transfer, low-weight edge pruning
- **API surface**: REST API with `POST /v1/memories`, `GET /v1/memories/:id`, `GET /v1/memories` (cursor pagination), `DELETE /v1/memories/:id`, `POST /v1/recall`, `GET /v1/health`, `GET /v1/stats`
- **Authentication**: Bearer token API keys with SHA-256 hashing, per-key rate limits
- **Rate limiting**: In-memory sliding-window per key, separate read/write buckets
- **Tenant isolation**: Row-Level Security on memory_nodes and edges, application-level owner_id filtering
- **Observability**: Structured JSON logging via structlog, ContextVar-based correlation_id and owner_id injection, Span timing with WARNING/ERROR thresholds, `?trace=true` retrieval diagnostics, Prometheus metrics (recall/write/embed/dreamer histograms, error counter, node/edge/activation gauges)
- **Error handling**: Standard error schema on all endpoints, Pydantic 422→400 remapping, generic 500 handler that logs but never leaks
- **Extension system**: Plugin loading via `ENGRAM_EXTENSIONS` env var
- **Test suite**: Unit tests (math, scoring, validation), integration tests (write/read/CRUD), tenant isolation tests, Dreamer tests, security tests (auth, injection, rate limiting)
- **Documentation**: README, ARCHITECTURE, API reference, SECURITY, EXTENSIONS, CONTRIBUTING, CHANGELOG
