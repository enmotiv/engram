# API Reference

Base URL: `http://localhost:8000/v1`

All endpoints except `/v1/health` require authentication via `Authorization: Bearer <api_key>`.

---

## Authentication

```
Authorization: Bearer eng_your_api_key_here
```

API keys are SHA-256 hashed before storage. The hash is compared on each request to resolve the owner. Revoked keys are rejected.

### Rate Limits

Per-key sliding window (default 60 seconds):
- **Writes** (POST, PUT, PATCH, DELETE): 100/minute
- **Reads** (GET): 500/minute

Rate limit info is returned in response headers:
```
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1709913600
```

When rate limited, the response includes:
```
Retry-After: 45
```

---

## POST /v1/memories

Create a new memory.

### Request

```json
{
  "content": "User prefers dark mode for all interfaces",
  "source_type": "conversation",
  "session_id": "optional-uuid",
  "metadata": {"key": "value"}
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| content | string | yes | 1-4096 chars |
| source_type | string | yes | conversation, event, observation, correction, system |
| session_id | uuid | no | Groups memories by session |
| metadata | object | no | Arbitrary JSON |

### Response (201)

```json
{
  "data": {
    "id": "01912345-6789-7abc-def0-123456789abc",
    "salience": 0.72
  }
}
```

### Errors

- **400** `INVALID_INPUT` — Validation failed
- **409** `DUPLICATE_CONTENT` — Content already exists (includes `existing_id`)

```bash
curl -X POST http://localhost:8000/v1/memories \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "source_type": "conversation"}'
```

---

## GET /v1/memories/{id}

Fetch a single memory by ID.

### Response (200)

```json
{
  "data": {
    "id": "01912345-...",
    "content": "User prefers dark mode",
    "content_hash": "a1b2c3...",
    "activation_level": 0.85,
    "salience": 0.72,
    "source_type": "conversation",
    "session_id": null,
    "metadata": {},
    "created_at": "2024-06-01T12:00:00+00:00",
    "last_accessed": "2024-06-01T14:30:00+00:00",
    "access_count": 3
  }
}
```

### Errors

- **404** `NOT_FOUND` — Memory does not exist or is deleted

```bash
curl http://localhost:8000/v1/memories/01912345-6789-7abc-def0-123456789abc \
  -H "Authorization: Bearer YOUR_KEY"
```

---

## GET /v1/memories

List memories with cursor pagination.

### Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| limit | int | 20 | 1-100 |
| cursor | string | null | ISO 8601 datetime from `next_cursor` |
| sort | string | created_at:desc | created_at:desc, created_at:asc, activation_level:desc |
| source_type | string | null | Comma-separated filter: conversation,event |

### Response (200)

```json
{
  "data": {
    "memories": [...],
    "next_cursor": "2024-06-01T12:00:00+00:00"
  }
}
```

`next_cursor` is `null` when there are no more pages.

Note: cursor pagination is designed for `created_at` sorts. When using `activation_level:desc`, cursor behavior is approximate.

```bash
curl "http://localhost:8000/v1/memories?limit=10&source_type=conversation" \
  -H "Authorization: Bearer YOUR_KEY"
```

---

## DELETE /v1/memories/{id}

Soft-delete a memory.

### Response (200)

```json
{
  "data": {
    "id": "01912345-...",
    "deleted": true
  }
}
```

### Errors

- **404** `NOT_FOUND` — Memory does not exist or already deleted

```bash
curl -X DELETE http://localhost:8000/v1/memories/01912345-6789-7abc-def0-123456789abc \
  -H "Authorization: Bearer YOUR_KEY"
```

---

## POST /v1/recall

Retrieve memories by semantic cue across all six dimensions.

### Request

```json
{
  "cue": "interface preferences",
  "top_k": 5,
  "min_convergence": 0.0,
  "include_edges": false
}
```

| Field | Type | Default | Constraints |
|-------|------|---------|-------------|
| cue | string | required | 1-4096 chars |
| top_k | int | 5 | 1-20 |
| min_convergence | float | 0.0 | >= 0.0 |
| include_edges | bool | false | Include edge graph data |

### Response (200)

```json
{
  "data": {
    "memories": [
      {
        "id": "01912345-...",
        "content": "User prefers dark mode",
        "content_hash": "a1b2c3...",
        "convergence_score": 3.45,
        "activation_level": 0.85,
        "dimension_scores": {
          "temporal": 0.12,
          "emotional": 0.45,
          "semantic": 0.92,
          "sensory": 0.78,
          "action": 0.33,
          "procedural": 0.05
        },
        "matched_axes": ["emotional", "semantic", "sensory"],
        "metadata": {},
        "salience": 0.72,
        "created_at": "2024-06-01T12:00:00+00:00",
        "last_accessed": "2024-06-01T14:30:00+00:00",
        "access_count": 3,
        "session_id": null,
        "source_type": "conversation"
      }
    ],
    "confidence": "high",
    "edges": []
  }
}
```

### Trace Mode

Add `?trace=true` to get retrieval diagnostics:

```json
{
  "data": {
    "memories": [...],
    "confidence": "high",
    "edges": [],
    "trace": {
      "correlation_id": "req_a1b2c3d4e5f6",
      "total_ms": 245.3,
      "embedding_ms": 180.5,
      "cue_preview": "interface preferences",
      "per_dimension_results": {
        "temporal": {"candidates": 20, "top_score": 0.45},
        "semantic": {"candidates": 20, "top_score": 0.92}
      },
      "unique_candidates": 35,
      "convergence_scores": [...],
      "spreading_activation": {
        "edges_loaded": 12,
        "excitatory_fired": 5,
        "inhibitory_fired": 1,
        "modulatory_fired": 0,
        "nodes_activated_by_spread": 3
      },
      "reconsolidation": {"nodes_boosted": 5, "edges_strengthened": 2},
      "post_filter": {"nodes_excluded": 0, "exclude_tags": []}
    }
  }
}
```

```bash
curl -X POST "http://localhost:8000/v1/recall?trace=true" \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"cue": "interface preferences", "top_k": 5}'
```

---

## GET /v1/health

Health check. No authentication required.

### Response (200)

```json
{
  "status": "healthy",
  "db": "connected",
  "openrouter": "reachable"
}
```

Returns 503 if database is disconnected. Returns "degraded" if OpenRouter is unreachable but DB is fine.

```bash
curl http://localhost:8000/v1/health
```

---

## GET /v1/stats

Per-owner statistics. Requires authentication.

### Response (200)

```json
{
  "data": {
    "node_count": 1523,
    "edge_count": 4210,
    "avg_activation": 0.6234,
    "unprocessed_nodes": 12
  }
}
```

```bash
curl http://localhost:8000/v1/stats \
  -H "Authorization: Bearer YOUR_KEY"
```

---

## Error Schema

All errors follow the same structure:

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Memory not found",
    "status": 404,
    "correlation_id": "req_a1b2c3d4e5f6"
  }
}
```

### Error Codes

| Code | Status | Description |
|------|--------|-------------|
| INVALID_INPUT | 400 | Request validation failed |
| UNAUTHORIZED | 401 | Missing or invalid API key |
| FORBIDDEN | 403 | Access denied |
| NOT_FOUND | 404 | Resource does not exist |
| METHOD_NOT_ALLOWED | 405 | HTTP method not supported |
| DUPLICATE_CONTENT | 409 | Content already exists (includes `existing_id`) |
| RATE_LIMITED | 429 | Too many requests (includes `Retry-After` header) |
| INTERNAL_ERROR | 500 | Server error (details logged, not returned) |
| SERVICE_UNAVAILABLE | 503 | Dependency down |

Every error response includes `X-Correlation-ID` in the response header and `correlation_id` in the body for support tracing.
