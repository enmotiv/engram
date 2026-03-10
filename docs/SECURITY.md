# Security

## Threat Model

Engram stores personal memories. The primary threats are:

1. **Cross-tenant access**: Owner A reading/modifying owner B's data
2. **API key compromise**: Unauthorized access to an owner's memories
3. **Content extraction**: Reversing embeddings to reconstruct memory content
4. **Injection attacks**: SQL injection, payload manipulation
5. **Denial of service**: Exhausting resources via API abuse

## Tenant Isolation

### Application Layer

Every database query includes `WHERE owner_id = $1` with parameterized values. There is no code path that queries across owners.

### Database Layer (Defense-in-Depth)

PostgreSQL Row-Level Security (RLS) is enabled on `memory_nodes` and `edges`:

```sql
CREATE POLICY owner_nodes ON memory_nodes
  USING (owner_id = current_setting('app.owner_id')::uuid)
  WITH CHECK (owner_id = current_setting('app.owner_id')::uuid);
```

The application sets `app.owner_id` via `SET` on each connection before any query.

**Production requirement**: The application must connect as a non-superuser role. Superusers bypass RLS. See below.

### Production Database Setup

```sql
CREATE ROLE engram_app WITH LOGIN PASSWORD '<strong-password>' NOSUPERUSER;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO engram_app;
```

Set `DATABASE_URL` to use `engram_app`. RLS applies automatically to non-superuser roles.

## API Key Security

- Keys are SHA-256 hashed before storage. Plaintext keys are never stored or logged.
- Key lookup queries filter `WHERE revoked_at IS NULL` to reject revoked keys.
- The raw key appears only in the `Authorization: Bearer <key>` header over HTTPS.
- Key hash comparison is performed in the database (constant-time at the DB level).

## Input Sanitization

- All SQL uses parameterized queries (`$1`, `$2`, ...). No string concatenation.
- Dynamic column names in ORDER BY and vector search are derived from a hardcoded allowlist (`AXES`, `_VALID_SORTS`), never from user input.
- The `tenant_connection` context manager validates the owner_id as a UUID before using it in a SET statement.
- Pydantic validates all request bodies. Invalid payloads are rejected before reaching application logic.

## Rate Limiting

- **Application layer**: Per-key sliding-window rate limiter. Default 100 writes/min, 500 reads/min. Configurable per key via `rate_limit_writes` and `rate_limit_reads` columns.
- **Production**: Add a second layer at the reverse proxy (nginx/Cloudflare) for IP-based rate limiting.

## Vector Reversal Risk

Embedding vectors are derived from memory content. Research has shown that embeddings can be approximately reversed to reconstruct original text. Therefore:

- **Vectors are PII.** Treat them with the same protections as the content itself.
- **Encrypt at rest.** Use filesystem-level or cloud-managed encryption for the PostgreSQL data directory.
- **Include in deletion.** When a memory is permanently purged, delete its vectors too.
- **Do not expose vectors via API.** The API returns scores, not raw vectors.

## Error Handling

- Error responses never include stack traces, SQL queries, or database details.
- All errors return a standard schema with a `correlation_id` for internal tracing.
- The generic 500 handler logs the full exception server-side but returns only "An unexpected error occurred" to the client.

## HTTPS

- **Production must reject HTTP.** Configure TLS termination at the reverse proxy.
- Do not redirect HTTP to HTTPS — reject it. Redirects can leak data in the initial plaintext request.

## Logging

- Auth failures are logged with correlation_id and key hash (never the raw key).
- Write operations are logged with node_id and operation type.
- Memory content is never logged (PII).
- All logs include correlation_id and owner_id via structlog processors.

## Dependency Audit

Run `pip audit` before every release:

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

Fix all critical and high vulnerabilities before shipping.

## Security Checklist

- [ ] Application connects as non-superuser role
- [ ] `sslmode=require` in DATABASE_URL
- [ ] TLS termination at reverse proxy, HTTP rejected
- [ ] CORS deny-all by default
- [ ] `pip audit` passes with no critical/high findings
- [ ] API key rotation process documented
- [ ] Backup encryption enabled
- [ ] Vector data included in deletion workflows

## Responsible Disclosure

If you find a security vulnerability, please report it privately. Do not open a public issue. Contact the maintainers directly.
