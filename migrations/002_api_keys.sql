-- migrations/002_api_keys.sql
-- API key authentication for Engram v1.0

CREATE TABLE api_keys (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id        UUID NOT NULL REFERENCES owners(id),
  key_hash        CHAR(64) NOT NULL UNIQUE,
  label           VARCHAR(255) NOT NULL DEFAULT '',
  rate_limit_writes INTEGER NOT NULL DEFAULT 100,
  rate_limit_reads  INTEGER NOT NULL DEFAULT 500,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  revoked_at      TIMESTAMPTZ
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash)
  WHERE revoked_at IS NULL;
CREATE INDEX idx_api_keys_owner ON api_keys(owner_id);
