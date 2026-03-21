-- Index for PATCH /v1/memories/by-meta/enmotiv_id/:value lookups.
-- Without this, every metadata update scans the full memory_nodes table.
-- At 1000+ users sharing one owner_id, that's 300k+ row scans per PATCH.

CREATE INDEX IF NOT EXISTS idx_nodes_enmotiv_id
  ON memory_nodes ((metadata->>'enmotiv_id'))
  WHERE metadata->>'enmotiv_id' IS NOT NULL;
