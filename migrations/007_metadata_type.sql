-- Phase 6: metadata_type discriminator column
--
-- Promoted field from the existing JSONB metadata column. Enables fast
-- type-filtered recall by composing a B-tree index with HNSW vector search.
--
-- Three-layer storage architecture:
--   1. Graph (edges + STDP) → relationships BETWEEN memories
--   2. JSONB metadata → memory-level flags (pinned, trust_level, enmotiv_id)
--   3. metadata_type → discriminator for type-filtered recall
--
-- External systems (e.g. enmotiv) use metadata_type to tag memories by
-- extension type (nutrition_log, project_task, clinical_note, etc.).
-- Engram does not interpret or validate these values — it indexes them
-- for pre-filtered vector search.
--
-- Extension payloads (calories, protein, task status) belong in the
-- external system's own typed tables, NOT in engram's JSONB metadata.
-- metadata_type is the bridge: engram filters by type, the external
-- system stores and queries the structured data.

ALTER TABLE memory_nodes
  ADD COLUMN IF NOT EXISTS metadata_type VARCHAR(50) DEFAULT NULL;

-- Partial B-tree index: only typed memories are indexed.
-- Composes with HNSW for filtered vector search:
--   partition prune (owner_id) → B-tree filter (metadata_type) → HNSW scan
-- Note: not using CONCURRENTLY — migration runner executes within a
-- transaction block. Safe at current scale; table has no typed rows yet.
CREATE INDEX IF NOT EXISTS idx_nodes_metadata_type
  ON memory_nodes (metadata_type)
  WHERE metadata_type IS NOT NULL AND is_deleted = FALSE;
