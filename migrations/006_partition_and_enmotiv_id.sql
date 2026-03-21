-- 006: Hash-partition memory_nodes + edges by owner_id (256 partitions)
-- Adds enmotiv_id as a real column for fast lookups without JSONB extraction.
--
-- With one user and a few hundred memories, the data copy is milliseconds.
-- Empty partitions cost nothing. This prevents a painful migration later.
--
-- Strategy: rename old tables, create partitioned replacements, copy data,
-- drop old tables, re-apply RLS.

-- -------------------------------------------------------------------------
-- Step 1: Add enmotiv_id column to existing table BEFORE partitioning
-- -------------------------------------------------------------------------

ALTER TABLE memory_nodes ADD COLUMN IF NOT EXISTS enmotiv_id VARCHAR(255);

-- Backfill from JSONB metadata
UPDATE memory_nodes
SET enmotiv_id = metadata->>'enmotiv_id'
WHERE metadata->>'enmotiv_id' IS NOT NULL AND enmotiv_id IS NULL;

-- -------------------------------------------------------------------------
-- Step 2: Disable RLS + drop policies on old tables
-- -------------------------------------------------------------------------

DROP POLICY IF EXISTS owner_nodes ON memory_nodes;
DROP POLICY IF EXISTS owner_edges ON edges;
ALTER TABLE memory_nodes DISABLE ROW LEVEL SECURITY;
ALTER TABLE edges DISABLE ROW LEVEL SECURITY;

-- -------------------------------------------------------------------------
-- Step 3: Rename old tables
-- -------------------------------------------------------------------------

ALTER TABLE edges RENAME TO edges_old;
ALTER TABLE memory_nodes RENAME TO memory_nodes_old;

-- Drop old indexes that reference old tables (they moved with rename)
DROP INDEX IF EXISTS idx_nodes_owner;
DROP INDEX IF EXISTS idx_nodes_session;
DROP INDEX IF EXISTS idx_nodes_unprocessed;
DROP INDEX IF EXISTS idx_vec_temporal;
DROP INDEX IF EXISTS idx_vec_emotional;
DROP INDEX IF EXISTS idx_vec_semantic;
DROP INDEX IF EXISTS idx_vec_sensory;
DROP INDEX IF EXISTS idx_vec_action;
DROP INDEX IF EXISTS idx_vec_procedural;
DROP INDEX IF EXISTS idx_nodes_enmotiv_id;
DROP INDEX IF EXISTS idx_edges_source_axis;
DROP INDEX IF EXISTS idx_edges_target_axis;
DROP INDEX IF EXISTS idx_edges_live;

-- -------------------------------------------------------------------------
-- Step 4: Create hash-partitioned memory_nodes (256 partitions)
-- -------------------------------------------------------------------------

CREATE TABLE memory_nodes (
  id                   UUID NOT NULL,
  owner_id             UUID NOT NULL REFERENCES owners(id),
  content              TEXT NOT NULL CHECK(char_length(content) <= 4096),
  content_hash         CHAR(64) NOT NULL,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_accessed        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  access_count         INTEGER NOT NULL DEFAULT 0,
  activation_level     REAL NOT NULL DEFAULT 1.0
                       CHECK(activation_level BETWEEN 0.0 AND 1.0),
  salience             REAL NOT NULL DEFAULT 0.5
                       CHECK(salience BETWEEN 0.0 AND 1.0),
  source_type          VARCHAR(20) NOT NULL CHECK(source_type IN
    ('conversation','event','observation','correction','system')),
  session_id           UUID,
  embedding_model      VARCHAR(100) NOT NULL,
  embedding_dimensions INTEGER NOT NULL,
  vec_temporal         VECTOR(1024) NOT NULL,
  vec_emotional        VECTOR(1024) NOT NULL,
  vec_semantic         VECTOR(1024) NOT NULL,
  vec_sensory          VECTOR(1024) NOT NULL,
  vec_action           VECTOR(1024) NOT NULL,
  vec_procedural       VECTOR(1024) NOT NULL,
  metadata             JSONB NOT NULL DEFAULT '{}'::jsonb,
  is_deleted           BOOLEAN NOT NULL DEFAULT FALSE,
  dreamer_processed    BOOLEAN NOT NULL DEFAULT FALSE,
  plasticity           REAL NOT NULL DEFAULT 0.8 CHECK(plasticity BETWEEN 0.1 AND 1.0),
  modification_count   INTEGER NOT NULL DEFAULT 0,
  enmotiv_id           VARCHAR(255),
  PRIMARY KEY (owner_id, id),
  UNIQUE (owner_id, content_hash)
  -- enmotiv_id uniqueness enforced via partial unique index below (NULLs allowed)
) PARTITION BY HASH (owner_id);

-- Generate 256 partitions
DO $$
BEGIN
  FOR i IN 0..255 LOOP
    EXECUTE format(
      'CREATE TABLE memory_nodes_p%s PARTITION OF memory_nodes '
      'FOR VALUES WITH (MODULUS 256, REMAINDER %s)',
      i, i
    );
  END LOOP;
END $$;

-- -------------------------------------------------------------------------
-- Step 5: Create hash-partitioned edges (256 partitions)
-- -------------------------------------------------------------------------

CREATE TABLE edges (
  id         UUID NOT NULL,
  owner_id   UUID NOT NULL,
  source_id  UUID NOT NULL,
  target_id  UUID NOT NULL,
  edge_type  VARCHAR(20) NOT NULL CHECK(edge_type IN
    ('excitatory','inhibitory','associative',
     'temporal','modulatory','structural')),
  axis       VARCHAR(20) NOT NULL CHECK(axis IN
    ('temporal','emotional','semantic',
     'sensory','action','procedural')),
  weight     REAL NOT NULL DEFAULT 0.5
             CHECK(weight BETWEEN 0.0 AND 1.0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (owner_id, id),
  FOREIGN KEY (owner_id, source_id) REFERENCES memory_nodes(owner_id, id),
  FOREIGN KEY (owner_id, target_id) REFERENCES memory_nodes(owner_id, id),
  UNIQUE(owner_id, source_id, target_id, edge_type, axis)
) PARTITION BY HASH (owner_id);

DO $$
BEGIN
  FOR i IN 0..255 LOOP
    EXECUTE format(
      'CREATE TABLE edges_p%s PARTITION OF edges '
      'FOR VALUES WITH (MODULUS 256, REMAINDER %s)',
      i, i
    );
  END LOOP;
END $$;

-- -------------------------------------------------------------------------
-- Step 6: Copy data from old tables
-- -------------------------------------------------------------------------

INSERT INTO memory_nodes (
  id, owner_id, content, content_hash, created_at, last_accessed,
  access_count, activation_level, salience, source_type, session_id,
  embedding_model, embedding_dimensions,
  vec_temporal, vec_emotional, vec_semantic, vec_sensory, vec_action, vec_procedural,
  metadata, is_deleted, dreamer_processed, plasticity, modification_count, enmotiv_id
)
SELECT
  id, owner_id, content, content_hash, created_at, last_accessed,
  access_count, activation_level, salience, source_type, session_id,
  embedding_model, embedding_dimensions,
  vec_temporal, vec_emotional, vec_semantic, vec_sensory, vec_action, vec_procedural,
  metadata, is_deleted, dreamer_processed, plasticity, modification_count, enmotiv_id
FROM memory_nodes_old;

INSERT INTO edges (
  id, owner_id, source_id, target_id, edge_type, axis, weight, created_at, updated_at
)
SELECT
  id, owner_id, source_id, target_id, edge_type, axis, weight, created_at, updated_at
FROM edges_old;

-- -------------------------------------------------------------------------
-- Step 7: Drop old tables
-- -------------------------------------------------------------------------

DROP TABLE edges_old;
DROP TABLE memory_nodes_old;

-- -------------------------------------------------------------------------
-- Step 8: Create indexes on partitioned tables
-- -------------------------------------------------------------------------

-- B-tree indexes (created on parent, auto-propagate to partitions)
CREATE INDEX idx_nodes_owner ON memory_nodes(owner_id);
CREATE INDEX idx_nodes_session ON memory_nodes(owner_id, session_id)
  WHERE session_id IS NOT NULL;
CREATE INDEX idx_nodes_unprocessed ON memory_nodes(owner_id)
  WHERE dreamer_processed = FALSE AND is_deleted = FALSE;
CREATE UNIQUE INDEX idx_nodes_enmotiv_id ON memory_nodes(owner_id, enmotiv_id)
  WHERE enmotiv_id IS NOT NULL;

-- HNSW indexes per dimension (on parent — each partition gets its own)
CREATE INDEX idx_vec_temporal ON memory_nodes USING hnsw
  (vec_temporal vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX idx_vec_emotional ON memory_nodes USING hnsw
  (vec_emotional vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX idx_vec_semantic ON memory_nodes USING hnsw
  (vec_semantic vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX idx_vec_sensory ON memory_nodes USING hnsw
  (vec_sensory vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX idx_vec_action ON memory_nodes USING hnsw
  (vec_action vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX idx_vec_procedural ON memory_nodes USING hnsw
  (vec_procedural vector_cosine_ops) WITH (m=16, ef_construction=64);

-- Edge indexes
CREATE INDEX idx_edges_source_axis ON edges(source_id, axis);
CREATE INDEX idx_edges_target_axis ON edges(target_id, axis);
CREATE INDEX idx_edges_live ON edges(weight) WHERE weight > 0.0;

-- -------------------------------------------------------------------------
-- Step 9: Re-enable RLS on partitioned tables
-- -------------------------------------------------------------------------

ALTER TABLE memory_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE edges ENABLE ROW LEVEL SECURITY;

CREATE POLICY owner_nodes ON memory_nodes
  USING (owner_id = current_setting('app.owner_id')::uuid)
  WITH CHECK (owner_id = current_setting('app.owner_id')::uuid);
CREATE POLICY owner_edges ON edges
  USING (owner_id = current_setting('app.owner_id')::uuid)
  WITH CHECK (owner_id = current_setting('app.owner_id')::uuid);

ALTER TABLE memory_nodes FORCE ROW LEVEL SECURITY;
ALTER TABLE edges FORCE ROW LEVEL SECURITY;
