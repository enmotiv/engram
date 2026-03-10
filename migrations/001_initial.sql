-- migrations/001_initial.sql
-- Engram v1.0 schema

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Owners (tenants)
CREATE TABLE owners (
  id         UUID PRIMARY KEY,
  label      VARCHAR(255) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Memory nodes
CREATE TABLE memory_nodes (
  id                   UUID PRIMARY KEY,
  owner_id             UUID NOT NULL REFERENCES owners(id),
  content              TEXT NOT NULL
                       CHECK(char_length(content) <= 4096),
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
  UNIQUE(owner_id, content_hash)
);

-- Indexes on memory_nodes
CREATE INDEX idx_nodes_owner ON memory_nodes(owner_id);
CREATE INDEX idx_nodes_session ON memory_nodes(owner_id, session_id)
  WHERE session_id IS NOT NULL;
CREATE INDEX idx_nodes_unprocessed ON memory_nodes(owner_id)
  WHERE dreamer_processed = FALSE AND is_deleted = FALSE;

-- HNSW per dimension (m=16, ef_construction=64)
CREATE INDEX idx_vec_temporal   ON memory_nodes USING hnsw
  (vec_temporal vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX idx_vec_emotional  ON memory_nodes USING hnsw
  (vec_emotional vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX idx_vec_semantic   ON memory_nodes USING hnsw
  (vec_semantic vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX idx_vec_sensory    ON memory_nodes USING hnsw
  (vec_sensory vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX idx_vec_action     ON memory_nodes USING hnsw
  (vec_action vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX idx_vec_procedural ON memory_nodes USING hnsw
  (vec_procedural vector_cosine_ops) WITH (m=16, ef_construction=64);

-- Per-axis edges
CREATE TABLE edges (
  id         UUID PRIMARY KEY,
  owner_id   UUID NOT NULL REFERENCES owners(id),
  source_id  UUID NOT NULL REFERENCES memory_nodes(id),
  target_id  UUID NOT NULL REFERENCES memory_nodes(id),
  edge_type  VARCHAR(20) NOT NULL CHECK(edge_type IN
    ('excitatory','inhibitory','associative',
     'temporal','modulatory')),
  axis       VARCHAR(20) NOT NULL CHECK(axis IN
    ('temporal','emotional','semantic',
     'sensory','action','procedural')),
  weight     REAL NOT NULL DEFAULT 0.5
             CHECK(weight BETWEEN 0.0 AND 1.0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(owner_id, source_id, target_id, edge_type, axis)
);

CREATE INDEX idx_edges_source_axis ON edges(source_id, axis);
CREATE INDEX idx_edges_target_axis ON edges(target_id, axis);
CREATE INDEX idx_edges_live ON edges(weight) WHERE weight > 0.0;

-- Row-Level Security
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
