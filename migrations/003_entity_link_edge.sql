-- migrations/003_entity_link_edge.sql
-- Add entity_link edge type (decay-exempt structural edges for entity intelligence)

-- Drop and recreate the CHECK constraint to include entity_link
ALTER TABLE edges DROP CONSTRAINT IF EXISTS edges_edge_type_check;
ALTER TABLE edges ADD CONSTRAINT edges_edge_type_check
  CHECK (edge_type IN (
    'excitatory','inhibitory','associative',
    'temporal','modulatory','entity_link'));
