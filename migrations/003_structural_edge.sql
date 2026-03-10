-- migrations/003_structural_edge.sql
-- Add structural edge type (client-managed, decay-exempt edges)

-- Drop and recreate the CHECK constraint to include structural
ALTER TABLE edges DROP CONSTRAINT IF EXISTS edges_edge_type_check;
ALTER TABLE edges ADD CONSTRAINT edges_edge_type_check
  CHECK (edge_type IN (
    'excitatory','inhibitory','associative',
    'temporal','modulatory','structural'));

-- Migrate any existing entity_link edges to structural
UPDATE edges SET edge_type = 'structural' WHERE edge_type = 'entity_link';
