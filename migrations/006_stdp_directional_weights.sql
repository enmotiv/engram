-- Phase 5: STDP directional weights
-- forward_weight = strength when traversing min(src,tgt) → max(src,tgt)
-- backward_weight = strength when traversing max(src,tgt) → min(src,tgt)
-- Defaults to symmetric (0.5/0.5), backfilled from current weight.

ALTER TABLE edges ADD COLUMN IF NOT EXISTS forward_weight REAL NOT NULL DEFAULT 0.5
  CHECK(forward_weight BETWEEN 0.0 AND 1.0);
ALTER TABLE edges ADD COLUMN IF NOT EXISTS backward_weight REAL NOT NULL DEFAULT 0.5
  CHECK(backward_weight BETWEEN 0.0 AND 1.0);

UPDATE edges SET forward_weight = weight, backward_weight = weight;
