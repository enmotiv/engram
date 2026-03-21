-- Phase 4: Metaplasticity columns
-- Additive migration — no existing column modifications.
-- Safe to deploy with flag off; columns sit at defaults until enabled.
--
-- Backfills existing memories using signals that are actually populated:
-- 1. access_count (recall history — may be 0 if recall path unused)
-- 2. edge count (dreamer classification history)
-- 3. age (older memories have had more time to consolidate)
-- 4. source_type (corrections are high-value from birth)
--
-- NOTE: metadata.trust_level is NOT used because Enmotiv's trust
-- progression updates Postgres but never syncs back to Engram metadata.
-- All Engram memories have their original write-time trust level (usually
-- 'provisional'), which doesn't reflect current trust state.

ALTER TABLE memory_nodes ADD COLUMN IF NOT EXISTS plasticity float NOT NULL DEFAULT 0.8;
ALTER TABLE memory_nodes ADD COLUMN IF NOT EXISTS modification_count int NOT NULL DEFAULT 0;

-- Backfill modification_count, then derive plasticity.
-- Each "modification" decrements plasticity by 0.02 (matching boost_nodes),
-- floored at 0.1.
UPDATE memory_nodes mn SET
  modification_count = sub.mod_count,
  plasticity = GREATEST(0.1, 0.8 - 0.02 * sub.mod_count)
FROM (
  SELECT
    mn2.id,
    -- Base: access_count + edges (direct interaction history)
    mn2.access_count + COALESCE(ec.edge_ct, 0)
    -- Age: 1 point per week, capped at 10 (~2.5 months to max)
    + LEAST(10, EXTRACT(EPOCH FROM (NOW() - mn2.created_at)) / 604800)::int
    -- Source type: corrections are established from birth
    + CASE
        WHEN mn2.source_type = 'correction' THEN 8
        WHEN mn2.source_type = 'observation' THEN 3
        WHEN mn2.source_type = 'event' THEN 2
        ELSE 0
      END
    AS mod_count
  FROM memory_nodes mn2
  LEFT JOIN (
    SELECT node_id, COUNT(*) AS edge_ct FROM (
      SELECT source_id AS node_id FROM edges
      UNION ALL
      SELECT target_id AS node_id FROM edges
    ) all_edges
    GROUP BY node_id
  ) ec ON ec.node_id = mn2.id
  WHERE mn2.is_deleted = FALSE
) sub
WHERE mn.id = sub.id;
