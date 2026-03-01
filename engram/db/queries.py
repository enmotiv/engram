"""Raw SQL queries for graph traversal."""

NEIGHBORHOOD_CTE = """
WITH RECURSIVE neighborhood AS (
    SELECT source_memory_id, target_memory_id, edge_type, weight, 1 as depth
    FROM edges
    WHERE source_memory_id = ANY(:seed_ids) AND namespace = :namespace
    UNION
    SELECT e.source_memory_id, e.target_memory_id, e.edge_type, e.weight, n.depth + 1
    FROM edges e
    JOIN neighborhood n ON e.source_memory_id = n.target_memory_id
    WHERE n.depth < :max_depth AND e.namespace = :namespace
)
SELECT DISTINCT source_memory_id, target_memory_id, edge_type, weight, depth
FROM neighborhood
"""
