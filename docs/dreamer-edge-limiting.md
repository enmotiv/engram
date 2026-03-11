# Dreamer Edge Limiting

> **Problem:** 34,457 edges across 440 nodes (78 per node). The Dreamer connected everything to everything, making spreading activation useless.
> **Root causes:** No minimum similarity for candidates, no max edges per node, LLM is too generous with classification, weight threshold too low (0.1), too many candidates per axis (10).

Claude Code already fixed:
- Weight threshold: 0.1 → 0.3 ✅
- Per-axis candidates: 10 → 5 ✅

Remaining fixes to apply to `engram/dreamer.py`:

## Fix 1: Minimum similarity threshold for candidates

In `_find_candidates()`, filter out candidates below a cosine similarity floor. Don't send barely-related memories to the LLM.

```python
# In _find_candidates(), after building the result list, before return:
MIN_CANDIDATE_SIMILARITY = 0.55
result = [c for c in result if c["score"] >= MIN_CANDIDATE_SIMILARITY]
```

## Fix 2: Max edges per node

In `process_new_memories()`, before the classification loop, check if the node already has enough edges:

```python
MAX_EDGES_PER_NODE = 20

async with tenant_connection(db_pool, owner_id) as conn:
    existing_edge_count = await conn.fetchval(
        "SELECT COUNT(*) FROM edges "
        "WHERE owner_id = $1 AND (source_id = $2 OR target_id = $2)",
        owner_id, mem_id,
    )

if existing_edge_count >= MAX_EDGES_PER_NODE:
    async with tenant_connection(db_pool, owner_id) as conn:
        await conn.execute(
            "UPDATE memory_nodes SET dreamer_processed = TRUE WHERE id = $1",
            mem_id,
        )
    stats["memories_processed"] += 1
    continue
```

## Fix 3: Stricter LLM prompt

Replace _CLASSIFICATION_PROMPT. Key additions:
- "Most memory pairs have NO meaningful relationship. Default to empty arrays."
- "A good classification produces edges on 1-2 axes at most."
- Explicit weight scale: 0.3 = weak but real, 0.5 = moderate, 0.7 = strong, 0.9 = direct cause/effect
- "Both are about the user's work" is NOT a relationship — explicitly calling out generic similarity

## Fix 4: Reduce semantic candidates from 30 to 15

In `_find_candidates()`, Strategy 1:
```python
"ORDER BY vec_semantic <=> $1 LIMIT 15",  # was 30
```

## Expected Impact

- Candidates per memory: ~30-35 (down from 90)
- After similarity filter (>0.55): ~15-20
- Edges per memory: ~5-10 (down from 78)
- Hard cap: 20 edges max per node
- Total for 4,100 nodes: ~20,000-40,000 edges (reasonable)

All changes go in engram/dreamer.py in the Engram repo.
Push to GHCR but do NOT redeploy until volume persistence is resolved.