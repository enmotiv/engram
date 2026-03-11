# Entity Intelligence: Living Summaries

Spec location: /mnt/user-data/outputs/entity-intelligence-spec.md (full version)
Also saved to: /Users/gloriafolaron/herd/enmotiv/docs/memory/entity-intelligence-spec.md (partial — needs full copy)

This spec covers both Engram and Enmotiv changes:

## Engram-side changes (this repo):
- entity_link → structural edge type (DONE via migration 003)
- entity-summary nodes pinned (skip activation decay) (DONE)
- Dreamer skips structural edges in decay (DONE)
- 2-hop spreading activation (DONE)

## Enmotiv-side changes (enmotiv repo):
- REPLAY prompt addition (entity extraction inline)
- Entity processing in task pipeline
- Known-entity fallback (name matching)
- Entity summary storage (inline from REPLAY, no second LLM call)
- Entity lifecycle events
- Entity-to-entity edges on co-occurrence
- Entity backfill script (DONE)
- Retrieval presentation (entity summaries as context blocks)