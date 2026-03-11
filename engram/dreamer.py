"""Backward compatibility — use engram.services.dreamer instead."""

from engram.services.dreamer import (
    decay_activations,
    decay_edge_weights,
    dedup_memories,
    process_new_memories,
    prune_edges,
    run_cycle,
)

__all__ = [
    "decay_activations",
    "decay_edge_weights",
    "dedup_memories",
    "process_new_memories",
    "prune_edges",
    "run_cycle",
]
