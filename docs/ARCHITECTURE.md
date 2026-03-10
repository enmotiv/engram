# Architecture

## Overview

Engram is a memory storage and retrieval service. It accepts text memories, embeds them across six perceptual dimensions, and retrieves them using convergence scoring with axis-aware spreading activation.

```
Client → API (FastAPI) → Write Path → PostgreSQL + pgvector
                       → Read Path  → Multi-axis search → Convergence → Spreading Activation → Response
                       → Dreamer    → Edge classification → Decay → Dedup → Prune
```

## Six Dimensions

Every memory is embedded into six separate vectors, each with a different perceptual prefix:

| Dimension | What It Captures |
|-----------|-----------------|
| **Temporal** | Sequential and temporal context |
| **Emotional** | Emotional tone, urgency, significance |
| **Semantic** | Core meaning and concepts |
| **Sensory** | Specific facts, names, numbers, identifiers |
| **Action** | Actions, goals, behavioral context |
| **Procedural** | Repeated patterns, routines, sequences |

Each dimension gets its own HNSW index in PostgreSQL via pgvector. This enables retrieval that matches how human memory works — a smell can trigger a memory that a keyword search would miss.

## Write Path

```
Content → Dedup Check → 6-Axis Embedding → Salience Scoring → DB Insert
```

1. **Dedup check**: SHA-256 content hash compared against existing nodes (per owner).
2. **Embedding**: One batched OpenRouter API call produces six vectors (one per dimension prefix + content).
3. **Salience**: Cosine similarity between the emotional vector and a cached "urgency anchor" vector. Range [0.1, 1.0].
4. **Insert**: Single INSERT with all six vectors, metadata, and salience score.

The DB connection is released during the embedding call (the slow part), then re-acquired for the fast insert.

## Read Path

```
Cue → 6-Axis Embedding → Per-Dimension Search → Convergence Scoring
    → Spreading Activation → Post-Filter → Reconsolidation → Response
```

### Multi-Axis Search

Six sequential `ORDER BY vec <=> cue LIMIT 20` queries, one per dimension. Results are unioned into a candidate set with per-axis scores.

### Convergence Scoring

For each candidate:
```
convergence = dims_matched × avg_score_across_matched_dims
adjusted = convergence × activation_level
```

A memory that matches on 4 dimensions with moderate scores ranks higher than one that matches perfectly on 1 dimension.

### Spreading Activation

One hop through the edge graph. Edges are loaded for seed nodes, filtered by matched axes:

- **Excitatory** edges: `delta = weight × src_activation × 0.5` (additive)
- **Inhibitory** edges: same magnitude, subtracted
- **Associative**: factor 0.25
- **Temporal**: factor 0.15
- **Modulatory**: `activation = current × weight` (multiplicative)

### Reconsolidation

After retrieval, returned nodes are boosted:
- `access_count += 1`
- `activation_level = LEAST(1.0, activation + 0.1)`
- Co-retrieved edges on shared axes are strengthened by 0.05

This implements memory reconsolidation — recalling a memory makes it stronger.

### Confidence

Based on the top result:
- **High**: 4+ dimensions matched AND convergence >= 3.0
- **Medium**: 2+ dimensions matched OR convergence >= 1.5
- **Low**: everything else

## The Dreamer

Background process that maintains the edge graph. Runs per-owner:

### Edge Classification

For each unprocessed memory, find candidates by vector similarity, then use an LLM to classify relationships on all six axes:

- **Excitatory**: A reinforces B
- **Inhibitory**: A contradicts or replaces B
- **Associative**: A and B are related but independent
- **Temporal**: A and B are sequential
- **Modulatory**: A changes how B should be interpreted

### Maintenance Jobs

| Job | Frequency | What It Does |
|-----|-----------|-------------|
| **Activation Decay** | Hourly | Nodes not accessed in 24h: `activation *= 0.95`, floor 0.01 |
| **Edge Decay** | Weekly | Edges not updated in 30 days: `weight *= 0.9` |
| **Dedup** | Weekly | Merge nodes with >0.95 semantic similarity, transfer edges |
| **Pruning** | Monthly | Delete edges with weight < 0.05 |

## Tenant Isolation

Every query filters by `owner_id`. PostgreSQL Row-Level Security policies enforce this at the database level as defense-in-depth. The `tenant_connection` context manager sets `app.owner_id` on each connection.

## Observability

- **Structured logging**: JSON via structlog with correlation_id and owner_id injected via ContextVars
- **Request tracing**: `?trace=true` on /recall returns per-stage timing and diagnostics
- **Prometheus metrics**: Histograms for recall/write/embed/dreamer latency, counters for errors, gauges for node/edge counts
- **Span timing**: Every pipeline stage is wrapped in a `Span` context manager that logs duration and warns on slowness
