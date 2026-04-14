# Extension Architecture: Three-Layer Storage Model

## Overview

Engram uses three distinct storage mechanisms, each serving a different purpose. Understanding which layer owns which data prevents overlap and keeps the architecture clean.

## The Three Layers

### 1. Graph (edges + STDP) — Relationships Between Memories

Edges represent how memories connect, reinforce, and decay over time. This is the nervous system layer.

- **What it stores:** Directional weighted connections between memory nodes
- **Edge types:** excitatory, inhibitory, associative, temporal, modulatory, structural
- **STDP:** Forward/backward weights for spike-timing-dependent plasticity
- **Owner:** Engram (Dreamer service classifies edges; reconsolidation strengthens co-retrieved edges)
- **Not for:** Data about a memory. Edges describe relationships, not attributes.

### 2. JSONB Metadata — Memory-Level Flags

The `metadata` JSONB column on each memory node holds flags that affect how engram treats the node. This is the per-node control surface.

- **What it stores:** Operational flags and cross-references
- **Examples:** `pinned` (prevents decay), `trust_level` (affects activation scaling), `enmotiv_id` (foreign key to external system)
- **Owner:** Written by external systems, read by engram's decay/retrieval logic
- **Not for:** Domain-specific structured payloads (calories, task status, clinical scores). Those belong in the external system's typed tables.

### 3. metadata_type — Extension Type Discriminator

The `metadata_type` VARCHAR(50) column is a promoted field that enables type-filtered recall. It tells engram's retrieval pipeline "only search within this subset" without engram interpreting the value.

- **What it stores:** A single string identifying the extension type (e.g., `nutrition_log`, `project_task`, `clinical_note`)
- **Why a column and not JSONB:** B-tree indexes compose cleanly with pgvector HNSW for filtered vector search. JSONB GIN indexes do not.
- **Query plan:** partition prune (owner_id) → B-tree filter (metadata_type) → HNSW vector scan
- **Owner:** Written by external systems at memory creation time. Engram indexes it, does not validate it.
- **Not for:** Storing the extension payload. The type discriminator says *what kind* of structured data exists; the data itself lives in the external system's typed tables.

## Where Extension Data Lives

External systems (like enmotiv) that define data type extensions follow this pattern:

| Data | Where | Why |
|------|-------|-----|
| Memory content + embeddings | Engram `memory_nodes` | Engram's core competency: dimensional encoding + vector search |
| Edge relationships | Engram `edges` | Engram's nervous system: STDP, spreading activation, decay |
| Memory-level flags | Engram `metadata` JSONB | Affects how engram treats the node (pinned, trust, cross-refs) |
| Extension type tag | Engram `metadata_type` column | Enables type-filtered recall at the vector search level |
| Structured extension payload | External system's typed tables | Real columns, real constraints, real indexes. SQL integrity. |

### Example: Nutrition Extension

```
Engram memory_nodes:
  content: "Had a large breakfast — eggs, toast, coffee. Felt energized."
  metadata_type: "nutrition_log"
  metadata: {"enmotiv_id": "abc-123", "trust_level": "emerging"}

Enmotiv nutrition_entries table:
  engram_memory_id: "..."
  enmotiv_memory_id: "abc-123"
  calories: 650
  protein_g: 28
  meal_type: "breakfast"
  logged_at: "2026-04-13T08:30:00Z"
```

**Query: "recall nutrition memories about energy"**
→ Engram recall with `metadata_type='nutrition_log'` + cue "energy"
→ Pre-filters to nutrition memories, then runs 6-axis vector search
→ Returns convergence-scored results

**Query: "average calories this week"**
→ Enmotiv queries its own `nutrition_entries` table
→ Standard SQL aggregation with real indexes
→ Engram not involved

**Query: "show all nutrition logs"**
→ Engram `GET /v1/memories?metadata_type=nutrition_log`
→ B-tree filtered pagination

## The Rule

> If engram reads it to decide how to treat a node, it's JSONB metadata.
> If the external system reads it to answer a domain question, it's a typed table.
> If it connects two nodes, it's an edge.
> If it discriminates what type of extension data exists, it's `metadata_type`.

When someone asks "where does this field go?" there should be exactly one answer.

## API Surface

### Creating typed memories

```
POST /v1/memories
{
  "content": "Had a large breakfast...",
  "source_type": "observation",
  "metadata_type": "nutrition_log",
  "metadata": {"enmotiv_id": "abc-123"}
}
```

### Type-filtered recall

```
POST /v1/recall
{
  "cue": "energy levels after eating",
  "metadata_type": "nutrition_log",
  "top_k": 5
}
```

`metadata_type` is applied as a **pre-filter** in the WHERE clause before HNSW vector search, not as a post-filter. This means only memories of the specified type are candidates for convergence scoring.

### Listing by type

```
GET /v1/memories?metadata_type=nutrition_log&sort=created_at:desc&limit=20
```

## Migration

Migration `007_metadata_type.sql` adds:
- `metadata_type VARCHAR(50) DEFAULT NULL` — nullable, existing memories unaffected
- Partial B-tree index on `metadata_type` where not null and not deleted
