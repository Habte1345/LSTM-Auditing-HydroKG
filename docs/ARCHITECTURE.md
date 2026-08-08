# Architecture

## Design rationale

HydroKG represents detected rule violations as structured, queryable relational facts,
linking each violation to its basin, rule, timestep, and hydroclimatic context, rather
than as a differentiable penalty term added to the training loss. This design choice
separates physical-consistency evaluation from the optimization objective: the model is
trained throughout on the unmodified basin-normalized NSE loss, and violation information
instead drives three mechanisms that operate outside the loss function entirely:

1. **Curriculum reweighting** (`hydrokg_enhancement.py::ViolationCurriculumSampler`) —
   determines which training examples the model sees in the following epoch, based on a
   graph query over each basin's accumulated violation count.
2. **Graph-analogy correction** (`hydrokg_enhancement.py::GraphAnalogyCorrector`) —
   modifies the model's output after the forward pass, based on a graph traversal to
   structurally similar, low-violation basins.
3. **Violation-history embedding** (`hydrokg_enhancement.py::build_embedding_matrix`) —
   modifies the model's input by exposing each basin's own violation profile as an
   auxiliary static feature.

None of these mechanisms require gradients to flow through a hydrologic constraint.

## Two operating modes

### Offline (post-processing) audit

```
Completed LSTM predictions
        │
        ▼
hydrokg_audit.OfflineAuditor
        │   evaluates R0-R6 against the full time series for each basin
        ▼
GraphStore.write_violations()  +  GraphStore.set_basin_metrics(kge, violation_burden)
        │
        ▼
hydrokg_evaluation.summarize_skill_trust   — basin-level skill-trust relationship
hydrokg_viz.plot_skill_trust_scatter       — corresponding figure
```

### Real-time (online) detection during fine-tuning

The real-time mechanism operates inside `EnhancedTrainingPipeline.fine_tune()`
(`hydrokg_enhancement.py`), scoped to the rules that a single training batch can support:

```
Training loop, one batch at a time
        │
        ▼
Batch output rescaled to physical units (mm/day)
        │
        ├── R0-R3 (daily-scale, no calendar context required): evaluated immediately
        │   against this batch's own output, detached from the loss and backward pass
        │
        └── R4-R6 (event- and annual-scale): not evaluated here; a full water-year of
            calendar-dated observations is required, which an isolated training
            sequence does not carry. These remain restricted to offline auditing,
            before and after training.
        │
        ▼
GraphStore accumulates R0-R3 violations continuously as they are detected
        │
        ▼
Between epochs: curriculum weights and violation embeddings are recomputed from the
graph's current state, reflecting the model's most recent training-time behavior
```

Real-time evaluation in this framework refers specifically to R0-R3; R4-R6 are audited
only offline. This scope follows from the temporal context each rule requires, not from
an arbitrary implementation choice.

## Graph backend

`hydrokg_graph.GraphStore` is an abstract interface with two implementations:

- `InMemoryGraphStore` — pandas-based, requires no server. Default backend for all CLI
  runs (`--graph_backend memory`).
- `Neo4jGraphStore` — Cypher-query backend using the official `neo4j` Python driver,
  intended for large-scale, multi-decade runs (`--graph_backend neo4j`; see
  `docker-compose.yml`).

Every rule, auditor, and enhancement mechanism is written against the `GraphStore`
interface, not against a specific backend, so switching between `memory` and `neo4j`
requires no change to rule or enhancement logic.

## Graph granularity

At 670 basins, multi-decade daily records, and seven rules, materializing every
(prediction, observation, rule-check) triple would produce on the order of 10⁸-10⁹ facts,
the overwhelming majority representing non-violations. `hydrokg_graph.py` therefore writes
only detected violations as graph facts. Curriculum reweighting, analogy correction, and
violation embeddings each require only the violation record, not the full daily series,
which remains available in the underlying data files produced by the baseline LSTM
pipeline.

## Interface to the baseline LSTM

`external/HydroAuditToolFrameowrk` is a git submodule containing the baseline LSTM
implementation (Kratzert et al., 2019) and is not modified. `src/hydrokg_adapters.py` is
the sole interface to this submodule: it adds the submodule root to `sys.path` and calls
its functions with configuration-driven arguments, so that changes to the submodule do not
require changes elsewhere in this codebase as long as its function signatures remain
stable.
