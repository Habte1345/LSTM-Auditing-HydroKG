# Architecture

## Why a knowledge graph, and why not a physics-informed loss

Physics-informed losses for hydrologic deep learning already exist: adding a
differentiable penalty term for water-balance closure or non-negativity to an LSTM's loss
function is well-trodden ground, and backpropagating through routing/storage-consistency
terms is expensive. That is explicitly *not* what this repository does anywhere.

HydroKG instead treats rule violations as **structured, queryable relational facts** —
which basin, which rule, which timestep, which aridity/land-cover class, how it relates to
other basins — and uses queries over that structure to drive three enhancement mechanisms
that sit entirely outside the loss function:

1. **Query-driven curriculum reweighting** (`src/hydrokg_enhancement.py::ViolationCurriculumSampler`) —
   changes which training examples the model sees next, based on a graph query.
2. **Graph-analogy correction** (`src/hydrokg_enhancement.py::GraphAnalogyCorrector`) —
   changes the model's output after the forward pass, based on a graph traversal to
   similar, low-violation basins.
3. **Violation-history embeddings** (`src/hydrokg_enhancement.py::build_embedding_matrix`) —
   changes the model's input, by exposing each basin's own violation profile as an
   auxiliary feature.

None of these require gradients to flow through a hydrologic constraint. This is the
architectural bet the whole repository is built around: leverage the graph's
relational/retrieval capability, not a PINN-style loss term.

## Two operating modes

### Offline (post-processing) audit

```
completed LSTM predictions (submodule's evaluate() pickle output)
        │
        ▼
hydrokg_audit.OfflineAuditor
        │  for each basin: run R0-R6 (hydrokg_rules.*) against the full time series
        ▼
GraphStore.write_violations()  +  GraphStore.set_basin_metrics(kge, violation_burden)
        │
        ▼
hydrokg_evaluation.summarize_skill_trust  →  is high KGE actually physically trustworthy?
hydrokg_viz.plot_skill_trust_scatter      →  the figure that shows the skill-trust gap
```

### Real-time (online) detection during fine-tuning

The real-time mechanism lives inline inside `EnhancedTrainingPipeline.fine_tune()`
(`src/hydrokg_enhancement.py`), scoped specifically to what training-time detection can
actually support:

```
Training loop, one batch at a time (inside fine_tune())
        │
        ▼
every batch's own forward-pass output, rescaled to physical mm/day
        │
        ├── R0-R3 (daily, no calendar context needed): checked immediately against
        │   this batch's own output -- zero extra forward passes, detached from the
        │   loss/backward pass entirely
        │
        └── R4, R5, R6: NOT evaluated here -- they need a full water-year of
            calendar-dated observations, which an isolated training sequence window
            doesn't carry. These remain audit-only (OfflineAuditor, before/after
            training), not real-time.
        │
        ▼
GraphStore accumulates R0-R3 violations continuously, written the instant they're detected
        │
        ▼
Between epochs: curriculum weights + violation embeddings (src/hydrokg_enhancement.py)
are recomputed from the graph's current state -- reflecting the model's own most recent
training-time behavior, not a frozen pre-training snapshot
```

This is a real, stated scope limit: "real-time" in this codebase means R0-R3 during
training, not all seven rules. State this explicitly in the manuscript rather than
implying uniform real-time treatment across the rule set.

## Graph backend: why two implementations exist

`hydrokg_graph.GraphStore` is an abstract interface with two implementations:

- `hydrokg_graph.InMemoryGraphStore` — plain pandas, no server required. The default
  backend for all CLI runs (`--graph_backend memory`).
- `hydrokg_graph.Neo4jGraphStore` — the production backend, using the official `neo4j`
  Python driver and real Cypher queries, intended for the full 670-basin, multi-decade
  CAMELS run and CIROH-scale operational use (`--graph_backend neo4j`; see
  `docker-compose.yml`).

Every rule, auditor, and enhancement mechanism is written against `GraphStore` only, never
against a specific backend, so switching from `memory` to `neo4j` requires no changes to
rule or enhancement logic. Validate `Neo4jGraphStore` against a live instance before
trusting it in production.

## Why only violations are graph nodes, not every daily value

At 670 basins × ~30 years × 7 rules, materializing every daily (prediction, observation,
rule-check) triple would be on the order of 10⁸–10⁹ facts, most of them "rule not
violated" — of little value to any downstream consumer. `src/hydrokg_graph.py`
documents this explicitly: only violations (the exception) are written as graph facts.
Curriculum reweighting, analogy correction, and violation embeddings all only need the
violation record, not the full daily series (which remains in the pandas
DataFrames/H5 files already produced by the submodule).

## Where the submodule fits

`external/HydroAuditToolFrameowrk` is a git submodule, **not modified in any way**. Its
hardcoded local paths (`camels_root`, `run_dir` in its committed `cfg.json` examples) and
lack of packaging (`setup.py`) are worked around entirely from `src/hydrokg_adapters.py`,
which is the single place that adds the submodule's root to `sys.path` and calls its
functions with config-driven arguments. If the submodule is ever updated upstream (e.g. a
new commit fixing its packaging), nothing in `src/` needs to change as long as the
function signatures in `data/datautils.py` and `Scripts/utils.py` stay stable.
