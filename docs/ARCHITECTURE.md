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

1. **Query-driven curriculum reweighting** (`hydrokg/enhancement/curriculum.py`) — changes
   which training examples the model sees next, based on a graph query.
2. **Graph-analogy correction** (`hydrokg/enhancement/graph_analogy_correction.py`) —
   changes the model's output after the forward pass, based on a graph traversal to
   similar, low-violation basins.
3. **Violation-history embeddings** (`hydrokg/enhancement/violation_embeddings.py`) —
   changes the model's input, by exposing each basin's own violation profile as an
   auxiliary feature.

None of these require gradients to flow through a hydrologic constraint. This is the
architectural bet the whole repository is built around, per an explicit project decision:
leverage the graph's relational/retrieval capability, not a PINN-style loss term.

## Two operating modes

### Offline (post-processing) audit

```
completed LSTM predictions (submodule's evaluate() pickle output)
        │
        ▼
hydrokg.audit.OfflineAuditor
        │  for each basin: run R0-R6 (hydrokg.rules.*) against the full time series
        ▼
GraphStore.write_violations()  +  GraphStore.set_basin_metrics(kge, violation_burden)
        │
        ▼
hydrokg.evaluation.skill_trust_analysis  →  is high KGE actually physically trustworthy?
hydrokg.viz.skill_trust_plots            →  the figure that shows the skill-trust gap
```

### Real-time (online) audit

```
LSTM inference loop, one prediction at a time
        │
        ▼
hydrokg.audit.RealtimeAuditor.ingest(basin, t, q_sim, q_obs, p)
        │
        ├── R0-R3 (daily): evaluated on this single row, immediately
        ├── R4 (event/water-year): evaluated once the water year closes
        └── R5, R6 (annual): evaluated once the water year closes
        │
        ▼
GraphStore accumulates violations continuously
        │
        ▼
hydrokg.enhancement.*  (curriculum / analogy correction / embeddings) consume the
graph's current state for the next training pass or the next inference call
```

The staging (daily immediately, event/annual only at window close) is a direct
implementation of the manuscript's requirement that "real-time auditing is staged rather
than simultaneous" — each rule activates exactly when its required temporal context
becomes available, not before.

## Graph backend: why two implementations exist

`hydrokg.graph.base.GraphStore` is an abstract interface with two implementations:

- `hydrokg.graph.memory_store.InMemoryGraphStore` — plain pandas, no server, used for
  every test in `tests/` and the `--demo` CLI paths. This is what was actually exercised
  and validated while building this repository (no Docker/Neo4j binary was available in
  that environment).
- `hydrokg.graph.neo4j_store.Neo4jGraphStore` — the production backend, using the
  official `neo4j` Python driver and real Cypher queries, intended for the full
  670-basin, multi-decade CAMELS run and CIROH-scale operational use.

Every rule, auditor, and enhancement mechanism is written against `GraphStore` only, never
against a specific backend, so switching from `memory` to `neo4j` (`--graph_backend` on
every CLI, or the `backend` argument to `hydrokg.graph.build_graph_store`) requires no
changes to rule or enhancement logic. `tests/test_neo4j_store.py` mirrors
`tests/test_graph_store_memory.py`'s exact assertions and is the acceptance test to run
against a live Neo4j instance before trusting the production backend.

## Why only violations are graph nodes, not every daily value

At 670 basins × ~30 years × 7 rules, materializing every daily (prediction, observation,
rule-check) triple would be on the order of 10⁸-10⁹ facts, most of them "rule not
violated" — of little value to any downstream consumer. `hydrokg/graph/schema.py`
documents this explicitly: only violations (the exception) are written as graph facts.
Curriculum reweighting, analogy correction, and violation embeddings all only need the
violation record, not the full daily series (which remains in the pandas
DataFrames/H5 files already produced by the submodule).

## Where the submodule fits

`external/HydroAuditToolFrameowrk` is a git submodule, **not modified in any way**. Its
hardcoded local paths (`camels_root`, `run_dir` in its committed `cfg.json` examples) and
lack of packaging (`setup.py`) are worked around entirely from `hydrokg/adapters/lstm_adapter.py`,
which is the single place that adds the submodule's root to `sys.path` and calls its
functions with config-driven arguments. If the submodule is ever updated upstream (e.g. a
new commit fixing its packaging), nothing in `hydrokg/` needs to change as long as the
function signatures in `data/datautils.py` and `Scripts/utils.py` stay stable.
