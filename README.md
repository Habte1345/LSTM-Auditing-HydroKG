# LSTM-Auditing-HydroKG

**Auditing and Improving LSTM Streamflow Predictions with Hydrologic Knowledge Graphs**

HydroKG is a knowledge-graph-based auditing and enhancement framework for data-driven
(LSTM) streamflow models. It targets the **skill-trust gap**: an LSTM can score well on
KGE/NSE while still violating basic hydrologic physics (negative flow, broken water
balance, mistimed peaks, Budyko inconsistency), and standard skill metrics do not surface
this.

HydroKG operationalizes seven physically interpretable rules (R0-R6) as a queryable
knowledge graph over predictions, observations, basin attributes, and time context, and
uses the graph itself — not a differentiable physics-informed loss term — as the
mechanism for model enhancement. This is the central methodological distinction from
physics-informed neural network (PINN) approaches: HydroKG does not add a hydrologic
penalty to the loss function. Physics-informed losses for hydrology already exist and are
computationally expensive to backpropagate through routing/storage terms. Instead, HydroKG
treats rule violations as **structured, queryable relational information** and uses that
structure for:

1. **Query-driven curriculum reweighting** — the graph is queried after each
   epoch/streaming window for violation clusters (by basin, rule, aridity class, land-cover
   class), and that query result determines training-sample weighting in the next pass.
2. **Graph-analogy correction at inference** — when a rule fires for basin *b* at time
   *t*, the graph is traversed to structurally similar, low-violation basins, and their
   behavior under comparable conditions informs a graph-derived correction, rather than a
   fixed physical clip.
3. **Violation-history embeddings** — each basin accumulates a violation profile vector
   from the graph, usable as an auxiliary feature for a fine-tuning pass.

## Repository status

**Alpha.** This repository implements the full architecture end-to-end against synthetic
data and an in-memory graph substitute (see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for why). The production path targets a real Neo4j instance (via `docker-compose.yml`) and
the pretrained LSTM in the
[`HydroAuditToolFrameowrk`](https://github.com/Habte1345/HydroAuditToolFrameowrk) submodule
run against CAMELS. Nothing here has been validated on a live 670-basin CAMELS run yet —
that is the next step once you have CAMELS data and a running Neo4j instance available.

## Why this is novel

No prior work applies knowledge-graph auditing to post-hoc or online evaluation of
hydrologic deep learning models. Existing physics-informed ML approaches for streamflow
encode physical constraints as differentiable loss terms (PINN-style). HydroKG instead
encodes physical rules as a **relational, queryable structure** that supports both
diagnostic auditing (offline mode) and non-differentiable, retrieval-based model
improvement (online mode) — treating the rule-violation graph as an active reasoning layer
rather than a loss regularizer.

## Two operating modes

| Mode | When it runs | What it does |
|---|---|---|
| **Offline audit** | After a completed LSTM training/evaluation run | Applies R0-R6 to the full prediction set, computes per-basin violation burden (Eq. 3), and compares it against KGE to quantify the skill-trust relationship |
| **Real-time audit** | Embedded into the training/inference loop | Evaluates rules in stages as their temporal context becomes available (R0-R3 per timestep, R4 per event window, R5-R6 per annual window), feeds violations into the graph continuously, and drives the three enhancement mechanisms above |

## Architecture at a glance

```
external/HydroAuditToolFrameowrk/   <- git submodule, UNTOUCHED (Kratzert et al. LSTM)
hydrokg/
  ontology/     RDF/OWL schema (Turtle) defining the graph's entities & relationships
  graph/        GraphStore interface + Neo4j backend + in-memory dev/test backend
  adapters/     Wraps the submodule's CLI/outputs without modifying it
  data/         Precipitation loading (CAMELS forcing) + ET-as-residual computation
  rules/        R0-R6, each a standalone, testable rule module
  audit/        Offline post-processing auditor + real-time staged auditor
  enhancement/  Curriculum reweighting, graph-analogy correction, violation embeddings
  evaluation/   Skill-trust analysis (Eq. 3) and enhancement metrics (Eq. 4-6)
  viz/          Publication-quality figures
  cli/          Entry points (`hydrokg-audit`, `hydrokg-enhance`, ...)
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/ONTOLOGY.md`](docs/ONTOLOGY.md),
[`docs/RULES.md`](docs/RULES.md), and [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for full
detail, including the specific places where the manuscript draft's rule definitions were
ambiguous or numerically underspecified and what this implementation assumes instead (flagged
explicitly rather than silently resolved).

## Installation

```bash
git clone --recurse-submodules https://github.com/<your-username>/LSTM-Auditing-HydroKG.git
cd LSTM-Auditing-HydroKG
pip install -e .
docker compose up -d neo4j   # production graph backend; optional for dev (in-memory works without it)
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## Quick start (synthetic data, no Neo4j, no CAMELS needed)

```bash
python -m hydrokg.cli.run_offline_audit --demo
```

This runs the full offline audit pipeline against synthetic basin predictions using the
in-memory graph backend, and prints/plots the skill-trust relationship. See
[`notebooks/demo_end_to_end.ipynb`](notebooks/demo_end_to_end.ipynb) for a walkthrough of
every stage, offline and real-time, both enhancement paths, and the resulting figures.

## Real usage against your CAMELS run

```bash
python -m hydrokg.cli.run_offline_audit \
  --config configs/default_config.yaml \
  --camels_root /path/to/CAMELS_US \
  --predictions_pickle external/HydroAuditToolFrameowrk/runs/<run_dir>/lstm_seed<seed>.p \
  --graph_backend neo4j
```

## Citation

If you use this framework, please cite the associated manuscript (in preparation):
*Auditing and Improving LSTM Streamflow Predictions with Hydrologic Knowledge Graphs*,
Dagne, H. and Mekonnen, M. (University of Alabama, Water Footprint Lab).

## License

MIT, consistent with the upstream `HydroAuditToolFrameowrk` submodule.
