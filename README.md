# LSTM-Auditing-HydroKG

**Auditing and Improving LSTM Streamflow Predictions with Hydrologic Knowledge Graphs**

HydroKG is a knowledge-graph-based auditing and enhancement framework for data-driven
(LSTM) streamflow models. It targets the **skill-trust gap**: an LSTM can score well on
KGE/NSE while still violating basic hydrologic physics (negative flow, broken water
balance, mistimed peaks, Budyko inconsistency), and standard skill metrics do not surface
this. On a real 670-basin CAMELS run, ~80% of basins scored KGE >= 0.5, and 100% of those
still violated at least one physical rule.

HydroKG operationalizes seven physically interpretable rules (R0-R6) as a queryable
knowledge graph over predictions, observations, basin attributes, and time context, and
uses the graph itself — not a differentiable physics-informed loss term — as the
mechanism for model enhancement. Physics-informed losses for hydrology already exist and
are expensive to backpropagate through routing/storage terms. HydroKG instead treats rule
violations as structured, queryable relational information and uses that structure for:

1. **Query-driven curriculum reweighting** — basins/days with more violations get
   resampled more often in training, recomputed between epochs from the graph's current
   state (not a frozen pre-training snapshot).
2. **Violation-history embeddings** — each basin's own violation-rate profile becomes an
   extra static input feature, injected via a copy of the LSTM's own `attributes.db`.
3. **Graph-analogy correction at inference** — flagged predictions are corrected using
   structurally similar, low-violation basins, applied once after training as an explicit
   post-processing step.

**Real-time scope, stated plainly:** only R0-R3 (the four rules needing just a single
timestep, no calendar-date window) are detected live during training, directly from every
batch's own forward pass, at zero extra inference cost. R4 (peak timing) and R5/R6 (mass
balance, Budyko) need a full water-year of calendar-dated data an isolated training
sequence can't carry — they remain audit-only (before/after training). See
`docs/ARCHITECTURE.md`.

## Repository layout

```
LSTM-Auditing-HydroKG/
├── src/hydrokg/            <- the package (9 files total, see below)
│   ├── rules.py             R0-R6 + base class + registry
│   ├── graph.py             GraphStore interface + in-memory + Neo4j backends + factory
│   ├── audit.py             OfflineAuditor + violation burden (Eq. 3)
│   ├── data.py              Precipitation loading, ET residual, aridity/land-cover strat
│   ├── adapters.py          The ONLY place that imports the untouched submodule
│   ├── enhancement.py       All 3 enhancement mechanisms + the training pipeline
│   ├── evaluation.py        KGE, skill-trust analysis, Eq. 4-6, stratified summaries
│   ├── viz.py               Publication-quality figures
│   ├── ontology/hydrokg_ontology.ttl   RDF/OWL schema (source of truth)
│   └── cli/
│       ├── run_offline_audit.py       `python -m hydrokg.cli.run_offline_audit`
│       └── run_enhanced_training.py   `python -m hydrokg.cli.run_enhanced_training`
├── external/HydroAuditToolFrameowrk/  <- git submodule, UNTOUCHED (Kratzert et al. LSTM)
├── scripts/                 Neo4j schema init, UAHPC SLURM submission template
├── data/                    Put/symlink your CAMELS_US dataset here (gitignored)
├── results/                 CLI output lands here (gitignored)
├── figures/                 Generated figures land here (gitignored)
├── notebook/                Real-data results analysis notebook (no synthetic data)
├── docs/                    ARCHITECTURE.md, ONTOLOGY.md, RULES.md, METHODOLOGY.md
├── configs/, docker-compose.yml, pyproject.toml
```

No `tests/` directory for now (removed intentionally — see project history; add it back
once the offline and real-time simulation runs are validated end to end).

## Installation

```bash
git clone --recurse-submodules https://github.com/<your-username>/LSTM-Auditing-HydroKG.git
cd LSTM-Auditing-HydroKG
pip install -e ".[torch]"          # torch, numba, h5py, scikit-learn -- needed for the real pipeline
docker compose up -d neo4j         # optional; in-memory backend needs no server
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## Usage — real data only, no demo mode

**1. Audit a completed LSTM run:**

```bash
python -m hydrokg.cli.run_offline_audit \
  --predictions_pickle external/HydroAuditToolFrameowrk/runs/<run_dir>/lstm_seed<seed>.p \
  --camels_root data/CAMELS_US \
  --stratification_db external/HydroAuditToolFrameowrk/runs/<run_dir>/attributes.db \
  --output_csv results/baseline_results.csv
```

**2. Run the full graph-guided enhancement pipeline** (baseline audit -> real-time
fine-tuning -> regenerate predictions -> graph-analogy correction -> final audit):

```bash
python -m hydrokg.cli.run_enhanced_training \
  --run_dir external/HydroAuditToolFrameowrk/runs/<run_dir> \
  --camels_root data/CAMELS_US \
  --predictions_pickle external/HydroAuditToolFrameowrk/runs/<run_dir>/lstm_seed<seed>.p \
  --n_epochs 3 \
  --output_prefix results/hydrokg_run1
```

**3. Analyze results:** open `notebook/analyze_results.ipynb`, point it at your
`results/hydrokg_run1_*` files, and it produces the skill-trust and enhancement figures
into `figures/`.

**HPC (SLURM):** see `scripts/run_enhancement_uahpc.slurm` for a submission template.

## Citation

If you use this framework, please cite the associated manuscript (in preparation):
*Auditing and Improving LSTM Streamflow Predictions with Hydrologic Knowledge Graphs*,
Dagne, H. and Mekonnen, M. (University of Alabama, Water Footprint Lab).

## License

MIT, consistent with the upstream `HydroAuditToolFrameowrk` submodule.
