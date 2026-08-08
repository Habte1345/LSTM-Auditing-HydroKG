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

## Repository layout — fully flat, no subfolders in src/ or scripts/

```
LSTM-Auditing-HydroKG/
├── src/
│   ├── hydrokg_rules.py         R0-R6 + base class + registry
│   ├── hydrokg_graph.py         GraphStore interface + in-memory + Neo4j backends
│   ├── hydrokg_audit.py         OfflineAuditor + violation burden (Eq. 3)
│   ├── hydrokg_data.py          Precipitation loading, ET residual, aridity/land-cover strat
│   ├── hydrokg_adapters.py      The ONLY file that imports the untouched submodule
│   ├── hydrokg_enhancement.py   All 3 enhancement mechanisms + the training pipeline
│   ├── hydrokg_evaluation.py    KGE, skill-trust analysis, Eq. 4-6, stratified summaries
│   ├── hydrokg_viz.py           Publication-quality figures
│   └── hydrokg_ontology.ttl     RDF/OWL schema (source of truth)
├── scripts/
│   ├── run_offline_audit.py         python scripts/run_offline_audit.py ...
│   ├── run_enhanced_training.py     python scripts/run_enhanced_training.py ...
│   ├── init_neo4j_schema.cypher     standalone schema init, mirrors hydrokg_graph.py
│   └── run_enhancement_uahpc.slurm  SLURM submission template
├── external/HydroAuditToolFrameowrk/  <- git submodule, UNTOUCHED (Kratzert et al. LSTM)
├── data/                    Put/symlink your CAMELS_US dataset here (gitignored)
├── results/                 CLI output lands here (gitignored)
├── figures/                 Generated figures land here (gitignored)
├── notebook/                Real-data results analysis notebook (no synthetic data)
├── docs/                    ARCHITECTURE.md, ONTOLOGY.md, RULES.md, METHODOLOGY.md
├── configs/, docker-compose.yml
├── requirements.txt, requirements-torch.txt, requirements-neo4j.txt
```

No `tests/` directory for now (removed intentionally — add back once the offline and
real-time simulation runs are validated end to end). No installed package either —
`scripts/*.py` add the sibling `src/` directory to `sys.path` themselves, so nothing
needs `pip install -e .`; just install the dependencies.

## Installation

```bash
git clone --recurse-submodules https://github.com/<your-username>/LSTM-Auditing-HydroKG.git
cd LSTM-Auditing-HydroKG
pip install -r requirements-torch.txt   # needed for the real fine-tuning pipeline
docker compose up -d neo4j              # optional; in-memory backend needs no server
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## Usage — real data only, no demo mode

**1. Audit a completed LSTM run:**

```bash
python scripts/run_offline_audit.py \
  --predictions_pickle external/HydroAuditToolFrameowrk/runs/<run_dir>/lstm_seed<seed>.p \
  --camels_root data/CAMELS_US \
  --stratification_db external/HydroAuditToolFrameowrk/runs/<run_dir>/attributes.db \
  --output_csv results/baseline_results.csv
```

**2. Run the full graph-guided enhancement pipeline** (baseline audit -> real-time
fine-tuning -> regenerate predictions -> graph-analogy correction -> final audit):

```bash
python scripts/run_enhanced_training.py \
  --run_dir external/HydroAuditToolFrameowrk/runs/<run_dir> \
  --camels_root data/CAMELS_US \
  --predictions_pickle external/HydroAuditToolFrameowrk/runs/<run_dir>/lstm_seed<seed>.p \
  --n_epochs 3 \
  --output_prefix results/hydrokg_run1
```

**3. Analyze results:** open `notebook/analyze_results.ipynb`, point it at your
`results/hydrokg_run1_*` files, and it produces the skill-trust and enhancement figures
into `figures/`.

**HPC (SLURM):** `sbatch scripts/run_enhancement_uahpc.slurm` (edit the paths at the top
of that file for your setup first).

## Citation

If you use this framework, please cite the associated manuscript (in preparation):
*Auditing and Improving LSTM Streamflow Predictions with Hydrologic Knowledge Graphs*,
Dagne, H. and Mekonnen, M. (University of Alabama, Water Footprint Lab).

## License

MIT, consistent with the upstream `HydroAuditToolFrameowrk` submodule.
