# LSTM-Auditing-HydroKG

**Auditing and Improving LSTM Streamflow Predictions with Hydrologic Knowledge Graphs**

HydroKG is a hydrology-specific knowledge graph framework that audits Long Short-Term
Memory (LSTM) streamflow predictions against seven physically interpretable rules and
uses detected violations to guide model enhancement. It addresses the skill-trust gap in
data-driven streamflow prediction: an LSTM can achieve high Kling-Gupta Efficiency (KGE)
while still violating basic hydrologic expectations, such as negative discharge, mistimed
peaks, or water-balance and Budyko inconsistency. Applied to 670 CAMELS basins, many
basins with moderate-to-high KGE were found to exhibit substantial rule violations, with
violation type and severity varying systematically with aridity and land cover.

HydroKG represents predictions, observations, basin attributes, and detected violations as
connected facts in a knowledge graph, implemented in the Resource Description Framework
(RDF), rather than as a modified loss function. Rule violations inform training through
three non-differentiable, graph-guided mechanisms:

1. **Curriculum reweighting** — each basin's cumulative violation count sets its sampling
   weight for the following training epoch, recomputed from the graph's current state
   after every epoch.
2. **Violation-history embedding** — each basin's complete seven-rule violation profile is
   added as a static input feature, giving the model direct access to its own violation
   history.
3. **Graph-analogy correction** — after training, flagged predictions are corrected once
   using structurally similar, low-violation basins.

None of these mechanisms modify the loss function; the model trains throughout on the
unmodified basin-normalized NSE loss.

**Scope of real-time detection.** Only four of the seven rules (R0-R3) can be evaluated
during training, since they require only the current simulated and observed discharge.
The remaining three rules (R4-R6) require an event window or a full annual cycle
unavailable within a single training batch, and are evaluated only through offline
auditing, before and after training. See `docs/ARCHITECTURE.md` for the full description
of both operating modes.

## Repository layout

```
LSTM-Auditing-HydroKG/
├── src/
│   ├── hydrokg_rules.py         Seven auditing rules (R0-R6) and rule registry
│   ├── hydrokg_graph.py         GraphStore interface, in-memory and Neo4j backends
│   ├── hydrokg_audit.py         Offline auditor and violation-burden computation (Eq. 3)
│   ├── hydrokg_data.py          Forcing data loading and aridity/land-cover stratification
│   ├── hydrokg_adapters.py      Interface to the baseline LSTM submodule
│   ├── hydrokg_enhancement.py   Enhancement mechanisms and fine-tuning pipeline
│   ├── hydrokg_evaluation.py    KGE, skill-trust analysis, enhancement metrics (Eq. 4-6)
│   ├── hydrokg_viz.py           Figure generation
│   └── hydrokg_ontology.ttl     RDF/OWL schema
├── scripts/
│   ├── run_offline_audit.py         Offline audit of a completed LSTM run
│   ├── run_enhanced_training.py     Full enhancement pipeline
│   ├── init_neo4j_schema.cypher     Standalone Neo4j schema initialization
│   └── run_enhancement_uahpc.slurm  SLURM submission template
├── external/HydroAuditToolFrameowrk/  Baseline LSTM implementation (git submodule)
├── data/                    CAMELS_US dataset location (not tracked)
├── results/                 Pipeline output (not tracked)
├── figures/                 Generated figures
├── notebook/                Results analysis notebook
├── docs/                    Architecture, ontology, rule specification, and methodology
├── configs/, docker-compose.yml
└── requirements.txt, requirements-torch.txt, requirements-neo4j.txt
```

`scripts/*.py` add the sibling `src/` directory to `sys.path` directly; no package
installation is required beyond the dependencies listed in `requirements*.txt`.

## Installation

```bash
git clone --recurse-submodules https://github.com/Habte1345/LSTM-Auditing-HydroKG.git
cd LSTM-Auditing-HydroKG
pip install -r requirements-torch.txt
docker compose up -d neo4j   # optional; the in-memory backend requires no server
```

If the repository was cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## Usage

**1. Audit a completed LSTM run:**

```bash
python scripts/run_offline_audit.py \
  --predictions_pickle external/HydroAuditToolFrameowrk/runs/<run_dir>/lstm_seed<seed>.p \
  --camels_root data/CAMELS_US \
  --stratification_db external/HydroAuditToolFrameowrk/runs/<run_dir>/attributes.db \
  --output_csv results/baseline_results.csv
```

**2. Run the full enhancement pipeline** (baseline audit, real-time fine-tuning,
prediction regeneration, graph-analogy correction, final audit):

```bash
python scripts/run_enhanced_training.py \
  --run_dir external/HydroAuditToolFrameowrk/runs/<run_dir> \
  --camels_root data/CAMELS_US \
  --predictions_pickle external/HydroAuditToolFrameowrk/runs/<run_dir>/lstm_seed<seed>.p \
  --n_epochs 3 \
  --output_prefix results/hydrokg_run1
```

**3. Analyze results:** open `notebook/notebook.ipynb`, point it at the resulting
`results/hydrokg_run1_*` files, and generate the skill-trust and enhancement figures.

**HPC (SLURM):** `sbatch scripts/run_enhancement_uahpc.slurm`, after editing the paths at
the top of the file for the target system.

## Citation

If you use this framework, please cite:

Tamiru, H., Wood, A. J., Akinade, B., Gong, J., Li, S., Guo, X., Loof, T., Holcomb, H.,
Davies, A., & Burian, S. J. (2026). *Auditing and Improving LSTM Streamflow Predictions
with Hydrologic Knowledge Graphs.* Geophysical Research Letters.

Code archive: https://doi.org/10.5281/zenodo.21707541

## Acknowledgment

This research was supported by the Cooperative Institute for Research to Operations in
Hydrology (CIROH) under award NA22NWS4320003 from the NOAA Cooperative Institute Program.
The statements, findings, conclusions, and recommendations are those of the authors and do
not necessarily reflect the opinions of NOAA.

## License

MIT, consistent with the upstream `HydroAuditToolFrameowrk` submodule.
