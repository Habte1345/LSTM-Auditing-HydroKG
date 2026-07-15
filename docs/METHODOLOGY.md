# Methodology → Code Mapping

For manuscript writing: exact pointers from each equation/concept in the methodology draft
to its implementation, so results in the paper can be traced back to a specific function.

| Manuscript item | Implementation |
|---|---|
| Eq. 1 — basin-normalized NSE training loss | Unmodified, from the submodule: `external/HydroAuditToolFrameowrk/Scripts/nseloss.py` |
| Eq. 2 — KGE | `src/hydrokg/evaluation.py::calc_kge` (added by HydroKG; the submodule only implements NSE-family metrics) |
| HydroKG construction (entities/relationships) | `src/hydrokg/ontology/hydrokg_ontology.ttl`, `src/hydrokg/graph.py`; see `docs/ONTOLOGY.md` |
| Seven rules R0-R6 | `src/hydrokg/rules.py (all seven Rule subclasses)`; see `docs/RULES.md` for every numeric choice not fully specified in the draft |
| Four violation classes (rollup) | `src/hydrokg/rules.py::VIOLATION_CLASS_TO_RULES` |
| Eq. 3 — basin violation burden $V_b$ | `src/hydrokg/audit.py::compute_violation_burden` |
| Dominant violation class per basin | `src/hydrokg/audit.py::dominant_violation_class` |
| Offline post-processing audit | `src/hydrokg/audit.py::OfflineAuditor` |
| Real-time R0-R3 detection during fine-tuning | `src/hydrokg/enhancement.py::EnhancedTrainingPipeline.fine_tune` (`_detect_online_violations`) |
| Skill-trust relationship (KGE vs $V_b$) | `src/hydrokg/evaluation.py` |
| Aridity/land-cover stratification | `src/hydrokg/data.py`, `src/hydrokg/evaluation.py` |
| Graph-guided enhancement (replaces a physics-informed loss term by design) | `src/hydrokg/enhancement.py` — see below |
| Eq. 4 — $\Delta KGE_b$ | `src/hydrokg/evaluation.py::compute_deltas` |
| Eq. 5 — $\Delta V_b$ | `src/hydrokg/evaluation.py::compute_deltas` |
| Eq. 6 — $P_{improved}$ | `src/hydrokg/evaluation.py::percent_improved` |
| Figure 1 (conceptual framework) | Not auto-generated; a static conceptual diagram, out of scope for this codebase |
| Figure 2 (real-time staged auditing) | Approximated by `fine_tune()`'s online R0-R3 detection; R4-R6 are NOT staged in real time in this codebase (need full water-year context) -- see `docs/ARCHITECTURE.md`'s "Real-time (online) detection during fine-tuning" section for the honest scope limit |

## The enhancement mechanism, precisely

The manuscript's methodology section describes the enhanced model as trained with "rule
violations... converted into rule-specific feedback." Per an explicit project decision
during development, this is **not** implemented as a physics-informed loss penalty (which
already exists in the literature and is expensive to backpropagate through). Instead,
"feedback" means three concrete, non-differentiable mechanisms, matching the three-part
mechanism agreed on with the PI:

1. `src/hydrokg/enhancement.py (ViolationCurriculumSampler)::ViolationCurriculumSampler` — graph query →
   training-sample reweighting (Eq. context: this determines *which* basin-days
   contribute to the next epoch's gradient, not the loss value itself).
2. `src/hydrokg/enhancement.py (GraphAnalogyCorrector)::GraphAnalogyCorrector` — graph
   traversal to structurally similar, low-violation basins → post-hoc correction of the
   model's raw output.
3. `src/hydrokg/enhancement.py (build_embedding_matrix)::build_embedding_matrix` — graph query →
   auxiliary static input feature for a fine-tuning pass.

`src/hydrokg/enhancement.py (EnhancedTrainingPipeline)::EnhancedTrainingPipeline` wires all three
together against the submodule's own `Model`/`CamelsH5` classes and `NSELoss` (unmodified).
**This integration module has not been executed end-to-end** — it requires a real CAMELS
data directory, a completed submodule training checkpoint, and PyTorch, none of which were
available while this repository was built. Each mechanism it calls into IS independently
tested against synthetic data (`tests/test_enhancement.py`). Validate the full pipeline
against a real run before reporting results from it.

## What still needs a real CAMELS + Neo4j run to validate

- The full 670-basin skill-trust relationship (only demonstrated on synthetic data here).
- `EnhancedTrainingPipeline.fine_tune()` end to end.
- `hydrokg.graph.Neo4jGraphStore` against a live server
  (`tests/test_neo4j_store.py` is ready to run once one exists).
- The actual magnitude of $\Delta KGE_b$ and $P_{improved}$ for the manuscript's results
  section — nothing in this repository should be read as a claim about what those numbers
  will be; the machinery to compute them from a real run is what this repository provides.
