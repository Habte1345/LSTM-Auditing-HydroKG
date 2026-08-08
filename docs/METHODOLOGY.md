# Methodology → Code Mapping

For manuscript writing: exact pointers from each equation/concept in the methodology draft
to its implementation, so results in the paper can be traced back to a specific function.

| Manuscript item | Implementation |
|---|---|
| Eq. 1 — basin-normalized NSE training loss | Unmodified, from the submodule: `external/HydroAuditToolFrameowrk/Scripts/nseloss.py` |
| Eq. 2 — KGE | `src/hydrokg_evaluation.py::calc_kge` (added by HydroKG; the submodule only implements NSE-family metrics) |
| HydroKG construction (entities/relationships) | `src/hydrokg_ontology.ttl`, `src/hydrokg_graph.py`; see `docs/ONTOLOGY.md` |
| Seven rules R0-R6 | `src/hydrokg_rules.py` (all seven `Rule` subclasses); see `docs/RULES.md` for every numeric choice not fully specified in the draft |
| Four violation classes (rollup) | `src/hydrokg_rules.py::VIOLATION_CLASS_TO_RULES` |
| Eq. 3 — basin violation burden $V_b$ | `src/hydrokg_audit.py::compute_violation_burden` |
| Dominant violation class per basin | `src/hydrokg_audit.py::dominant_violation_class` |
| Offline post-processing audit | `src/hydrokg_audit.py::OfflineAuditor` |
| Real-time R0-R3 detection during fine-tuning | `src/hydrokg_enhancement.py::EnhancedTrainingPipeline.fine_tune` (`_detect_online_violations`) |
| Skill-trust relationship (KGE vs $V_b$) | `src/hydrokg_evaluation.py` |
| Aridity/land-cover stratification | `src/hydrokg_data.py`, `src/hydrokg_evaluation.py` |
| Graph-guided enhancement (replaces a physics-informed loss term by design) | `src/hydrokg_enhancement.py` — see below |
| Eq. 4 — $\Delta KGE_b$ | `src/hydrokg_evaluation.py::compute_deltas` |
| Eq. 5 — $\Delta V_b$ | `src/hydrokg_evaluation.py::compute_deltas` |
| Eq. 6 — $P_{improved}$ | `src/hydrokg_evaluation.py::percent_improved` |
| Figure 1 (conceptual framework) | Not auto-generated; a static conceptual diagram, out of scope for this codebase |
| Figure 2 (real-time staged auditing) | Approximated by `fine_tune()`'s online R0-R3 detection; R4-R6 are NOT staged in real time in this codebase (need full water-year context) — see `docs/ARCHITECTURE.md`'s "Real-time (online) detection during fine-tuning" section for the honest scope limit |

## The enhancement mechanism, precisely

The manuscript's methodology section describes the enhanced model as trained with "rule
violations... converted into rule-specific feedback." This is **not** implemented as a
physics-informed loss penalty (which already exists in the literature and is expensive to
backpropagate through). Instead, "feedback" means three concrete, non-differentiable
mechanisms:

1. `src/hydrokg_enhancement.py::ViolationCurriculumSampler` — graph query → training-sample
   reweighting (this determines *which* basin-days contribute to the next epoch's
   gradient, not the loss value itself).
2. `src/hydrokg_enhancement.py::GraphAnalogyCorrector` — graph traversal to structurally
   similar, low-violation basins → post-hoc correction of the model's raw output.
3. `src/hydrokg_enhancement.py::build_embedding_matrix` — graph query → auxiliary static
   input feature for a fine-tuning pass.

`src/hydrokg_enhancement.py::EnhancedTrainingPipeline` wires all three together against the
submodule's own `Model`/`CamelsH5` classes and `NSELoss` (unmodified). Treat any specific
number produced by a given run of this pipeline as tied to that run's configuration
(basins, epochs, learning rate, hardware) — not as a general guarantee of what the
mechanism will produce elsewhere.

## What to verify before citing numbers in the manuscript

- The full skill-trust relationship and enhancement deltas ($\Delta KGE_b$, $\Delta V_b$,
  $P_{improved}$) for the basins actually included in a given run — read them from that
  run's own `*_baseline_results.csv` / `*_enhanced_results.csv`, not from this document.
- `hydrokg_graph.Neo4jGraphStore` against a live server, if used — the in-memory backend
  is the default and requires no separate validation step.
- Whether the dominant-violation-class pattern (e.g. timing failures) reflects a genuine
  hydrologic finding or a threshold-sensitivity artifact (R4's lag-day threshold, R2/R3's
  thresholds) — see `docs/RULES.md` for every choice this codebase made where the
  manuscript draft was numerically underspecified.
- Whether a comparison against a physics-informed-loss baseline is needed before
  submission — this codebase does not include that comparison arm.
