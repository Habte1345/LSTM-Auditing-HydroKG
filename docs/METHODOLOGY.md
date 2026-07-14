# Methodology → Code Mapping

For manuscript writing: exact pointers from each equation/concept in the methodology draft
to its implementation, so results in the paper can be traced back to a specific function.

| Manuscript item | Implementation |
|---|---|
| Eq. 1 — basin-normalized NSE training loss | Unmodified, from the submodule: `external/HydroAuditToolFrameowrk/Scripts/nseloss.py` |
| Eq. 2 — KGE | `hydrokg/evaluation/metrics.py::calc_kge` (added by HydroKG; the submodule only implements NSE-family metrics) |
| HydroKG construction (entities/relationships) | `hydrokg/ontology/hydrokg_ontology.ttl`, `hydrokg/graph/schema.py`; see `docs/ONTOLOGY.md` |
| Seven rules R0-R6 | `hydrokg/rules/r0_negative_flow.py` … `r6_budyko.py`; see `docs/RULES.md` for every numeric choice not fully specified in the draft |
| Four violation classes (rollup) | `hydrokg/rules/registry.py::VIOLATION_CLASS_TO_RULES` |
| Eq. 3 — basin violation burden $V_b$ | `hydrokg/audit/violation_burden.py::compute_violation_burden` |
| Dominant violation class per basin | `hydrokg/audit/violation_burden.py::dominant_violation_class` |
| Offline post-processing audit | `hydrokg/audit/offline_auditor.py::OfflineAuditor` |
| Real-time staged audit | `hydrokg/audit/realtime_auditor.py::RealtimeAuditor` |
| Skill-trust relationship (KGE vs $V_b$) | `hydrokg/evaluation/skill_trust_analysis.py` |
| Aridity/land-cover stratification | `hydrokg/data/basin_attributes.py`, `hydrokg/evaluation/stratification.py` |
| Graph-guided enhancement (replaces a physics-informed loss term by design) | `hydrokg/enhancement/` — see below |
| Eq. 4 — $\Delta KGE_b$ | `hydrokg/evaluation/enhancement_metrics.py::compute_deltas` |
| Eq. 5 — $\Delta V_b$ | `hydrokg/evaluation/enhancement_metrics.py::compute_deltas` |
| Eq. 6 — $P_{improved}$ | `hydrokg/evaluation/enhancement_metrics.py::percent_improved` |
| Figure 1 (conceptual framework) | Not auto-generated; a static conceptual diagram, out of scope for this codebase |
| Figure 2 (real-time staged auditing) | Directly implemented by `RealtimeAuditor`'s daily-vs-event-vs-annual staging logic |

## The enhancement mechanism, precisely

The manuscript's methodology section describes the enhanced model as trained with "rule
violations... converted into rule-specific feedback." Per an explicit project decision
during development, this is **not** implemented as a physics-informed loss penalty (which
already exists in the literature and is expensive to backpropagate through). Instead,
"feedback" means three concrete, non-differentiable mechanisms, matching the three-part
mechanism agreed on with the PI:

1. `hydrokg/enhancement/curriculum.py::ViolationCurriculumSampler` — graph query →
   training-sample reweighting (Eq. context: this determines *which* basin-days
   contribute to the next epoch's gradient, not the loss value itself).
2. `hydrokg/enhancement/graph_analogy_correction.py::GraphAnalogyCorrector` — graph
   traversal to structurally similar, low-violation basins → post-hoc correction of the
   model's raw output.
3. `hydrokg/enhancement/violation_embeddings.py::build_embedding_matrix` — graph query →
   auxiliary static input feature for a fine-tuning pass.

`hydrokg/enhancement/enhanced_training.py::EnhancedTrainingPipeline` wires all three
together against the submodule's own `Model`/`CamelsH5` classes and `NSELoss` (unmodified).
**This integration module has not been executed end-to-end** — it requires a real CAMELS
data directory, a completed submodule training checkpoint, and PyTorch, none of which were
available while this repository was built. Each mechanism it calls into IS independently
tested against synthetic data (`tests/test_enhancement.py`). Validate the full pipeline
against a real run before reporting results from it.

## What still needs a real CAMELS + Neo4j run to validate

- The full 670-basin skill-trust relationship (only demonstrated on synthetic data here).
- `EnhancedTrainingPipeline.fine_tune()` end to end.
- `hydrokg.graph.neo4j_store.Neo4jGraphStore` against a live server
  (`tests/test_neo4j_store.py` is ready to run once one exists).
- The actual magnitude of $\Delta KGE_b$ and $P_{improved}$ for the manuscript's results
  section — nothing in this repository should be read as a claim about what those numbers
  will be; the machinery to compute them from a real run is what this repository provides.
