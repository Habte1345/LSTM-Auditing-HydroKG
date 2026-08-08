# Methodology-to-Code Correspondence

This document maps each methodological component and equation in Tamiru et al. (2026),
*Auditing and Improving LSTM Streamflow Predictions with Hydrologic Knowledge Graphs*, to
its implementation in this repository, to support reproducibility and independent
verification of the reported results.

| Manuscript component | Implementation |
|---|---|
| Eq. 1 — basin-normalized NSE training loss | `external/HydroAuditToolFrameowrk/Scripts/nseloss.py` (baseline LSTM, unmodified) |
| Eq. 2 — Kling-Gupta Efficiency (KGE) | `src/hydrokg_evaluation.py::calc_kge` |
| HydroKG construction (entities and relationships) | `src/hydrokg_ontology.ttl`, `src/hydrokg_graph.py`; see `docs/ONTOLOGY.md` |
| Seven auditing rules, R0-R6 | `src/hydrokg_rules.py`; see `docs/RULES.md` for the full rule specification |
| Four violation classes | `src/hydrokg_rules.py::VIOLATION_CLASS_TO_RULES` |
| Eq. 3 — basin violation burden, $V_b$ | `src/hydrokg_audit.py::compute_violation_burden` |
| Dominant violation class per basin | `src/hydrokg_audit.py::dominant_violation_class` |
| Offline post-processing audit | `src/hydrokg_audit.py::OfflineAuditor` |
| Real-time detection during fine-tuning (R0-R3) | `src/hydrokg_enhancement.py::EnhancedTrainingPipeline.fine_tune` |
| Skill-trust relationship (KGE vs. $V_b$) | `src/hydrokg_evaluation.py` |
| Aridity and land-cover stratification | `src/hydrokg_data.py`, `src/hydrokg_evaluation.py` |
| Enhancement mechanisms | `src/hydrokg_enhancement.py` |
| Eq. 4 — $\Delta KGE_b$ | `src/hydrokg_evaluation.py::compute_deltas` |
| Eq. 5 — $\Delta V_b$ | `src/hydrokg_evaluation.py::compute_deltas` |
| Eq. 6 — $P_{improved}$ | `src/hydrokg_evaluation.py::percent_improved` |
| Figure 2 (real-time auditing framework) | `src/hydrokg_enhancement.py::EnhancedTrainingPipeline.fine_tune`; R4-R6 are evaluated only offline, consistent with the rule scope described in the manuscript |

## Enhancement mechanism

The enhancement procedure is implemented as three concrete, non-differentiable
mechanisms, not as a physics-informed loss term:

1. `ViolationCurriculumSampler` — a graph query over each basin's accumulated R0-R3
   violation count determines that basin's sampling weight for the following epoch. This
   affects which basin-days contribute to the next epoch's gradient, not the value of the
   loss itself.
2. `GraphAnalogyCorrector` — a graph traversal to structurally similar, low-violation
   basins provides a post-hoc correction of flagged model output, applied once after
   fine-tuning completes.
3. `build_embedding_matrix` — a graph query over each basin's complete seven-rule
   violation profile produces an auxiliary static input feature used during fine-tuning.

`EnhancedTrainingPipeline` combines all three mechanisms with the baseline model's
existing architecture and the unmodified NSE loss. Reported skill and violation-burden
values are specific to the basins, epochs, learning rate, and hardware configuration of a
given run; they are recorded per run in that run's own
`*_baseline_results.csv`/`*_enhanced_results.csv` output files.

## Known scope limitations

- Real-time detection is restricted to R0-R3, since R4-R6 require an event window or full
  annual cycle unavailable within a single training batch. This is a structural property
  of the available temporal context, not an implementation gap.
- Rule thresholds not fully constrained by prior literature (R1-R3's magnitude
  thresholds, R2's zero-flow definition, R4's peak-lag tolerance) were set based on
  hydrologic reasoning and validated on real data; see `docs/RULES.md` for the complete
  derivation of each threshold.
- This repository does not include a physics-informed-loss comparison arm.
