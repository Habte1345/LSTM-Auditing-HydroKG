# Physical Auditing Rules (R0-R6)

This document is the implementation-facing companion to the manuscript's rule table. Every
rule below flags exactly what the manuscript specifies **except** where explicitly noted —
those notes are places where the manuscript draft was numerically underspecified and this
implementation had to make (and document) a concrete choice. Report these choices in the
manuscript rather than leaving them as this codebase's silent defaults.

| Rule | Name | Failure type | Class | Scale | Implementation |
|---|---|---|---|---|---|
| R0 | Negative flow | Physical failure | Physical impossibility | Daily | `src/hydrokg/rules.py::NegativeFlowRule` |
| R1 | Extreme ratio | Predictive error | Magnitude failure | Daily | `src/hydrokg/rules.py::ExtremeRatioRule` |
| R2 | Zero-flow collapse | Predictive error | Physical impossibility | Daily | `src/hydrokg/rules.py::ZeroFlowCollapseRule` |
| R3 | High relative error | Predictive error | Magnitude failure | Daily | `src/hydrokg/rules.py::HighRelativeErrorRule` |
| R4 | Peak-timing error | Predictive error | Timing failure | Event/window | `src/hydrokg/rules.py::PeakTimingRule` |
| R5 | Annual mass balance | Physical failure | Budget-scale failure | Annual | `src/hydrokg/rules.py::MassBalanceRule` |
| R6 | Budyko consistency | Physical failure | Budget-scale failure | Annual | `src/hydrokg/rules.py::BudykoConsistencyRule` |

## R0 — Negative flow
Unambiguous: `Q_sim < 0`. No tunable parameter.

## R1 — Extreme ratio
`Q_sim/Q_obs < 0.2` or `> 5`, evaluated only when `Q_obs > 0`. Thresholds match the
manuscript exactly. Configurable via `low_ratio`/`high_ratio`.

## R2 — Zero-flow collapse
**Manuscript gap:** "Q_sim ≈ 0, but Q_obs is large enough" has no numeric definition in the
draft — hydrology has no universal constant for "large enough" (basin-scale dependent).
**This implementation's resolution:** `Q_sim < 0.01 mm/day` AND `Q_obs > 0.10 × basin's own
long-term mean Q_obs`. Both are configurable (`sim_zero_abs`, `obs_large_frac`) and should
be reported and justified as a deliberate methodological choice in the manuscript, not left
implicit.

## R3 — High relative error
**Manuscript gap:** the draft states `|Q_sim - Q_obs|/Q_obs > 0`, which as literally written
flags virtually every timestep with any nonzero error (relative error is essentially never
exactly zero) — almost certainly a placeholder/typo rather than an intended condition.
**This implementation's resolution:** a real threshold, default `1.0` (error exceeds 100%
of observed flow), configurable via `relative_error_threshold`. This should sit clearly
above R1's extreme-ratio bounds so R3 is not simply redundant with R1 — tune per basin
regime as needed, and report the chosen value explicitly.

## R4 — Peak-timing error
`|t_peak_sim - t_peak_obs| > 2 days` (manuscript default, configurable via `max_lag_days`).
**Windowing choice:** the manuscript says "event/window based" without specifying the
window. This implementation uses USGS water years (Oct 1 - Sep 30) as the comparison
window — the standard hydrologic convention for annual peak-flow analysis, and one that
avoids splitting a single flood event across a calendar-year boundary in most CONUS
basins. Swap `_water_year_windows` for a storm-event-detection routine if a shorter,
event-specific window is preferred; the peak-lag comparison logic itself is unchanged.

## R5 — Annual mass balance
Manuscript condition: `P - Q - ET = 0`, `dS ≈ 0`, flag when `Q_sim_mean > P_mean` over a
water year. Per project decision, ET is **not** sourced from an external product (e.g.
GLEAM) anywhere in this rule — R5 only needs `P` and `Q_sim`, since the unambiguous
physical check ("the model manufactures more water than fell as precipitation") doesn't
require ET at all. Requires a near-complete water year (≥300 days) to avoid false
positives from partial-year data.

## R6 — Budyko consistency
Manuscript condition: `ET_sim/P > 1`, where `ET_sim = P - Q_sim`.
**Scope limitation, stated explicitly:** without an independent PET/aridity-index product
(again, per the project decision not to source external ET/PET), the *full* Budyko curve —
relating `ET/P` to the aridity index `PET/P` — cannot be evaluated. Only the physical bound
`0 <= ET_sim/P <= 1` is checked, which is exactly the manuscript's stated condition. This
means R6 as implemented **cannot** catch a basin whose water totals are physically
plausible in aggregate but land in the wrong place on the actual Budyko curve for its
climate — only whether the simulated ET/P ratio violates the hard physical bound. If/when
an independent PET product is added to the pipeline, upgrade this rule to test against
Fu's equation (or another parametric Budyko curve) rather than just its bounds.

## Where ET comes from for reporting (not for the rules themselves)
`src/hydrokg/data.py` computes a long-term water-balance ET residual
(`ET = P̄ - Q̄_obs`, assuming `dS/dt ≈ 0` over multi-decade CAMELS records — the standard
large-sample-hydrology assumption) for basin-level diagnostics and figures. **R5 and R6
do not use this function** — they operate on `P` and `Q_sim` directly over annual windows,
deliberately avoiding compounding a noisy short-window ET residual into a per-year audit.

## Real-time staging (R0-R3 vs R4 vs R5/R6)
`src/hydrokg/rules.py` defines `DAILY_RULES`, `EVENT_RULES`, `ANNUAL_RULES`. Only
`DAILY_RULES` (R0-R3) are evaluated in real time, inside
`EnhancedTrainingPipeline.fine_tune()`'s training loop (`src/hydrokg/enhancement.py`) --
directly against every batch's own forward-pass output, no extra inference needed. R4
(event) and R5/R6 (annual) require a full water-year of calendar-dated observations
that an isolated training sequence window doesn't carry, so they remain evaluated only
by the offline `OfflineAuditor` (before and after training), never in real time. State
this scope limit explicitly rather than implying uniform real-time treatment.
