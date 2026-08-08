# Physical Auditing Rules (R0-R6)

This document specifies the seven physically interpretable rules used to audit LSTM
streamflow predictions, including the numeric thresholds and their justification, as
implemented in `src/hydrokg_rules.py`.

| Rule | Name | Failure type | Violation class | Temporal scale | Implementation |
|---|---|---|---|---|---|
| R0 | Negative flow | Physical failure | Physical impossibility | Daily | `NegativeFlowRule` |
| R1 | Extreme ratio | Predictive error | Magnitude failure | Daily | `ExtremeRatioRule` |
| R2 | Zero-flow collapse | Predictive error | Physical impossibility | Daily | `ZeroFlowCollapseRule` |
| R3 | High relative error | Predictive error | Magnitude failure | Daily | `HighRelativeErrorRule` |
| R4 | Peak-timing error | Predictive error | Timing failure | Event window | `PeakTimingRule` |
| R5 | Annual mass balance | Physical failure | Budget-scale failure | Annual | `MassBalanceRule` |
| R6 | Budyko consistency | Physical failure | Budget-scale failure | Annual | `BudykoConsistencyRule` |

## R0 — Negative flow

Flags `Q_sim < 0`. No tunable parameter; negative discharge is physically impossible under
any basin condition.

## R1 — Extreme ratio

Flags `Q_sim / Q_obs < 0.2` or `> 5`, evaluated only when `Q_obs > 0`. Thresholds are
configurable via `low_ratio` and `high_ratio`.

**Low-flow floor.** R1 additionally requires `Q_obs` to exceed 5% of the basin's own
long-term mean flow (`min_flow_frac`). Without this floor, a small absolute prediction
error at very low observed flow produces an arbitrarily large ratio, since the denominator
is near zero, flagging low-flow and baseflow periods for magnitude errors unrelated to
genuine physical inconsistency. A controlled test confirmed this directly: a fixed 5%
relative error at high flow triggers no violation, while an identical absolute-scale error
at low flow triggers a violation on every affected day. Without the floor, this behavior
alone accounted for 82.8% of all rule violations recorded across all seven rules in a
647-basin evaluation, and for 95.5% of basins' dominant violation class, reflecting which
basins have more low-flow days rather than which basins violate distinct physical
expectations. The floor removes this artifact from the audit.

## R2 — Zero-flow collapse

Flags `Q_sim < 0.01 mm/day` when `Q_obs > 0.10 ×` the basin's own long-term mean `Q_obs`.
Both thresholds (`sim_zero_abs`, `obs_large_frac`) are basin-relative rather than fixed
absolute values, since streamflow magnitude varies by orders of magnitude across the
CAMELS basin set and a single fixed threshold would not be physically comparable across
basins.

## R3 — High relative error

Flags `|Q_sim - Q_obs| / Q_obs > 1.0` (`relative_error_threshold`), corresponding to a
prediction error exceeding 100% of observed flow. This threshold is set above R1's
extreme-ratio bounds so that R3 identifies large-magnitude errors distinct from, rather
than redundant with, R1.

**Low-flow floor.** R3 uses the same basin-relative low-flow floor as R1 (`min_flow_frac`),
for the same reason: without it, relative-error inflation at low observed flow, rather
than genuinely large prediction errors, accounted for the majority of flagged violations
in the same 647-basin evaluation described above.

## R4 — Peak-timing error

Flags a mistimed annual peak when `|t_peak,sim - t_peak,obs| > 2 days` (`max_lag_days`),
compared within USGS water years (October 1 - September 30). The water-year window is the
standard convention for annual peak-flow analysis and avoids splitting a single flood
event across a calendar-year boundary in most CONUS basins.

## R5 — Annual mass balance

Flags `Q_sim,mean > P_mean` over a water year, the condition under which a model
generates more streamflow than fell as precipitation, a direct violation of mass
conservation. This check uses only precipitation and simulated discharge; it does not
require an independent evapotranspiration (ET) product, since the physical bound being
tested does not depend on ET. A water year is required to have at least 300 days of valid
data to avoid false positives from incomplete records.

## R6 — Budyko consistency

Flags `ET_sim / P > 1` or `< 0`, where `ET_sim = P - Q_sim`, the physical bound that
evapotranspiration cannot exceed available precipitation or be negative. This rule
evaluates the physical bound of the Budyko framework rather than the full parametric
Budyko curve, since evaluating consistency with a specific curve (e.g., Fu's equation)
requires an independent potential evapotranspiration or aridity-index product not sourced
in this pipeline. As implemented, R6 identifies basins whose annual water balance is
physically impossible in aggregate, but does not identify basins whose water balance is
physically plausible yet inconsistent with the expected Budyko relationship for their
climate.

## Evapotranspiration used for reporting, not for R5/R6

`src/hydrokg_data.py` computes a long-term water-balance ET residual
(`ET = P̄ - Q̄_obs`, assuming negligible long-term storage change) for basin-level
diagnostics and figures. R5 and R6 do not use this residual; both operate directly on
precipitation and simulated discharge over annual windows, avoiding the introduction of a
noisy short-window ET estimate into a per-year audit.

## Real-time evaluation scope

`src/hydrokg_rules.py` defines `DAILY_RULES` (R0-R3), `EVENT_RULES` (R4), and
`ANNUAL_RULES` (R5-R6). Only `DAILY_RULES` are evaluated in real time, inside
`EnhancedTrainingPipeline.fine_tune()`, directly against each batch's own forward-pass
output. R4-R6 require a full water-year of calendar-dated observations that an isolated
training sequence does not carry, and are therefore evaluated only by the offline
`OfflineAuditor`, before and after training.
