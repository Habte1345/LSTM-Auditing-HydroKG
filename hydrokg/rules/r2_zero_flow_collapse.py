"""R2 - Zero-flow collapse (grouped with physical impossibility in the manuscript's
4-class rollup, predictive-error rule table entry; daily timestep).

Violation condition per the manuscript: "Q_sim ~ 0, but Q_obs is large enough."
This is NOT numerically specified in the draft -- "large enough" has no fixed value in
hydrology (it is basin-scale dependent: 0.5 mm/day is trivial for a large humid basin and
a major event for a small arid one). This implementation resolves it as basin-relative:

    q_sim < sim_zero_abs                                  (near-zero simulated flow)
    AND q_obs > obs_large_frac * mean(q_obs over the full evaluated period for this basin)

Defaults: sim_zero_abs=0.01 (mm/day), obs_large_frac=0.10 (i.e. observed flow at least 10%
of the basin's own long-term mean). Both are configurable and should be reported/justified
explicitly in the manuscript rather than left as this implementation's silent default.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from hydrokg.rules.base import Rule


class ZeroFlowCollapseRule(Rule):
    rule_id = "R2"
    required_scale = "daily"

    def __init__(self, sim_zero_abs: float = 0.01, obs_large_frac: float = 0.10, **params):
        super().__init__(sim_zero_abs=sim_zero_abs, obs_large_frac=obs_large_frac, **params)
        self.sim_zero_abs = sim_zero_abs
        self.obs_large_frac = obs_large_frac

    def evaluate(self, basin_id, df: pd.DataFrame, aridity_class: Optional[str] = None,
                 landcover_class: Optional[str] = None):
        violations = []
        valid = df[["qobs", "qsim"]].dropna()
        if valid.empty:
            return violations
        basin_mean_obs = valid["qobs"].mean()
        obs_threshold = self.obs_large_frac * basin_mean_obs
        mask = (valid["qsim"] < self.sim_zero_abs) & (valid["qobs"] > obs_threshold)
        for ts, row in valid.loc[mask].iterrows():
            violations.append(self._make_violation(
                basin_id, ts, row["qsim"], row["qobs"], magnitude=row["qobs"] - row["qsim"],
                aridity_class=aridity_class, landcover_class=landcover_class,
                obs_threshold=float(obs_threshold),
            ))
        return violations
