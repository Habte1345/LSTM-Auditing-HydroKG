"""R6 - Budyko consistency (physical failure, long-term window).

Manuscript condition: ET_sim/P > 1, where ET_sim = P - Q_sim.

Note on scope: without an independent PET/aridity-index product (per project decision,
we are not sourcing external ET/PET here), the full Budyko curve -- which relates
ET/P to the aridity index PET/P -- cannot be evaluated; only the physical bound
0 <= ET_sim/P <= 1 can be checked model-agnostically. That bound is exactly the
manuscript's stated condition, so this rule implements it as written: ET_sim/P > 1
(model claims more evapotranspiration than fell as precipitation, impossible) OR
ET_sim/P < 0 (model claims negative ET, i.e. simulated runoff exceeds precipitation,
which is the same physical impossibility as R5 but expressed as an ET ratio rather than a
runoff comparison). If/when an independent PET product is added, upgrade this rule to test
against the full Budyko curve (e.g. Fu's equation) rather than just its bounds -- that
would let R6 catch physically *plausible* water totals that still land in the wrong
place on the curve, which this bounds-only version cannot.

Input df must have columns: qobs, qsim, p (see hydrokg/data/forcing_loader.py).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from hydrokg.rules.base import Rule


def _water_year(ts: pd.Timestamp) -> int:
    return ts.year + 1 if ts.month >= 10 else ts.year


class BudykoConsistencyRule(Rule):
    rule_id = "R6"
    required_scale = "annual"

    def n_evaluable(self, df: pd.DataFrame) -> int:
        valid = df[["qobs", "qsim", "p"]].dropna()
        if valid.empty:
            return 0
        return int(valid.index.to_series().apply(_water_year).nunique())

    def evaluate(self, basin_id, df: pd.DataFrame, aridity_class: Optional[str] = None,
                 landcover_class: Optional[str] = None):
        violations = []
        valid = df[["qobs", "qsim", "p"]].dropna()
        if valid.empty:
            return violations
        water_years = valid.index.to_series().apply(_water_year)
        for wy, window in valid.groupby(water_years):
            if len(window) < 300:
                continue
            p_mean = window["p"].mean()
            if p_mean <= 0:
                continue
            q_sim_mean = window["qsim"].mean()
            et_sim = p_mean - q_sim_mean
            ratio = et_sim / p_mean
            if ratio > 1.0 or ratio < 0.0:
                last_ts = window.index.max()
                violations.append(self._make_violation(
                    basin_id, last_ts, q_sim_mean, window["qobs"].mean(),
                    magnitude=ratio,
                    aridity_class=aridity_class, landcover_class=landcover_class,
                    annual_window=f"WY{wy}",
                ))
        return violations
