"""R5 - Annual mass balance (physical failure, annual/rolling-year window).

Manuscript condition: P - Q - ET = 0, dS ~ 0, and flags when Q_sim_mean > P_mean.

Per project decision, ET is not sourced from an external product (e.g. GLEAM) here --
it is computed as the long-term water-balance residual ET_obs = P - Q_obs (storage change
assumed ~0 over annual/multi-annual windows, the standard large-sample hydrology
assumption; see hydrokg/data/et_water_balance.py). R5 itself only needs P and Q_sim: it
flags a basin-year as violating mass balance whenever the simulated annual runoff ratio
exceeds 1 (i.e. the model manufactures more water than fell as precipitation), which is
the unambiguous, non-negotiable physical check in the manuscript's condition regardless of
how ET is estimated.

Input df must have columns: qobs, qsim, p (precipitation, same units as discharge --
typically mm/day after basin-area normalization; see hydrokg/data/forcing_loader.py).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from hydrokg.rules.base import Rule


def _water_year(ts: pd.Timestamp) -> int:
    return ts.year + 1 if ts.month >= 10 else ts.year


class MassBalanceRule(Rule):
    rule_id = "R5"
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
            if len(window) < 300:  # require a near-complete water year (~365 days)
                continue
            q_sim_mean = window["qsim"].mean()
            p_mean = window["p"].mean()
            if q_sim_mean > p_mean:
                last_ts = window.index.max()
                violations.append(self._make_violation(
                    basin_id, last_ts, q_sim_mean, window["qobs"].mean(),
                    magnitude=q_sim_mean - p_mean,
                    aridity_class=aridity_class, landcover_class=landcover_class,
                    annual_window=f"WY{wy}",
                ))
        return violations
