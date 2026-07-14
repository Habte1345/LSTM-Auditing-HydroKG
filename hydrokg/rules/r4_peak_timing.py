"""R4 - Peak-timing error (timing failure, event/window-based).

Violation condition: |t_peak_sim - t_peak_obs| > max_lag_days (default 2, per manuscript).

Windowing: the manuscript does not specify the event window explicitly ("event/window
based"). This implementation uses USGS water years (Oct 1 - Sep 30) as the window,
which is the standard hydrologic convention for annual peak-flow comparison and avoids
splitting a single flood event across a calendar-year boundary in most CONUS basins. If a
storm-event-detection window (rather than a fixed water year) is preferred, swap the
`_water_year_windows` grouping for an event-detection routine -- the rest of the rule
logic (peak-lag comparison) is unchanged.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from hydrokg.rules.base import Rule


def _water_year(ts: pd.Timestamp) -> int:
    return ts.year + 1 if ts.month >= 10 else ts.year


class PeakTimingRule(Rule):
    rule_id = "R4"
    required_scale = "event"

    def __init__(self, max_lag_days: int = 2, **params):
        super().__init__(max_lag_days=max_lag_days, **params)
        self.max_lag_days = max_lag_days

    def n_evaluable(self, df: pd.DataFrame) -> int:
        valid = df[["qobs", "qsim"]].dropna()
        if valid.empty:
            return 0
        water_years = valid.index.to_series().apply(_water_year)
        return int(water_years.nunique())

    def evaluate(self, basin_id, df: pd.DataFrame, aridity_class: Optional[str] = None,
                 landcover_class: Optional[str] = None):
        violations = []
        valid = df[["qobs", "qsim"]].dropna()
        if valid.empty:
            return violations
        water_years = valid.index.to_series().apply(_water_year)
        for wy, window in valid.groupby(water_years):
            if window["qobs"].isna().all() or window["qsim"].isna().all():
                continue
            t_peak_obs = window["qobs"].idxmax()
            t_peak_sim = window["qsim"].idxmax()
            lag_days = abs((t_peak_sim - t_peak_obs).days)
            if lag_days > self.max_lag_days:
                violations.append(self._make_violation(
                    basin_id, t_peak_sim, window.loc[t_peak_sim, "qsim"],
                    window.loc[t_peak_obs, "qobs"], magnitude=lag_days,
                    aridity_class=aridity_class, landcover_class=landcover_class,
                    event_window=f"WY{wy}",
                ))
        return violations
