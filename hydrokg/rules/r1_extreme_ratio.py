"""R1 - Extreme ratio (magnitude failure, daily timestep).

Violation condition (only evaluated when q_obs > 0): q_sim/q_obs < low_ratio or > high_ratio.
Defaults (low_ratio=0.2, high_ratio=5) are exactly as specified in the manuscript draft.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from hydrokg.rules.base import Rule


class ExtremeRatioRule(Rule):
    rule_id = "R1"
    required_scale = "daily"

    def __init__(self, low_ratio: float = 0.2, high_ratio: float = 5.0, **params):
        super().__init__(low_ratio=low_ratio, high_ratio=high_ratio, **params)
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio

    def n_evaluable(self, df: pd.DataFrame) -> int:
        return int(((df["qobs"] > 0) & df["qsim"].notna()).sum())

    def evaluate(self, basin_id, df: pd.DataFrame, aridity_class: Optional[str] = None,
                 landcover_class: Optional[str] = None):
        violations = []
        valid = df[(df["qobs"] > 0) & df["qsim"].notna()]
        ratio = valid["qsim"] / valid["qobs"]
        mask = (ratio < self.low_ratio) | (ratio > self.high_ratio)
        for ts, row in valid.loc[mask].iterrows():
            r = row["qsim"] / row["qobs"]
            violations.append(self._make_violation(
                basin_id, ts, row["qsim"], row["qobs"], magnitude=r,
                aridity_class=aridity_class, landcover_class=landcover_class,
            ))
        return violations
