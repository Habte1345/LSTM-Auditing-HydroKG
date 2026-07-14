"""R3 - High relative error (magnitude failure, daily timestep).

The manuscript draft states the condition as |Q_sim - Q_obs| / Q_obs > 0, which is a
placeholder/typo -- as literally written it flags every single timestep with any
nonzero error at all, since relative error is virtually never exactly 0. This
implementation instead uses a real threshold (default 1.0, i.e. the absolute error
exceeds 100% of the observed flow), which should be reported and justified in the
manuscript as a deliberate choice, not left implicit. This threshold should sit clearly
above R1's ratio-based extremes so R3 is not simply redundant with R1; tune per basin
regime if needed.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from hydrokg.rules.base import Rule


class HighRelativeErrorRule(Rule):
    rule_id = "R3"
    required_scale = "daily"

    def __init__(self, relative_error_threshold: float = 1.0, **params):
        super().__init__(relative_error_threshold=relative_error_threshold, **params)
        self.threshold = relative_error_threshold

    def n_evaluable(self, df: pd.DataFrame) -> int:
        return int(((df["qobs"] > 0) & df["qsim"].notna()).sum())

    def evaluate(self, basin_id, df: pd.DataFrame, aridity_class: Optional[str] = None,
                 landcover_class: Optional[str] = None):
        violations = []
        valid = df[(df["qobs"] > 0) & df["qsim"].notna()]
        rel_err = (valid["qsim"] - valid["qobs"]).abs() / valid["qobs"]
        mask = rel_err > self.threshold
        for ts, row in valid.loc[mask].iterrows():
            re = abs(row["qsim"] - row["qobs"]) / row["qobs"]
            violations.append(self._make_violation(
                basin_id, ts, row["qsim"], row["qobs"], magnitude=re,
                aridity_class=aridity_class, landcover_class=landcover_class,
            ))
        return violations
