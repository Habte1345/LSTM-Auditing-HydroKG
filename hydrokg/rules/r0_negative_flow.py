"""R0 - Negative flow (physical impossibility, daily timestep).

Violation condition: Q_sim < 0. Unambiguous, no tunable threshold needed.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from hydrokg.rules.base import Rule


class NegativeFlowRule(Rule):
    rule_id = "R0"
    required_scale = "daily"

    def evaluate(self, basin_id, df: pd.DataFrame, aridity_class: Optional[str] = None,
                 landcover_class: Optional[str] = None):
        violations = []
        mask = df["qsim"] < 0
        for ts, row in df.loc[mask].iterrows():
            violations.append(self._make_violation(
                basin_id, ts, row["qsim"], row["qobs"], magnitude=row["qsim"],
                aridity_class=aridity_class, landcover_class=landcover_class,
            ))
        return violations
