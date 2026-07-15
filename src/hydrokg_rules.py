"""
R0-R6: the seven physically interpretable auditing rules, plus the base Rule class and
the registry tying them together. Merged into one file (was 9 files) since each rule is
short and they're always used together.

Each rule takes a per-basin daily DataFrame (columns: qobs, qsim, and for R5/R6
additionally 'p') plus static context (basin_id, aridity_class, landcover_class) and
returns a list of hydrokg.graph.ViolationRecord. Rules are stateless, pure functions of
their input DataFrame -- this is what lets the same rule implementation serve both the
offline auditor (applied to a complete DataFrame after the fact) and the online
detection inside enhanced_training.fine_tune() (applied to single training batches).

See docs/RULES.md for the full table and every place a numeric threshold in the
manuscript draft was ambiguous/underspecified and what this implementation assumes
instead (R2's "large enough" threshold, R3's real relative-error threshold, R4's
water-year windowing choice) -- flagged explicitly rather than resolved silently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import pandas as pd

from hydrokg_graph import ViolationRecord

# ============================================================================
# Base class
# ============================================================================


class Rule(ABC):
    rule_id: str
    required_scale: str  # "daily", "event", "annual"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def evaluate(
        self,
        basin_id: str,
        df: pd.DataFrame,
        aridity_class: Optional[str] = None,
        landcover_class: Optional[str] = None,
    ) -> list[ViolationRecord]:
        """Return one ViolationRecord per detected violation."""

    def n_evaluable(self, df: pd.DataFrame) -> int:
        """Number of valid opportunities for this rule to fire, for Eq. 3 normalization.
        Default: one opportunity per row with non-null qobs/qsim. Overridden by
        event/annual-scale rules (R4, R5, R6)."""
        return int(df[["qobs", "qsim"]].dropna().shape[0])

    def _make_violation(self, basin_id, ts, q_sim, q_obs, magnitude, aridity_class=None,
                         landcover_class=None, annual_window=None, event_window=None,
                         **extra) -> ViolationRecord:
        ts_date = ts.date() if hasattr(ts, "date") else ts
        return ViolationRecord(
            basin_id=basin_id,
            rule_id=self.rule_id,
            timestamp=ts_date if isinstance(ts_date, date) else date.fromisoformat(str(ts_date)),
            q_sim=float(q_sim),
            q_obs=float(q_obs),
            magnitude=float(magnitude),
            aridity_class=aridity_class,
            landcover_class=landcover_class,
            annual_window=annual_window,
            event_window=event_window,
            extra=extra,
        )


def _water_year(ts: pd.Timestamp) -> int:
    return ts.year + 1 if ts.month >= 10 else ts.year


# ============================================================================
# R0 - Negative flow (physical impossibility, daily)
# Violation: Q_sim < 0. Unambiguous, no tunable threshold.
# ============================================================================


class NegativeFlowRule(Rule):
    rule_id = "R0"
    required_scale = "daily"

    def evaluate(self, basin_id, df: pd.DataFrame, aridity_class=None, landcover_class=None):
        violations = []
        mask = df["qsim"] < 0
        for ts, row in df.loc[mask].iterrows():
            violations.append(self._make_violation(
                basin_id, ts, row["qsim"], row["qobs"], magnitude=row["qsim"],
                aridity_class=aridity_class, landcover_class=landcover_class,
            ))
        return violations


# ============================================================================
# R1 - Extreme ratio (magnitude failure, daily)
# Violation (only when q_obs > 0): q_sim/q_obs < low_ratio or > high_ratio.
# Defaults (0.2, 5.0) match the manuscript exactly.
# ============================================================================


class ExtremeRatioRule(Rule):
    rule_id = "R1"
    required_scale = "daily"

    def __init__(self, low_ratio: float = 0.2, high_ratio: float = 5.0, **params):
        super().__init__(low_ratio=low_ratio, high_ratio=high_ratio, **params)
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio

    def n_evaluable(self, df: pd.DataFrame) -> int:
        return int(((df["qobs"] > 0) & df["qsim"].notna()).sum())

    def evaluate(self, basin_id, df: pd.DataFrame, aridity_class=None, landcover_class=None):
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


# ============================================================================
# R2 - Zero-flow collapse (daily)
# Manuscript gap: "Q_sim ~ 0, but Q_obs is large enough" has no numeric definition.
# Resolved here as basin-relative: q_sim < sim_zero_abs AND q_obs > obs_large_frac *
# basin's own long-term mean q_obs. See docs/RULES.md.
# ============================================================================


class ZeroFlowCollapseRule(Rule):
    rule_id = "R2"
    required_scale = "daily"

    def __init__(self, sim_zero_abs: float = 0.01, obs_large_frac: float = 0.10, **params):
        super().__init__(sim_zero_abs=sim_zero_abs, obs_large_frac=obs_large_frac, **params)
        self.sim_zero_abs = sim_zero_abs
        self.obs_large_frac = obs_large_frac

    def evaluate(self, basin_id, df: pd.DataFrame, aridity_class=None, landcover_class=None):
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


# ============================================================================
# R3 - High relative error (magnitude failure, daily)
# Manuscript gap: stated as |Q_sim-Q_obs|/Q_obs > 0, which as literally written flags
# nearly every timestep (almost certainly a placeholder/typo). Resolved with a real
# threshold, default 1.0 (error exceeds 100% of observed flow). See docs/RULES.md.
# ============================================================================


class HighRelativeErrorRule(Rule):
    rule_id = "R3"
    required_scale = "daily"

    def __init__(self, relative_error_threshold: float = 1.0, **params):
        super().__init__(relative_error_threshold=relative_error_threshold, **params)
        self.threshold = relative_error_threshold

    def n_evaluable(self, df: pd.DataFrame) -> int:
        return int(((df["qobs"] > 0) & df["qsim"].notna()).sum())

    def evaluate(self, basin_id, df: pd.DataFrame, aridity_class=None, landcover_class=None):
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


# ============================================================================
# R4 - Peak-timing error (timing failure, event/water-year window)
# Violation: |t_peak_sim - t_peak_obs| > max_lag_days (default 2, per manuscript).
# Windowing choice: USGS water years (Oct 1 - Sep 30), the standard hydrologic
# convention -- not specified in the manuscript draft. See docs/RULES.md.
# ============================================================================


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
        return int(valid.index.to_series().apply(_water_year).nunique())

    def evaluate(self, basin_id, df: pd.DataFrame, aridity_class=None, landcover_class=None):
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


# ============================================================================
# R5 - Annual mass balance (budget-scale failure, annual window)
# Manuscript: P - Q - ET = 0, dS ~ 0, flag when Q_sim_mean > P_mean over a water year.
# Per project decision, ET is not sourced externally -- R5 only needs P and Q_sim,
# since that's the unambiguous physical check regardless of how ET is estimated.
# ============================================================================


class MassBalanceRule(Rule):
    rule_id = "R5"
    required_scale = "annual"

    def n_evaluable(self, df: pd.DataFrame) -> int:
        if "p" not in df.columns:
            return 0
        valid = df[["qobs", "qsim", "p"]].dropna()
        if valid.empty:
            return 0
        return int(valid.index.to_series().apply(_water_year).nunique())

    def evaluate(self, basin_id, df: pd.DataFrame, aridity_class=None, landcover_class=None):
        violations = []
        if "p" not in df.columns:
            return violations
        valid = df[["qobs", "qsim", "p"]].dropna()
        if valid.empty:
            return violations
        water_years = valid.index.to_series().apply(_water_year)
        for wy, window in valid.groupby(water_years):
            if len(window) < 300:  # require a near-complete water year
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


# ============================================================================
# R6 - Budyko consistency (budget-scale failure, annual window)
# Manuscript: ET_sim/P > 1, where ET_sim = P - Q_sim. Without an independent PET
# product, only the physical bound 0 <= ET_sim/P <= 1 can be checked -- the full
# Budyko curve is out of scope here. See docs/RULES.md.
# ============================================================================


class BudykoConsistencyRule(Rule):
    rule_id = "R6"
    required_scale = "annual"

    def n_evaluable(self, df: pd.DataFrame) -> int:
        if "p" not in df.columns:
            return 0
        valid = df[["qobs", "qsim", "p"]].dropna()
        if valid.empty:
            return 0
        return int(valid.index.to_series().apply(_water_year).nunique())

    def evaluate(self, basin_id, df: pd.DataFrame, aridity_class=None, landcover_class=None):
        violations = []
        if "p" not in df.columns:
            return violations
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


# ============================================================================
# Registry
# ============================================================================

RULE_CLASSES: dict[str, type[Rule]] = {
    "R0": NegativeFlowRule,
    "R1": ExtremeRatioRule,
    "R2": ZeroFlowCollapseRule,
    "R3": HighRelativeErrorRule,
    "R4": PeakTimingRule,
    "R5": MassBalanceRule,
    "R6": BudykoConsistencyRule,
}

# Rules requiring only daily context vs. progressively longer temporal context -- used
# by enhanced_training.py's online detection to know which rules can run mid-training.
DAILY_RULES = ["R0", "R1", "R2", "R3"]
EVENT_RULES = ["R4"]
ANNUAL_RULES = ["R5", "R6"]


def build_all_rules(**rule_params) -> dict[str, Rule]:
    """Instantiate all seven rules. rule_params: e.g. {'R1': {'low_ratio': 0.15}}."""
    return {
        rule_id: cls(**rule_params.get(rule_id, {}))
        for rule_id, cls in RULE_CLASSES.items()
    }
