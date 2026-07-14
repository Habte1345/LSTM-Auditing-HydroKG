"""
Abstract Rule interface.

Each rule (R0-R6) takes a per-basin daily DataFrame (columns at least: qobs, qsim, and for
R5/R6 additionally 'p' and 'et' -- see hydrokg.data) plus static context (basin_id,
aridity_class, landcover_class) and returns a list of hydrokg.graph.schema.ViolationRecord.

Rules are intentionally basin-scoped, stateless, pure functions of their input DataFrame --
this is what lets the same rule implementation serve both the offline auditor (applied to a
complete DataFrame after the fact) and the real-time auditor (applied to a growing window as
data streams in, staged by required temporal context; see hydrokg/audit/realtime_auditor.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import pandas as pd

from hydrokg.graph.schema import ViolationRecord


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
        event/annual-scale rules (R4, R5, R6).
        """
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
