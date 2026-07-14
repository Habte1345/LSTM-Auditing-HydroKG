"""
Real-time (online) audit mode.

Implements the manuscript's staging: point-wise rules (R0-R3) are evaluated at every
timestep as predictions arrive; R4 needs a full event/water-year window; R5/R6 need a
full annual window. Rather than re-deriving separate streaming logic per rule, this
auditor buffers each basin's incoming (timestamp, qsim, qobs, p) rows and re-uses the
exact same Rule.evaluate() implementations as the offline auditor (hydrokg.rules.*) --
daily rules are evaluated on each new row's 1-row frame, while R4/R5/R6 are evaluated
once a water-year boundary is crossed, on that just-closed water year's buffered rows.
This keeps a single source of truth for rule logic between offline and online modes,
which the manuscript's design explicitly requires ("staged rather than simultaneous").
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from hydrokg.graph.base import GraphStore
from hydrokg.rules.registry import ANNUAL_RULES, DAILY_RULES, EVENT_RULES, build_all_rules


def _water_year(ts: pd.Timestamp) -> int:
    return ts.year + 1 if ts.month >= 10 else ts.year


class RealtimeAuditor:

    def __init__(self, graph_store: GraphStore, rule_params: Optional[dict] = None,
                 flush_every: int = 500):
        self.graph = graph_store
        self.rules = build_all_rules(**(rule_params or {}))
        self.graph.initialize_schema()
        self._buffers: dict[str, list[dict]] = {}
        self._current_water_year: dict[str, int] = {}
        self._flush_every = flush_every
        self._since_flush = 0
        self._registered: set[str] = set()

    def register_basin(self, basin_id: str, aridity_class: Optional[str] = None,
                        landcover_class: Optional[str] = None) -> None:
        self.graph.register_catchment(basin_id, aridity_class, landcover_class)
        self._registered.add(basin_id)
        self._buffers.setdefault(basin_id, [])

    def ingest(
        self,
        basin_id: str,
        timestamp,
        q_sim: float,
        q_obs: float,
        p: Optional[float] = None,
        aridity_class: Optional[str] = None,
        landcover_class: Optional[str] = None,
    ) -> int:
        """Feed one new prediction. Returns the number of violations written for this call
        (daily rules only; event/annual rules fire in batches on window close, see
        `_maybe_close_window`).
        """
        if basin_id not in self._registered:
            self.register_basin(basin_id, aridity_class, landcover_class)

        ts = pd.Timestamp(timestamp)
        row = {"qobs": q_obs, "qsim": q_sim, "p": p}
        self._buffers[basin_id].append({"timestamp": ts, **row})

        # --- daily-scale rules: evaluate immediately on this single row ---
        one_row_df = pd.DataFrame([row], index=[ts])
        n_written = 0
        for rule_id in DAILY_RULES:
            violations = self.rules[rule_id].evaluate(basin_id, one_row_df, aridity_class, landcover_class)
            n_written += self.graph.write_violations(violations)

        # --- event/annual-scale rules: evaluate only when their window closes ---
        wy = _water_year(ts)
        prev_wy = self._current_water_year.get(basin_id)
        if prev_wy is not None and wy != prev_wy:
            n_written += self._close_water_year(basin_id, prev_wy, aridity_class, landcover_class)
        self._current_water_year[basin_id] = wy

        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            self._trim_buffers()
            self._since_flush = 0

        return n_written

    def _close_water_year(self, basin_id: str, closed_wy: int, aridity_class, landcover_class) -> int:
        buf = self._buffers.get(basin_id, [])
        if not buf:
            return 0
        df = pd.DataFrame(buf).set_index("timestamp")
        window = df[df.index.to_series().apply(_water_year) == closed_wy]
        if window.empty:
            return 0
        n_written = 0
        for rule_id in EVENT_RULES + ANNUAL_RULES:
            violations = self.rules[rule_id].evaluate(basin_id, window, aridity_class, landcover_class)
            n_written += self.graph.write_violations(violations)
        return n_written

    def flush_all(self) -> int:
        """Force-evaluate event/annual rules on whatever partial window remains buffered
        for every basin (call at the end of a streaming session)."""
        n_written = 0
        for basin_id, wy in list(self._current_water_year.items()):
            n_written += self._close_water_year(basin_id, wy, None, None)
        return n_written

    def _trim_buffers(self, keep_days: int = 400) -> None:
        """Bound memory use: only the current + previous water year are needed for R4-R6,
        so older rows can be dropped once their window has already been evaluated."""
        for basin_id, buf in self._buffers.items():
            if len(buf) > keep_days:
                self._buffers[basin_id] = buf[-keep_days:]
