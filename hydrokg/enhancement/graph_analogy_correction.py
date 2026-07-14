"""
Enhancement mechanism 2/3: graph-analogy correction at inference.

When a daily point-wise rule (R0-R3) fires for basin b at time t, HydroKG traverses to
structurally similar basins (same AridityClass/LandCoverClass, low historical violation
rate for that specific rule -- GraphStore.query_analog_basins) and uses THEIR behavior,
scaled to basin b's own flow regime, to produce a correction. This is a retrieval-based
correction, not a fixed physical clip (e.g. "set negative flow to 0") and not a
differentiable penalty -- it is applied once, at inference, after the LSTM has already
produced its raw prediction.

Scope: intended for the daily, point-wise rules (R0-R3), where "what would a similar,
well-behaved basin have predicted under comparable relative conditions" is a meaningful
question. R4 (peak timing) and R5/R6 (annual mass balance/Budyko) are event/annual-scale
by construction and are not corrected point-by-point here -- they feed back only through
the curriculum reweighting mechanism (hydrokg.enhancement.curriculum) and offline
diagnostics, not per-timestep correction.

Mechanism: each analog basin's prediction at a comparable time is expressed relative to
that basin's OWN long-term median flow (a scale-free ratio), then that ratio is
transferred to basin b's own long-term median flow. This avoids naively comparing raw
discharge across basins of very different size/climate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from hydrokg.graph.base import GraphStore


class GraphAnalogyCorrector:

    def __init__(self, graph_store: GraphStore, basin_dataframes: dict[str, pd.DataFrame],
                 min_analog_median: float = 1e-3):
        """
        Parameters
        ----------
        basin_dataframes : {basin_id: DataFrame(qobs, qsim)} for ALL basins the corrector
            may be asked to draw analogs from (typically the full training-basin set).
        """
        self.graph = graph_store
        self.basin_dataframes = basin_dataframes
        self.min_analog_median = min_analog_median
        self._median_cache: dict[str, float] = {}

    def _basin_median_qsim(self, basin_id: str) -> float:
        if basin_id not in self._median_cache:
            df = self.basin_dataframes.get(basin_id)
            median = float(df["qsim"].median()) if df is not None and not df.empty else np.nan
            self._median_cache[basin_id] = max(median, self.min_analog_median) if not np.isnan(median) else np.nan
        return self._median_cache[basin_id]

    def correct(
        self,
        basin_id: str,
        timestamp,
        q_sim_raw: float,
        rule_id: str,
        aridity_class: Optional[str],
        landcover_class: Optional[str],
        top_k: int = 5,
    ) -> tuple[float, dict]:
        """
        Returns
        -------
        (corrected_value, info) where info records which analogs were used and the
        correction ratio actually applied, for auditability -- every correction should be
        traceable, matching the rest of HydroKG's design philosophy.
        """
        analogs = self.graph.query_analog_basins(
            basin_id, rule_id, aridity_class, landcover_class, top_k=top_k
        )
        if not analogs:
            # No analogs found (e.g. isolated aridity/land-cover class): fall back to a
            # conservative physical floor rather than leaving an impossible value.
            fallback = max(q_sim_raw, 0.0) if rule_id in ("R0", "R2") else q_sim_raw
            return fallback, {"method": "no_analogs_fallback", "analogs": []}

        ts = pd.Timestamp(timestamp)
        ratios = []
        used = []
        for analog_id, score in analogs:
            analog_df = self.basin_dataframes.get(analog_id)
            if analog_df is None or ts not in analog_df.index:
                continue
            analog_median = self._basin_median_qsim(analog_id)
            if np.isnan(analog_median) or analog_median <= 0:
                continue
            analog_qsim_t = analog_df.loc[ts, "qsim"]
            ratios.append(analog_qsim_t / analog_median)
            used.append(analog_id)

        if not ratios:
            fallback = max(q_sim_raw, 0.0) if rule_id in ("R0", "R2") else q_sim_raw
            return fallback, {"method": "no_temporal_overlap_fallback", "analogs": [a for a, _ in analogs]}

        target_median = self._basin_median_qsim(basin_id)
        if np.isnan(target_median):
            fallback = max(q_sim_raw, 0.0) if rule_id in ("R0", "R2") else q_sim_raw
            return fallback, {"method": "no_target_median_fallback", "analogs": used}

        analog_ratio = float(np.median(ratios))
        corrected = target_median * analog_ratio

        # never let the correction itself introduce a new physical impossibility
        if rule_id in ("R0", "R2"):
            corrected = max(corrected, 0.0)

        return corrected, {
            "method": "graph_analogy",
            "analogs": used,
            "analog_ratio": analog_ratio,
            "target_median": target_median,
            "raw_value": q_sim_raw,
        }
