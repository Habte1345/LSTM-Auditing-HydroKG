"""
Offline audit mode: HydroKG applied as a post-processing layer over a completed LSTM
prediction set (the submodule's evaluate() output). Implements the manuscript's
"Offline post-processing audit" section end to end: per-basin rule application, Eq. 3
violation burden, dominant violation class, and KGE-vs-burden comparison.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from hydrokg.audit.violation_burden import compute_violation_burden, dominant_violation_class
from hydrokg.evaluation.metrics import calc_kge
from hydrokg.graph.base import GraphStore
from hydrokg.rules.registry import build_all_rules

logger = logging.getLogger(__name__)


class OfflineAuditor:

    def __init__(self, graph_store: GraphStore, rule_params: Optional[dict] = None):
        self.graph = graph_store
        self.rules = build_all_rules(**(rule_params or {}))
        self.graph.initialize_schema()

    def audit_basin(
        self,
        basin_id: str,
        df: pd.DataFrame,
        aridity_class: Optional[str] = None,
        landcover_class: Optional[str] = None,
    ) -> dict:
        """
        Parameters
        ----------
        df : DataFrame indexed by date with columns qobs, qsim, and (for R5/R6) p.
             Rows missing 'p' are silently skipped by R5/R6 (see Rule.n_evaluable).

        Returns
        -------
        dict with kge, violation_burden, dominant_class, violation_counts, n_evaluable
        """
        self.graph.register_catchment(basin_id, aridity_class, landcover_class)

        violation_counts: dict[str, int] = {}
        n_evaluable: dict[str, int] = {}

        for rule_id, rule in self.rules.items():
            violations = rule.evaluate(basin_id, df, aridity_class, landcover_class)
            n_written = self.graph.write_violations(violations)
            violation_counts[rule_id] = n_written
            n_evaluable[rule_id] = rule.n_evaluable(df)

        valid = df[["qobs", "qsim"]].dropna()
        kge = calc_kge(valid["qobs"].values, valid["qsim"].values) if not valid.empty else float("nan")
        burden = compute_violation_burden(violation_counts, n_evaluable)
        dom_class = dominant_violation_class(violation_counts)

        self.graph.set_basin_metrics(basin_id, kge=kge, violation_burden=burden)

        return {
            "basin_id": basin_id,
            "kge": kge,
            "violation_burden": burden,
            "dominant_class": dom_class,
            "violation_counts": violation_counts,
            "n_evaluable": n_evaluable,
            "aridity_class": aridity_class,
            "landcover_class": landcover_class,
        }

    def audit_all(
        self,
        basin_dataframes: dict[str, pd.DataFrame],
        stratification: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        basin_dataframes : {basin_id: DataFrame(qobs, qsim[, p])}
        stratification : optional DataFrame indexed by basin_id with columns
            aridity_class, landcover_class (see hydrokg.data.basin_attributes)

        Returns
        -------
        DataFrame, one row per basin, ready for skill-trust analysis and stratified
        summaries (hydrokg.evaluation.skill_trust_analysis, .stratification).
        """
        rows = []
        for basin_id, df in basin_dataframes.items():
            aridity_class = landcover_class = None
            if stratification is not None and basin_id in stratification.index:
                aridity_class = stratification.loc[basin_id].get("aridity_class")
                landcover_class = stratification.loc[basin_id].get("landcover_class")
            try:
                result = self.audit_basin(basin_id, df, aridity_class, landcover_class)
                rows.append(result)
            except Exception as exc:  # noqa: BLE001 - one bad basin should not kill the run
                logger.warning("Audit failed for basin %s: %s", basin_id, exc)
        return pd.DataFrame(rows)
