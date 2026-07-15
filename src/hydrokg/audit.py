"""
Offline post-processing audit: applies R0-R6 to a completed LSTM's predictions and
computes the Eq. 3 violation burden. Merged from 3 files (offline_auditor.py,
violation_burden.py, plus the now-removed realtime_auditor.py).

Note on real-time detection: the actual "real-time" mechanism used by the real study
lives in hydrokg/enhanced_training.py (fine_tune()'s online R0-R3 detection against
every training batch's own forward pass). An earlier, more generic streaming auditor
(RealtimeAuditor, staged evaluation as data arrives) was removed here because it was
never actually used by the real pipeline -- only its own standalone demo called it.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import pandas as pd

from hydrokg.evaluation import calc_kge
from hydrokg.graph import GraphStore, RULE_IDS, RULE_METADATA
from hydrokg.rules import build_all_rules

logger = logging.getLogger(__name__)


# ============================================================================
# Eq. 3: violation burden
# ============================================================================


def compute_violation_burden(violation_counts: dict[str, int], n_evaluable: dict[str, int]) -> float:
    """
    V_b = (1/7) * sum_r [ N_violations(b, r) / N_evaluable(b, r) ]

    Rules with n_evaluable == 0 (e.g. too few years for an annual rule) are excluded
    from the average rather than treated as a 0/0 term.
    """
    terms = []
    for rule_id in RULE_IDS:
        n_eval = n_evaluable.get(rule_id, 0)
        if n_eval <= 0:
            continue
        n_viol = violation_counts.get(rule_id, 0)
        terms.append(n_viol / n_eval)
    if not terms:
        return float("nan")
    return sum(terms) / len(RULE_IDS) if len(terms) == len(RULE_IDS) else sum(terms) / len(terms)


def dominant_violation_class(violation_counts: dict[str, int]) -> Optional[str]:
    """The violation class (of the 4) contributing the largest share of a basin's
    total violations."""
    class_totals: dict[str, int] = defaultdict(int)
    for rule_id, count in violation_counts.items():
        if rule_id in RULE_METADATA:
            class_totals[RULE_METADATA[rule_id]["violation_class"]] += count
    if not class_totals:
        return None
    return max(class_totals, key=class_totals.get)


# ============================================================================
# Offline auditor
# ============================================================================


class OfflineAuditor:

    def __init__(self, graph_store: GraphStore, rule_params: Optional[dict] = None):
        self.graph = graph_store
        self.rules = build_all_rules(**(rule_params or {}))
        self.graph.initialize_schema()

    def audit_basin(self, basin_id: str, df: pd.DataFrame, aridity_class: Optional[str] = None,
                     landcover_class: Optional[str] = None) -> dict:
        """
        Parameters
        ----------
        df : DataFrame indexed by date with columns qobs, qsim, and (for R5/R6) p.

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
            "basin_id": basin_id, "kge": kge, "violation_burden": burden,
            "dominant_class": dom_class, "violation_counts": violation_counts,
            "n_evaluable": n_evaluable, "aridity_class": aridity_class,
            "landcover_class": landcover_class,
        }

    def audit_all(self, basin_dataframes: dict[str, pd.DataFrame],
                   stratification: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        basin_dataframes : {basin_id: DataFrame(qobs, qsim[, p])}
        stratification : optional DataFrame indexed by basin_id with columns
            aridity_class, landcover_class

        Returns
        -------
        DataFrame, one row per basin.
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
