"""
Eq. 4: delta_KGE_b = KGE_HydroKG_b - KGE_LSTM_b
Eq. 5: delta_V_b   = V_LSTM_b - V_HydroKG_b   (positive = improved physical consistency)
Eq. 6: P_improved  = 100 * count(delta_KGE_b > 0) / N_basins
"""

from __future__ import annotations

import pandas as pd


def compute_deltas(baseline: pd.DataFrame, enhanced: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    baseline, enhanced : DataFrames indexed or column-matched by basin_id, each with
        columns kge, violation_burden (typically OfflineAuditor.audit_all() output run
        once on the traditional LSTM and once on the HydroKG-enhanced LSTM).

    Returns
    -------
    DataFrame indexed by basin_id with delta_kge, delta_violation_burden.
    """
    base = baseline.set_index("basin_id")[["kge", "violation_burden"]]
    enh = enhanced.set_index("basin_id")[["kge", "violation_burden"]]
    joined = base.join(enh, lsuffix="_lstm", rsuffix="_hydrokg", how="inner")
    joined["delta_kge"] = joined["kge_hydrokg"] - joined["kge_lstm"]
    joined["delta_violation_burden"] = joined["violation_burden_lstm"] - joined["violation_burden_hydrokg"]
    return joined


def percent_improved(deltas: pd.DataFrame, column: str = "delta_kge") -> float:
    """Eq. 6, generalized to also apply to delta_violation_burden if desired."""
    valid = deltas[column].dropna()
    if valid.empty:
        return float("nan")
    return 100.0 * (valid > 0).sum() / len(valid)


def enhancement_summary(baseline: pd.DataFrame, enhanced: pd.DataFrame) -> dict:
    deltas = compute_deltas(baseline, enhanced)
    return {
        "n_basins": len(deltas),
        "pct_improved_skill": percent_improved(deltas, "delta_kge"),
        "pct_improved_physical_consistency": percent_improved(deltas, "delta_violation_burden"),
        "mean_delta_kge": float(deltas["delta_kge"].mean()),
        "mean_delta_violation_burden": float(deltas["delta_violation_burden"].mean()),
        "deltas": deltas,
    }
