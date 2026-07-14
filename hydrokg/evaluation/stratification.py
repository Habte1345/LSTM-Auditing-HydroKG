"""Aridity/land-cover stratified summaries of violation patterns and enhancement gains,
per the manuscript's "stratified by aridity class and dominant vegetation type" design."""

from __future__ import annotations

import pandas as pd


def stratified_violation_summary(audit_results: pd.DataFrame, by: str = "aridity_class") -> pd.DataFrame:
    """Mean KGE and violation burden per stratification class, plus basin counts."""
    if by not in audit_results.columns:
        raise KeyError(f"'{by}' not in audit_results columns: {list(audit_results.columns)}")
    return (
        audit_results.groupby(by, observed=True)
        .agg(
            n_basins=("basin_id", "count"),
            mean_kge=("kge", "mean"),
            mean_violation_burden=("violation_burden", "mean"),
        )
        .reset_index()
        .sort_values("mean_violation_burden", ascending=False)
    )


def stratified_dominant_class_counts(audit_results: pd.DataFrame, by: str = "aridity_class") -> pd.DataFrame:
    """Cross-tab of dominant violation class (physical impossibility / magnitude /
    timing / budget-scale) against a stratification variable."""
    if by not in audit_results.columns or "dominant_class" not in audit_results.columns:
        raise KeyError("audit_results must include both the stratification column and 'dominant_class'")
    return pd.crosstab(audit_results[by], audit_results["dominant_class"])
