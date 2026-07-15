"""
Evaluation: KGE (Eq. 2), skill-trust relationship analysis, enhancement metrics
(Eq. 4-6: delta KGE, delta violation burden, percent improved), and aridity/land-cover
stratified summaries. Merged from 4 files.

No dependency on hydrokg.audit or hydrokg.rules -- this module only operates on
DataFrames that audit.py's OfflineAuditor already produced, keeping it a leaf module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================================
# KGE (Eq. 2) -- the submodule only provides NSE-family metrics; this is HydroKG's own
# addition since the manuscript's skill-trust analysis is framed around KGE specifically.
# ============================================================================


def calc_kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta Efficiency: 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)."""
    obs = np.asarray(obs).flatten()
    sim = np.asarray(sim).flatten()
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs, sim = obs[mask], sim[mask]
    if len(obs) < 2 or np.std(obs) == 0:
        return float("nan")
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else float("nan")
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def kge_components(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Return r, alpha, beta, kge as a dict, for diagnostics/figures."""
    obs = np.asarray(obs).flatten()
    sim = np.asarray(sim).flatten()
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs, sim = obs[mask], sim[mask]
    if len(obs) < 2 or np.std(obs) == 0:
        return {"r": float("nan"), "alpha": float("nan"), "beta": float("nan"), "kge": float("nan")}
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else float("nan")
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {"r": float(r), "alpha": float(alpha), "beta": float(beta), "kge": float(kge)}


# ============================================================================
# Skill-trust relationship analysis
# ============================================================================


def skill_trust_correlation(audit_results: pd.DataFrame) -> dict:
    """Spearman correlation between KGE and violation burden across basins."""
    valid = audit_results[["kge", "violation_burden"]].dropna()
    if len(valid) < 3:
        return {"n_basins": len(valid), "spearman_r": float("nan"), "p_value": float("nan")}
    from scipy.stats import spearmanr
    r, p = spearmanr(valid["kge"], valid["violation_burden"])
    return {"n_basins": len(valid), "spearman_r": float(r), "p_value": float(p)}


def high_skill_high_violation_basins(audit_results: pd.DataFrame, kge_threshold: float = 0.5,
                                      burden_threshold: float = 0.05) -> pd.DataFrame:
    """Basins that look good on KGE but still violate physical rules at a non-trivial rate."""
    return audit_results[
        (audit_results["kge"] >= kge_threshold) & (audit_results["violation_burden"] >= burden_threshold)
    ].sort_values("violation_burden", ascending=False)


def summarize_skill_trust(audit_results: pd.DataFrame, kge_threshold: float = 0.5) -> dict:
    """Headline numbers for the manuscript's skill-trust framing."""
    total = len(audit_results)
    high_skill = audit_results[audit_results["kge"] >= kge_threshold]
    pct_high_skill = 100.0 * len(high_skill) / total if total else float("nan")
    pct_high_skill_with_violations = (
        100.0 * (high_skill["violation_burden"] > 0).sum() / len(high_skill)
        if len(high_skill) else float("nan")
    )
    return {
        "n_basins": total,
        "pct_basins_kge_above_threshold": pct_high_skill,
        "pct_of_those_with_any_violation": pct_high_skill_with_violations,
        "mean_violation_burden_high_skill": float(high_skill["violation_burden"].mean()) if len(high_skill) else float("nan"),
        "correlation": skill_trust_correlation(audit_results),
    }


# ============================================================================
# Eq. 4-6: enhancement metrics
# ============================================================================


def compute_deltas(baseline: pd.DataFrame, enhanced: pd.DataFrame) -> pd.DataFrame:
    """
    Eq. 4: delta_KGE_b = KGE_HydroKG_b - KGE_LSTM_b
    Eq. 5: delta_V_b   = V_LSTM_b - V_HydroKG_b  (positive = improved physical consistency)

    Parameters
    ----------
    baseline, enhanced : DataFrames with columns basin_id, kge, violation_burden
        (typically OfflineAuditor.audit_all() output run on the traditional LSTM and on
        the HydroKG-enhanced LSTM respectively).
    """
    base = baseline.set_index("basin_id")[["kge", "violation_burden"]]
    enh = enhanced.set_index("basin_id")[["kge", "violation_burden"]]
    joined = base.join(enh, lsuffix="_lstm", rsuffix="_hydrokg", how="inner")
    joined["delta_kge"] = joined["kge_hydrokg"] - joined["kge_lstm"]
    joined["delta_violation_burden"] = joined["violation_burden_lstm"] - joined["violation_burden_hydrokg"]
    return joined


def percent_improved(deltas: pd.DataFrame, column: str = "delta_kge") -> float:
    """Eq. 6, generalized to also apply to delta_violation_burden."""
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


# ============================================================================
# Aridity/land-cover stratified summaries
# ============================================================================


def stratified_violation_summary(audit_results: pd.DataFrame, by: str = "aridity_class") -> pd.DataFrame:
    """Mean KGE and violation burden per stratification class, plus basin counts."""
    if by not in audit_results.columns:
        raise KeyError(f"'{by}' not in audit_results columns: {list(audit_results.columns)}")
    return (
        audit_results.groupby(by, observed=True)
        .agg(n_basins=("basin_id", "count"), mean_kge=("kge", "mean"),
             mean_violation_burden=("violation_burden", "mean"))
        .reset_index().sort_values("mean_violation_burden", ascending=False)
    )


def stratified_dominant_class_counts(audit_results: pd.DataFrame, by: str = "aridity_class") -> pd.DataFrame:
    """Cross-tab of dominant violation class against a stratification variable."""
    if by not in audit_results.columns or "dominant_class" not in audit_results.columns:
        raise KeyError("audit_results must include both the stratification column and 'dominant_class'")
    return pd.crosstab(audit_results[by], audit_results["dominant_class"])
