"""
Quantifies the skill-trust relationship: does high KGE imply low physical violation
burden? Uses the OfflineAuditor.audit_all() output (one row per basin: kge,
violation_burden, dominant_class, ...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def skill_trust_correlation(audit_results: pd.DataFrame) -> dict:
    """Spearman correlation between KGE and violation burden across basins (Spearman,
    not Pearson, since violation burden is a bounded, non-normally-distributed rate)."""
    valid = audit_results[["kge", "violation_burden"]].dropna()
    if len(valid) < 3:
        return {"n_basins": len(valid), "spearman_r": float("nan"), "p_value": float("nan")}
    from scipy.stats import spearmanr

    r, p = spearmanr(valid["kge"], valid["violation_burden"])
    return {"n_basins": len(valid), "spearman_r": float(r), "p_value": float(p)}


def high_skill_high_violation_basins(audit_results: pd.DataFrame, kge_threshold: float = 0.5,
                                      burden_threshold: float = 0.05) -> pd.DataFrame:
    """The core skill-trust-gap diagnostic: basins that look good on KGE but still
    violate physical rules at a non-trivial rate."""
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
