"""
ET as a water-balance residual (project decision: no external ET/PET product, e.g. GLEAM,
is used here -- unlike the GRACE/GLEAM-based CWatM calibration work, this framework treats
ET purely as P - Q so that R5/R6 need nothing beyond what's already in the LSTM
input/output plus CAMELS forcing).

    ET_obs(basin) = mean(P) - mean(Q_obs)     over a long-term (multi-annual) window

This assumes negligible long-term storage change (dS/dt ~ 0), the standard large-sample
hydrology assumption for multi-decade CAMELS records (Sankarasubramanian & Vogel, 2002).
It should NOT be applied over short/annual windows for individual dry or wet years, where
storage change can be a large fraction of the balance -- R5/R6 in this codebase therefore
use simulated quantities directly (Q_sim, P) rather than relying on a per-year ET_obs
estimate, precisely to avoid compounding a noisy short-window residual into the audit.
This module is provided for basin-level, long-term ET reporting/diagnostics (e.g. for
stratification and figures) rather than as an input to the rules themselves.
"""

from __future__ import annotations

import pandas as pd


def long_term_et_residual(p: pd.Series, qobs: pd.Series, min_years: int = 5) -> float:
    """Long-term mean ET (mm/day) as the water-balance residual P - Q_obs.

    Raises if fewer than `min_years` of overlapping daily records are available, since the
    dS/dt ~ 0 assumption is unreliable over short periods.
    """
    aligned = pd.DataFrame({"p": p, "qobs": qobs}).dropna()
    n_years = aligned.index.to_series().dt.year.nunique() if not aligned.empty else 0
    if n_years < min_years:
        raise ValueError(
            f"Only {n_years} years of overlapping P/Qobs data; need >= {min_years} for a "
            "stable long-term water-balance ET residual."
        )
    return float(aligned["p"].mean() - aligned["qobs"].mean())


def long_term_runoff_ratio(p: pd.Series, q: pd.Series) -> float:
    """mean(Q) / mean(P) -- used for basin aridity/water-balance sanity checks in figures."""
    aligned = pd.DataFrame({"p": p, "q": q}).dropna()
    p_mean = aligned["p"].mean()
    if p_mean <= 0:
        return float("nan")
    return float(aligned["q"].mean() / p_mean)
