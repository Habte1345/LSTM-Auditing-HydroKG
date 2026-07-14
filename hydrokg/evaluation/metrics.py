"""KGE (Eq. 2 in the manuscript). The submodule (Scripts/metrics.py) only provides
NSE-family metrics (Kratzert et al. 2019 trained/evaluated on NSE) -- KGE is HydroKG's own
addition since the manuscript's skill-trust analysis is framed around KGE specifically.
"""

from __future__ import annotations

import numpy as np


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
