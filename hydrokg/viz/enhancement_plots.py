"""Before/after (traditional LSTM vs HydroKG-enhanced LSTM) comparison figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_delta_kge_distribution(deltas: pd.DataFrame, ax=None, save_path: str | None = None):
    """Histogram of per-basin delta_KGE (Eq. 4), with a reference line at 0."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4), dpi=150)

    ax.hist(deltas["delta_kge"].dropna(), bins=30, color="#2F9E44", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(r"$\Delta$KGE (HydroKG-enhanced $-$ traditional LSTM)")
    ax.set_ylabel("Number of basins")
    ax.set_title("Distribution of skill change after enhancement")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return ax


def plot_skill_vs_consistency_gain(deltas: pd.DataFrame, ax=None, save_path: str | None = None):
    """delta_KGE (x) vs delta_violation_burden (y) -- the quadrant plot showing whether
    enhancement improves skill and physical consistency together (Eq. 4 & 5 jointly)."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)

    ax.scatter(deltas["delta_kge"], deltas["delta_violation_burden"], s=22,
               color="#345995", alpha=0.7, edgecolors="none")
    ax.axhline(0, color="grey", linewidth=1, alpha=0.6)
    ax.axvline(0, color="grey", linewidth=1, alpha=0.6)
    ax.set_xlabel(r"$\Delta$KGE")
    ax.set_ylabel(r"$\Delta V_b$ (positive = more physically consistent)")
    ax.set_title("Joint skill and physical-consistency improvement")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return ax
