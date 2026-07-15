"""
Publication-quality figures: skill-trust scatter, rule/stratified violation summaries,
and before/after enhancement comparisons. Merged from 3 files. Matplotlib only, no
seaborn dependency; clean layout, minimal clutter, no gridlines by default.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hydrokg_graph import RULE_IDS, RULE_METADATA

_CLASS_COLORS = {
    "PhysicalImpossibility": "#B23A48",
    "MagnitudeFailure": "#D68C45",
    "TimingFailure": "#4C6EF5",
    "BudgetScaleFailure": "#2F9E44",
}


# ============================================================================
# Skill-trust relationship
# ============================================================================


def plot_skill_trust_scatter(audit_results: pd.DataFrame, kge_threshold: float = 0.5,
                              ax=None, save_path: str | None = None):
    """KGE (x) vs violation burden (y), colored by dominant violation class."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    for cls, color in _CLASS_COLORS.items():
        subset = audit_results[audit_results.get("dominant_class") == cls]
        if subset.empty:
            continue
        ax.scatter(subset["kge"], subset["violation_burden"], s=22, color=color,
                   alpha=0.75, label=cls, edgecolors="none")

    ax.axvline(kge_threshold, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("KGE")
    ax.set_ylabel("Violation burden $V_b$")
    ax.set_title("Skill-trust relationship across basins")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return ax


# ============================================================================
# Violation summaries (by rule, by stratification class)
# ============================================================================


def plot_rule_violation_counts(violation_counts_by_basin: dict[str, dict[str, int]],
                                ax=None, save_path: str | None = None):
    """Total violation count per rule (R0-R6), summed across all basins."""
    totals = {rule_id: 0 for rule_id in RULE_IDS}
    for basin_counts in violation_counts_by_basin.values():
        for rule_id, count in basin_counts.items():
            totals[rule_id] = totals.get(rule_id, 0) + count

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    labels = [f"{rid}\n{RULE_METADATA[rid]['name']}" for rid in RULE_IDS]
    values = [totals[rid] for rid in RULE_IDS]
    ax.bar(labels, values, color="#345995", width=0.6)
    ax.set_ylabel("Total violations")
    ax.set_title("Violation counts by rule")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return ax


def plot_stratified_burden(stratified_summary: pd.DataFrame, by_column: str,
                            ax=None, save_path: str | None = None):
    """Mean violation burden per stratification class (e.g. aridity_class)."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4), dpi=150)

    ax.bar(stratified_summary[by_column].astype(str), stratified_summary["mean_violation_burden"],
           color="#7A4988", width=0.6)
    ax.set_ylabel("Mean violation burden $V_b$")
    ax.set_title(f"Violation burden by {by_column}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return ax


# ============================================================================
# Before/after enhancement comparison
# ============================================================================


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
    """delta_KGE (x) vs delta_violation_burden (y) -- joint skill/consistency gain."""
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
