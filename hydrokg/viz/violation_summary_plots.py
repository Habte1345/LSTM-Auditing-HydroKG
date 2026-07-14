"""Rule-level and stratified violation summary figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from hydrokg.graph.schema import RULE_IDS, RULE_METADATA


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
