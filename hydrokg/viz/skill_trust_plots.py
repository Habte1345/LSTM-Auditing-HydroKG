"""Publication-quality skill-trust figures. Clean layout, minimal clutter, no gridlines
by default, matplotlib only (no seaborn dependency)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

_CLASS_COLORS = {
    "PhysicalImpossibility": "#B23A48",
    "MagnitudeFailure": "#D68C45",
    "TimingFailure": "#4C6EF5",
    "BudgetScaleFailure": "#2F9E44",
}


def plot_skill_trust_scatter(audit_results: pd.DataFrame, kge_threshold: float = 0.5,
                              ax=None, save_path: str | None = None):
    """KGE (x) vs violation burden (y), colored by dominant violation class -- the
    figure that makes the skill-trust gap visible at a glance."""
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
