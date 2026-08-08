"""
Combined 8-panel figure: (a)-(d) from the 2x2 rule/aridity/skill-trust figure,
(e)-(h) from the statistical-evidence figure (heatmap, CDFs, forest plot).
Every panel's own plotting code is preserved exactly as originally written --
only grid placement and panel letters changed. Data loading is merged once
(both source scripts loaded the same CSVs/CAMELS feather separately).

Usage: edit the CSV paths if needed, then run this file.
"""
import sys
from pathlib import Path

try:
    repo_root = Path(__file__).resolve().parent
except NameError:
    repo_root = Path("/bighome/hdagne1/LSTM-Auditing-HydroKG")

sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import ast
import tempfile

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams.update({
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "savefig.dpi": 600,
})

# ============================================================
# 1. LOAD AND PAIR YOUR REAL BASELINE + ENHANCED RESULTS (shared by all 8 panels)
# ============================================================
BASELINE_CSV = repo_root / "results" / "hydrokg_run2_REAUDITED_baseline_results.csv"
ENHANCED_CSV = repo_root / "results" / "hydrokg_run2_REAUDITED_enhanced_results.csv"

rule_cols = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
rule_to_class = {
    "R0": "Physical impossibility", "R2": "Physical impossibility",
    "R1": "Magnitude failure", "R3": "Magnitude failure",
    "R4": "Timing failure",
    "R5": "Budget failure", "R6": "Budget failure",
}
class_order = ["Physical impossibility", "Magnitude failure", "Timing failure", "Budget failure"]


def load_stage(csv_path):
    raw = pd.read_csv(csv_path, dtype={"basin_id": str})
    raw["basin_id"] = raw["basin_id"].str.zfill(8)
    raw["violation_counts"] = raw["violation_counts"].apply(ast.literal_eval)
    raw["n_evaluable"] = raw["n_evaluable"].apply(ast.literal_eval)
    rule_rates = raw.apply(
        lambda row: pd.Series({r: row["violation_counts"].get(r, 0) / row["n_evaluable"].get(r, 1)
                                if row["n_evaluable"].get(r, 0) > 0 else np.nan for r in rule_cols}),
        axis=1,
    )
    out = pd.concat([raw[["basin_id", "kge", "violation_burden", "aridity_class", "violation_counts"]],
                      rule_rates], axis=1)
    return out.set_index("basin_id")


baseline = load_stage(BASELINE_CSV)
enhanced = load_stage(ENHANCED_CSV)
common = baseline.index.intersection(enhanced.index)
print(f"Paired basins: {len(common)}")
baseline, enhanced = baseline.loc[common], enhanced.loc[common]

delta_kge = enhanced["kge"] - baseline["kge"]
delta_burden = baseline["violation_burden"] - enhanced["violation_burden"]
aridity_class = baseline["aridity_class"]
aridity_order = [c for c in ["humid", "sub_humid", "semi_arid", "arid"] if c in aridity_class.unique()]
aridity_display = {"humid": "Humid", "sub_humid": "Sub-humid", "semi_arid": "Semi-arid", "arid": "Arid"}

jet = mpl.colormaps["jet"]
aridity_jet_frac = {"humid": 0.18, "sub_humid": 0.38, "semi_arid": 0.62, "arid": 0.85}
aridity_colors = {c: jet(aridity_jet_frac[c]) for c in aridity_order}
rule_colors_map = {
    "Physical impossibility": "#3b78d8", "Magnitude failure": "#ff8c00",
    "Timing failure": "#4caf50", "Budget failure": "#8a4cc2",
}
model_colors = {"Baseline": "magenta", "HydroKG": "darkgreen"}

# ============================================================
# 2. LAND-COVER, fresh from the authoritative CAMELS attribute (dom_land_cover) --
# fetched ONCE, used by both the (d) scatter and the (e) heatmap
# ============================================================
camels_url = ("https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/"
              "data/contents/camels_attributes_v2.0.feather")
tmp = tempfile.NamedTemporaryFile(suffix=".feather", delete=False)
tmp.write(requests.get(camels_url).content)
basins_geo = gpd.read_feather(tmp.name).reset_index(drop=False)
basins_geo["gauge_id"] = basins_geo["gauge_id"].astype(str).str.zfill(8)
basins_geo = basins_geo.set_index("gauge_id")

land_cover_mapping = {
    "Croplands": "CL/NVM", "cropland/natural vegetation mosaic": "CL/NVM",
    "Deciduous Broadleaf Forest": "DBF", "Evergreen Needleleaf Forest": "EF",
    "Evergreen Broadleaf Forest": "EF", "Mixed Forests": "MF",
    "Grasslands": "GL", "Savannas": "WS + SL", "Woody Savannas": "WS + SL",
    "Closed Shrublands": "WS + SL", "Open Shrublands": "WS + SL",
}
veg_order = ["CL/NVM", "DBF", "EF", "MF", "GL", "WS + SL"]
veg_marker_map = {"CL/NVM": "o", "DBF": "s", "EF": "^", "MF": "D", "GL": "P", "WS + SL": "*"}
landcover_grouped = basins_geo["dom_land_cover"].astype(str).str.strip().map(land_cover_mapping).reindex(common)
print("Land-cover basins matched:", landcover_grouped.notna().sum(), "of", len(common))

# ============================================================
# 3. (a) RULE-GROUP CONTRIBUTION TO TOTAL REDUCTION
# ============================================================
baseline_mean_rates = baseline[rule_cols].mean()
enhanced_mean_rates = enhanced[rule_cols].mean()
abs_reduction_per_rule = (baseline_mean_rates - enhanced_mean_rates)
class_abs_reduction = pd.Series(0.0, index=class_order)
for r in rule_cols:
    class_abs_reduction[rule_to_class[r]] += max(abs_reduction_per_rule[r], 0)
class_pct_contribution = 100 * class_abs_reduction / class_abs_reduction.sum()
print("\nRule-group contribution to total (positive) reduction:\n", class_pct_contribution)

# ============================================================
# 4. (b) DOMINANT RULE COMPOSITION BY ARIDITY CLASS
# ============================================================
def class_totals(violation_counts: dict) -> pd.Series:
    return pd.Series({cls: sum(violation_counts.get(r, 0) for r, c in rule_to_class.items() if c == cls)
                       for cls in class_order})

class_df = baseline["violation_counts"].apply(class_totals)
class_df["aridity_class"] = aridity_class
rule_comp = class_df.groupby("aridity_class", observed=True)[class_order].sum()
rule_comp = rule_comp.div(rule_comp.sum(axis=1), axis=0).T[aridity_order]
print("\nDominant rule composition by aridity class (fractions):\n", rule_comp)

# ============================================================
# 5. (e) HEATMAP data: aridity x land-cover interaction, mean delta_kge per cell
# ============================================================
interaction_df = pd.DataFrame({
    "aridity": aridity_class, "landcover": landcover_grouped, "delta_kge": delta_kge,
}).dropna()
heat = interaction_df.pivot_table(index="aridity", columns="landcover", values="delta_kge", aggfunc="mean")
heat = heat.reindex(index=aridity_order, columns=veg_order)
counts = interaction_df.pivot_table(index="aridity", columns="landcover", values="delta_kge", aggfunc="count")
counts = counts.reindex(index=aridity_order, columns=veg_order)

# ============================================================
# 6. (h) BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================
rng = np.random.default_rng(0)


def bootstrap_ci(values, n_boot=2000, ci=95):
    values = np.asarray(values)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return values.mean(), lo, hi


kge_ci = {c: bootstrap_ci(delta_kge[aridity_class == c].dropna().values) for c in aridity_order}
vb_ci = {c: bootstrap_ci(delta_burden[aridity_class == c].dropna().values) for c in aridity_order}

# ============================================================
# 7. FIGURE: 4 rows x 2 columns, (a)-(d) then (e)-(h)
# ============================================================
fig, axes = plt.subplots(4, 2, figsize=(9.5, 13), gridspec_kw={"wspace": 0.5, "hspace": 0.65}, dpi=300)

# --- (a) Rule-group contribution bar chart -- UNCHANGED from the 2x2 figure ---
ax = axes[0, 0]
ax.set_title("(a)", loc="left", pad=6)
sorted_contribution = class_pct_contribution.sort_values(ascending=True)
bar_colors_a = [rule_colors_map[g] for g in sorted_contribution.index]

bars_a = ax.barh(sorted_contribution.index, sorted_contribution.values, color=bar_colors_a,
                 edgecolor="black", linewidth=0.9)

ax.set_yticklabels([
    label.replace(" ", "\n", 1)
    for label in sorted_contribution.index
])

for b, val in zip(bars_a, sorted_contribution.values):
    ax.text(val + 1.0, b.get_y() + b.get_height() / 2, f"{val:.1f}%",
            va="center", fontsize=10, fontweight="bold")

ax.set_xlim(0, max(sorted_contribution.values) * 1.18)
ax.set_xlabel("% of total violation reduction")
ax.grid(axis="x", linestyle="--", alpha=0.25)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# --- (b) Dominant rule composition by aridity class -- UNCHANGED ---
ax = axes[0, 1]
bottom = np.zeros(len(aridity_order))
aridity_labels = [aridity_display[c] for c in aridity_order]
for rule_class in rule_comp.index:
    vals = rule_comp.loc[rule_class, aridity_order].values
    ax.bar(aridity_labels, vals, bottom=bottom, color=rule_colors_map[rule_class],
           edgecolor="white", linewidth=0.8, label=rule_class)
    bottom += vals
ax.set_ylim(0, 1.0)
ax.set_ylabel("Fraction of total violations")
ax.set_title("(b)", loc="left", pad=6)
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.tick_params(axis="x", labelrotation=45)
ax.set_axisbelow(True)
ax.legend(loc="upper right", ncol=1, frameon=1, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# --- (c) Skill improvement (delta KGE) by aridity class -- UNCHANGED ---
ax = axes[1, 0]
data = [delta_kge[aridity_class == c].dropna().values for c in aridity_order]
bp = ax.boxplot(data, patch_artist=True, widths=0.45, showfliers=False,
                 medianprops=dict(color="black", linewidth=1.1),
                 whiskerprops=dict(color="black", linewidth=0.9),
                 capprops=dict(color="black", linewidth=0.9))
for b, c in zip(bp["boxes"], aridity_order):
    b.set(facecolor=aridity_colors[c], edgecolor="black", linewidth=1.0, alpha=0.85)
ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
ax.set_xticks(np.arange(1, len(aridity_order) + 1))
ax.set_xticklabels(aridity_labels)
ax.tick_params(axis="x", labelrotation=45)
ax.set_ylabel(r"$\Delta$KGE")
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.set_axisbelow(True)
ax.set_title("(c)", loc="left", pad=6)
ax.spines[["top", "right"]].set_visible(False)

# --- (d) Skill-trust improvement space -- UNCHANGED ---
ax = axes[1, 1]
for veg in veg_order:
    for arid in aridity_order:
        mask = (landcover_grouped == veg) & (aridity_class == arid)
        if mask.sum() == 0:
            continue
        ax.scatter(delta_kge[mask], delta_burden[mask], s=30, color=aridity_colors[arid],
                   marker=veg_marker_map[veg], edgecolor="black", linewidth=0.30, alpha=0.82)
ax.axvline(0, color="0.4", linestyle="--", linewidth=1.0)
ax.axhline(0, color="0.4", linestyle="--", linewidth=1.0)
ax.set_xlabel(r"$\Delta$KGE")
ax.set_ylabel(r"$\Delta V_b$")
ax.set_xlim(-0.5, 1)
ax.set_title("(d)", loc="left", pad=6)
arid_handles = [Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=aridity_colors[c],
                       markeredgecolor="black", markersize=7, label=aridity_display[c]) for c in aridity_order]
leg1 = ax.legend(handles=arid_handles, title="Aridity classes", loc="upper left", frameon=0, fontsize=8)
ax.add_artist(leg1)
veg_handles = [Line2D([0], [0], marker=veg_marker_map[v], linestyle="None", color="black",
                      markerfacecolor="white", markersize=7, label=v) for v in veg_order]
ax.legend(handles=veg_handles, title="Vegetation types", loc="lower right", frameon=0, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# --- (e) Heatmap: aridity x land-cover interaction -- UNCHANGED from the
# statistical-evidence figure (was panel (a) there) ---
ax = axes[2, 0]
vmax = np.nanmax(np.abs(heat.values))
im = ax.imshow(heat.values, cmap="bwr_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(veg_order)))
ax.set_xticklabels(veg_order, rotation=45)
ax.set_yticks(range(len(aridity_order)))
ax.set_yticklabels([aridity_display[c] for c in aridity_order])
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        val = heat.values[i, j]
        n = counts.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:+.2f}\n(n={int(n)})", ha="center", va="center", fontsize=8,
                     color="white" if abs(val) > vmax * 0.5 else "black")
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"Mean $\Delta$KGE")
ax.set_title("(e)", loc="left")

# --- (f) Empirical CDF: KGE -- UNCHANGED (was panel (b) there) ---
ax = axes[2, 1]
for stage_name, df, color in [("Baseline", baseline, model_colors["Baseline"]),
                                ("HydroKG", enhanced, model_colors["HydroKG"])]:
    sorted_vals = np.sort(df["kge"].dropna().values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, color=color, linewidth=2, label=stage_name)
ax.set_xlabel("KGE")
ax.set_ylabel("Cumulative fraction")
ax.set_xlim(-1, 1)
ax.set_title("(f)", loc="left")
ax.legend(frameon=False, loc="upper left")
ax.grid(alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)

# --- (g) Empirical CDF: violation burden -- UNCHANGED (was panel (c) there) ---
ax = axes[3, 0]
for stage_name, df, color in [("Baseline", baseline, model_colors["Baseline"]),
                                ("HydroKG", enhanced, model_colors["HydroKG"])]:
    sorted_vals = np.sort(df["violation_burden"].dropna().values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, color=color, linewidth=2, label=stage_name)
ax.set_xlabel("Violation burden $V_b$")
ax.set_ylabel("Cumulative fraction")
ax.set_title("(g)", loc="left")
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)


# --- (h) Forest plot: bootstrap 95% CI -- UNCHANGED (was panel (d) there) ---
ax = axes[3, 1]
y_positions = np.arange(len(aridity_order))
offset = 0.15

for metric_name, ci_dict, color, off in [(r"$\Delta$KGE", kge_ci, "magenta", -offset),
                                         (r"$\Delta V_b$", vb_ci, "darkgreen", offset)]:
    means = [ci_dict[c][0] for c in aridity_order]
    los = [ci_dict[c][0] - ci_dict[c][1] for c in aridity_order]
    his = [ci_dict[c][2] - ci_dict[c][0] for c in aridity_order]

    ax.errorbar(means, y_positions + off, xerr=[los, his], fmt="o",
                color=color, capsize=4, markersize=7, linewidth=1.6)

ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_yticks(y_positions)
ax.set_yticklabels([aridity_display[c] for c in aridity_order])
ax.set_xlabel("Mean improvement")
ax.set_title("(h)", loc="left")

legend_handles = [
    Line2D([0], [0], color="magenta", linewidth=1.6, label=r"$\Delta$KGE"),
    Line2D([0], [0], color="darkgreen", linewidth=1.6, label=r"$\Delta V_b$")
]

ax.legend(handles=legend_handles, frameon=False, loc="best")
ax.grid(axis="x", alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

save_path = repo_root / "figures" / "Figure_8panel_combined.png"
fig.savefig(save_path, dpi=600, bbox_inches="tight")
plt.show()
print(f"\nSaved to {save_path}")