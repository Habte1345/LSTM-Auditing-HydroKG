import sys
from pathlib import Path

repo_root = Path("/bighome/hdagne1/LSTM-Auditing-HydroKG")
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import tempfile
import requests

from src.hydrokg_graph import VIOLATION_CLASS_TO_RULES
from src.hydrokg_evaluation import compute_deltas

# =========================================================
# STYLE
# =========================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9

jet = plt.get_cmap("jet")

# =========================================================
# LOAD YOUR REAL RESULTS
# =========================================================
RESULT_PREFIX = str(repo_root / "results" / "hydrokg_run2")

def _load_audit_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["violation_counts"] = df["violation_counts"].apply(ast.literal_eval)
    df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
    df = df.drop(columns=["aridity_class", "landcover_class"], errors="ignore")
    return df

baseline_results = _load_audit_csv(f"{RESULT_PREFIX}_baseline_results.csv")
enhanced_results = _load_audit_csv(f"{RESULT_PREFIX}_enhanced_results.csv")

# =========================================================
# LOAD CAMELS ATTRIBUTES + GEOMETRY
# =========================================================
url = (
    "https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/"
    "data/contents/camels_attributes_v2.0.feather"
)
tmp = tempfile.NamedTemporaryFile(suffix=".feather", delete=False)
tmp.write(requests.get(url).content)
basins = gpd.read_feather(tmp.name).reset_index()

if basins.crs is None:
    basins = basins.set_crs("EPSG:4326")

basins_proj = basins.to_crs(5070)
basins_proj["geometry_points"] = basins_proj.geometry.centroid
basins_points = basins_proj.set_geometry("geometry_points").to_crs("EPSG:4326")
basins_points["gauge_id"] = basins_points["gauge_id"].astype(str).str.zfill(8)

# =========================================================
# ARIDITY CLASSIFICATION -- quantile-based
# =========================================================
# =========================================================
# ARIDITY CLASSIFICATION -- fixed literature thresholds (UNEP-style P/PET bins)
# Reverts from quantile-based bins. Note: with this CAMELS sample, these thresholds
# put ~670 of 671 basins into "Humid", leaving "Arid" with a single basin -- the
# same imbalance flagged earlier. Proceeding as requested; if the single-basin
# "Arid" box looks statistically odd in panels (b)/(e)/(f), consider merging
# Arid + Semi-arid into one category, or note the imbalance in the figure caption.
# =========================================================
basins_points["AI"] = 1.0 / basins_points["aridity"]

bins = [0, 0.42, 0.8, 1.2, np.inf]
aridity_order = ["Arid", "Semi-arid", "Dry sub-humid", "Humid"]
aridity_labels = ["Arid", "Semi-arid", "Dry\nsub-humid", "Humid"]

basins_points["aridity_class"] = pd.cut(
    basins_points["AI"], bins=bins, labels=aridity_order, include_lowest=True
)
basins_points = basins_points.dropna(subset=["aridity_class"])
print(basins_points["aridity_class"].value_counts())

# =========================================================
# LAND-COVER MAPPING
# =========================================================
land_cover_mapping = {
    "Croplands": "CL/NVM", "cropland/natural vegetation mosaic": "CL/NVM",
    "Deciduous Broadleaf Forest": "DBF", "Evergreen Needleleaf Forest": "EF",
    "Evergreen Broadleaf Forest": "EF", "Mixed Forests": "MF",
    "Grasslands": "GL", "Savannas": "WS + SL", "Woody Savannas": "WS + SL",
    "Closed Shrublands": "WS + SL", "Open Shrublands": "WS + SL",
}
basins_points["dom_land_cover_short"] = basins_points["dom_land_cover"].map(land_cover_mapping)
basins_points = basins_points.dropna(subset=["dom_land_cover_short"])
veg_order = ["CL/NVM", "DBF", "EF", "MF", "GL", "WS + SL"]

# =========================================================
# COLORS -- all drawn automatically from jet via even spacing, nothing hand-picked
# =========================================================
rule_class_order = ["Physical impossibility", "Magnitude failure", "Timing failure", "Budget failure"]

aridity_colors = dict(zip(aridity_order, jet(np.linspace(0, 1, len(aridity_order)))))
model_colors = dict(zip(["Traditional LSTM", "HydroKG-enhanced LSTM"], jet(np.linspace(0, 1, 2))))
rule_colors = dict(zip(rule_class_order, jet(np.linspace(0, 1, len(rule_class_order)))))

veg_marker_map = {"CL/NVM": "o", "DBF": "s", "EF": "^", "MF": "D", "GL": "P", "WS + SL": "*"}

# =========================================================
# MERGE REAL RESULTS ONTO CAMELS ATTRIBUTES
# =========================================================
strat_cols = basins_points[["gauge_id", "aridity_class", "dom_land_cover_short"]]

baseline_merged = baseline_results.merge(strat_cols, left_on="basin_id", right_on="gauge_id", how="inner")
enhanced_merged = enhanced_results.merge(strat_cols, left_on="basin_id", right_on="gauge_id", how="inner")
print(f"Matched {len(baseline_merged)} baseline / {len(enhanced_merged)} enhanced basins to CAMELS attributes")

deltas = compute_deltas(baseline_results, enhanced_results)
deltas = deltas.join(strat_cols.set_index("gauge_id"), how="inner")

# =========================================================
# REAL violation-class breakdown per basin, explicitly ordered
# =========================================================
class_display_names = {
    "PhysicalImpossibility": "Physical impossibility", "MagnitudeFailure": "Magnitude failure",
    "TimingFailure": "Timing failure", "BudgetScaleFailure": "Budget failure",
}

def class_totals(violation_counts: dict) -> pd.Series:
    return pd.Series({
        class_display_names[cls]: sum(violation_counts.get(r, 0) for r in rule_ids)
        for cls, rule_ids in VIOLATION_CLASS_TO_RULES.items()
    })

class_df = baseline_merged["violation_counts"].apply(class_totals)
baseline_merged = pd.concat([baseline_merged.reset_index(drop=True), class_df.reset_index(drop=True)], axis=1)

rule_comp = baseline_merged.groupby("aridity_class", observed=True)[rule_class_order].sum()
rule_comp = rule_comp.div(rule_comp.sum(axis=1), axis=0).T[aridity_order]
rule_comp = rule_comp.reindex(rule_class_order)

# =========================================================
# HELPER: grouped (Traditional vs Enhanced) boxplot
# =========================================================
def grouped_boxplot(ax, data_dict, categories, display_labels, ylim, ylabel, title):
    pos = np.arange(len(categories))
    width = 0.28
    left_data = [data_dict["Traditional LSTM"][c] for c in categories]
    right_data = [data_dict["HydroKG-enhanced LSTM"][c] for c in categories]

    bp1 = ax.boxplot(left_data, positions=pos - width / 2, widths=0.24, patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(right_data, positions=pos + width / 2, widths=0.24, patch_artist=True, showfliers=False)

    for b in bp1["boxes"]:
        b.set(facecolor=model_colors["Traditional LSTM"], edgecolor="black", linewidth=1.0, alpha=0.85)
    for b in bp2["boxes"]:
        b.set(facecolor=model_colors["HydroKG-enhanced LSTM"], edgecolor="black", linewidth=1.0, alpha=0.85)
    for bp in [bp1, bp2]:
        for key in ["whiskers", "caps", "medians"]:
            for obj in bp[key]:
                obj.set(color="black", linewidth=1.0)

    ax.set_xticks(pos)
    ax.set_xticklabels(display_labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=model_colors["Traditional LSTM"], edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor=model_colors["HydroKG-enhanced LSTM"], edgecolor="black"),
    ]
    ax.legend(handles, ["Traditional LSTM", "HydroKG-enhanced LSTM"], loc="upper left", frameon=False)

viol_aridity = {
    "Traditional LSTM": {c: baseline_merged.loc[baseline_merged["aridity_class"] == c, "violation_burden"].values
                          for c in aridity_order},
    "HydroKG-enhanced LSTM": {c: enhanced_merged.loc[enhanced_merged["aridity_class"] == c, "violation_burden"].values
                              for c in aridity_order},
}
viol_veg = {
    "Traditional LSTM": {c: baseline_merged.loc[baseline_merged["dom_land_cover_short"] == c, "violation_burden"].values
                         for c in veg_order},
    "HydroKG-enhanced LSTM": {c: enhanced_merged.loc[enhanced_merged["dom_land_cover_short"] == c, "violation_burden"].values
                              for c in veg_order},
}
skill_improve = {c: deltas.loc[deltas["aridity_class"] == c, "delta_kge"].values for c in aridity_order}
scatter_df = deltas.reset_index().rename(
    columns={"aridity_class": "Aridity", "dom_land_cover_short": "Vegetation",
             "delta_kge": "dKGE", "delta_violation_burden": "ViolationReduction"}
)

# =========================================================
# FIGURE
# =========================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Violations and Skill Improvement Across Hydrologic Regimes",
             fontsize=20, fontweight="bold", y=0.98)

# --- (a) CAMELS basins by aridity class -- automatic jet coloring via geopandas ---
ax = axes[0, 0]
states_all = gpd.read_file(str(repo_root / "data" / "States_shapefile" / "States_shapefile.shp")).to_crs(5070)
states = states_all[~states_all["State_Name"].isin(["ALASKA", "HAWAII"])]
states.boundary.plot(ax=ax, edgecolor="black", linewidth=0.2)

basins_5070 = basins_points.to_crs(5070)
basins_5070.plot(
    ax=ax, column="aridity_class", cmap="coolwarm_r", categorical=True, markersize=30,
    marker="o", linewidth=0.8, edgecolor="k", alpha=0.7, legend=True,
    legend_kwds={"frameon": False, "ncol": 4, "loc": "lower left", "fontsize": 9},
)
leg = ax.get_legend()
if leg is not None:
    leg.set_title(None)
ax.set_title(f"(a) Audited CAMELS basins by aridity class (n={len(basins_points)})", loc="left")
ax.set_axis_off()

# --- (b)/(c) Violation burden, Traditional vs Enhanced ---
grouped_boxplot(axes[0, 1], viol_aridity, aridity_order, aridity_labels, ylim=(0, 0.8),
                ylabel="Normalized violation burden", title="(b) Violation burden by aridity class")
grouped_boxplot(axes[0, 2], viol_veg, veg_order, veg_order, ylim=(0, 0.75),
                ylabel="Normalized violation burden", title="(c) Violation burden by land-cover class")

# --- (d) Dominant rule composition by aridity class ---
ax = axes[1, 0]
bottom = np.zeros(len(aridity_order))
bar_colors = [rule_colors[r] for r in rule_comp.index]
for rule, color in zip(rule_comp.index, bar_colors):
    vals = rule_comp.loc[rule, aridity_order].values
    ax.bar(aridity_labels, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.8)
    bottom += vals
ax.set_ylim(0, 1.0)
ax.set_ylabel("Fraction of total violations")
ax.set_title("(d) Dominant rule composition by aridity class (traditional LSTM)", loc="left")
ax.grid(axis="y", linestyle="--", alpha=0.25)
legend_handles = [Patch(facecolor=c, edgecolor="white", label=r) for r, c in zip(rule_comp.index, bar_colors)]
ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=True)

# --- (e) Real delta KGE by aridity class ---
ax = axes[1, 1]
data = [skill_improve[c] for c in aridity_order]
bp = ax.boxplot(data, patch_artist=True, widths=0.45, showfliers=False)
box_colors = [aridity_colors[c] for c in aridity_order]
for b, c in zip(bp["boxes"], box_colors):
    b.set(facecolor=c, edgecolor="black", linewidth=1.0, alpha=0.85)
for key in ["whiskers", "caps", "medians"]:
    for obj in bp[key]:
        obj.set(color="black", linewidth=1.0)
ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
ax.set_xticks(np.arange(1, 5))
ax.set_xticklabels(aridity_labels)
ax.set_ylabel(r"$\Delta$KGE (HydroKG - Traditional)")
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.set_title("(e) Skill improvement by aridity class", loc="left")

# --- (f) Real skill-trust improvement space -- legends INSET, matching reference ---
ax = axes[1, 2]
for veg in veg_order:
    for arid in aridity_order:
        sub = scatter_df[(scatter_df["Vegetation"] == veg) & (scatter_df["Aridity"] == arid)]
        if sub.empty:
            continue
        ax.scatter(sub["dKGE"], sub["ViolationReduction"], s=30, color=aridity_colors[arid],
                   marker=veg_marker_map[veg], edgecolor="black", linewidth=0.30, alpha=0.82)

ax.axvline(0, color="0.4", linestyle="--", linewidth=1.0)
ax.axhline(0, color="0.4", linestyle="--", linewidth=1.0)
ax.set_xlim(-0.5, 1)
ax.set_xlabel(r"$\Delta$KGE")
ax.set_ylabel(r"$\Delta V_b$ (violation-burden reduction)")
ax.set_title("(f) Skill-trust improvement space", loc="left")
ax.text(0.76, 0.94, "Improved skill\nand trust", transform=ax.transAxes, ha="center", va="center",
        fontsize=10, fontstyle="italic")

arid_handles = [Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=aridity_colors[a],
                       markeredgecolor="black", markersize=7, label=a) for a in aridity_order]
leg1 = ax.legend(handles=arid_handles, title="Aridity", loc="upper left", frameon=True, fontsize=8)
ax.add_artist(leg1)
veg_handles = [Line2D([0], [0], marker=veg_marker_map[v], linestyle="None", color="black",
                      markerfacecolor="white", markersize=7, label=v) for v in veg_order]
ax.legend(handles=veg_handles, title="Land cover", loc="lower right", frameon=True, fontsize=8)

for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

plt.tight_layout(rect=[0, 0.03, 0.95, 0.96])
plt.savefig(str(repo_root / "figures" / "violations_skill_full_panel.png"), bbox_inches="tight")
plt.show()