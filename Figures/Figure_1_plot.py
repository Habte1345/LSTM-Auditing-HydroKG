import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.lines import Line2D
import tempfile
import requests

from hydrokg.rules.registry import VIOLATION_CLASS_TO_RULES

# =========================================================
# GLOBAL STYLE
# =========================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.9,
    "savefig.dpi": 600,
})

# =========================================================
# INPUTS YOU ALREADY HAVE
# =========================================================
# `results` = the DataFrame from auditor.audit_all(...) / your run_offline_audit output
# (columns: basin_id, kge, violation_burden, dominant_class, violation_counts, ...)
#
# `enhanced_results` = the SAME kind of DataFrame, but from auditing the HydroKG-enhanced
# LSTM's predictions. Set to None until you've actually run the enhancement pipeline
# (hydrokg.enhancement.enhanced_training.EnhancedTrainingPipeline) and re-audited its output.
enhanced_results = None  # <-- replace once you have it; do not fabricate this

# =========================================================
# LOAD CAMELS BASIN ATTRIBUTES + GEOMETRY (real, unchanged from your script)
# =========================================================
url = (
    "https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/"
    "data/contents/camels_attributes_v2.0.feather"
)
tmp = tempfile.NamedTemporaryFile(suffix=".feather", delete=False)
tmp.write(requests.get(url).content)
basins = gpd.read_feather(tmp.name).reset_index(drop=False)
if basins.crs is None:
    basins = basins.set_crs("EPSG:4326")

states_all = gpd.read_file(
    r"F:\Data\Mopex_Boundaries\CONUS_shape\States_shapefile.shp"
).to_crs(5070)
states_5070 = states_all[~states_all["State_Name"].isin(["ALASKA", "HAWAII"])].to_crs(5070)

basins_5070 = basins.to_crs(5070).copy()
basins_5070["geometry_points"] = basins_5070.geometry.centroid
basins_5070 = basins_5070.set_geometry("geometry_points")

# =========================================================
# MERGE YOUR REAL AUDIT RESULTS ONTO CAMELS ATTRIBUTES
# =========================================================
results = results.copy()
results["gauge_id"] = results["basin_id"].astype(str).str.zfill(8)
basins_5070["gauge_id"] = basins_5070["gauge_id"].astype(str).str.zfill(8)

merged = basins_5070.merge(results, on="gauge_id", how="inner")
print(f"Matched {len(merged)} of {len(results)} audited basins to CAMELS attributes")

# Aridity classification: AI = P/PET = 1/aridity (CAMELS 'aridity' is PET/P)
merged["AI"] = 1 / merged["aridity"]
bins = [0, 0.20, 0.50, 0.65, np.inf]
aridity_order = ["Arid", "Semi-arid", "Dry sub-humid", "Humid"]
merged["aridity_class"] = pd.cut(merged["AI"], bins=bins, labels=aridity_order, include_lowest=True)
merged = merged.dropna(subset=["aridity_class"])
aridity_display = {"Arid": "Arid", "Semi-arid": "Semi-arid",
                    "Dry sub-humid": "Dry\nsub-humid", "Humid": "Humid"}

aridity_colors_map = dict(zip(aridity_order, plt.cm.jet(np.linspace(0, 1, len(aridity_order)))))

# Land cover mapping (real, from your script)
land_cover_mapping = {
    "Croplands": "CL/NVM", "cropland/natural vegetation mosaic": "CL/NVM",
    "Deciduous Broadleaf Forest": "DBF", "Evergreen Needleleaf Forest": "EF",
    "Evergreen Broadleaf Forest": "EF", "Mixed Forests": "MF",
    "Grasslands": "GL", "Savannas": "WS + SL", "Woody Savannas": "WS + SL",
    "Closed Shrublands": "WS + SL", "Open Shrublands": "WS + SL",
}
merged["dom_land_cover_short"] = merged["dom_land_cover"].map(land_cover_mapping)
merged = merged.dropna(subset=["dom_land_cover_short"])
veg_order = ["CL/NVM", "DBF", "EF", "MF", "GL", "WS + SL"]

model_colors = {"Traditional LSTM": "#377eb8", "HydroKG-enhanced LSTM": "#ff7f00"}

# =========================================================
# REAL VIOLATION-CLASS BREAKDOWN PER BASIN (from violation_counts)
# =========================================================
class_display_names = {
    "PhysicalImpossibility": "Physical impossibility",
    "MagnitudeFailure": "Magnitude failure",
    "TimingFailure": "Timing failure",
    "BudgetScaleFailure": "Budget failure",
}
rule_colors = {
    "Physical impossibility": aridity_colors_map["Arid"],
    "Magnitude failure": aridity_colors_map["Semi-arid"],
    "Timing failure": aridity_colors_map["Dry sub-humid"],
    "Budget failure": aridity_colors_map["Humid"],
}

def class_totals(violation_counts: dict) -> pd.Series:
    return pd.Series({
        class_display_names[cls]: sum(violation_counts.get(r, 0) for r in rule_ids)
        for cls, rule_ids in VIOLATION_CLASS_TO_RULES.items()
    })

class_df = merged["violation_counts"].apply(class_totals)
merged = pd.concat([merged.reset_index(drop=True), class_df.reset_index(drop=True)], axis=1)

rule_comp = (
    merged.groupby("aridity_class", observed=True)[list(class_display_names.values())]
    .sum()
)
rule_comp = rule_comp.div(rule_comp.sum(axis=1), axis=0).T[aridity_order]  # fractions, rules x aridity

# =========================================================
# HELPER: single-series boxplot (one model only -- that's all we have so far)
# =========================================================
def single_boxplot(ax, data_by_category, categories, display_labels, ylabel, title, colors):
    data = [data_by_category[c].dropna().values for c in categories]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
                     medianprops=dict(color="black", linewidth=1.1),
                     whiskerprops=dict(color="black", linewidth=0.9),
                     capprops=dict(color="black", linewidth=0.9))
    for box, cat in zip(bp["boxes"], categories):
        box.set(facecolor=colors[cat], edgecolor="black", linewidth=0.9, alpha=0.85)
    ax.set_xticks(np.arange(1, len(categories) + 1))
    ax.set_xticklabels(display_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=6)
    ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)

# =========================================================
# FIGURE
# =========================================================
fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2),
                          gridspec_kw={"wspace": 0.25, "hspace": 0.34}, dpi=200)

# --- (a) CAMELS basins by aridity class, restricted to your audited/matched basins ---
ax = axes[0, 0]
for cls in aridity_order:
    sub = merged[merged["aridity_class"] == cls]
    ax.scatter(sub.geometry.x, sub.geometry.y, s=22, color=aridity_colors_map[cls],
               edgecolor="black", linewidth=0.20, alpha=0.85, label=cls, zorder=2)
states_5070.boundary.plot(ax=ax, edgecolor="black", linewidth=0.35, zorder=3)
ax.set_title(f"(a) Audited CAMELS basins by aridity class (n={len(merged)})", loc="left", pad=6)
ax.set_axis_off()
handles = [Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=aridity_colors_map[c],
                  markeredgecolor="black", markeredgewidth=0.4, markersize=6, label=c)
           for c in aridity_order]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.10),
          frameon=False, ncol=4, fontsize=8, handletextpad=0.6, columnspacing=0.8)

# --- (b) Real violation burden by aridity class (traditional LSTM only) ---
viol_by_aridity = {c: merged.loc[merged["aridity_class"] == c, "violation_burden"] for c in aridity_order}
single_boxplot(axes[0, 1], viol_by_aridity, aridity_order,
               [aridity_display[c] for c in aridity_order],
               "Violation burden $V_b$", "(b) Violation burden by aridity class (traditional LSTM)",
               aridity_colors_map)

# --- (c) Real violation burden by land-cover class (traditional LSTM only) ---
veg_colors = dict(zip(veg_order, plt.cm.tab10(np.linspace(0, 1, len(veg_order)))))
viol_by_veg = {c: merged.loc[merged["dom_land_cover_short"] == c, "violation_burden"] for c in veg_order}
single_boxplot(axes[0, 2], viol_by_veg, veg_order, veg_order,
               "Violation burden $V_b$", "(c) Violation burden by land-cover class (traditional LSTM)",
               veg_colors)

# --- (d) Real dominant-rule composition by aridity class ---
ax = axes[1, 0]
bottom = np.zeros(len(aridity_order))
for rule in rule_comp.index:
    vals = rule_comp.loc[rule, aridity_order].values
    ax.bar([aridity_display[c] for c in aridity_order], vals, bottom=bottom,
           color=rule_colors[rule], edgecolor="white", linewidth=0.7, label=rule)
    bottom += vals
ax.set_ylim(0, 1.0)
ax.set_ylabel("Fraction of total violations")
ax.set_title("(d) Dominant rule composition by aridity class", loc="left", pad=6)
ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True,
          framealpha=0.95, edgecolor="0.7", fontsize=8)

# --- (e)/(f): require enhanced_results (traditional LSTM alone can't populate these) ---
for ax, letter, title in [
    (axes[1, 1], "e", "Skill improvement by aridity class"),
    (axes[1, 2], "f", "Skill-trust improvement space"),
]:
    ax.set_title(f"({letter}) {title}", loc="left", pad=6)
    if enhanced_results is None:
        ax.text(0.5, 0.5, "Pending: run HydroKG-enhanced LSTM\nand re-audit to populate",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color="0.35",
                fontstyle="italic")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        # merge enhanced_results in the same way as `merged` above, compute delta_kge and
        # delta_violation_burden per basin (hydrokg.evaluation.enhancement_metrics.compute_deltas),
        # then reproduce your original (e)/(f) plotting logic against those real deltas.
        pass

for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

plt.tight_layout(rect=[0, 0.035, 1, 0.955])
plt.show()