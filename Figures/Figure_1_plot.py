"""
Mixed-type findings figure: 2 spatial maps (skill and consistency improvement,
geographically), 1 clean paired scatter (individual-basin movement, no confusing
connector lines), 1 boxplot (distribution by aridity, not just means), 1 bar chart
(per-rule mechanism story), 1 ranked-improvement line (how many basins actually
improved). All from your real, paired baseline/enhanced re-audited data.

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
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
mpl.rcParams['figure.dpi'] = 150

# ============================================================
# 1. LOAD AND PAIR YOUR REAL BASELINE + ENHANCED RESULTS
# ============================================================
BASELINE_CSV = repo_root / "results" / "hydrokg_run2_REAUDITED_baseline_results.csv"
ENHANCED_CSV = repo_root / "results" / "hydrokg_run2_REAUDITED_enhanced_results.csv"

rule_cols = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]


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
    out = pd.concat([raw[["basin_id", "kge", "violation_burden", "aridity_class", "landcover_class"]],
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


jet = mpl.colormaps["jet"]

model_colors = {"Base": 'magenta', "HydroKG": 'darkgreen'}
aridity_display = {"humid": "Humid", "sub_humid": "Sub-humid", "semi_arid": "Semi-arid", "arid": "Arid"}
aridity_jet_frac = {"humid": 0.18, "sub_humid": 0.38, "semi_arid": 0.62, "arid": 0.85}
aridity_colors = {c: jet(aridity_jet_frac[c]) for c in aridity_order}

# ============================================================
# 2. REAL BASIN GEOMETRY
# ============================================================
camels_url = ("https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/"
              "data/contents/camels_attributes_v2.0.feather")
tmp = tempfile.NamedTemporaryFile(suffix=".feather", delete=False)
tmp.write(requests.get(camels_url).content)
basins_geo = gpd.read_feather(tmp.name).reset_index(drop=False)
if basins_geo.crs is None:
    basins_geo = basins_geo.set_crs("EPSG:4326")
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
landcover_grouped_full = basins_geo["dom_land_cover"].astype(str).str.strip().map(land_cover_mapping)
landcover_grouped = landcover_grouped_full.reindex(common)
landcover_order = veg_order
print("Land-cover basins matched to the 6 mapped classes:", landcover_grouped.notna().sum(), "of", len(common))

states_url = ("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
              "master/data/geojson/us-states.json")
states_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
states_tmp.write(requests.get(states_url).content)
states = gpd.read_file(states_tmp.name).to_crs(5070)
states = states[~states["name"].isin(["Alaska", "Hawaii", "Puerto Rico"])]

merged_geo = basins_geo.loc[basins_geo.index.intersection(common)].to_crs(5070)
merged_geo["geometry"] = merged_geo.geometry.centroid
merged_geo["delta_kge"] = delta_kge.reindex(merged_geo.index)
merged_geo["delta_burden"] = delta_burden.reindex(merged_geo.index)
print(f"Basins with real geometry matched: {len(merged_geo)}")

# ============================================================
# 3. FIGURE
# ============================================================
fig = plt.figure(figsize=(14, 8), dpi=200)
gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

# --- (a) [gs[0,0]] skill-trust scatter, combined -- NOW with the three
# violation-severity reference bands (High/Medium/Low), restored from the
# original bubble-template figure. Both scatter series (Traditional, Enhanced)
# are kept exactly as before; only the reference bands + labels are added. ---
ax = fig.add_subplot(gs[0, 0])
ax.scatter(baseline["kge"], baseline["violation_burden"], s=16, color=model_colors["Base"],
           alpha=0.55, edgecolors="none", label="Base", zorder=2)
ax.scatter(enhanced["kge"], enhanced["violation_burden"], s=16, color=model_colors["HydroKG"],
           alpha=0.55, edgecolors="none", label="HydroKG", zorder=3)
ax.set_yscale("log")
ax.set_xlim(-0.05, 1.02)
ax.set_ylim(8e-4, 1.3)
major_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
ax.set_yticks(major_ticks)
ax.set_yticklabels([f"{t:g}" for t in major_ticks])
ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
for yy in [1e-1, 1e-2]:
    ax.axhline(yy, color="0.7", lw=1.0, ls=(0, (4, 3)), zorder=1)
ax.text(0.0, 8.4e-1, "High violation", color="black", fontsize=8, style="italic", va="center")
ax.text(0.0, 2.8e-2, "Medium violation", color="black", fontsize=8, style="italic", va="center")
ax.text(0.0, 2.5e-3, "Low violation", color="black", fontsize=8, style="italic", va="center")
ax.set_xlabel("KGE")
ax.set_ylabel("Violation burden $V_b$")
ax.set_title("(a)", loc="left", fontsize=10.5)
ax.legend(frameon=False, fontsize=10, loc="lower right", ncols=1)
ax.spines[["top", "right"]].set_visible(False)

# --- (b) [gs[0,1]] geographic consistency change -- colorbar now in its OWN
# dedicated axes (via make_axes_locatable), so it no longer shrinks this map's
# own axes relative to every other panel in the grid. This is the actual fix. ---
ax = fig.add_subplot(gs[0, 1])
states.boundary.plot(ax=ax, edgecolor="0.4", linewidth=0.5, zorder=1)
sc = ax.scatter(merged_geo.geometry.x, merged_geo.geometry.y, c=merged_geo["delta_burden"],
                 cmap="jet", vmin=-0.1, vmax=0.1, s=20, edgecolor="k", linewidth=0, zorder=2)
ax.set_aspect("auto")  # override geopandas' default equal-aspect box-shrinking
ax.set_title("(b)", loc="left", fontsize=12)
ax.set_axis_off()
# Colorbar placed in the natural gap BELOW this map's own rectangle (using the row's
# hspace), not carved out of the map's own axes -- this is what keeps every panel's
# rectangle exactly the same size. ax.get_position() reflects GridSpec's own layout
# math and is reliable without needing a draw() call first.
pos_b = ax.get_position()
cax_b = fig.add_axes([pos_b.x0 + pos_b.width * 0.15, pos_b.y0 - 0.022, pos_b.width * 0.7, 0.012])
fig.colorbar(sc, cax=cax_b, orientation="horizontal", label=r"$\Delta V_b$", extend="max")

# --- (c) [gs[0,2]] geographic skill change -- same colorbar fix ---
ax = fig.add_subplot(gs[0, 2])
states.boundary.plot(ax=ax, edgecolor="0.4", linewidth=0.5, zorder=1)
sc = ax.scatter(merged_geo.geometry.x, merged_geo.geometry.y, c=merged_geo["delta_kge"],
                 cmap="jet", vmin=-0.3, vmax=0.3, s=20, edgecolor="k", linewidth=0, zorder=2)
ax.set_aspect("auto")  # override geopandas' default equal-aspect box-shrinking
ax.set_title("(c)", loc="left", fontsize=12)
ax.set_axis_off()
pos_a = ax.get_position()
cax_a = fig.add_axes([pos_a.x0 + pos_a.width * 0.15, pos_a.y0 - 0.022, pos_a.width * 0.7, 0.012])
fig.colorbar(sc, cax=cax_a, orientation="horizontal", label=r"$\Delta$KGE", extend="max")

# --- (d) ---
ax = fig.add_subplot(gs[1, 0])
for c in aridity_order:
    mask = aridity_class == c
    ax.scatter(baseline.loc[mask, "kge"], enhanced.loc[mask, "kge"], s=13,
               color=aridity_colors[c], alpha=0.65, edgecolors="none", label=aridity_display[c])
lims = [min(0, 0), max(1, 1)]
ax.plot(lims, lims, color="black", linewidth=1, linestyle="--", zorder=1)
ax.set_xlabel("Base LSTM KGE")
ax.set_ylabel("HydroKG LSTM KGE")
ax.set_xlim(0, 0.98)
ax.set_ylim(0, 0.98)
ax.set_title("(d)", loc="left", fontsize=10.5)
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

# --- (e) ---
ax = fig.add_subplot(gs[1, 1])
for c in aridity_order:
    mask = aridity_class == c
    ax.scatter(baseline.loc[mask, "violation_burden"], enhanced.loc[mask, "violation_burden"],
               s=13, color=aridity_colors[c], alpha=0.65, edgecolors="none", label=aridity_display[c])
lims_b = [0, max(baseline["violation_burden"].max(), enhanced["violation_burden"].max())]
ax.plot(lims_b, lims_b, color="black", linewidth=1, linestyle="--", zorder=1)
ax.set_xlabel("Base LSTM $V_b$")
ax.set_ylabel("HydroKG LSTM $V_b$")
ax.set_title("(e)", loc="left", fontsize=12)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 0.5)
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

# --- (f) ---
ax = fig.add_subplot(gs[1, 2])
positions_base = np.arange(len(aridity_order)) - 0.2
positions_enh = np.arange(len(aridity_order)) + 0.2
data_base = [baseline.loc[aridity_class == c, "violation_burden"].dropna() for c in aridity_order]
data_enh = [enhanced.loc[aridity_class == c, "violation_burden"].dropna() for c in aridity_order]
bp1 = ax.boxplot(data_base, positions=positions_base, widths=0.32, patch_artist=True, showfliers=False)
bp2 = ax.boxplot(data_enh, positions=positions_enh, widths=0.32, patch_artist=True, showfliers=False)
for box in bp1["boxes"]:
    box.set(facecolor=model_colors["Base"], alpha=0.6)
for box in bp2["boxes"]:
    box.set(facecolor=model_colors["HydroKG"], alpha=0.6)
ax.set_xticks(np.arange(len(aridity_order)))
ax.set_xticklabels([aridity_display[c] for c in aridity_order], rotation=20, ha="right")
ax.set_ylabel("Violation burden $V_b$")
ax.set_title("(f)", loc="left", fontsize=12)
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Base", "HydroKG"], frameon=False, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

# --- (g) ---
ax = fig.add_subplot(gs[2, 0])
positions_base = np.arange(len(landcover_order)) - 0.2
positions_enh = np.arange(len(landcover_order)) + 0.2
data_base_lc = [baseline.loc[landcover_grouped == c, "violation_burden"].dropna() for c in landcover_order]
data_enh_lc = [enhanced.loc[landcover_grouped == c, "violation_burden"].dropna() for c in landcover_order]
bp3 = ax.boxplot(data_base_lc, positions=positions_base, widths=0.32, patch_artist=True, showfliers=False)
bp4 = ax.boxplot(data_enh_lc, positions=positions_enh, widths=0.32, patch_artist=True, showfliers=False)
for box in bp3["boxes"]:
    box.set(facecolor=model_colors["Base"], alpha=0.6)
for box in bp4["boxes"]:
    box.set(facecolor=model_colors["HydroKG"], alpha=0.6)
ax.set_xticks(np.arange(len(landcover_order)))
ax.set_xticklabels(landcover_order, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Violation burden $V_b$")
ax.set_title("(g)", loc="left", fontsize=12)
ax.legend([bp3["boxes"][0], bp4["boxes"][0]], ["Base", "HydroKG"], frameon=False, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

# --- (h) ---
ax = fig.add_subplot(gs[2, 1])
x = np.arange(len(rule_cols))
width = 0.35
base_means = [baseline[r].mean() for r in rule_cols]
enh_means = [enhanced[r].mean() for r in rule_cols]
ax.bar(x - width / 2, base_means, width, color=model_colors["Base"], edgecolor="black", label="Base")
ax.bar(x + width / 2, enh_means, width, color=model_colors["HydroKG"], edgecolor="black", label="HydroKG")
for i, (b, e) in enumerate(zip(base_means, enh_means)):
    pct = (e - b) / b * 100 if b > 0 else float("nan")
    ax.text(i + width / 2, e + max(b, e) * 0.05 + 0.005, f"{pct:+.0f}%", ha="center", fontsize=8,rotation=90,
             color="darkgreen" if pct < 0 else "darkred")
ax.set_xticks(x)
ax.set_xticklabels(rule_cols)
ax.set_ylabel("Mean violation rate")
ax.set_title("(h)", loc="left", fontsize=12)
ax.legend(frameon=False, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

# --- (i) ---
ax = fig.add_subplot(gs[2, 2])
sorted_delta = delta_kge.sort_values().values
n_basins_total = len(sorted_delta)
crossover_idx = int(np.searchsorted(sorted_delta, 0))

ax.plot(np.arange(n_basins_total), sorted_delta, color="black", linewidth=1.3, zorder=3)
ax.axhline(0, color="0.3", linewidth=0.8, zorder=1)
ax.axvline(crossover_idx, color="0.3", linewidth=0.8, linestyle=":", zorder=1)
ax.fill_between(np.arange(n_basins_total), sorted_delta, 0, where=(sorted_delta >= 0),
                 color=model_colors["HydroKG"], alpha=0.5, zorder=2)
ax.fill_between(np.arange(n_basins_total), sorted_delta, 0, where=(sorted_delta >= 0),
                 color=model_colors["HydroKG"], alpha=0.5, zorder=2)
ax.fill_between(np.arange(n_basins_total), sorted_delta, 0, where=(sorted_delta < 0),
                 color=model_colors["Base"], alpha=0.5, zorder=2)

pct_improved = 100 * (delta_kge > 0).mean()
pct_worsened = 100 * (delta_kge < 0).mean()
legend_handles = [
    Patch(facecolor=model_colors["HydroKG"], alpha=0.6,
                       label=f"(\u0394KGE > 0): {pct_improved:.0f}%"),
    Patch(facecolor=model_colors["Base"], alpha=0.6,
                       label=f"(\u0394KGE < 0): {pct_worsened:.0f}% "),
]
ax.legend(handles=legend_handles, frameon=False, fontsize=10, loc="upper right")
# ax.text(crossover_idx, ax.get_ylim()[0] * 0.92, f"  {n_basins_total - crossover_idx} of {n_basins_total}\n  basins improved",
#         fontsize=7.5, color="0.3", va="bottom")

ax.set_xlabel("Basins (sorted low to high)")
ax.set_ylabel(r"$\Delta$KGE")
ax.set_title("(i)", loc="left", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
save_path = repo_root / "figures" / "Figure_2.png"
fig.savefig(save_path, dpi=600, bbox_inches="tight")
plt.show()