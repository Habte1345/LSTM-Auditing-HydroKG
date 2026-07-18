import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogFormatterMathtext
from matplotlib import cm
import matplotlib as mpl
import numpy as np

# =========================
# Global style
# =========================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']

# mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['figure.dpi'] = 200

# ============================================================
# 1. LOAD YOUR BASIN DATA
# ============================================================
np.random.seed(42)

total_basins = 318   # <<<<<< SINGLE DENOMINATOR USED EVERYWHERE

df = pd.DataFrame({
    "basin_id": [f"basin_{i+1:04d}" for i in range(total_basins)],
    "skill": np.clip(np.random.beta(3.0, 1.8, total_basins), 0, 1),
    "burden": 10 ** np.random.uniform(-3.0, 0.0, total_basins),
    "R0": np.random.poisson(0.8, total_basins),
    "R1": np.random.poisson(1.6, total_basins),
    "R2": np.random.poisson(0.7, total_basins),
    "R3": np.random.poisson(1.5, total_basins),
    "R4": np.random.poisson(1.0, total_basins),
    "R5": np.random.poisson(0.9, total_basins),
    "R6": np.random.poisson(0.8, total_basins),
})

# ============================================================
# 2. FIXED TEMPLATE BUBBLE LAYOUT
# ============================================================
# ============================================================
# MORE DENSE TEMPLATE BUBBLE LAYOUT
# ============================================================


x_template = np.array([
    0.10, 0.18, 0.25, 0.32, 0.40, 0.50, 0.60, 0.72, 0.85, 0.95,
    0.12, 0.22, 0.35, 0.48, 0.62, 0.78, 0.90,
    0.15, 0.28, 0.45, 0.58, 0.70, 0.82, 0.92,
    0.20, 0.38, 0.55, 0.68, 0.80, 0.90
])

y_template = np.array([
    0.90, 0.45, 0.28, 0.18, 0.12, 0.30, 0.18, 0.07, 0.03, 0.16,
    0.08, 0.035, 0.018, 0.012, 0.010, 0.015, 0.002,
    0.25, 0.13, 0.07, 0.035, 0.020, 0.012, 0.006,
    0.015, 0.008, 0.004, 0.0025, 0.0018, 0.0012
])

classes_template = np.array([
    "physical", "physical", "magnitude", "magnitude", "budget",
    "magnitude", "magnitude", "timing", "timing", "physical",
    "budget", "magnitude", "budget", "timing", "timing", "budget", "budget",
    "physical", "magnitude", "magnitude", "timing", "timing", "budget", "budget",
    "magnitude", "budget", "timing", "budget", "timing", "budget"
])

template = pd.DataFrame({
    "bubble_id": np.arange(1, len(x_template) + 1),
    "x": x_template,
    "y": y_template,
    "dominant_class": classes_template
})

# ============================================================
# 3. COLORS
# ============================================================
jet = cm.get_cmap("jet")

color_map = {
    "physical":  jet(0.05),
    "magnitude": jet(0.32),
    "timing":    jet(0.68),
    "budget":    jet(0.95),
}

class_order = ["physical", "magnitude", "timing", "budget"]

class_labels_long = {
    "physical": "R0/R2",
    "magnitude": "R1/R3",
    "timing": "R4",
    "budget": "R5/R6"
}

# ============================================================
# 4. PREPARE BASIN CLASSIFICATION
# ============================================================
rule_cols = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]

df = df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["skill", "burden"] + rule_cols
).copy()

df = df[(df["skill"] >= 0) & (df["skill"] <= 1)]
df = df[df["burden"] > 0].copy()

df["physical"]  = df["R0"] + df["R2"]
df["magnitude"] = df["R1"] + df["R3"]
df["timing"]    = df["R4"]
df["budget"]    = df["R5"] + df["R6"]

merged_cols = ["physical", "magnitude", "timing", "budget"]
df["total_violations"] = df[rule_cols].sum(axis=1)

# Keep this only if you intentionally want to exclude zero-violation basins
df = df[df["total_violations"] > 0].copy()

df["dominant_class"] = df[merged_cols].idxmax(axis=1)

# ============================================================
# 5. ASSIGN EACH BASIN TO NEAREST TEMPLATE BUBBLE
# ============================================================
def assign_to_template_bubble(row, template_df):
    sub = template_df[template_df["dominant_class"] == row["dominant_class"]].copy()

    dx = (sub["x"].values - row["skill"]) / 0.10
    dy = (np.log10(sub["y"].values) - np.log10(row["burden"])) / 0.30

    dist = dx**2 + dy**2
    return sub.iloc[np.argmin(dist)]["bubble_id"]

df["bubble_id"] = df.apply(assign_to_template_bubble, axis=1, template_df=template)

# ============================================================
# 6. SUMMARIZE BASINS PER BUBBLE
# ============================================================
bubble_counts = df.groupby("bubble_id").size().rename("n_basins")

bubble_stats = template.merge(bubble_counts, on="bubble_id", how="left")
bubble_stats["n_basins"] = bubble_stats["n_basins"].fillna(0).astype(int)
bubble_stats["pct_basins"] = 100.0 * bubble_stats["n_basins"] / total_basins

min_size = 70
max_size = 850

if bubble_stats["n_basins"].max() > 0:
    bubble_stats["plot_size"] = min_size + (
        np.sqrt(bubble_stats["n_basins"] / bubble_stats["n_basins"].max())
    ) * (max_size - min_size)
else:
    bubble_stats["plot_size"] = min_size

# ============================================================
# 7. SUMMARY STATISTICS — SAME DENOMINATOR
# ============================================================
class_counts = df["dominant_class"].value_counts().reindex(class_order, fill_value=0)
class_pct = 100.0 * class_counts / total_basins

# Individual rule dominance for merged-rule shares
df["dominant_rule"] = df[rule_cols].idxmax(axis=1)
rule_counts = df["dominant_rule"].value_counts().reindex(rule_cols, fill_value=0)
rule_pct = 100.0 * rule_counts / total_basins

# ============================================================
# 8. FIGURE STYLE
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.titlesize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

fig = plt.figure(figsize=(10, 12), dpi=200)

# ============================================================
# 9. PANEL A: SKILL–TRUST SPACE
# ============================================================
ax = fig.add_axes([0.08, 0.39, 0.48, 0.48])

plot_df = bubble_stats[bubble_stats["n_basins"] > 0].copy()

ax.scatter(
    plot_df["x"],
    plot_df["y"],
    s=plot_df["plot_size"],
    c=[color_map[c] for c in plot_df["dominant_class"]],
    edgecolor="white",
    linewidth=1.2,
    zorder=3
)

ax.set_yscale("log")
ax.set_xlim(0.0, 1.02)
ax.set_ylim(8e-4, 1.5)

ax.set_xlabel("Predictive skill, KGE", fontsize=22)
ax.set_ylabel("Physical violation burden\n(normalized violation frequency)", fontsize=22)
ax.set_title("a) Predictive skills across rules", fontsize=22)

ax.set_xticks(np.linspace(0, 1, 6))
ax.set_yticks([1e0, 1e-1, 1e-2, 1e-3])
ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.tick_params(labelsize=15)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.6)
ax.spines["bottom"].set_linewidth(1.6)

for yy in [1e-1, 1e-2]:
    ax.axhline(yy, color="0.7", lw=1.2, ls=(0, (4, 3)), zorder=1)

ax.text(0.05, 1.2e-1, "High violation",
        color="red", fontsize=14, style="italic", va="center")

ax.text(0.05, 1.18e-2, "Medium violation",
        color="darkorange", fontsize=14, style="italic", va="center")

ax.text(0.05, 2.5e-3, "Low violation",
        color="orange", fontsize=14, style="italic", va="center")

# ============================================================
# 10. RIGHT PANEL AREA
# ============================================================
leg = fig.add_axes([0.6, 0.29, 0.5, 0.65])
leg.axis("off")
leg.set_xlim(0, 1)
leg.set_ylim(0, 1)

# ============================================================
# 11. PANEL B: DOMINANT RULE-CLASS SUMMARY
# ============================================================
ax_hist = fig.add_axes([0.64, 0.65, 0.4, 0.17])

ax_hist.bar(
    np.arange(len(class_order)),
    class_pct.values,
    color=[color_map[c] for c in class_order],
    edgecolor="0.35",
    linewidth=0.8
)

for i, (cnt, pct) in enumerate(zip(class_counts.values, class_pct.values)):
    ax_hist.text(
        i,
        pct + max(class_pct.values) * 0.03 + 0.2,
        f"{cnt}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=11
    )

ax_hist.set_xticks(np.arange(len(class_order)))
ax_hist.set_xticklabels([class_labels_long[c] for c in class_order], fontsize=12)
ax_hist.set_ylabel("% basins", fontsize=13)
ax_hist.set_title("b) Dominant rule-class summary\n\n", fontsize=22, pad=4)
ax_hist.tick_params(axis="y", labelsize=12)

ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)

# ============================================================
# 12. MERGED-RULE SHARES — SAME DENOMINATOR
# ============================================================
share_text = (
    "Merged-rule shares ("
    f"R0 / R2 = {rule_pct['R0']:.2f}% / {rule_pct['R2']:.2f}%; "
    f"R1 / R3 = {rule_pct['R1']:.2f}% / {rule_pct['R3']:.2f}%; "
    f"R5 / R6 = {rule_pct['R5']:.2f}% / {rule_pct['R6']:.2f}%)"
)

fig.text(
    0.08,
    0.31,
    share_text,
    ha="left",
    va="top",
    fontsize=18,
    color="black",
    family="Times New Roman"
)

# ============================================================
# 13. CUSTOM LEGEND AREA
# ============================================================
leg.text(0.04, 0.45, "Dominant rules", fontsize=18, va="center")

entries = [
    ("R0/R2 physical impossibility", color_map["physical"]),
    ("R1/R3 magnitude failure", color_map["magnitude"]),
    ("R4 timing failure", color_map["timing"]),
    ("R5/R6 budget-scale failure", color_map["budget"]),
]

ys = [0.40, 0.34, 0.28, 0.22]

for (label, col), yy in zip(entries, ys):
    leg.scatter([0.07], [yy], s=190, color=col, edgecolor="0.3")
    leg.text(0.12, yy, label, fontsize=14, va="center")

# ============================================================
# 14. BASIN SIZE LEGEND — SAME DENOMINATOR
# ============================================================
leg.text(0.68, 0.45, "Basins (size)", fontsize=18, va="center")

# Choose fixed reference counts
example_counts = np.array([10, 50, 100])

def size_from_count(c):
    if bubble_stats["n_basins"].max() == 0:
        return min_size

    return min_size + (
        np.sqrt(c / bubble_stats["n_basins"].max())
    ) * (max_size - min_size)

example_sizes = [size_from_count(c) for c in example_counts]
example_labels = [f"{c} ({100*c/total_basins:.1f}%)" for c in example_counts]

yy_leg = [0.40, 0.34, 0.28]

for lab, ss, yy in zip(example_labels, example_sizes, yy_leg):
    leg.scatter([0.74], [yy], s=ss, color="gray", edgecolor="0.4")
    leg.text(0.81, yy, lab, fontsize=14, va="center")

plt.tight_layout()

plt.savefig("HydroKG_skill_trust_rule_summary.png", dpi=600, bbox_inches="tight")
plt.show()

# ============================================================
# CHECK CONSISTENCY
# ============================================================
print("Original total basins:", total_basins)
print("Valid basins after filtering:", len(df))
print("Dominant class counts:")
print(class_counts)
# print("Dominant class percentages using total_basins:")
print(class_pct)