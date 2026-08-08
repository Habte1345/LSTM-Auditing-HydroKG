import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrow
import matplotlib as mpl
import numpy as np

# =========================
# Global style
# =========================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']

# mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['figure.dpi'] = 200

# =========================
# Helper functions
# =========================
def add_round_box(ax, x, y, w, h, edgecolor='black', linewidth=1.6, radius=0.02, facecolor='white'):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor
    )
    ax.add_patch(box)
    return box

def add_box(ax, x, y, w, h, edgecolor='black', linewidth=1.0, facecolor='white'):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.003,rounding_size=0.007",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor
    )
    ax.add_patch(rect)
    return rect

def interp_color(c1, c2, t):
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return tuple((1 - t) * c1 + t * c2)

# =========================
# Figure canvas
# =========================
fig, ax = plt.subplots(figsize=(18, 8), dpi=200)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Colors
# Colors: sequential crimson reds and blues
blue_border = '#08306b'
red_border  = "#dc8e1a"

blue_light = '#deebf7'
blue_mid   = '#6baed6'
blue_dark  = '#08519c'

red_light = '#fee0d2'
red_mid   = '#fb6a4a'
red_dark  = "#a5530f"
# =========================
# Top panel: LSTM model
# =========================
top_x, top_y, top_w, top_h = 0.08, 0.72, 0.82, 0.23
add_round_box(ax, top_x, top_y, top_w, top_h, edgecolor=blue_border, linewidth=1.8, radius=0.02)

ax.text(0.49, 0.92, "Training LSTM model is progressing ...",
        ha='center', va='center', fontsize=24, fontweight='bold', color='black')

ax.text(0.16, 0.865, "Simulation running", ha='center', va='center', fontsize=22, color='black')

# Progress bar
bar_x, bar_y, bar_w, bar_h = 0.30, 0.848, 0.54, 0.022
add_round_box(ax, bar_x, bar_y, bar_w, bar_h, edgecolor='gray', linewidth=1.2, radius=0.007, facecolor='white')
progress = 0.65
ax.add_patch(FancyBboxPatch(
    (bar_x, bar_y), bar_w * progress, bar_h,
    boxstyle="round,pad=0.001,rounding_size=0.007",
    linewidth=0, facecolor="#0634ed"
))

# Time boxes
box_y = 0.785
label_y = 0.735
box_w = 0.13
box_h = 0.045
time_xs = [0.12, 0.29, 0.46, 0.59, 0.73]
time_labels = [r"$t_1$", r"$t_2$", r"$t_3$", r"$\cdots$", r"$t_n$"]
q_labels = [r"$Q_{\mathrm{sim}}(t_1)$", r"$Q_{\mathrm{sim}}(t_2)$", r"$Q_{\mathrm{sim}}(t_3)$", r"$\cdots$", r"$Q_{\mathrm{sim}}(t_n)$"]

for i, x in enumerate(time_xs):
    if i != 3:
        add_box(ax, x, box_y, box_w, box_h, edgecolor='black', linewidth=1.0, facecolor='white')
        ax.text(x + box_w/2, box_y + box_h/2, time_labels[i], ha='center', va='center', fontsize=24, color='black')
        ax.text(x + box_w/2, label_y, q_labels[i], ha='center', va='center', fontsize=18, color='black')
    else:
        ax.text(x + 0.06, box_y + box_h/2, time_labels[i], ha='center', va='center', fontsize=28, color='black')
        ax.text(x + 0.06, label_y, q_labels[i], ha='center', va='center', fontsize=20, color='black')

# Down arrow
ax.add_patch(FancyArrow(
    0.49, 0.705, 0, -0.06,
    width=0.0035, head_width=0.018, head_length=0.02,
    color=blue_border, length_includes_head=True
))

# =========================
# Bottom panel: HydroKG auditing layer
# =========================
bot_x, bot_y, bot_w, bot_h = 0.08, 0.22, 0.82, 0.40
add_round_box(ax, bot_x, bot_y, bot_w, bot_h, edgecolor='green', linewidth=1.8, radius=0.02)

ax.text(0.49, 0.58, "HydroKG Real-time auditing layer",
        ha='center', va='center', fontsize=24, fontweight='bold', color='black')
ax.text(0.49, 0.538, "(Rules evaluate new predictions when applicable)",
        ha='center', va='center', fontsize=16, color='blue')

# Layout for heatmap
label_left = 0.10
divider_x = 0.25
grid_left = 0.255
grid_right = 0.90
grid_top = 0.49
grid_bottom = 0.235

# Divider line
ax.plot([divider_x, divider_x], [grid_bottom, grid_top], color='black', lw=1.4)

# Row labels
rules = [
    "R0: Negative flow",
    "R1: Extreme ratio",
    "R2: Zero-flow collapse",
    "R3: High relative error",
    "R4: Peak timing error",
    "R5: Mass balance",
    "R6 Budyko deviation"
]

n_rows = len(rules)
n_cols = 6
row_h = (grid_top - grid_bottom) / n_rows
col_w = (grid_right - grid_left) / n_cols

# Column labels
col_labels = [r"$t_1$", r"$t_2$", r"$t_3$", r"$t_4$", r"$\cdots$", r"$t_n$"]
for j, lab in enumerate(col_labels):
    x = grid_left + j * col_w + col_w/2
    ax.text(x, grid_top + 0.018, lab, ha='center', va='center', fontsize=20, color='black')

for i, rule in enumerate(rules):
    y = grid_top - (i + 0.5) * row_h
    ax.text(label_left, y, rule, ha='left', va='center', fontsize=16, color='black')
# Heatmap values
# 0 = waiting / information (blue)
# 1 = active auditing (crimson red)
# Values progress from light to dark as time continues

from matplotlib import cm

# =========================
# Jet colormap
# =========================
jet = cm.get_cmap("jet")  # Use 'coolwarm' for a more distinct blue-red gradient

# Heatmap values
# 0 = waiting / information
# 1 = active auditing
heatmap_mode = np.array([
    [1, 1, 1, 1, 1, 1],  # R0
    [1, 1, 1, 1, 1, 1],  # R1
    [1, 1, 1, 1, 1, 1],  # R2
    [1, 1, 1, 1, 1, 1],  # R3
    [0, 0, 0, 1, 1, 1],  # R4
    [0, 0, 0, 1, 1, 1],  # R5
    [0, 0, 0, 0, 0, 1],  # R6
], dtype=int)

# Shade intensity from left to right
active_t = [0.10, 0.28, 0.45, 0.62, 0.78, 0.95]
wait_t   = [0.15, 0.35, 0.55, 0.70, 0.82, 0.92]

for i in range(n_rows):
    for j in range(n_cols):
        x = grid_left + j * col_w
        y = grid_top - (i + 1) * row_h

        if heatmap_mode[i, j] == 1:
            # warm side of jet: green → yellow → red
            color = jet(0.45 + 0.50 * active_t[j])
        else:
            # cool side of jet: blue → cyan
            color = jet(0.05 + 0.30 * wait_t[j])

        ax.add_patch(Rectangle(
            (x + 0.002, y + 0.004),
            col_w - 0.004, row_h - 0.008,
            facecolor=color,
            edgecolor='white',
            linewidth=1.3
        ))

# =========================
# Legend
# =========================
leg_x, leg_y, leg_w, leg_h = 0.24, 0.08, 0.50, 0.09
add_round_box(ax, leg_x, leg_y, leg_w, leg_h, edgecolor='gray', linewidth=1.2, radius=0.01)

bx0 = leg_x + 0.015
by0 = leg_y + 0.05
sw = 0.055
sh = 0.035

# waiting / information: cool jet colors
for k in range(4):
    ax.add_patch(Rectangle(
        (bx0 + k * sw, by0),
        sw * 0.98, sh,
        facecolor=jet(0.05 + 0.30 * (k / 3)),
        edgecolor='white',
        linewidth=1
    ))

# active auditing: warm jet colors
rx0 = leg_x + 0.29
for k in range(4):
    ax.add_patch(Rectangle(
        (rx0 + k * sw, by0),
        sw * 0.98, sh,
        facecolor=jet(0.45 + 0.50 * (k / 3)),
        edgecolor='white',
        linewidth=1
    ))

ax.text(bx0 + 0.11, leg_y + 0.022, "waiting / information",
        ha='center', va='center', fontsize=18, color='black')
ax.text(rx0 + 0.13, leg_y + 0.022, "active auditing",
        ha='center', va='center', fontsize=18, color='black')

# =========================
# Save and show
# =========================
# plt.savefig("HydroKG_online_auditing_framework.png", dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()