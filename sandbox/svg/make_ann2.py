"""Redraw img/ann2.svg: one hidden unit j receiving all 64 inputs.

Same visual language as the new ann.svg (green inputs, blue hidden unit, text as paths),
but with larger and darker labels so it stays readable in a half-width slide column.
Notation follows the lecture: z_j (pre-activation), h_j (activation), b_j (bias).

Run:  python3 make_ann2.py     (needs matplotlib)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

matplotlib.rcParams["svg.fonttype"] = "path"
matplotlib.rcParams["mathtext.fontset"] = "stixsans"

C_IN, C_HID = "#d9ffd9", "#e3e3ff"
C_EDGE, C_TEXT, C_BIAS = "#8a8a8a", "#333333", "#7a7a7a"
FS = 16

fig, ax = plt.subplots(figsize=(6.6, 5.6))
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-0.3, 10.2)
ax.set_ylim(-0.1, 8.7)

# --- inputs -------------------------------------------------------------
X_IN, R_IN = 1.2, 0.5
y_in = [7.0, 5.6, 4.2, 2.2]                    # x1, x2, x3, x64
labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_{64}$"]
inputs = []
for y, lab in zip(y_in, labels):
    c = Circle((X_IN, y), R_IN, facecolor=C_IN, edgecolor="black", lw=1.8, zorder=3)
    ax.add_patch(c)
    ax.text(X_IN, y, lab, ha="center", va="center", fontsize=FS, zorder=4)
    inputs.append(c)
for dy in (-0.28, 0, 0.28):                     # vertical dots between x3 and x64
    ax.add_patch(Circle((X_IN, 3.2 + dy), 0.045, facecolor="black", zorder=3))

ax.text(-0.1, 8.35, r"Input vector $\mathbf{x}$", ha="left", va="center", fontsize=FS + 1)

# bracket + size under the inputs
bx0, bx1, by = X_IN - 0.75, X_IN + 0.75, 1.35
ax.plot([bx0, bx0, bx1, bx1], [by + 0.25, by, by, by + 0.25], color="black", lw=1.6)
ax.text(X_IN, by - 0.2, r"$\mathbf{x}\in\mathbb{R}^{64}$", ha="center", va="top", fontsize=FS)

# --- hidden unit --------------------------------------------------------
UX, UY, UR = 6.1, 4.9, 1.0
unit = Circle((UX, UY), UR, facecolor=C_HID, edgecolor="black", lw=2.2, zorder=3)
ax.add_patch(unit)
ax.text(UX, UY, r"$\Sigma$", ha="center", va="center", fontsize=34, zorder=4)
ax.text(UX, UY + UR + 0.35, r"Hidden unit $j$", ha="center", va="bottom", fontsize=FS + 1)

# --- weights: input i -> unit j -----------------------------------------
w_labels = [r"$w_{j,1}$", r"$w_{j,2}$", r"$w_{j,3}$", r"$w_{j,64}$"]
t = 0.42
for c, y, lab in zip(inputs, y_in, w_labels):
    ax.add_patch(FancyArrowPatch(
        (X_IN, y), (UX, UY), arrowstyle="-|>", mutation_scale=16, lw=1.5,
        color=C_EDGE, patchA=c, patchB=unit, shrinkA=0, shrinkB=0, zorder=1))
    mx = X_IN + t * (UX - X_IN)
    my = y + t * (UY - y)
    off = -0.5 if y < UY - 1 else 0.32          # label below the line for the lowest input
    ax.text(mx, my + off, lab, ha="center", va="center", fontsize=FS - 1, color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.9),
            zorder=2)

# --- bias ---------------------------------------------------------------
ax.add_patch(FancyArrowPatch(
    (UX, 3.05), (UX, UY - UR), arrowstyle="-|>", mutation_scale=14, lw=1.4, ls="--",
    color=C_BIAS, patchB=unit, shrinkA=0, shrinkB=0, zorder=1))
ax.text(UX, 2.85, r"$b_j$", ha="center", va="top", fontsize=FS, color=C_BIAS)

# --- output: sigma and h_j ----------------------------------------------
ax.add_patch(FancyArrowPatch(
    (UX + UR, UY), (9.6, UY), arrowstyle="-|>", mutation_scale=16, lw=1.8,
    color="black", patchA=unit, shrinkA=0, shrinkB=0, zorder=1))
ax.text(UX + UR + 1.15, UY + 0.3, r"$\sigma(\cdot)$", ha="center", va="bottom", fontsize=FS + 4)
ax.text(9.95, UY, r"$h_j$", ha="center", va="center", fontsize=FS + 2)
ax.text(8.9, UY - 0.45, "to next layer", ha="center", va="top", fontsize=FS - 4, color=C_BIAS)

# --- equations ----------------------------------------------------------
ax.text(6.9, 1.55, r"$z_j=\sum_{i=1}^{64} w_{j,i}\,x_i + b_j$", ha="center", va="center", fontsize=FS + 1)
ax.text(6.9, 0.45, r"$h_j=\sigma(z_j)$", ha="center", va="center", fontsize=FS + 1)

fig.savefig("ann2_new.svg", bbox_inches="tight", pad_inches=0.05, transparent=True)
#fig.savefig("ann2_new_preview.png", bbox_inches="tight", pad_inches=0.05, dpi=110, facecolor="white")
print("ok")
