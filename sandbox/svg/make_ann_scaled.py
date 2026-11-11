"""64 -> 128 -> 64 -> 10 network; column height grows with the layer size.

Height of a column = H_MAX * (n / n_max) ** POWER
  POWER = 1.0 -> strictly proportional (output layer becomes tiny)
  POWER = 0.5 -> proportional to sqrt(n)  (default; readable for 10 vs 128)

Run:  python3 make_ann_scaled.py     (needs matplotlib)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

matplotlib.rcParams["svg.fonttype"] = "path"
matplotlib.rcParams["mathtext.fontset"] = "stixsans"

POWER = 0.5
H_MAX = 5.6          # height of the tallest column (128 units)
R = 0.15             # neuron radius
X_GAP = 3.0
STEP = 0.95          # vertical distance between the displayed top neurons

# (name, math label, fill colour, layer size)
layers = [
    ("Input",    r"$\mathbf{x}\in\mathbb{R}^{64}$",       "#d9ffd9", 64),
    ("Hidden 1", r"$\mathbf{h}_1\in\mathbb{R}^{128}$",    "#e3e3ff", 128),
    ("Hidden 2", r"$\mathbf{h}_2\in\mathbb{R}^{64}$",     "#e3e3ff", 64),
    ("Output",   r"$\hat{\mathbf{y}}\in\mathbb{R}^{10}$", "#ffd9d9", 10),
]
weight_labels = [
    r"$\mathbf{W}_1,\,\mathbf{b}_1$",
    r"$\mathbf{W}_2,\,\mathbf{b}_2$",
    r"$\mathbf{W}_3,\,\mathbf{b}_3$",
]

n_max = max(l[3] for l in layers)


def column(height):
    """Neuron y-positions and ellipsis centre for a column of the given height,
    centred on y = 0: top neurons, an ellipsis, then one neuron at the bottom."""
    top, bottom = height / 2, -height / 2
    n_top = 3 if height >= 3.0 else 2
    ys = [top - i * min(STEP, 0.4 * height) for i in range(n_top)] + [bottom]
    dots_y = (ys[n_top - 1] + bottom) / 2
    return ys, dots_y


cols = []
for name, math, fill, n in layers:
    h = H_MAX * (n / n_max) ** POWER
    ys, dots_y = column(h)
    cols.append((ys, dots_y, h))

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.set_aspect("equal")
ax.axis("off")
xs = [i * X_GAP for i in range(len(layers))]

# edges
for li in range(len(layers) - 1):
    for y1 in cols[li][0]:
        for y2 in cols[li + 1][0]:
            ax.add_patch(FancyArrowPatch(
                (xs[li] + R, y1), (xs[li + 1] - R, y2),
                arrowstyle="-|>", mutation_scale=7, lw=0.8,
                color="#c8c8c8", shrinkA=0, shrinkB=0, zorder=1))

# neurons and ellipses
for li, (name, math, fill, n) in enumerate(layers):
    x = xs[li]
    ys, dots_y, h = cols[li]
    for y in ys:
        ax.add_patch(Circle((x, y), R, facecolor=fill, edgecolor="black", lw=1.6, zorder=3))
    for dy in (-0.15, 0, 0.15):
        ax.add_patch(Circle((x, dots_y + dy), 0.035, facecolor="black", zorder=3))

# labels: same baseline for every column, below the tallest one
y_label = -H_MAX / 2 - 0.45
for li, (name, math, fill, n) in enumerate(layers):
    ax.text(xs[li], y_label, name, ha="center", va="top", fontsize=15)
    ax.text(xs[li], y_label - 0.5, math, ha="center", va="top", fontsize=15)

for li, lab in enumerate(weight_labels):
    ax.text((xs[li] + xs[li + 1]) / 2, H_MAX / 2 + 0.5, lab,
            ha="center", va="bottom", fontsize=15)

ax.set_xlim(-0.6, xs[-1] + 0.6)
ax.set_ylim(y_label - 1.3, H_MAX / 2 + 1.2)

fig.savefig("ann_2hidden_scaled.svg", bbox_inches="tight", pad_inches=0.05, transparent=True)
fig.savefig("ann_2hidden_scaled_preview.png", bbox_inches="tight", pad_inches=0.05, dpi=130, facecolor="white")
print("heights:", [round(c[2], 2) for c in cols])
