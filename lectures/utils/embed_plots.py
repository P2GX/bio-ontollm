import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


BLUE = "#4472C4"
LIGHT_BLUE = "#AED6F1"
DARK_BLUE = "#5DADE2"
EDGE_COLOR = "black"
RED = "#C0392B"

  

def cars_pca(figsize=(10, 8)):
    np.random.seed(42)
    n_cars = 60

    # Generate synthetic data for dimensions
    # X-axis: Size (1 = Small/Compact, 5 = Large/SUV)
    size = np.random.uniform(1.0, 5.0, n_cars)

    # Y-axis: Color spectrum (1 = Warm tones/Red, 5 = Cool tones/Blue)
    color_spectrum = np.random.uniform(1.0, 5.0, n_cars)

    # Z-axis: Price (in thousands, influenced somewhat by size for realism)
    price = size * 15 + np.random.normal(0, 8, n_cars)
    price = np.clip(price, 10, 100) # Keep within a realistic range

    # Create the 3D figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    # Scatter plot: map the 'color_spectrum' variable to actual marker colors 
    # using a colormap (e.g., 'coolwarm' matches our red-to-blue axis intent)
    scatter = ax.scatter(
        xs=size, 
        ys=color_spectrum, 
        zs=price, 
        c=color_spectrum, 
        cmap='coolwarm', 
        s=70, 
        edgecolors='k', 
        alpha=0.85
    )

    # Set informative labels for the semantic vector space
    ax.set_xlabel(r'Size (Compact $\rightarrow$ Large)', labelpad=10, fontsize=11)
    ax.set_ylabel(r'Color Spectrum (Warm $\rightarrow$ Cool)', labelpad=10, fontsize=11)
    ax.set_zlabel('Price ($k)', labelpad=10, fontsize=11)
    # Add a color bar representing the color dimension
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label('Color Metric', fontsize=10)

    # Adjust viewing angle for an optimal 3D perspective
    ax.view_init(elev=25, azim=135)

    plt.tight_layout()
    plt.show()

def w2v_analogy(figsize=(11, 5)):
    """
    Recreates the classic word2vec analogy diagram
    """

    def draw_arrow(ax, start, end, color):
        ax.annotate(
            "",
            xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2.2,
                mutation_scale=22,
                shrinkA=0, shrinkB=0,
            ),
            zorder=2,
        )

    def draw_label(ax, xy, text, ha="center", va="center", dx=0.0, dy=0.25):
        ax.text(
            xy[0] + dx, xy[1] + dy, text,
            fontsize=13, fontweight="bold",
            ha=ha, va=va, zorder=3,
        )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

    # ---------------------------------------------------------------- LEFT PANEL
    points_left = {
        "MAN":   (0.0, 4.0),
        "WOMAN": (2.0, 7.0),
        "UNCLE": (2.5, 1.0),
        "AUNT":  (5.0, 4.0),
        "KING":  (1.0, -4.0),
        "QUEEN": (3.5, -1.0),
    }

    pairs_left = [("MAN", "WOMAN"), ("UNCLE", "AUNT"), ("KING", "QUEEN")]

    for start_name, end_name in pairs_left:
        draw_arrow(ax_left, points_left[start_name], points_left[end_name], BLUE)

    label_offsets_left = {
        "MAN":   dict(dx=-0.35, dy=0.0, ha="right", va="center"),
        "WOMAN": dict(dx=0.0, dy=0.35, ha="center", va="bottom"),
        "UNCLE": dict(dx=0.0, dy=-0.35, ha="center", va="top"),
        "AUNT":  dict(dx=0.0, dy=0.35, ha="center", va="bottom"),
        "KING":  dict(dx=-0.35, dy=0.0, ha="right", va="center"),
        "QUEEN": dict(dx=0.0, dy=-0.35, ha="center", va="top"),
    }

    for name, xy in points_left.items():
        off = label_offsets_left[name]
        draw_label(ax_left, xy, name, **off)

    # ---------------------------------------------------------------- RIGHT PANEL
    points_right = {
        "KING":   (3.0, 0.0),
        "KINGS":  (1.0, 3.0),
        "QUEEN":  (5.5, 1.5),
        "QUEENS": (3.5, 4.5),
    }

    # Blue = "plural" vectors, Red = "gender" vectors
    draw_arrow(ax_right, points_right["KING"], points_right["KINGS"], BLUE)
    draw_arrow(ax_right, points_right["KING"], points_right["QUEEN"], BLUE)
    draw_arrow(ax_right, points_right["KINGS"], points_right["QUEENS"], RED)
    draw_arrow(ax_right, points_right["QUEEN"], points_right["QUEENS"], RED)

    label_offsets_right = {
        "KING":   dict(dx=0.0, dy=-0.35, ha="center", va="top"),
        "KINGS":  dict(dx=-0.35, dy=0.0, ha="right", va="center"),
        "QUEEN":  dict(dx=0.35, dy=0.0, ha="left", va="center"),
        "QUEENS": dict(dx=0.0, dy=0.35, ha="center", va="bottom"),
    }

    for name, xy in points_right.items():
        off = label_offsets_right[name]
        draw_label(ax_right, xy, name, **off)

    # ---------------------------------------------------------------- STYLING
    for ax, pts in [(ax_left, points_left), (ax_right, points_right)]:
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
        ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

    # vertical divider between panels
    fig.subplots_adjust(wspace=0.05)
    line = plt.Line2D([0.505, 0.505], [0.05, 0.95], transform=fig.transFigure,
                    color="black", linewidth=1)
    fig.add_artist(line)

    plt.tight_layout()


def plot_skipgram(figsize=(7, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)

    # --- Node positions -------------------------------------------------------
    root_pos = (2.5, 0.0)
    root_word = "loves"

    leaf_words = ["the", "man", "his", "dog"]
    leaf_xs = [0.0, 1.4, 3.8, 5.2]
    leaf_y = 2.0
    leaf_positions = [(x, leaf_y) for x in leaf_xs]

    node_radius = 0.32

    # --- Draw arrows from root to each leaf (drawn first, so nodes sit on top)
    for (lx, ly) in leaf_positions:
        dx, dy = lx - root_pos[0], ly - root_pos[1]
        dist = (dx**2 + dy**2) ** 0.5
        ux, uy = dx / dist, dy / dist
        start = (root_pos[0] + ux * node_radius, root_pos[1] + uy * node_radius)
        end = (lx - ux * node_radius, ly - uy * node_radius)

        ax.annotate(
            "",
            xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle="-|>",
                color=EDGE_COLOR,
                lw=1.4,
                mutation_scale=16,
                shrinkA=0, shrinkB=0,
            ),
            zorder=2,
        )
    # mark the center of the root node (e.g. to denote it's the head/center word)
    ax.plot(root_pos[0], leaf_y, marker="o", markersize=6,
        markerfacecolor="black", markeredgecolor="black", zorder=5)
    # --- Draw leaf nodes --------------------------------------------------------
    for word, (lx, ly) in zip(leaf_words, leaf_positions):
        circle = Circle((lx, ly), node_radius, facecolor=LIGHT_BLUE,
                        edgecolor=EDGE_COLOR, linewidth=1.2, zorder=3)
        ax.add_patch(circle)
        ax.text(lx, ly, word, ha="center", va="center", fontsize=11, zorder=4)

    # --- Draw root node ----------------------------------------------------------
    root_circle = Circle(root_pos, node_radius, facecolor=DARK_BLUE,
                        edgecolor=EDGE_COLOR, linewidth=1.2, zorder=3)
    ax.add_patch(root_circle)
    ax.text(root_pos[0], root_pos[1] -node_radius - 0.3, root_word,
            ha="center", va="top", fontsize=12)

    # --- Styling -----------------------------------------------------------------
    ax.set_xlim(-0.8, 6.0)
    ax.set_ylim(-1.0, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
