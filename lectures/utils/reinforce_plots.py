
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path


def pipeline():
    """
    "Multi-Stage Training of a Language Model" pipeline
    """

    

    # ---------------------------------------------------------------------------
    # Palette
    # ---------------------------------------------------------------------------
    BG = "#faf3e3"
    DARK = "#1f3d3d"
    ORANGE = "#f0a03c"
    ORANGE_DARK = "#d6862a"
    TEAL_LIGHT = "#bcd9cf"
    TEAL_MED = "#8fc0b0"
    TEAL_PAPER = "#dce9df"
    TEXT_DARK = "#2b3a3a"

    FIG_W, FIG_H = 13, 7.2
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # ---------------------------------------------------------------------------
    # Title
    # ---------------------------------------------------------------------------
    ax.text(
        6.5, 6.7, "Multi-Stage Training of a Language Model",
        ha="center", va="center", fontsize=24, fontweight="bold", color=TEXT_DARK,
    )

    # ---------------------------------------------------------------------------
    # Layout: 5 columns of icons with arrows between them, labels below,
    # and a second row of "sub-labels" under an arrow for the first four.
    # ---------------------------------------------------------------------------
    centers_x = [1.2, 3.7, 6.2, 8.7, 11.2]
    icon_cy = 4.9
    label_y = 3.55
    sub_arrow_top = 3.15
    sub_arrow_bot = 2.55
    sub_label_y = 2.05

    titles = ["Pre-training", "Fine-tuning", "Instruction\ntuning", "RLHF", "RLHF"]
    sub_labels = ["Unlabeled\ndata", "Task-specific\ndataset", "Instructions\ndataset", "Human\nfeedback", None]


    def arrow_between(x1, x2, y):
        ax.annotate(
            "", xy=(x2 - 0.55, y), xytext=(x1 + 0.55, y),
            arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.2, mutation_scale=22),
        )


    # --- Icon 1: stack of books -------------------------------------------------
    def draw_books(cx, cy):
        w, h = 1.15, 0.32
        for i, dy in enumerate([-0.42, 0.0, 0.42]):
            rect = patches.FancyBboxPatch(
                (cx - w / 2, cy + dy - h / 2), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=2.2, edgecolor=DARK, facecolor=ORANGE, zorder=3,
            )
            ax.add_patch(rect)
            # spine highlight line
            ax.plot([cx - w / 2 + 0.12, cx + w / 2 - 0.12],
                    [cy + dy + h / 2 - 0.08] * 2, color=ORANGE_DARK, lw=1.4, zorder=4)


    # --- Icon 2: database cylinder ---------------------------------------------
    def draw_database(cx, cy):
        w, h = 1.3, 1.5
        ell_h = 0.28
        body = patches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h - ell_h,
            boxstyle="square,pad=0", linewidth=2.2,
            edgecolor=DARK, facecolor=TEAL_MED, zorder=2,
        )
        ax.add_patch(body)
        top = patches.Ellipse((cx, cy + h / 2 - ell_h), w, ell_h * 2,
                            linewidth=2.2, edgecolor=DARK, facecolor=TEAL_MED, zorder=3)
        bottom = patches.Ellipse((cx, cy - h / 2), w, ell_h * 2,
                                linewidth=2.2, edgecolor=DARK, facecolor=TEAL_MED, zorder=3)
        ax.add_patch(bottom)
        ax.add_patch(top)
        # small grid of squares to suggest "data"
        sq = 0.16
        gap = 0.09
        start_x = cx - (sq * 3 + gap * 2) / 2
        start_y = cy - 0.05
        for row in range(2):
            for col in range(3):
                x0 = start_x + col * (sq + gap)
                y0 = start_y - row * (sq + gap)
                ax.add_patch(patches.Rectangle(
                    (x0, y0), sq, sq, linewidth=1.6,
                    edgecolor=DARK, facecolor=BG, zorder=4,
                ))


    # --- Icon 3: neural network --------------------------------------------------
    def draw_network(cx, cy):
        w, h = 1.2, 1.6
        frame = patches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=2.2, edgecolor=DARK, facecolor=TEAL_LIGHT, zorder=2,
        )
        ax.add_patch(frame)
        r = 0.09
        top_y = cy + h / 2 - 0.32
        bot_y = cy - h / 2 + 0.32
        top_xs = [cx - 0.3, cx, cx + 0.3]
        bot_xs = [cx - 0.3, cx, cx + 0.3]
        for tx in top_xs:
            for bx in bot_xs:
                ax.plot([tx, bx], [top_y, bot_y], color=DARK, lw=1.3, zorder=3)
        for tx in top_xs:
            ax.add_patch(patches.Circle((tx, top_y), r, facecolor=TEAL_LIGHT,
                                        edgecolor=DARK, lw=2, zorder=4))
        for bx in bot_xs:
            ax.add_patch(patches.Circle((bx, bot_y), r, facecolor=TEAL_LIGHT,
                                        edgecolor=DARK, lw=2, zorder=4))


    # --- Icon 4: document with cursor -------------------------------------------
    def draw_document(cx, cy):
        w, h = 1.05, 1.5
        doc = patches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=2.2, edgecolor=DARK, facecolor=TEAL_PAPER, zorder=2,
        )
        ax.add_patch(doc)
        for i, dy in enumerate([0.42, 0.16, -0.1, -0.36]):
            line_w = w * (0.72 if i % 2 == 0 else 0.55)
            ax.plot([cx - line_w / 2, cx + line_w / 2], [cy + dy] * 2,
                    color=DARK, lw=2.2, zorder=3, solid_capstyle="round")
        # cursor arrow (simple triangle) bottom-right
        cur_x, cur_y = cx + 0.28, cy - 0.55
        verts = [(cur_x, cur_y), (cur_x + 0.4, cur_y - 0.32),
                (cur_x + 0.22, cur_y - 0.34), (cur_x + 0.32, cur_y - 0.6),
                (cur_x + 0.18, cur_y - 0.53), (cur_x + 0.1, cur_y - 0.28),
                (cur_x, cur_y)]
        poly = patches.Polygon(verts, closed=True, facecolor="white",
                                edgecolor=DARK, lw=2, zorder=5)
        ax.add_patch(poly)


    # --- Icon 5: chat bubble with heart + person ---------------------------------
    def draw_feedback(cx, cy):
        # speech bubble
        bw, bh = 1.15, 0.75
        by = cy + 0.55
        bubble = patches.FancyBboxPatch(
            (cx - bw / 2, by - bh / 2), bw, bh,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=2.2, edgecolor=DARK, facecolor=TEAL_MED, zorder=2,
        )
        ax.add_patch(bubble)
        tail = patches.Polygon(
            [(cx - 0.12, by - bh / 2 + 0.02), (cx - 0.32, by - bh / 2 - 0.28),
            (cx + 0.1, by - bh / 2 + 0.05)],
            closed=True, facecolor=TEAL_MED, edgecolor=DARK, lw=2.2, zorder=1,
        )
        ax.add_patch(tail)
        # heart inside bubble (parametric heart curve)
        import math
        hx, hy = cx, by
        steps = 60
        ts = [i / steps * 2 * math.pi for i in range(steps + 1)]
        hx_list = [0.02 * (16 * math.sin(t_) ** 3) for t_ in ts]
        hy_list = [0.02 * (13 * math.cos(t_) - 5 * math.cos(2 * t_) - 2 * math.cos(3 * t_) - math.cos(4 * t_)) for t_ in ts]
        heart_x = [hx + v for v in hx_list]
        heart_y = [hy + v for v in hy_list]
        ax.fill(heart_x, heart_y, color=ORANGE, edgecolor=ORANGE_DARK, lw=1.2, zorder=3)

        # person icon below
        py = cy - 0.45
        ax.add_patch(patches.Circle((cx, py + 0.28), 0.18, facecolor=TEAL_MED,
                                    edgecolor=DARK, lw=2.2, zorder=3))
        body = patches.FancyBboxPatch(
            (cx - 0.32, py - 0.35), 0.64, 0.5,
            boxstyle="round,pad=0.02,rounding_size=0.28",
            linewidth=2.2, edgecolor=DARK, facecolor=TEAL_MED, zorder=3,
        )
        ax.add_patch(body)


    icon_fns = [draw_books, draw_database, draw_network, draw_document, draw_feedback]

    # ---------------------------------------------------------------------------
    # Draw icons, arrows, and titles
    # ---------------------------------------------------------------------------
    for cx, fn in zip(centers_x, icon_fns):
        fn(cx, icon_cy)

    for i in range(len(centers_x) - 1):
        arrow_between(centers_x[i], centers_x[i + 1], icon_cy)

    for cx, title in zip(centers_x, titles):
        ax.text(cx, label_y, title, ha="center", va="top",
                fontsize=15.5, fontweight="bold", color=TEXT_DARK)

    # sub-arrows + sub-labels (skip the last column, matching the reference image)
    for cx, sub in zip(centers_x, sub_labels):
        if sub is None:
            continue
        ax.annotate(
            "", xy=(cx, sub_arrow_bot), xytext=(cx, sub_arrow_top),
            arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2, mutation_scale=16),
        )
        ax.text(cx, sub_label_y, sub, ha="center", va="top",
                fontsize=13.5, color=TEXT_DARK)

    plt.tight_layout()
