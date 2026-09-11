
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

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


def training_stages(include=None, fig_width=9.5, row_height=2.0, row_gap=0.28,
                     margin_top=0.3, margin_bottom=0.25):

    # ---------------------------------------------------------------- STYLE ---
    NAVY = "#0B2545"
    BLUE = "#1F6FEB"
    LIGHT_BLUE = "#EAF2FF"
    DARK = "#1A1A1A"
    GREY = "#6B7280"
    WHITE = "#FFFFFF"

    PANEL_BORDER = BLUE
    PANEL_FACE = WHITE
    BADGE_FACE = BLUE

    FONT_TITLE = "DejaVu Sans"
    FIG_WIDTH = 9.5
    ROW_HEIGHT = 2.0          # height of each stage panel, in inches
    ROW_GAP = 0.28            # gap between panels
    MARGIN_TOP = 0.9          # space for the headline
    MARGIN_BOTTOM = 0.25


    # ------------------------------------------------------------- HELPERS ---
    def rounded_panel(ax, x, y, w, h, face=PANEL_FACE, edge=PANEL_BORDER, lw=2.0, radius=0.06):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw, edgecolor=edge, facecolor=face, zorder=2,
            clip_on=False
        )
        ax.add_patch(box)
        return box


    def stage_badge(ax, x, y, w, h, text):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.05",
            linewidth=0, facecolor=BADGE_FACE, zorder=4,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=WHITE, fontsize=13, fontweight="bold", zorder=5)


    def person_icon(ax, cx, cy, scale=1.0, color=DARK, label=None, label_dy=-0.34):
        head_r = 0.075 * scale
        ax.add_patch(Circle((cx, cy + 0.10 * scale), head_r, facecolor=color, edgecolor="none", zorder=4))
        body = FancyBboxPatch(
            (cx - 0.11 * scale, cy - 0.14 * scale), 0.22 * scale, 0.20 * scale,
            boxstyle="round,pad=0,rounding_size=0.06", facecolor=color, edgecolor="none", zorder=4,
        )
        ax.add_patch(body)
        if label:
            ax.text(cx, cy + label_dy * scale, label, ha="center", va="top",
                    fontsize=8, color=GREY, zorder=4)


    def llm_box(ax, cx, cy, w=0.95, h=0.62, label="LLM", sublabel=None):
        box = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0,rounding_size=0.08",
            linewidth=1.6, edgecolor=BLUE, facecolor=LIGHT_BLUE, zorder=4,
        )
        ax.add_patch(box)
        ax.text(cx, cy + 0.03, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=NAVY, zorder=5)
        if sublabel:
            ax.text(cx, cy - h / 2 - 0.10, sublabel, ha="center", va="top",
                    fontsize=7.5, color=GREY, zorder=4)


    def data_stack_icon(ax, cx, cy, label=None):
        # three small "folder/card" rectangles stacked
        offsets = [0.10, 0.0, -0.10]
        for i, dy in enumerate(offsets):
            r = FancyBboxPatch(
                (cx - 0.16, cy + dy - 0.045), 0.32, 0.09,
                boxstyle="round,pad=0,rounding_size=0.02",
                linewidth=1.2, edgecolor=BLUE, facecolor=WHITE, zorder=4 + i,
            )
            ax.add_patch(r)
        if label:
            ax.text(cx, cy - 0.28, label, ha="center", va="top", fontsize=7.5, color=GREY, zorder=6)


    def chat_bubble_icon(ax, cx, cy, label=None):
        box = FancyBboxPatch(
            (cx - 0.18, cy - 0.10), 0.36, 0.22,
            boxstyle="round,pad=0,rounding_size=0.05",
            linewidth=1.4, edgecolor=DARK, facecolor=WHITE, zorder=4,
        )
        ax.add_patch(box)
        tail = [(cx - 0.06, cy - 0.10), (cx - 0.12, cy - 0.19), (cx + 0.01, cy - 0.10)]
        ax.add_patch(plt.Polygon(tail, closed=True, facecolor=WHITE, edgecolor=DARK, linewidth=1.4, zorder=4))
        if label:
            ax.text(cx, cy - 0.30, label, ha="center", va="top", fontsize=7.5, color=GREY, zorder=6)


    def doc_check_icon(ax, cx, cy, ok=True, label=None):
        box = FancyBboxPatch(
            (cx - 0.14, cy - 0.19), 0.28, 0.38,
            boxstyle="round,pad=0,rounding_size=0.03",
            linewidth=1.3, edgecolor=DARK, facecolor=WHITE, zorder=4,
        )
        ax.add_patch(box)
        for dy in (0.08, 0.0, -0.08):
            ax.plot([cx - 0.08, cx + 0.08], [cy + dy, cy + dy], color=GREY, lw=1.0, zorder=5)
        badge_color = "#16A34A" if ok else "#DC2626"
        ax.add_patch(Circle((cx + 0.12, cy + 0.16), 0.055, facecolor=badge_color, edgecolor=WHITE, linewidth=1.2, zorder=6))
        mark = "\u2713" if ok else "\u2715"
        ax.text(cx + 0.12, cy + 0.16, mark, ha="center", va="center", fontsize=6.5, color=WHITE, zorder=7)
        if label:
            ax.text(cx, cy - 0.30, label, ha="center", va="top", fontsize=7.5, color=GREY, zorder=6)


    def dashed_arrow(ax, p0, p1, label=None, label_dy=0.10, curve=0.0):
        arrow = FancyArrowPatch(
            p0, p1,
            connectionstyle=f"arc3,rad={curve}",
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.3, linestyle=(0, (3, 2)), color=DARK, zorder=3,
        )
        ax.add_patch(arrow)
        if label:
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + label_dy
            ax.text(mx, my, label, ha="center", va="bottom", fontsize=7.5, color=DARK, zorder=6)


    # --------------------------------------------------------------- FLOWS ---
    # Each flow function draws the inside of one stage panel.
    # (ax, x0, y0, w, h) describe the panel's plot-local rectangle;
    # text_x is where the stage title/description column ends and the
    # flow diagram begins.

    def flow_stage0(ax, x0, y0, w, h, fx):
        cy = y0 + h / 2
        person_icon(ax, fx + 0.35, cy + 0.05, label="User")
        ax.text(fx + 0.70, cy + 0.30, "Query", ha="center", fontsize=8, color=DARK)
        dashed_arrow(ax, (fx + 0.48, cy + 0.10), (fx + 1.05, cy + 0.10))
        llm_box(ax, fx + 1.75, cy, w=1.0, sublabel="Untrained model\n(random weights)")
        ax.text(fx + 2.55, cy + 0.30, "Output", ha="center", fontsize=8, color=DARK)
        dashed_arrow(ax, (fx + 2.25, cy + 0.10), (fx + 2.85, cy + 0.10))
        out_box = rounded_panel(ax, fx + 2.85, cy - 0.32, 1.15, 0.64,
                                face=WHITE, edge=DARK, lw=1.2, radius=0.05)
        ax.text(fx + 2.85 + 0.575, cy, "incoherent /\nrandom text", ha="center", va="center",
                fontsize=7.5, color=GREY)


    def flow_stage1(ax, x0, y0, w, h, fx):
        cy = y0 + h / 2 + 0.05
        data_stack_icon(ax, fx + 0.30, cy, label="Raw text\ncorpus")
        dashed_arrow(ax, (fx + 0.50, cy), (fx + 0.95, cy), label="train")
        llm_box(ax, fx + 1.60, cy, sublabel="Untrained model")
        dashed_arrow(ax, (fx + 2.10, cy), (fx + 2.55, cy))
        llm_box(ax, fx + 3.20, cy, sublabel="Pre-trained model")
        ax.text(fx + 1.55, y0 + 0.16,
                "Learns statistical patterns of language by predicting\nthe next token over massive text corpora.",
                ha="center", fontsize=7.3, color=GREY, style="italic")


    def flow_stage2(ax, x0, y0, w, h, fx):
        cy = y0 + h / 2 + 0.05
        chat_bubble_icon(ax, fx + 0.30, cy, label="Instruction /\nresponse pairs")
        dashed_arrow(ax, (fx + 0.52, cy), (fx + 0.95, cy))
        llm_box(ax, fx + 1.60, cy, sublabel="Pre-trained model")
        dashed_arrow(ax, (fx + 2.10, cy), (fx + 2.55, cy))
        llm_box(ax, fx + 3.20, cy, sublabel="Instruction-tuned\nmodel")
        ax.text(fx + 1.55, y0 + 0.16,
                "Learns to follow instructions and hold a\nconversational, helpful role.",
                ha="center", fontsize=7.3, color=GREY, style="italic")


    def flow_stage3(ax, x0, y0, w, h, fx):
        cy = y0 + h / 2 + 0.05
        person_icon(ax, fx + 0.25, cy, label="User")
        dashed_arrow(ax, (fx + 0.38, cy), (fx + 0.80, cy))
        llm_box(ax, fx + 1.35, cy, w=0.9, sublabel="Instruction-tuned\nmodel")
        dashed_arrow(ax, (fx + 1.80, cy + 0.12), (fx + 2.35, cy + 0.30))
        dashed_arrow(ax, (fx + 1.80, cy - 0.12), (fx + 2.35, cy - 0.30))
        doc_check_icon(ax, fx + 2.55, cy + 0.34, ok=True, label="Response A")
        doc_check_icon(ax, fx + 2.55, cy - 0.34, ok=False, label="Response B")
        person_icon(ax, fx + 3.10, cy, label="Human /\nreward model")
        dashed_arrow(ax, (fx + 2.75, cy + 0.30), (fx + 2.98, cy + 0.08), curve=-0.2)
        dashed_arrow(ax, (fx + 2.75, cy - 0.30), (fx + 2.98, cy - 0.08), curve=0.2)
        dashed_arrow(ax, (fx + 3.28, cy), (fx + 3.65, cy), label="preferred\nresponse")
        pref_box = rounded_panel(ax, fx + 3.65, cy - 0.24, 0.55, 0.48,
                                face=WHITE, edge=DARK, lw=1.0, radius=0.05)
        ax.text(fx + 3.65 + 0.275, cy, "pref.\npair", ha="center", va="center", fontsize=6.8, color=GREY)


    def flow_stage4(ax, x0, y0, w, h, fx):
        cy = y0 + h / 2 + 0.05
        chat_bubble_icon(ax, fx + 0.30, cy, label="Reasoning task\n+ ground truth")
        dashed_arrow(ax, (fx + 0.52, cy), (fx + 0.95, cy))
        llm_box(ax, fx + 1.60, cy, sublabel="Preference-tuned\nmodel")
        dashed_arrow(ax, (fx + 2.10, cy), (fx + 2.50, cy))
        doc_check_icon(ax, fx + 2.70, cy, ok=True, label="Verified\nresponse")
        dashed_arrow(ax, (fx + 2.90, cy), (fx + 3.30, cy))
        ax.text(fx + 3.65, cy, "Update weights\nto increase reward\nfor correct answers",
                ha="center", va="center", fontsize=7.3, color=GREY)


    STAGES = [
        dict(number="Stage 0", title="Randomly\nInitialised Model", flow=flow_stage0),
        dict(number="Stage 1", title="Pre-Training", flow=flow_stage1),
        dict(number="Stage 2", title="Instruction\nFine-Tuning", flow=flow_stage2),
        dict(number="Stage 3", title="Preference\nFine-Tuning", flow=flow_stage3),
        dict(number="Stage 4", title="Reasoning\nFine-Tuning", flow=flow_stage4),
    ]


    # ---------------------------------------------------------------- MAIN ---
    if include is not None:
        STAGES = [s for s in STAGES if s["number"] in include]
    n = len(STAGES)
    fig_height = margin_top + n * row_height + (n - 1) * row_gap + margin_bottom
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, FIG_WIDTH)
    ax.set_ylim(0, fig_height)
    ax.axis("off")
    ax.set_aspect("equal")

    panel_x = 0.35
    panel_w = FIG_WIDTH - 2 * panel_x

    y_cursor = fig_height - MARGIN_TOP
    for stage in STAGES:
        y0 = y_cursor - ROW_HEIGHT
        rounded_panel(ax, panel_x, y0, panel_w, ROW_HEIGHT)

        # left-hand label column: badge + title
        badge_w, badge_h = 1.15, 0.34
        badge_x = panel_x + 0.25
        badge_y = y0 + ROW_HEIGHT - 0.30 - badge_h
        stage_badge(ax, badge_x, badge_y, badge_w, badge_h, stage["number"])
        ax.text(badge_x, badge_y - 0.16, stage["title"],
                ha="left", va="top", fontsize=11.5, fontweight="bold", color=DARK,
                linespacing=1.3)

        # flow diagram starts after the label column
        flow_x = panel_x + 1.75
        stage["flow"](ax, panel_x, y0, panel_w, ROW_HEIGHT, flow_x)

        y_cursor -= (ROW_HEIGHT + ROW_GAP)

    ax = fig.add_axes([0.005, 0.01, 0.99, 0.98])  # tiny margin, not exactly full-bleed
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")
    ax.set_aspect("equal")
   
    return fig
