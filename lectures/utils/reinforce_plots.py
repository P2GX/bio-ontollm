
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch,Rectangle
from matplotlib.lines import Line2D


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

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch


def agent_env_loop(
    agent_label="Agent",
    agent_formula=r"$\pi_\theta(\cdot)$",
    env_label=("Target", "Environment"),
    left_labels=(r"$s_t$", r"$r_t$"),
    right_label=r"$a_t$",
    agent_color="#E8A33D",
    env_color="#2E8B57",
    line_color="#2B2B6B",
    text_color="#1A1A2E",
    fig_width=5.2,
    fig_height=3.6,
):
    """
    Draw a simple agent-environment reinforcement-learning loop diagram:
    an orange "Agent" ellipse on top, a green "Environment" box below,
    connected by a rectangular loop (state/reward flowing up the left
    side into the agent, action flowing down the right side into the
    environment).

    All text, colors, and sizing are parameters so the same layout can
    be reused for different labelings (e.g. multi-agent, different
    reward notation, etc.). Returns the matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_aspect("equal")

    # ---- Agent ellipse ----
    agent = Ellipse((5.5, 5.3), width=3.6, height=1.7,
                     facecolor=agent_color, edgecolor="none", zorder=3)
    ax.add_patch(agent)
    ax.text(5.5, 5.65, agent_label, ha="center", va="center",
            fontsize=13, fontweight="bold", color=text_color, zorder=4)
    ax.text(5.5, 5.05, agent_formula, ha="center", va="center",
            fontsize=13, color=text_color, zorder=4)

    # ---- Target Environment box ----
    env = FancyBboxPatch((3.55, 1.1), 4.0, 1.6, boxstyle="square,pad=0",
                          facecolor=env_color, edgecolor="none", zorder=3)
    ax.add_patch(env)
    ax.text(5.55, 2.15, env_label[0], ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", zorder=4)
    ax.text(5.55, 1.6, env_label[1], ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", zorder=4)

    # ---- Left loop: environment -> up -> agent (state / reward) ----
    left_x = 2.0
    ax.plot([left_x, left_x], [1.9, 5.3], color=line_color, lw=1.6, zorder=2)
    ax.plot([left_x, 3.7], [1.9, 1.9], color=line_color, lw=1.6, zorder=2)
    arrow_left = FancyArrowPatch((left_x, 5.3), (3.7, 5.3), arrowstyle="-|>",
                                  mutation_scale=16, linewidth=1.6,
                                  color=line_color, zorder=2)
    ax.add_patch(arrow_left)

    ax.text(1.55, 4.1, left_labels[0], ha="center", va="center",
            fontsize=13, color=text_color)
    ax.text(1.55, 3.3, left_labels[1], ha="center", va="center",
            fontsize=13, color=text_color)

    # ---- Right loop: agent -> down -> environment (action) ----
    right_x = 9.0
    ax.plot([right_x, right_x], [1.9, 5.3], color=line_color, lw=1.6, zorder=2)
    ax.plot([7.3, right_x], [5.3, 5.3], color=line_color, lw=1.6, zorder=2)
    arrow_right = FancyArrowPatch((right_x, 1.9), (7.3, 1.9), arrowstyle="-|>",
                                   mutation_scale=16, linewidth=1.6,
                                   color=line_color, zorder=2)
    ax.add_patch(arrow_right)
    ax.text(9.45, 3.6, right_label, ha="center", va="center",
            fontsize=13, color=text_color)

    fig.tight_layout(pad=0.3)
    return fig


def preference_plot():
    
    NAVY = "#1A1A2E"
    BORDER = "#2B2B6B"
    GREEN = "#22C55E"
    GREEN_FACE = "#DCFCE7"
    PINK = "#EC4899"
    PINK_FACE = "#FCE7F3"
    GREY = "#6B7280"
    WHITE = "#FFFFFF"


    def _token_width(token, char_w=0.11, min_w=0.45, pad=0.18):
        return max(min_w, len(token) * char_w + pad)


    def _draw_token_row(ax, tokens, y, label, box_h=0.5, gap=0.08,
                        start_x=1.7, highlight_last=None):
        """Draw one row of token boxes; returns the x position after the row."""
        ax.text(start_x - 0.15, y, label, ha="right", va="center",
                fontsize=11, fontweight="bold", color=NAVY)

        x = start_x
        for i, tok in enumerate(tokens):
            w = _token_width(tok)
            is_last = (i == len(tokens) - 1)
            if is_last and highlight_last is not None:
                face, edge, lw = highlight_last["face"], highlight_last["edge"], 2.4
            else:
                face, edge, lw = WHITE, BORDER, 1.2
            box = FancyBboxPatch(
                (x, y - box_h / 2), w, box_h,
                boxstyle="round,pad=0,rounding_size=0.05",
                linewidth=lw, edgecolor=edge, facecolor=face, zorder=3,
            )
            ax.add_patch(box)
            ax.text(x + w / 2, y, tok, ha="center", va="center",
                    fontsize=10, color=NAVY, zorder=4)
            x += w + gap
        return x


    def preference_rm_diagram(
        chosen_tokens,
        rejected_tokens,
        title="Training a Preference RM: Pairwise Comparison at EOS",
        loss_text=r"Loss: $\mathcal{L} = -\log \sigma(r_c - r_r)$   |   Only score difference matters",
        fig_width=11.5,
        fig_height=3.8,
    ):
        """
        Draw a token-box diagram illustrating preference reward-model training:
        a "Chosen" row and a "Rejected" row of tokens, each ending in an EOS
        token highlighted in green (chosen) or pink (rejected), with a legend,
        loss formula, and caption underneath.

        chosen_tokens / rejected_tokens: lists of strings, e.g.
            ["<|eos|>", "What", "is", "12", "\u00d7", "8", "?",
            "The", "answer", "is", "96", ".", "<|eos|>"]

        Returns the matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(0, fig_width)
        ax.set_ylim(0, fig_height)
        ax.axis("off")

        top_y = fig_height - 0.55
        ax.text(fig_width / 2, top_y, title, ha="center", va="center",
                fontsize=14, fontweight="bold", color=NAVY)

        # legend, upper right
        leg_x = fig_width - 1.9
        leg_y1 = top_y - 0.05
        leg_y2 = leg_y1 - 0.32
        ax.add_patch(Rectangle((leg_x, leg_y1 - 0.09), 0.22, 0.18,
                                facecolor=GREEN_FACE, edgecolor=GREEN, linewidth=1.6, zorder=4))
        ax.text(leg_x + 0.32, leg_y1, "Supervised", ha="left", va="center", fontsize=8.5, color=NAVY)
        ax.add_patch(Rectangle((leg_x, leg_y2 - 0.09), 0.22, 0.18,
                                facecolor=PINK_FACE, edgecolor=PINK, linewidth=1.6, zorder=4))
        ax.text(leg_x + 0.32, leg_y2, "Rejected", ha="left", va="center", fontsize=8.5, color=NAVY)

        row1_y = top_y - 1.05
        row2_y = row1_y - 0.85

        _draw_token_row(ax, chosen_tokens, row1_y, "Chosen:",
                        highlight_last={"face": GREEN_FACE, "edge": GREEN})
        _draw_token_row(ax, rejected_tokens, row2_y, "Rejected:",
                        highlight_last={"face": PINK_FACE, "edge": PINK})

        ax.text(fig_width / 2, row2_y - 0.55, loss_text,
                ha="center", va="center", fontsize=10, style="italic", color=NAVY)
        #ax.text(0.15, row2_y - 1.05, caption=None, ha="left", va="center", fontsize=8.5, color=GREY)

        fig.tight_layout(pad=0.4)
        return fig
    chosen = ["<|eos|>", "What", "is", "12", "\u00d7", "8", "?",
                  "The", "answer", "is", "96", ".", "<|eos|>"]
    rejected = ["<|eos|>", "What", "is", "12", "\u00d7", "8", "?",
                    "The", "answer", "is", "84", ".", "<|eos|>"]
    fig = preference_rm_diagram(chosen, rejected)
    return fig

