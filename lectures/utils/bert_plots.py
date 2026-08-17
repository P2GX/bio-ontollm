import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

BLACK      = "#111111"
MAGENTA    = "#C4187A"
RED        = "#D3202A"
BLUE       = "#2255CC"
GRAY       = "#666666"
BERT_YELLOW    = "#F2DA8A"
CLASSIFIER_BG  = "#F7F2D8"
BAR_TEAL   = "#A9D8D0"
BAR_PINK   = "#F4C6CE"
SENT_A_BG  = "#DAD6EE"
SENT_B_BG  = "#F6CBA4"
BOX_EDGE   = "#111111"


def bert_nsp(figsize=(10.6, 4.6)):
    """
    Recreates a slide-style figure explaining BERT's Next Sentence Prediction (NSP) task:
    left column = title + bullet explanation, right column = architecture diagram
    (input tokens -> BERT -> [CLS] representation -> binary classifier -> IsNext/NotNext).
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    # =========================================================
    # LEFT COLUMN — title + bullets
    # =========================================================

    def rich_text(x, y, parts, ax, fontsize=21, weight="bold", va="top", ha="left"):
        """Draw a line built from (text, color) pairs, left to right, word-wrapped manually."""
        renderer = fig.canvas.get_renderer()
        cur_x = x
        for text, color in parts:
            t = ax.text(cur_x, y, text, color=color, fontsize=fontsize,
                        fontweight=weight, va=va, ha=ha, family="DejaVu Sans")
            fig.canvas.draw()  # force layout so we can measure
            bbox = t.get_window_extent(renderer=renderer)
            bbox_data = bbox.transformed(ax.transData.inverted())
            cur_x = bbox_data.x1 + 0.06
        return cur_x


    # ---- BERT block ----
    bert_x0, bert_x1 = 0.3, 7.3
    bert_y0, bert_y1 = 1.55, 2.75
    bert_box = FancyBboxPatch((bert_x0, bert_y0), bert_x1 - bert_x0, bert_y1 - bert_y0,
                            boxstyle="round,pad=0.02,rounding_size=0.18",
                            linewidth=2, edgecolor=BOX_EDGE, facecolor=BERT_YELLOW, zorder=2)
    ax.add_patch(bert_box)
    ax.text((bert_x0 + bert_x1) / 2, (bert_y0 + bert_y1) / 2, "BERT",
            fontsize=30, fontweight="bold", color=BLACK, ha="center", va="center", zorder=3)

    # ---- CLS -> Binary Classifier arrow ----
    cls_x = 0.55
    ax.add_patch(FancyArrowPatch((cls_x, bert_y1), (cls_x, 3.55),
                                arrowstyle="-|>", mutation_scale=18,
                                linewidth=2.4, color=RED, zorder=2))

    # ---- Binary classifier box ----
    clf_x0, clf_x1 = 0.15, 1.85
    clf_y0, clf_y1 = 3.55, 4.35
    clf_box = FancyBboxPatch((clf_x0, clf_y0), clf_x1 - clf_x0, clf_y1 - clf_y0,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            linewidth=1.6, edgecolor=BOX_EDGE, facecolor=CLASSIFIER_BG, zorder=2)
    ax.add_patch(clf_box)
    ax.text((clf_x0 + clf_x1) / 2, clf_y1 - 0.28, "Binary", fontsize=15, fontweight="bold",
            ha="center", va="center", color=BLACK)
    ax.text((clf_x0 + clf_x1) / 2, clf_y0 + 0.25, "Classifier", fontsize=15, fontweight="bold",
            ha="center", va="center", color=BLACK)

    # ---- Classifier -> bar chart arrow ----
    ax.add_patch(FancyArrowPatch((clf_x1 + 0.05, (clf_y0 + clf_y1) / 2), (5.55, (clf_y0 + clf_y1) / 2),
                                arrowstyle="-|>", mutation_scale=20,
                                linewidth=2.6, color=BLACK, zorder=2))

    # ---- mini bar chart ----
    bar_base_y = 3.60
    ax.add_line(Line2D([5.65, 6.55], [bar_base_y, bar_base_y], color=BLACK, linewidth=1.8))
    ax.add_line(Line2D([5.65, 5.65], [bar_base_y, 4.35], color=BLACK, linewidth=1.8))
    ax.add_patch(Rectangle((5.75, bar_base_y), 0.28, 0.68, facecolor=BAR_TEAL, edgecolor=BLACK, linewidth=1.2))
    ax.add_patch(Rectangle((6.08, bar_base_y), 0.28, 0.20, facecolor=BAR_PINK, edgecolor=BLACK, linewidth=1.2))

    ax.text(3.75, 4.30, "IsNext", fontsize=15, fontweight="bold", ha="left", va="center", color=BLACK)
    ax.text(3.75, 3.98, "or", fontsize=13, ha="left", va="center", color=GRAY)
    ax.text(3.75, 3.66, "NotNext", fontsize=15, fontweight="bold", ha="left", va="center", color=BLACK)

    # ---- input token boxes + arrows into BERT ----
    tokens = [
        ("[CLS]", None, RED, cls_x, 0.55),
    ]

    def token_box(x0, width, y0, y1, text, facecolor, textcolor, arrow_color, fontsize=13,
                bold=False, box=True):
        xc = x0 + width / 2
        if box:
            ax.add_patch(Rectangle((x0, y0), width, y1 - y0, facecolor=facecolor,
                                    edgecolor="none", zorder=1))
        ax.text(xc, (y0 + y1) / 2, text, fontsize=fontsize, ha="center", va="center",
                color=textcolor, fontweight="bold" if bold else "normal", zorder=2)
        ax.add_patch(FancyArrowPatch((xc, y1 + 0.03), (xc, bert_y0),
                                    arrowstyle="-|>", mutation_scale=13,
                                    linewidth=1.8, color=arrow_color, zorder=2))
        return xc

    tok_y0, tok_y1 = 0.55, 1.05

    # [CLS] (no fill box, just red bracket text)
    xc_cls = token_box(cls_x - 0.30, 0.60, tok_y0, tok_y1, "[CLS]", None, RED, RED,
                        fontsize=14, bold=True, box=False)

    # Sentence A box
    sentA_x0, sentA_x1 = 1.10, 3.05
    xc_a = token_box(sentA_x0, sentA_x1 - sentA_x0, tok_y0, tok_y1, "The sky is blue.",
                    SENT_A_BG, BLACK, BLACK, fontsize=13)
    # extra internal up-arrows across the sentence A box (visual only, 4 arrows total incl. edges)
    for fx in (0.18, 0.42, 0.66, 0.90):
        xpos = sentA_x0 + fx * (sentA_x1 - sentA_x0)
        ax.add_patch(FancyArrowPatch((xpos, tok_y1 + 0.03), (xpos, bert_y0),
                                    arrowstyle="-|>", mutation_scale=11,
                                    linewidth=1.5, color=BLACK, zorder=2))

    # [SEP]
    sep_x0 = sentA_x1 + 0.20
    xc_sep = token_box(sep_x0, 0.58, tok_y0, tok_y1, "[SEP]", None, BLACK, BLACK,
                        fontsize=14, bold=True, box=False)

    # Sentence B box
    sentB_x0 = sep_x0 + 0.80
    sentB_x1 = 7.15
    xc_b = token_box(sentB_x0, sentB_x1 - sentB_x0, tok_y0, tok_y1, "It looks beautiful today.",
                    SENT_B_BG, BLACK, BLACK, fontsize=13)
    for fx in (0.13, 0.34, 0.55, 0.76, 0.94):
        xpos = sentB_x0 + fx * (sentB_x1 - sentB_x0)
        ax.add_patch(FancyArrowPatch((xpos, tok_y1 + 0.03), (xpos, bert_y0),
                                    arrowstyle="-|>", mutation_scale=11,
                                    linewidth=1.5, color=BLACK, zorder=2))

    # ---- Sentence A / Sentence B labels ----
    ax.text((sentA_x0 + sentA_x1) / 2, 0.20, "Sentence A", fontsize=15, fontweight="bold",
            color=BLUE, ha="center", va="top")
    ax.text((sentB_x0 + sentB_x1) / 2, 0.20, "Sentence B", fontsize=15, fontweight="bold",
            color=BLUE, ha="center", va="top")

    plt.tight_layout()
