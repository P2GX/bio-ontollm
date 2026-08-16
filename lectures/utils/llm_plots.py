import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.path import Path
import numpy as np

def plot_activations(show_gelu=False):
    """
    Plot ReLU and GELU activations
    """
    x = np.linspace(-3, 3, 200)
    relu = np.maximum(0, x)
    leaky = np.where(x > 0, x, 0.1 * x)
    # Approximation of GELU
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    ax.plot(x, relu, label='ReLU', color='#0000ff', lw=3)
    ax.plot(x, leaky, label='Leaky ReLU', color='#ff0000', ls='--', lw=2)
    if show_gelu:
        ax.plot(x, gelu, label='GELU', color='#00aa00', lw=2)

    ax.set_aspect('equal') # Prevents the "squeezed" look
    ax.legend()
    ax.grid(True, alpha=0.3)


def draw_gelu_block():
    """
    Plot the FFN block of BERT with its GELU unit fan out
    """
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Vertices for trapezoids
    top_trap = [[20, 80], [80, 80], [100, 60], [0, 60]]
    bot_trap = [[0, 30], [100, 30], [80, 10], [20, 10]]

    ax.add_patch(patches.Polygon(top_trap, fc="#F06292", ec="black"))
    ax.add_patch(patches.Rectangle((0, 35), 100, 20, fc="#B2EBF2", ec="black"))
    ax.add_patch(patches.Polygon(bot_trap, fc="#F06292", ec="black"))

    # Labels
    ax.text(50, 45, "GELU", ha='center', weight='bold', size=14)
    
    # Arrows
    arrow_data = [(50, 92, 50, 80), (50, 60, 50, 54), (50, 35, 50, 29), (50, 10, 50, 0)]
    for x1, y1, x2, y2 in arrow_data:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2))



def draw_encoder_compact(highlight=None):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10) 
    ax.axis('off')

    colors = {
        'mha': '#fdba74', 'ffn': '#93c5fd', 'norm': '#e2e8f0',
        'text': '#333333', 'highlight': '#1D9A73', 'residual': '#64748b'
    }

    def draw_box(y, text, color, key):
        is_hl = (highlight == key)
        rect = patches.FancyBboxPatch((2.5, y), 5, 0.8, boxstyle="round,pad=0.1", 
                                      linewidth=3 if is_hl else 1.5, 
                                      edgecolor=colors['highlight'] if is_hl else '#333333', 
                                      facecolor=color)
        ax.add_patch(rect)
        ax.text(5, y + 0.4, text, ha='center', va='center', fontsize=10, fontweight='bold')

    ax.plot([5, 5], [0.2, 0.8], color='#333', lw=1.5)
    ax.plot([3.5, 6.5], [0.8, 0.8], color='#333', lw=1.5)
    for x, label in zip([3.5, 5, 6.5], ['K', 'Q', 'V']):
        ax.plot([x, x], [0.8, 1.5], color='#333', lw=1.5)
        ax.text(x, 1.1, label, ha='center', va='center', fontsize=9, fontweight='bold', backgroundcolor='white')

    # --- MHA ---
    draw_box(1.5, "Multi-Head Attention", colors['mha'], 'mha')
    
    # Residual 1
    ax.plot([5, 1, 1], [0.6, 0.6, 3.2], color=colors['residual'], lw=1.2, ls='--')
    ax.annotate('', xy=(2.5, 3.2), xytext=(1, 3.2), arrowprops=dict(arrowstyle='->', color=colors['residual'], ls='--'))

    # --- Add & Norm 1 ---
    draw_box(2.8, "Add & Norm", colors['norm'], 'norm1')
    ax.annotate('', xy=(5, 2.8), xytext=(5, 2.3), arrowprops=dict(arrowstyle='->', lw=1.5))

    # --- FFN ---
    draw_box(4.5, "Feed Forward (FFN)", colors['ffn'], 'ffn')
    
    # Residual 2
    ax.plot([5, 1, 1], [4.0, 4.0, 6.2], color=colors['residual'], lw=1.2, ls='--')
    ax.annotate('', xy=(2.5, 6.2), xytext=(1, 6.2), arrowprops=dict(arrowstyle='->', color=colors['residual'], ls='--'))

    # --- Add & Norm 2 ---
    draw_box(5.8, "Add & Norm", colors['norm'], 'norm2')
    ax.annotate('', xy=(5, 5.8), xytext=(5, 5.3), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Output
    ax.annotate('', xy=(5, 7.5), xytext=(5, 6.6), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(5, 7.8, "Output to Next Layer", ha='center', fontsize=9, fontweight='bold')

def decoder_loop(figsize=(8, 3)):
    """
    Recreates a 3-step autoregressive decoding diagram:
    Step 1: Process prompt          -> predicts "on"
    Step 2: Append and reprocess    -> predicts "the"
    Step 3: Continue generation     -> predicts "mat"
    Result: full generated sequence

    Usage:
        python draw_decoding_steps.py
    Produces: decoding_steps.svg and decoding_steps.png
    """
    # ---------------------------------------------------------------------
    # Colors (matched approximately to the reference image)
    BLUE_FACE   = "#c7d2fe"
    BLUE_EDGE   = "#4338ca"
    GREEN_FACE  = "#6ee7b7"
    GREEN_EDGE  = "#059669"
    BAR_COLOR   = "#7dd3fc"
    TEXT_GREEN  = "#059669"
    TEXT_GRAY   = "#374151"

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis("off")

    TOKEN_W, TOKEN_H = 1.0, 0.65
    TOKEN_GAP = 0.15

    def draw_tokens(x0, y0, tokens, colors):
        """tokens: list of strings; colors: list of (face, edge) tuples.
        Returns x position right after the last token (for arrow start)."""
        x = x0
        for tok, (face, edge) in zip(tokens, colors):
            box = FancyBboxPatch((x, y0), TOKEN_W, TOKEN_H,
                                boxstyle="round,pad=0.05,rounding_size=0.08",
                                linewidth=1.6, edgecolor=edge, facecolor=face)
            ax.add_patch(box)
            ax.text(x + TOKEN_W / 2, y0 + TOKEN_H / 2, tok,
                    ha="center", va="center", fontsize=10, color="#1f2937")
            x += TOKEN_W + TOKEN_GAP
        return x

    def draw_arrow(x0, x1, y):
        arrow = FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                mutation_scale=18, linewidth=1.8, color="#111827")
        ax.add_patch(arrow)

    def draw_bars(x0, y0, labels, heights, highlight_idx, max_h=1.4, bar_w=0.45, gap=0.25):
        """Draws a small bar chart. y0 is the baseline. Returns (x_left, x_right)."""
        x = x0
        for i, (lab, h) in enumerate(zip(labels, heights)):
            color = BAR_COLOR
            alpha = 1.0 if i == highlight_idx else 0.55
            ax.add_patch(Rectangle((x, y0), bar_w, h * max_h,
                                    facecolor=color, edgecolor="none", alpha=alpha))
            ax.text(x + bar_w / 2, y0 - 0.18, lab, ha="center", va="top", fontsize=8.5,
                    color=TEXT_GRAY)
            x += bar_w + gap
        return x0, x

    def draw_row(y_base, step_label, step_sublabel, tokens, colors,
                bar_labels, bar_heights, highlight_idx, predicted_word):
        # Step label
        ax.text(0.2, y_base + TOKEN_H + 0.55, step_label, fontsize=13, fontweight="bold",
                color=TEXT_GRAY, va="bottom")
        ax.text(1.55, y_base + TOKEN_H + 0.55, step_sublabel, fontsize=12,
                color=TEXT_GRAY, va="bottom")

        # Tokens
        x_end = draw_tokens(1.1, y_base, tokens, colors)

        # Arrow from tokens to bar chart
        arrow_y = y_base + TOKEN_H / 2
        draw_arrow(x_end + 0.1, x_end + 1.0, arrow_y)

        # "P(next)" label
        bars_x0 = x_end + 1.2
        ax.text(bars_x0, y_base + 1.55, "P(next)", fontsize=10, color=TEXT_GRAY, ha="left")

        # Bars
        _, bars_x1 = draw_bars(bars_x0 + 0.3, y_base, bar_labels, bar_heights,
                                highlight_idx, max_h=1.4)

        # Predicted word (green, quoted)
        ax.text(bars_x1 + 0.3, y_base + 0.75, f'"{predicted_word}"',
                fontsize=12, color=TEXT_GREEN, fontweight="bold", va="center")

    # ---------------------------------------------------------------------
    BLUE = (BLUE_FACE, BLUE_EDGE)
    GREEN = (GREEN_FACE, GREEN_EDGE)

    # Step 1
    draw_row(
        y_base=10.6,
        step_label="Step 1:", step_sublabel="Process prompt",
        tokens=["The", "cat", "sat"], colors=[BLUE, BLUE, BLUE],
        bar_labels=["the", "a", "on", "in"], bar_heights=[0.12, 0.28, 1.0, 0.45],
        highlight_idx=2, predicted_word="on",
    )

    # Step 2
    draw_row(
        y_base=7.3,
        step_label="Step 2:", step_sublabel="Append and reprocess",
        tokens=["The", "cat", "sat", "on"], colors=[BLUE, BLUE, BLUE, GREEN],
        bar_labels=["the", "a", "my", "his"], bar_heights=[1.0, 0.22, 0.2, 0.08],
        highlight_idx=0, predicted_word="the",
    )

    # Step 3
    draw_row(
        y_base=4.0,
        step_label="Step 3:", step_sublabel="Continue generation",
        tokens=["The", "cat", "sat", "on", "the"], colors=[BLUE, BLUE, BLUE, GREEN, GREEN],
        bar_labels=["mat", "floor", "bed", "couch"], bar_heights=[0.75, 0.55, 0.4, 0.62],
        highlight_idx=0, predicted_word="mat",
    )

    # --- Result row --------------------------------------------------------
    result_y = 0.6
    ax.text(0.2, result_y + TOKEN_H + 0.3, "Result:", fontsize=13, fontweight="bold",
            color=TEXT_GRAY, va="bottom")
    draw_tokens(1.1, result_y,
                ["The", "cat", "sat", "on", "the", "mat"],
                [BLUE, BLUE, BLUE, GREEN, GREEN, GREEN])

    plt.tight_layout()

