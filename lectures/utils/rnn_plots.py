from matplotlib import patches, path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.path import Path
from matplotlib.patches import Rectangle, FancyArrow, FancyBboxPatch,  FancyArrowPatch, Polygon
import matplotlib.patches as mpatches

def basic_rnn(figsize=(10,5)):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 2.2]})
    fig.patch.set_facecolor('white')
    node_radius = 0.15
    box_props = dict(boxstyle="circle,pad=0.3", facecolor="#ebf5fb", edgecolor="#2980b9", linewidth=2)
    font_style = dict(fontsize=12, fontweight='bold', ha='center', va='center')
    arrow_props = dict(arrowstyle="->", lw=2, color="#34495e", mutation_scale=15)

    # =============================================================================
    # 1. LEFT PANEL: FOLDED ARCHITECTURE (Input -> Hidden (Self-loop) -> Output)
    # =============================================================================
    ax1.set_title("Folded RNN Representation", fontsize=12, fontweight='bold', pad=20)
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.axis('off')

    # Node positions
    x_in, y_in = 0.5, 0.0
    x_hid, y_hid = 0.5, 1.0
    x_out, y_out = 0.5, 2.0

    # Annotate Nodes as boxes to create a clean schematic look
    ax1.text(x_in, y_in, "Input\n$x_t$", **font_style, bbox=box_props)
    ax1.text(x_hid, y_hid, "Hidden\n$h_t$", **font_style, bbox=dict(boxstyle="circle,pad=0.3", facecolor="#e8f8f5", edgecolor="#16a085", linewidth=2))
    ax1.text(x_out, y_out, "Output\n$y_t$", **font_style, bbox=box_props)

    # Draw straight arrows (accounting for offsets so they don't pierce text box centers)
    ax1.annotate("", xy=(x_hid, y_hid - 0.22), xytext=(x_in, y_in + 0.22), arrowprops=arrow_props)
    ax1.annotate("", xy=(x_out, y_out - 0.22), xytext=(x_hid, y_hid + 0.22), arrowprops=arrow_props)

    # Recursive Self-Loop Arrow on the Hidden Layer
    ax1.annotate("", 
             xy=(x_hid + 0.11, y_hid - 0.11),     # Target endpoint (bottom-right edge of circle)
             xytext=(x_hid + 0.11, y_hid + 0.11), # Starting point (top-right edge of circle)
             arrowprops=dict(arrowstyle="->", 
                             lw=2, 
                             color="#e67e22", 
                             connectionstyle="arc3,rad=-3.0", # Large negative radius forces a wide external loop
                             mutation_scale=15))
    ax1.text(x_hid + 0.55, y_hid, "$W_{hh}$", fontsize=11, color="#d35400", ha='center', va='center')
    ax1.text(x_hid - 0.2, y_in + 0.5, "$W_{xh}$", fontsize=10, color="#34495e", ha='center')
    ax1.text(x_hid - 0.2, y_hid + 0.5, "$W_{hy}$", fontsize=10, color="#34495e", ha='center')


    # =============================================================================
    # 2. RIGHT PANEL: UNROLLED ARCHITECTURE (t-1 -> t -> t+1)
    # =============================================================================
    ax2.set_title("Unrolled Network Through Time", fontsize=12, fontweight='bold', pad=20)
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 2.5)
    ax2.axis('off')

    time_steps = [r"$t-1$", r"$t$", r"$t+1$"]
    x_coords = [0.0, 1.0, 2.0]

    for i, (ts, x) in enumerate(zip(time_steps, x_coords)):
        # Render Nodes across time steps
        ax2.text(x, y_in, f"Input\n$x_{{{ts[1:-1]}}}$", **font_style, bbox=box_props)
        ax2.text(x, y_hid, f"Hidden\n$h_{{{ts[1:-1]}}}$", **font_style, bbox=dict(boxstyle="circle,pad=0.3", facecolor="#e8f8f5", edgecolor="#16a085", linewidth=2))
        ax2.text(x, y_out, f"Output\n$y_{{{ts[1:-1]}}}$", **font_style, bbox=box_props)
        
        # Internal layer arrows (Feed-Forward path)
        ax2.annotate("", xy=(x, y_hid - 0.22), xytext=(x, y_in + 0.22), arrowprops=arrow_props)
        ax2.annotate("", xy=(x, y_out - 0.22), xytext=(x, y_hid + 0.22), arrowprops=arrow_props)
        
        # Temporal Recurrent links between adjacent hidden nodes
        if i < len(x_coords) - 1:
            ax2.annotate("", xy=(x_coords[i+1] - 0.25, y_hid), xytext=(x + 0.25, y_hid),
                        arrowprops=dict(arrowstyle="->", lw=2, color="#e67e22", mutation_scale=15))
            ax2.text(x + 0.5, y_hid + 0.15, "$W_{hh}$", fontsize=11, color="#d35400", ha='center')

    plt.tight_layout()


def plot_tanh(figsize=(5.5, 3.5)):
    
    z = np.linspace(-4, 4, 200)
    activation = np.tanh(z)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')

    # Plot the tanh curve
    ax.plot(z, activation, color='#16a085', lw=2.5, label=r'$\tanh(z)$')

    # Axis styling & zero alignment
    ax.axhline(0, color='#34495e', lw=0.8, ls='--')
    ax.axvline(0, color='#34495e', lw=0.8, ls='--')
    ax.axhline(1, color='#e74c3c', lw=1, ls=':', alpha=0.7)
    ax.axhline(-1, color='#e74c3c', lw=1, ls=':', alpha=0.7)

    # Labels & Limits
    ax.set_title("Hyperbolic Tangent (tanh)", fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()


BLUE = "#9DC3E6"
ORANGE = "#F4B183"
GREEN = "#A9D18E"
EDGE = "black"
BOX_W, BOX_H = 0.62, 0.42          # input / output rectangles
ELL_W, ELL_H = 0.62, 0.42          # hidden-state ellipses
COL_GAP = 1.15                     # horizontal spacing between columns
ROW_INPUT, ROW_HIDDEN, ROW_OUTPUT = 0.0, 1.15, 2.3
 
 
def rounded_box(ax, x, y, color):
    box = FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=EDGE, facecolor=color, zorder=3,
    )
    ax.add_patch(box)
 
 
def ellipse(ax, x, y, color):
    e = Ellipse((x, y), ELL_W, ELL_H, linewidth=1.4,
                edgecolor=EDGE, facecolor=color, zorder=3)
    ax.add_patch(e)
 
 
def arrow(ax, xy_from, xy_to):
    a = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle="-|>", mutation_scale=13,
        linewidth=1.3, color="#404040", zorder=2,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)
 
 
def draw_topology(ax, title, n_hidden, input_positions, output_positions):
    """
    n_hidden        : number of hidden-state columns to draw
    input_positions : set/list of hidden-column indices (0-based) that have
                       an input box feeding into them
    output_positions: set/list of hidden-column indices (0-based) that have
                       an output box coming out of them
    """
    xs = [i * COL_GAP for i in range(n_hidden)]
 
    # hidden-to-hidden recurrent arrows (only if more than one hidden step)
    for i in range(n_hidden - 1):
        arrow(ax, (xs[i] + ELL_W / 2, ROW_HIDDEN), (xs[i + 1] - ELL_W / 2, ROW_HIDDEN))
 
    for i, x in enumerate(xs):
        ellipse(ax, x, ROW_HIDDEN, ORANGE)
 
        if i in input_positions:
            rounded_box(ax, x, ROW_INPUT, BLUE)
            arrow(ax, (x, ROW_INPUT + BOX_H / 2), (x, ROW_HIDDEN - ELL_H / 2))
 
        if i in output_positions:
            rounded_box(ax, x, ROW_OUTPUT, GREEN)
            arrow(ax, (x, ROW_HIDDEN + ELL_H / 2), (x, ROW_OUTPUT - BOX_H / 2))
 
    ax.set_xlim(-0.9, xs[-1] + 0.9)
    ax.set_ylim(-0.9, ROW_OUTPUT + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)

def plot_sequence_models(figsize=(15, 9)):
    # ----------------------------------------------------------------------
    # Figure layout: 2 rows x 3 columns, one panel left blank
    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    draw_topology(
        axes[0, 0], "one-to-one\n(vanilla feedforward net)",
        n_hidden=1, input_positions={0}, output_positions={0},
    )
    
    draw_topology(
        axes[0, 1], "many-to-one",
        n_hidden=3, input_positions={0, 1, 2}, output_positions={2},
    )
    
    draw_topology(
        axes[0, 2], "one-to-many",
        n_hidden=3, input_positions={0}, output_positions={0, 1, 2},
    )
    
    draw_topology(
        axes[1, 0], "many-to-many\n(synced)",
        n_hidden=3, input_positions={0, 1, 2}, output_positions={0, 1, 2},
    )
    
    draw_topology(
        axes[1, 1], "many-to-many\n(encoder-decoder)",
        n_hidden=5, input_positions={0, 1, 2}, output_positions={2, 3, 4},
    )
    
    # blank the unused 6th panel and add the source citation there instead
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02, 0.5,
        "Figure adapted from:\nSebastian Raschka, Vahid Mirjalili.\n"
        "Python Machine Learning, 3rd Edition.\n"
        "Birmingham, UK: Packt Publishing, 2019.",
        fontsize=10, color="#555555", va="center",
    )
    
    fig.suptitle("RNN input/output topologies", fontsize=18, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    


def rnn_basic_idea(figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.set_facecolor('#f0f0f0')  # Set light gray background
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')  # Turn off the axis spines and ticks

    # --- Define Block Parameters ---
    block_width = 2
    block_height = 1.5
    center_x = 5
    # Vertical positions
    y_x_block = 1.5
    y_rnn_block = 4.5
    y_y_block = 7.5

    # Colors (approximate from image)
    light_pink = '#E8C6C6'
    green = '#3B7A1A'
    light_blue = '#BCCCE6'

    # --- Function to Add Blocks with Centered Text ---
    def add_block(ax, x, y, w, h, color, text, text_color='white', fontsize=16, weight='bold'):
        # Create the rectangle
        rect = patches.Rectangle((x - w/2, y - h/2), w, h, 
                                 linewidth=1, edgecolor='black', 
                                 facecolor=color)
        ax.add_patch(rect)
        # Add centered text
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=fontsize, color=text_color, fontweight=weight)
        return rect

    # --- Add the Blocks ---
    # X input block
    add_block(ax, center_x, y_x_block, block_width, block_height, light_pink, 'x', text_color='black')
    
    # RNN block
    add_block(ax, center_x, y_rnn_block, block_width, block_height, green, 'RNN', fontsize=10)
    
    # Y output block
    add_block(ax, center_x, y_y_block, block_width, block_height, light_blue, 'y', text_color='black')

    # --- Add Arrows and Self-Loop ---
    
    # Helper for standard arrows
    def draw_arrow(ax, start_x, start_y, end_y):
        ax.annotate('', xy=(start_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(facecolor='black', edgecolor='black', 
                                    arrowstyle='->', lw=1.5))

    # Input to RNN arrow (x -> RNN)
    draw_arrow(ax, center_x, y_x_block + block_height/2, y_rnn_block - block_height/2)
    
    # RNN to Output arrow (RNN -> y)
    draw_arrow(ax, center_x, y_rnn_block + block_height/2, y_y_block - block_height/2)
    
    # Self-loop arrow (RNN -> RNN)
    # Calculate start and end points on the right side of the RNN block
    loop_start_x = center_x + block_width/2
    loop_end_x = center_x + block_width/2
    loop_y_center = y_rnn_block
    
    # Use a curved path (cubic bezier spline) for the self-loop
    # The path control points create the loop shape
    # We extend it further to the right as in the image
    control_point_offset = 3.5 # How far out the loop goes
    
    verts = [
        (loop_start_x, loop_y_center),                  # P0: Start point
        (loop_start_x + control_point_offset, loop_y_center), # P1: Control point 1 (right)
        (loop_start_x + control_point_offset, loop_y_center), # P2: Control point 2 (tip)
        (loop_start_x, loop_y_center - block_height/4)  # P3: End point (slightly below middle)
    ]
    
    # Manually add the tip of the arrow using a small polygon patch
    arrow_head_path = patches.RegularPolygon((loop_start_x, loop_y_center - block_height/4), 
                                            3, radius=0.12, orientation=np.pi*1.5, 
                                            facecolor='black', edgecolor='black', zorder=10)
    ax.add_patch(arrow_head_path)
    # Draw the curve of the self-loop
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path_data = path.Path(verts, codes)
    patch_loop = patches.PathPatch(path_data, facecolor='none', edgecolor='black', lw=1.5)
    ax.add_patch(patch_loop)
    
    # Optional: Add the tiny red circle near the self-loop, as in the original image
    circle_red = patches.Circle((loop_start_x + 1.5, loop_y_center - 0.2), 0.08, facecolor='red', edgecolor='none', zorder=15, alpha=0.6)
    ax.add_patch(circle_red)

    # --- Finalize and Display ---
    plt.tight_layout()


def sentiment_analysis_basic_idea(figsize=(9, 2.2)):
    """
    sentiment_pipeline.py

    Recreates a simple horizontal pipeline diagram:
    quoted input text --> [ Sentiment Analysis ] --> output label
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ----------------------------------------------------------------------
    # Left: quoted input text
    # ----------------------------------------------------------------------
    input_text = (
        '"A startlingly inept film that offers overblown,\n'
        'wall-to-wall action without a hint of\n'
        'wit, coherence, style, or originality."'
    )
    ax.text(0.0, 0.5, input_text, fontsize=11, ha="left", va="center",
            linespacing=1.4)

    # ----------------------------------------------------------------------
    # Middle: gray "Sentiment Analysis" box
    # ----------------------------------------------------------------------
    box_x, box_w = 4.3, 2.6
    box_y, box_h = 0.15, 0.7

    box = Rectangle((box_x, box_y), box_w, box_h,
                    facecolor="#A6A6A6", edgecolor="#7F7F7F", linewidth=1.0, zorder=2)
    ax.add_patch(box)
    ax.text(box_x + box_w / 2, box_y + box_h / 2, "Sentiment Analysis",
            fontsize=11, ha="center", va="center", zorder=3)

    # ----------------------------------------------------------------------
    # Right: output label
    # ----------------------------------------------------------------------
    ax.text(7.85, 0.5, "Negative", fontsize=11, ha="left", va="center")

    # ----------------------------------------------------------------------
    # Arrows
    # ----------------------------------------------------------------------
    def block_arrow(x_start, x_end, y):
        ax.add_patch(FancyArrow(
            x_start, y, x_end - x_start, 0,
            width=0.09, head_width=0.28, head_length=0.18,
            length_includes_head=True,
            facecolor="#5B9BD5", edgecolor="#5B9BD5", zorder=2,
        ))

    block_arrow(3.05, 4.2, 0.5)
    block_arrow(7.05, 7.75, 0.5)

    # ----------------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------------
    ax.set_xlim(-0.1, 9.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")

    plt.tight_layout()

def sentiment_embedding(figsize=(11, 3.2)):
    """
    embedding_pipeline.py

    Recreates a pipeline diagram: a stack of word-embedding vectors
    (word label, row index, D-dimensional dot vector) feeding into a
    "Sentiment Analysis" box, producing an output label.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ----------------------------------------------------------------------
    # Word-vector rows
    # ----------------------------------------------------------------------
    words = ["A", "startlingly", None, "originality"]   # None -> vertical "..." row
    indices = ["1", "2", None, "20"]

    n_dots = 16          # dots per vector (visual only, not tied to real D)
    vec_x0 = 1.75        # left edge of the blue vector bar
    vec_w = 2.7          # width of the blue vector bar
    row_h = 0.34         # height of each vector bar
    row_ys = [2.55, 2.05, 1.30, 0.35]   # y-position (bottom) of each row, incl. gap for "..."

    BLUE = "#5B9BD5"
    DOT = "black"

    for (word, idx, y) in zip(words, indices, row_ys):
        if word is None:
            # vertical ellipsis between row 2 and the last row
            for dy in (0.55, 0.40, 0.25):
                ax.plot(vec_x0 + vec_w / 2, y + dy, marker="o",
                        markersize=4, color="black", zorder=4)
            continue

        ax.text(-0.05, y + row_h / 2, word, fontsize=12, ha="left", va="center")
        ax.text(1.4, y + row_h / 2, idx, fontsize=12, ha="right", va="center")

        bar = Rectangle((vec_x0, y), vec_w, row_h,
                        facecolor=BLUE, edgecolor="#3D6E9E", linewidth=1.0, zorder=2)
        ax.add_patch(bar)

        dot_xs = [vec_x0 + vec_w * (k + 0.5) / n_dots for k in range(n_dots)]
        ax.scatter(dot_xs, [y + row_h / 2] * n_dots, s=26, color=DOT, zorder=3)

    ax.text(vec_x0 + vec_w / 2, 3.05, "D-dimensional vector",
            fontsize=12, ha="center", va="center")

    # ----------------------------------------------------------------------
    # Middle: gray "Sentiment Analysis" box
    # ----------------------------------------------------------------------
    box_x, box_w = 6.6, 2.9
    box_y, box_h = 0.9, 1.7

    box = Rectangle((box_x, box_y), box_w, box_h,
                    facecolor="#A6B3C5", edgecolor="#7F8FA6", linewidth=1.0, zorder=2)
    ax.add_patch(box)
    ax.text(box_x + box_w / 2, box_y + box_h / 2, "Sentiment Analysis",
            fontsize=13, ha="center", va="center", zorder=3)

    # ----------------------------------------------------------------------
    # Right: output label
    # ----------------------------------------------------------------------
    ax.text(10.35, box_y + box_h / 2, "Negative", fontsize=13, ha="left", va="center")

    # ----------------------------------------------------------------------
    # Arrows
    # ----------------------------------------------------------------------
    def block_arrow(x_start, x_end, y):
        ax.add_patch(FancyArrow(
            x_start, y, x_end - x_start, 0,
            width=0.11, head_width=0.34, head_length=0.22,
            length_includes_head=True,
            facecolor=BLUE, edgecolor=BLUE, zorder=2,
        ))

    block_arrow(4.85, 6.35, box_y + box_h / 2)
    block_arrow(9.5, 10.15, box_y + box_h / 2)

    # ----------------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------------
    ax.set_xlim(-0.4, 11.2)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    plt.tight_layout()


def unrolled_rnn_sentiment(figsize=(15, 6)):
    """
    rnn_unrolled_classifier.py

    Recreates an unrolled-RNN diagram for sentiment classification:
    input words -> x_t vectors -> recurrent hidden states h_t (chained by W^H,
    fed by W^X) -> ... -> Binary Softmax Classifier -> output probabilities,
    with a curly brace marking "Max Sequence Length" under the input words.
    """



    BLUE = "#AFC6E9"
    BLUE_EDGE = "#5B87B5"
    ORANGE = "#F2A97E"
    ORANGE_EDGE = "#D97F44"
    YELLOW = "#F5E08A"
    YELLOW_EDGE = "#C9A72E"
    ARROW_BLUE = "#5B9BD5"
    BLACK = "black"
    fig, ax = plt.subplots(figsize=figsize)

    # ----------------------------------------------------------------------
    # Column layout for the three explicit time steps
    # ----------------------------------------------------------------------
    col_x = [1.0, 4.2, 7.4]              # x-centers for t-1, t, t+1
    h_labels = [r"$h_{t-1}$", r"$h_t$", r"$h_{t+1}$"]
    x_labels = [r"$x_{t-1}$", r"$x_t$", r"$x_{t+1}$"]
    wx_labels = [r"$W^{X}_{t-1}$", r"$W^{X}_t$", r"$W^{X}_{t+1}$"]
    words = ["The", "movie", "was"]

    h_w, h_h = 1.0, 1.5
    h_y = 3.1
    x_w, x_h = 1.3, 0.62
    x_y = 0.55

    # ----------------------------------------------------------------------
    # Draw hidden-state cells (orange rounded boxes) + input cells (blue)
    # ----------------------------------------------------------------------
    for i, xc in enumerate(col_x):
        # hidden state box
        hbox = FancyBboxPatch(
            (xc - h_w / 2, h_y), h_w, h_h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.3, edgecolor=ORANGE_EDGE, facecolor=ORANGE, zorder=3,
        )
        ax.add_patch(hbox)
        ax.text(xc, h_y + h_h / 2, h_labels[i], fontsize=15, ha="center", va="center", zorder=4)

        # input box
        xbox = Rectangle(
            (xc - x_w / 2, x_y), x_w, x_h,
            linewidth=1.3, edgecolor=BLUE_EDGE, facecolor=BLUE, zorder=3,
        )
        ax.add_patch(xbox)
        ax.text(xc, x_y + x_h / 2, x_labels[i], fontsize=14, ha="center", va="center", zorder=4)

        # word label under input box
        ax.text(xc, x_y - 0.35, words[i], fontsize=15, ha="center", va="center")

        # arrow: x box (top) -> straight up into hidden box (bottom)
        wx_arrow = FancyArrowPatch(
            (xc - h_w * 0.22, x_y + x_h), (xc - h_w * 0.22, h_y),
            arrowstyle="-|>", mutation_scale=16,
            linewidth=1.6, color=ARROW_BLUE, zorder=2,
            shrinkA=0, shrinkB=0,
        )
        ax.add_patch(wx_arrow)
        ax.text(xc - 0.85, (x_y + x_h + h_y) / 2, wx_labels[i],
                fontsize=13, ha="center", va="center")

    # ----------------------------------------------------------------------
    # W^H arrows chaining the hidden states
    # ----------------------------------------------------------------------
    def straight_arrow(x_from, x_to, y, color=ARROW_BLUE, lw=1.8, mscale=16):
        a = FancyArrowPatch(
            (x_from, y), (x_to, y),
            arrowstyle="-|>", mutation_scale=mscale,
            linewidth=lw, color=color, zorder=2,
            shrinkA=0, shrinkB=0,
        )
        ax.add_patch(a)


    h_mid_y = h_y + h_h * 0.72

    # arrow entering the first hidden cell from the left
    straight_arrow(col_x[0] - h_w / 2 - 0.9, col_x[0] - h_w / 2, h_mid_y)
    ax.text((col_x[0] - h_w / 2 - 0.9 + col_x[0] - h_w / 2) / 2, h_mid_y + 0.28,
            r"$W^{H}$", fontsize=14, ha="center", va="center")

    for i in range(len(col_x) - 1):
        x_from = col_x[i] + h_w / 2
        x_to = col_x[i + 1] - h_w / 2
        straight_arrow(x_from, x_to, h_mid_y)
        ax.text((x_from + x_to) / 2, h_mid_y + 0.28, r"$W^{H}$",
                fontsize=14, ha="center", va="center")

    # ----------------------------------------------------------------------
    # Ellipsis, then arrow to classifier, then arrow to output
    # ----------------------------------------------------------------------
    last_x = col_x[-1] + h_w / 2
    dots_x = last_x + 1.1
    straight_arrow(last_x, dots_x - 0.35, h_mid_y)

    for dx in (-0.18, 0.0, 0.18):
        ax.plot(dots_x + dx, h_mid_y, marker="o", markersize=6, color="black", zorder=4)

    clf_x0 = dots_x + 0.55
    clf_w, clf_h = 2.1, 1.7
    clf_y = h_mid_y - clf_h / 2
    straight_arrow(dots_x + 0.35, clf_x0, h_mid_y)

    clf_box = Rectangle(
        (clf_x0, clf_y), clf_w, clf_h,
        linewidth=1.3, edgecolor=YELLOW_EDGE, facecolor=YELLOW, zorder=3,
    )
    ax.add_patch(clf_box)
    ax.text(clf_x0 + clf_w / 2, clf_y + clf_h / 2,
            "Binary\nSoftmax\nClassifier", fontsize=14, ha="center", va="center", zorder=4)

    out_x0 = clf_x0 + clf_w + 0.9
    straight_arrow(clf_x0 + clf_w, out_x0 - 0.15, h_mid_y)
    ax.text(out_x0, h_mid_y, "[ 0.09, .91 ]", fontsize=15, ha="left", va="center")

    # ----------------------------------------------------------------------
    # Curly brace under the input words + "Max Sequence Length" label
    # ----------------------------------------------------------------------
    def curly_brace(ax, x_start, x_end, y_top, depth=0.35, lw=1.8, color="black"):
        x_mid = (x_start + x_end) / 2
        dx = (x_end - x_start) * 0.22

        verts = [
            (x_start, y_top),
            (x_start, y_top - depth * 0.9), (x_mid - dx, y_top - depth * 0.9), (x_mid, y_top - depth * 1.6),
            (x_mid + dx, y_top - depth * 0.9), (x_end, y_top - depth * 0.9), (x_end, y_top),
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                Path.CURVE4, Path.CURVE4, Path.LINETO]
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor="none", edgecolor=color, linewidth=lw, zorder=2)
        ax.add_patch(patch)
        return x_mid, y_top - depth * 1.6


    brace_y = x_y - 0.65
    brace_x_start = col_x[0] - x_w / 2
    brace_x_end = col_x[-1] + x_w / 2
    _, tip_y = curly_brace(ax, brace_x_start, brace_x_end, brace_y, depth=0.35)
    ax.text((brace_x_start + brace_x_end) / 2, tip_y - 0.35, "Max Sequence Length",
            fontsize=15, ha="center", va="center")

    # ----------------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------------
    ax.set_xlim(-1.2, out_x0 + 2.0)
    ax.set_ylim(tip_y - 0.9, h_y + h_h + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()


def rnn_language_model(figsize=(6, 6.5)):
    """
    rnn_language_model_diagram.py

    Recreates panel (b): a chain of embeddings e_{t-2..t} feeding through
    weight boxes W into hidden states h_{t-2..t}, chained by weight boxes U,
    with the final hidden state h_t projected through a flared weight box V
    into an output y_hat_t.
    """

    CYAN = "#AEEEEE"
    CYAN_EDGE = "#3AA0A0"
    WHITE = "white"
    EDGE = "black"

    fig, ax = plt.subplots(figsize=figsize)

    # ----------------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------------
    col_x = [0.0, 2.2, 4.4]                 # x positions for t-2, t-1, t
    e_labels = [r"$e_{t-2}$", r"$e_{t-1}$", r"$e_t$"]
    h_labels = [r"$h_{t-2}$", r"$h_{t-1}$", r"$h_t$"]

    e_y = 0.0
    w_y0, w_h = 0.75, 1.15
    h_y = 2.55
    ell_w, ell_h = 1.35, 0.75    # circle (ellipse) size
    sq_w = 0.9                   # W/U square width

    # ----------------------------------------------------------------------
    # Bottom row: e_t embeddings, W boxes, h_t hidden states
    # ----------------------------------------------------------------------
    for xc, e_lab, h_lab in zip(col_x, e_labels, h_labels):
        # embedding circle
        ax.add_patch(Ellipse((xc, e_y), ell_w, ell_h, facecolor=WHITE,
                            edgecolor=EDGE, linewidth=1.3, zorder=3))
        ax.text(xc, e_y, e_lab, fontsize=13, ha="center", va="center", zorder=4)

        # W square
        ax.add_patch(Rectangle((xc - sq_w / 2, w_y0), sq_w, w_h,
                                facecolor=CYAN, edgecolor=CYAN_EDGE, linewidth=1.3, zorder=2))
        ax.text(xc, w_y0 + w_h / 2, "W", fontsize=14, ha="center", va="center", zorder=4)

        # hidden state circle
        ax.add_patch(Ellipse((xc, h_y), ell_w, ell_h, facecolor=WHITE,
                            edgecolor=EDGE, linewidth=1.3, zorder=3))
        ax.text(xc, h_y, h_lab, fontsize=13, ha="center", va="center", zorder=4)

    # ----------------------------------------------------------------------
    # U boxes chaining the hidden states horizontally
    # ----------------------------------------------------------------------
    u_w = 0.6
    for i in range(len(col_x) - 1):
        xc = (col_x[i] + col_x[i + 1]) / 2
        ax.add_patch(Rectangle((xc - u_w / 2, h_y - u_w / 2), u_w, u_w,
                                facecolor=CYAN, edgecolor=CYAN_EDGE, linewidth=1.3, zorder=4))
        ax.text(xc, h_y, "U", fontsize=13, ha="center", va="center", zorder=5)

    # ----------------------------------------------------------------------
    # V trapezoid (flares wider toward the top) above the last hidden state,
    # and the final output circle y_hat_t
    # ----------------------------------------------------------------------
    xc = col_x[-1]
    v_y0, v_y1 = h_y + ell_h / 2 + 0.15, h_y + ell_h / 2 + 1.7
    v_bottom_w, v_top_w = sq_w, 1.5

    v_poly = Polygon(
        [
            (xc - v_bottom_w / 2, v_y0),
            (xc + v_bottom_w / 2, v_y0),
            (xc + v_top_w / 2, v_y1),
            (xc - v_top_w / 2, v_y1),
        ],
        closed=True, facecolor=CYAN, edgecolor=CYAN_EDGE, linewidth=1.3, zorder=2,
    )
    ax.add_patch(v_poly)
    ax.text(xc, (v_y0 + v_y1) / 2, "V", fontsize=14, ha="center", va="center", zorder=4)

    y_hat_y = v_y1 + ell_h / 2 + 0.15
    ax.add_patch(Ellipse((xc, y_hat_y), ell_w, ell_h, facecolor=WHITE,
                        edgecolor=EDGE, linewidth=1.3, zorder=3))
    ax.text(xc, y_hat_y, r"$\hat{y}_t$", fontsize=13, ha="center", va="center", zorder=4)

    # ----------------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------------
    ax.set_xlim(col_x[0] - 1.6, col_x[-1] + 1.4)
    ax.set_ylim(e_y - ell_h / 2 - 0.3, y_hat_y + ell_h / 2 + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
