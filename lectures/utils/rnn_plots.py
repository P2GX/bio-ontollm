from matplotlib import patches, path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import Circle,Rectangle, FancyArrow, FancyBboxPatch,  FancyArrowPatch, Polygon,FancyBboxPatch, Rectangle, Ellipse, PathPatch
import matplotlib.patches as mpatches



ARROW_BLUE = "#3C6FB0"
PURPLE = "#8C8CE0"
PURPLE_EDGE = "#5A5AC0"
RED_SHADES = ["#8B1A1A", "#C0392B", "#E63946"]
WHITE = "white"
BLUE = "#9DC3E6"
BLUE_CELL = "#AACBEA"
BLUE_EDGE = "black"
ORANGE = "#F2B98B"
ORANGE_EDGE = "#2255AA"
GREEN = "#A9D18E"
GREEN_CELL = "#8FCB7A"
GREEN_EDGE = "#4E9A33"
PURPLE = "#B7B7EA"
PURPLE_EDGE = "#8E8ED6"
PINK = "#F4B6B6"
PINK_EDGE = "#D98888"
WHITE = "white"
BLACK = "black"

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

def rnn_next_word(figsize=(14, 6.5)):
    """
    Recreates the full RNN-language-model training diagram: a chain of input
    word embeddings feeding into RNN cells (inside a shared "RNN" block),
    each producing a softmax distribution over the vocabulary, each compared
    against the true next word via a per-step cross-entropy loss box, with
    the average loss formula at the right.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ----------------------------------------------------------------------
    # Columns / words
    # ----------------------------------------------------------------------
    words_in = ["So", "long", "and", "thanks", "for"]
    words_next = ["long", "and", "thanks", "for", "all"]
    col_x = [1.6, 3.5, 5.4, 7.3, 9.2]
    dots_x = 10.7

    # ----------------------------------------------------------------------
    # Row y-coordinates
    # ----------------------------------------------------------------------
    word_y = 0.0
    emb_y0, emb_y1 = 0.55, 1.55          # embedding capsule span
    rnn_y0, rnn_y1 = 2.35, 3.15          # RNN square span
    sm_y0, sm_y1 = 3.75, 4.55            # softmax icon span
    loss_y0, loss_y1 = 5.15, 5.85        # loss box span
    nextword_y = 6.35

    sq_w = 0.62   # RNN square width
    emb_w = 0.42  # embedding capsule width

    # ----------------------------------------------------------------------
    # Shared purple background block (covers RNN + softmax rows)
    # ----------------------------------------------------------------------
    block_x0 = col_x[0] - 0.85
    block_x1 = dots_x + 0.55
    block = FancyBboxPatch(
        (block_x0, rnn_y0 - 0.35), block_x1 - block_x0, (sm_y1 + 0.3) - (rnn_y0 - 0.35),
        boxstyle="round,pad=0.02,rounding_size=0.3",
        linewidth=1.2, edgecolor=PURPLE_EDGE, facecolor=PURPLE, zorder=1,
    )
    ax.add_patch(block)

    # ----------------------------------------------------------------------
    # Helper: small histogram/bar-chart icon
    # ----------------------------------------------------------------------
    rng = np.random.default_rng(3)

    def histogram_icon(ax, xc, y0, y1, w=0.55):
        box = FancyBboxPatch(
            (xc - w / 2, y0), w, y1 - y0,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            linewidth=1.0, edgecolor="#999999", facecolor=WHITE, zorder=3,
        )
        ax.add_patch(box)
        n_bars = 6
        heights = rng.uniform(0.25, 0.85, n_bars) * (y1 - y0 - 0.12)
        bar_w = (w - 0.14) / n_bars
        base = y0 + 0.06
        for i, h in enumerate(heights):
            bx = xc - w / 2 + 0.07 + i * bar_w
            ax.add_patch(Rectangle((bx, base), bar_w * 0.7, h,
                                    facecolor="black", edgecolor="none", zorder=4))

    # ----------------------------------------------------------------------
    # Helper: straight arrow
    # ----------------------------------------------------------------------
    def arrow(xy_from, xy_to, color="black", lw=1.5, mscale=14, zorder=3):
        a = FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=mscale,
                            linewidth=lw, color=color, zorder=zorder,
                            shrinkA=0, shrinkB=0)
        ax.add_patch(a)

    # ----------------------------------------------------------------------
    # Build each column
    # ----------------------------------------------------------------------
    for i, xc in enumerate(col_x):
        # word (bottom)
        ax.text(xc, word_y, words_in[i], fontsize=13, ha="center", va="center")

        # embedding capsule with 3 stacked dots
        capsule = FancyBboxPatch(
            (xc - emb_w / 2, emb_y0), emb_w, emb_y1 - emb_y0,
            boxstyle="round,pad=0.0,rounding_size=0.2",
            linewidth=1.2, edgecolor="#555555", facecolor="none", zorder=3,
        )
        ax.add_patch(capsule)
        dot_ys = np.linspace(emb_y0 + 0.22, emb_y1 - 0.22, 3)
        for dy, color in zip(dot_ys, RED_SHADES):
            ax.add_patch(Ellipse((xc, dy), 0.22, 0.22, facecolor=color,
                                edgecolor="black", linewidth=0.6, zorder=4))

        arrow((xc, emb_y1), (xc, rnn_y0))

        # RNN square
        rnn_box = Rectangle((xc - sq_w / 2, rnn_y0), sq_w, rnn_y1 - rnn_y0,
                            facecolor=WHITE, edgecolor="black", linewidth=1.2, zorder=3)
        ax.add_patch(rnn_box)

        # recurrent arrow to next RNN cell
        if i < len(col_x) - 1:
            arrow((xc + sq_w / 2, (rnn_y0 + rnn_y1) / 2),
                (col_x[i + 1] - sq_w / 2, (rnn_y0 + rnn_y1) / 2))
        if i == 0:
            ax.text((xc + col_x[1]) / 2 - 0.35, (rnn_y0 + rnn_y1) / 2 + 0.25,
                    "h", fontsize=13, ha="center", va="center", style="italic")

        arrow((xc, rnn_y1), (xc, sm_y0))
        if i == 0:
            ax.text(xc - 0.55, (rnn_y1 + sm_y0) / 2, "Vh", fontsize=12,
                    ha="center", va="center", style="italic")

        # softmax histogram icon
        histogram_icon(ax, xc, sm_y0, sm_y1)

        arrow((xc, sm_y1), (xc, loss_y0))
        if i == 0:
            ax.text(xc - 0.3, (sm_y1 + loss_y0) / 2, "$y$", fontsize=13,
                    ha="center", va="center", style="italic")

        # loss box
        loss_box = FancyBboxPatch(
            (xc - 0.55, loss_y0), 1.1, loss_y1 - loss_y0,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.0, edgecolor=PINK_EDGE, facecolor=PINK, zorder=3,
        )
        ax.add_patch(loss_box)
        ax.text(xc, (loss_y0 + loss_y1) / 2, rf"$-\log \hat{{y}}_{{\mathrm{{{words_next[i]}}}}}$",
                fontsize=10.5, ha="center", va="center", zorder=4)

        arrow((xc, loss_y1), (xc, nextword_y - 0.25))

        # next-word label
        ax.text(xc, nextword_y, words_next[i], fontsize=13, ha="center", va="center")

    # ----------------------------------------------------------------------
    # Ellipsis column ("...") at each row
    # ----------------------------------------------------------------------
    for y in [word_y, (emb_y0 + emb_y1) / 2, (rnn_y0 + rnn_y1) / 2,
            (sm_y0 + sm_y1) / 2, (loss_y0 + loss_y1) / 2, nextword_y]:
        ax.text(dots_x, y, "...", fontsize=16, ha="center", va="center", zorder=3)

    # extend the recurrent arrow from the last RNN cell to the dots
    arrow((col_x[-1] + sq_w / 2, (rnn_y0 + rnn_y1) / 2),
        (dots_x - 0.3, (rnn_y0 + rnn_y1) / 2))

    # ----------------------------------------------------------------------
    # Average-loss formula, far right
    # ----------------------------------------------------------------------
    formula_x = dots_x + 2.0
    ax.text(formula_x, (loss_y0 + loss_y1) / 2,
            r"$\dfrac{1}{T}\sum_{t=1}^{T} L_{CE}$",
            fontsize=15, ha="center", va="center")

    # ----------------------------------------------------------------------
    # Row labels (left margin)
    # ----------------------------------------------------------------------
    row_labels = [
        ("Next word", nextword_y),
        ("Loss", (loss_y0 + loss_y1) / 2),
        ("Softmax over\nVocabulary", (sm_y0 + sm_y1) / 2),
        ("RNN", (rnn_y0 + rnn_y1) / 2),
        ("Input\nEmbeddings", (emb_y0 + emb_y1) / 2),
    ]
    label_x = block_x0 - 0.5
    for text, y in row_labels:
        ax.text(label_x, y, text, fontsize=12, ha="right", va="center", zorder=5)

    # ----------------------------------------------------------------------
    # Outer border
    # ----------------------------------------------------------------------
    xlim = (label_x - 2.4, formula_x + 1.5)
    ylim = (word_y - 0.6, nextword_y + 0.6)
    border = Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
                        facecolor="none", edgecolor="black", linewidth=1.2, zorder=0)
    ax.add_patch(border)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()


def rnn_autoregression(figsize=(9, 5.2)):
    """
    rnn_autoregressive_generation.py

    Recreates a textbook-style figure: autoregressive generation with an
    RNN-based neural language model. Each timestep column shows an input
    word -> embedding -> RNN cell -> softmax -> sampled word, with dashed
    vertical separators between timesteps and a figure caption at the bottom.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ----------------------------------------------------------------------
    # Columns / words
    # ----------------------------------------------------------------------
    words_in = ["<s>", "So", "long", "and"]
    words_sampled = ["So", "long", "and", "?"]
    col_x = [1.75, 3.45, 5.15, 6.85]

    # ----------------------------------------------------------------------
    # Row y-coordinates
    # ----------------------------------------------------------------------
    input_y = 0.0
    emb_y0, emb_y1 = 0.55, 1.35
    rnn_y0, rnn_y1 = 2.15, 2.95
    sm_y0, sm_y1 = 3.55, 4.2
    sampled_y = 4.85

    sq_w = 0.55

    # ----------------------------------------------------------------------
    # Shared purple RNN background block
    # ----------------------------------------------------------------------
    block_x0 = col_x[0] - 1.15
    block_x1 = col_x[-1] + 1.35
    block = FancyBboxPatch(
        (block_x0, rnn_y0 - 0.35), block_x1 - block_x0, (rnn_y1 - rnn_y0) + 0.7,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        linewidth=1.2, edgecolor=PURPLE_EDGE, facecolor=PURPLE, zorder=1,
    )
    ax.add_patch(block)
    ax.text(block_x0 + 0.45, (rnn_y0 + rnn_y1) / 2, "RNN", fontsize=15,
            ha="center", va="center", fontweight="bold", zorder=2, color="black")

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    rng = np.random.default_rng(5)

    def histogram_pill(ax, xc, y0, y1, w=1.0):
        h = y1 - y0
        box = FancyBboxPatch(
            (xc - w / 2, y0), w, h,
            boxstyle="round,pad=0.01,rounding_size=" + str(h / 2),
            linewidth=1.1, edgecolor="black", facecolor=WHITE, zorder=3,
        )
        ax.add_patch(box)
        n_bars = 6
        heights = rng.uniform(0.25, 0.85, n_bars) * (h - 0.14)
        bar_w = (w - 0.3) / n_bars
        base = y0 + 0.07
        for i, hh in enumerate(heights):
            bx = xc - w / 2 + 0.15 + i * bar_w
            ax.add_patch(Rectangle((bx, base), bar_w * 0.65, hh,
                                    facecolor="black", edgecolor="none", zorder=4))


    def arrow(xy_from, xy_to, color="black", lw=1.5, mscale=14, zorder=3):
        a = FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=mscale,
                            linewidth=lw, color=color, zorder=zorder,
                            shrinkA=0, shrinkB=0)
        ax.add_patch(a)

    # ----------------------------------------------------------------------
    # Build each column
    # ----------------------------------------------------------------------
    for i, xc in enumerate(col_x):
        ax.text(xc, input_y, words_in[i], fontsize=14, ha="center", va="center")

        # embedding: 3 stacked dots, no outline
        dot_ys = np.linspace(emb_y0 + 0.14, emb_y1 - 0.14, 3)
        for dy, color in zip(dot_ys, RED_SHADES):
            ax.add_patch(Ellipse((xc, dy), 0.26, 0.26, facecolor=color,
                                edgecolor="black", linewidth=0.7, zorder=4))

        arrow((xc, emb_y1 + 0.05), (xc, rnn_y0))

        # RNN square
        rnn_box = Rectangle((xc - sq_w / 2, rnn_y0), sq_w, rnn_y1 - rnn_y0,
                            facecolor=WHITE, edgecolor="black", linewidth=1.2, zorder=3)
        ax.add_patch(rnn_box)

        if i < len(col_x) - 1:
            arrow((xc + sq_w / 2, (rnn_y0 + rnn_y1) / 2),
                (col_x[i + 1] - sq_w / 2, (rnn_y0 + rnn_y1) / 2))

        arrow((xc, rnn_y1), (xc, sm_y0))

        histogram_pill(ax, xc, sm_y0, sm_y1)

        arrow((xc, sm_y1), (xc, sampled_y - 0.3))

        ax.text(xc, sampled_y, words_sampled[i], fontsize=15, ha="center", va="center")

    # arrow continuing right past the last RNN cell
    arrow((col_x[-1] + sq_w / 2, (rnn_y0 + rnn_y1) / 2),
        (block_x1 - 0.15, (rnn_y0 + rnn_y1) / 2))

    # ----------------------------------------------------------------------
    # Dashed vertical timestep separators
    # ----------------------------------------------------------------------
    sep_top = sampled_y + 0.45
    sep_bottom = input_y - 0.35
    sep_xs = [(col_x[i] + col_x[i + 1]) / 2 for i in range(len(col_x) - 1)]
    for sx in sep_xs:
        ax.plot([sx, sx], [sep_bottom, sep_top], linestyle="--", color="black",
                linewidth=1.1, zorder=5)

    # ----------------------------------------------------------------------
    # Row labels (left margin)
    # ----------------------------------------------------------------------
    row_labels = [
        ("Sampled Word", sampled_y),
        ("Softmax", (sm_y0 + sm_y1) / 2),
        ("Embedding", (emb_y0 + emb_y1) / 2),
        ("Input Word", input_y),
    ]
    label_x = block_x0 - 0.5
    for text, y in row_labels:
        ax.text(label_x, y, text, fontsize=13, ha="right", va="center", zorder=5)

    # ----------------------------------------------------------------------
    # Outer border + caption
    # ----------------------------------------------------------------------
    xlim = (label_x - 2.3, block_x1 + 0.4)
    ylim = (sep_bottom - 0.9, sep_top + 0.15)


    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

def encode_decode(figsize=(12, 3.2)):
    """
    encoder_decoder_pipeline.py

    Recreates a simple encoder-decoder diagram:
    Input Text -> [Encoder] -> Context Vector -> [Decoder] -> Summary
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ----------------------------------------------------------------------
    # Input Text box
    # ----------------------------------------------------------------------
    in_x, in_y, in_w, in_h = 0.0, 0.85, 1.9, 0.7
    ax.add_patch(Rectangle((in_x, in_y), in_w, in_h,
                            facecolor="white", edgecolor="black", linewidth=1.5, zorder=3))
    ax.text(in_x + in_w / 2, in_y + in_h / 2, "Input Text ..", fontsize=15,
            ha="center", va="center", zorder=4)

    # ----------------------------------------------------------------------
    # Encoder box
    # ----------------------------------------------------------------------
    enc_x, enc_y, enc_w, enc_h = 3.1, 0.15, 2.9, 2.0
    ax.add_patch(FancyBboxPatch(
        (enc_x, enc_y), enc_w, enc_h,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        linewidth=1.8, edgecolor=GREEN_EDGE, facecolor=GREEN, zorder=3,
    ))
    ax.text(enc_x + enc_w / 2, enc_y + enc_h / 2, "Encoder", fontsize=18,
            ha="center", va="center", fontweight="bold", zorder=4)

    # ----------------------------------------------------------------------
    # Context Vector box
    # ----------------------------------------------------------------------
    ctx_x, ctx_y, ctx_w, ctx_h = 6.85, 0.35, 0.55, 1.6
    ax.add_patch(Rectangle((ctx_x, ctx_y), ctx_w, ctx_h,
                            facecolor=ORANGE, edgecolor=ORANGE_EDGE, linewidth=1.8, zorder=3))
    values = ["0.1", "0.8", "-0.3", "0.6", "0.1"]
    val_ys = [ctx_y + ctx_h * (0.9 - 0.2 * i) for i in range(len(values))]
    for v, vy in zip(values, val_ys):
        ax.text(ctx_x + ctx_w / 2, vy, v, fontsize=11, ha="center", va="center",
                fontweight="bold", zorder=4)

    ax.text(ctx_x + ctx_w / 2, ctx_y - 0.35, "Context Vector", fontsize=15,
            ha="center", va="center", fontweight="bold")

    # ----------------------------------------------------------------------
    # Decoder box
    # ----------------------------------------------------------------------
    dec_x, dec_y, dec_w, dec_h = 7.9, 0.15, 2.9, 2.0
    ax.add_patch(FancyBboxPatch(
        (dec_x, dec_y), dec_w, dec_h,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        linewidth=1.8, edgecolor=BLUE_EDGE, facecolor=BLUE, zorder=3,
    ))
    ax.text(dec_x + dec_w / 2, dec_y + dec_h / 2, "Decoder", fontsize=18,
            ha="center", va="center", fontweight="bold", zorder=4)

    # ----------------------------------------------------------------------
    # Summary box
    # ----------------------------------------------------------------------
    sum_x, sum_y, sum_w, sum_h = 11.6, 0.85, 1.75, 0.7
    ax.add_patch(Rectangle((sum_x, sum_y), sum_w, sum_h,
                            facecolor="white", edgecolor="black", linewidth=1.5, zorder=3))
    ax.text(sum_x + sum_w / 2, sum_y + sum_h / 2, "Output...", fontsize=15,
            ha="center", va="center", zorder=4)

    # ----------------------------------------------------------------------
    # Block arrows
    # ----------------------------------------------------------------------
    def block_arrow(x_start, x_end, y):
        ax.add_patch(FancyArrow(
            x_start, y, x_end - x_start, 0,
            width=0.16, head_width=0.42, head_length=0.22,
            length_includes_head=True,
            facecolor=ARROW_BLUE, edgecolor=ARROW_BLUE, zorder=2,
        ))

    mid_y = 1.2
    block_arrow(in_x + in_w + 0.1, enc_x - 0.1, mid_y)
    block_arrow(enc_x + enc_w + 0.1, ctx_x - 0.1, mid_y)
    block_arrow(ctx_x + ctx_w + 0.1, dec_x - 0.1, mid_y)
    block_arrow(dec_x + dec_w + 0.1, sum_x - 0.1, mid_y)

    # ----------------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------------
    ax.set_xlim(-0.3, sum_x + sum_w + 0.3)
    ax.set_ylim(-0.7, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

def doughnut(figsize=(15, 6.5)):
    """
    encoder_decoder_mt_diagram.py

    Recreates the classic encoder-decoder machine-translation figure:
    source words (English) are encoded by an RNN into a final hidden state
    h_n, which then seeds an autoregressive decoder that predicts target
    words (Spanish) one at a time, each predicted word being fed back in
    as the next input word.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ----------------------------------------------------------------------
    # Columns
    # ----------------------------------------------------------------------
    src_words = ["Ich", "bin", "ein", "Berliner"]
    tgt_input_words = ["<s>", "I", "am", "a", "doughnut"]
    tgt_pred_words = ["I", "am", "a", "doughnut", "</s>"]

    n_src, n_tgt = len(src_words), len(tgt_input_words)
    col_x = [1.0 + 1.55 * i for i in range(n_src)]
    gap = 0.9
    tgt_start = col_x[-1] + 1.55 + gap
    col_x += [tgt_start + 1.55 * i for i in range(n_tgt)]

    # ----------------------------------------------------------------------
    # Row y-coordinates
    # ----------------------------------------------------------------------
    word_y = 0.0
    emb_y0, emb_y1 = 0.5, 1.3
    hid_y0, hid_y1 = 1.95, 2.65
    sm_y0, sm_y1 = 3.35, 4.0
    pred_y = 4.6
    sq_w = 0.65

    # ----------------------------------------------------------------------
    # Purple background block (spans hidden-layer row across all columns)
    # ----------------------------------------------------------------------
    block_x0 = col_x[0] - 0.85
    block_x1 = col_x[-1] + 0.85
    block = FancyBboxPatch(
        (block_x0, hid_y0 - 0.35), block_x1 - block_x0, (hid_y1 - hid_y0) + 0.7,
        boxstyle="round,pad=0.02,rounding_size=0.3",
        linewidth=1.2, edgecolor=PURPLE_EDGE, facecolor=PURPLE, zorder=1,
    )
    ax.add_patch(block)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    rng = np.random.default_rng(9)

    def arrow(xy_from, xy_to, lw=1.6, mscale=13, color="black", zorder=3, style="-"):
        a = FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=mscale,
                            linewidth=lw, color=color, zorder=zorder,
                            linestyle=style, shrinkA=0, shrinkB=0)
        ax.add_patch(a)


    def histogram_pill(ax, xc, y0, y1, w=0.85):
        h = y1 - y0
        box = FancyBboxPatch(
            (xc - w / 2, y0), w, h,
            boxstyle="round,pad=0.01,rounding_size=" + str(h / 2),
            linewidth=1.0, edgecolor="black", facecolor=WHITE, zorder=3,
        )
        ax.add_patch(box)
        n_bars = 6
        heights = rng.uniform(0.25, 0.85, n_bars) * (h - 0.12)
        bar_w = (w - 0.26) / n_bars
        base = y0 + 0.06
        for i, hh in enumerate(heights):
            bx = xc - w / 2 + 0.13 + i * bar_w
            ax.add_patch(Rectangle((bx, base), bar_w * 0.65, hh,
                                    facecolor="black", edgecolor="none", zorder=4))


    def curly_brace(ax, x_start, x_end, y_ref, depth=0.32, lw=1.6, direction="down"):
        """direction='down': tip points below y_ref (brace sits above its label).
        direction='up':   tip points above y_ref (brace sits below its label)."""
        sign = -1 if direction == "down" else 1
        x_mid = (x_start + x_end) / 2
        dx = (x_end - x_start) * 0.22
        verts = [
            (x_start, y_ref),
            (x_start, y_ref + sign * depth * 0.9), (x_mid - dx, y_ref + sign * depth * 0.9),
            (x_mid, y_ref + sign * depth * 1.6),
            (x_mid + dx, y_ref + sign * depth * 0.9), (x_end, y_ref + sign * depth * 0.9),
            (x_end, y_ref),
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                Path.CURVE4, Path.CURVE4, Path.LINETO]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor="none",
                                edgecolor="black", linewidth=lw, zorder=2))
        return x_mid, y_ref + sign * depth * 1.6

    # ----------------------------------------------------------------------
    # Build all 9 columns
    # ----------------------------------------------------------------------
    all_words_bottom = src_words + tgt_input_words
    is_source = [True] * n_src + [False] * n_tgt

    for i, xc in enumerate(col_x):
        color = "#2255CC" if is_source[i] else "#CC2222"
        ax.text(xc, word_y, all_words_bottom[i], fontsize=13, ha="center", va="center",
                color=color, fontweight="bold")

        dot_ys = np.linspace(emb_y0 + 0.13, emb_y1 - 0.13, 3)
        for dy, c in zip(dot_ys, RED_SHADES):
            ax.add_patch(Ellipse((xc, dy), 0.24, 0.24, facecolor=c, edgecolor="black",
                                linewidth=0.6, zorder=4))
        arrow((xc, emb_y1 + 0.05), (xc, hid_y0))

        is_hn = (i == n_src - 1)
        cell_color = GREEN_CELL if is_hn else BLUE_CELL
        cell_edge = GREEN_EDGE if is_hn else BLUE_EDGE
        ax.add_patch(Rectangle((xc - sq_w / 2, hid_y0), sq_w, hid_y1 - hid_y0,
                                facecolor=cell_color, edgecolor=cell_edge, linewidth=1.4, zorder=3))
        if is_hn:
            ax.text(xc, (hid_y0 + hid_y1) / 2, r"$h_n$", fontsize=13, ha="center",
                    va="center", zorder=4, color="black")

        if i < len(col_x) - 1:
            arrow((xc + sq_w / 2, (hid_y0 + hid_y1) / 2),
                (col_x[i + 1] - sq_w / 2, (hid_y0 + hid_y1) / 2))

        if not is_source[i]:
            arrow((xc, hid_y1), (xc, sm_y0))
            histogram_pill(ax, xc, sm_y0, sm_y1)
            arrow((xc, sm_y1), (xc, pred_y - 0.28))
            j = i - n_src
            ax.text(xc, pred_y, tgt_pred_words[j], fontsize=14, ha="center",
                    va="center", color="#CC2222", fontweight="bold")

    ax.text((col_x[0] + col_x[n_src - 1]) / 2, (sm_y0 + sm_y1) / 2,
            "(output of source is ignored)", fontsize=11, ha="center", va="center", zorder=3)

    # ----------------------------------------------------------------------
    # Dashed vertical separators between target columns
    # ----------------------------------------------------------------------
    sep_top = pred_y + 0.4
    sep_bottom = word_y - 0.3
    for i in range(n_src, len(col_x) - 1):
        sx = (col_x[i] + col_x[i + 1]) / 2
        ax.plot([sx, sx], [sep_bottom, sep_top], linestyle="--", color="black",
                linewidth=1.0, zorder=5)

    # curved dashed feedback arrows: predicted word (top) -> next input word (bottom)
    for i in range(n_src, len(col_x) - 1):
        x_from, x_to = col_x[i], col_x[i + 1]
        conn = FancyArrowPatch(
            (x_from, word_y + 0.12), (x_to, word_y + 0.12),
            connectionstyle="arc3,rad=-0.5",
            arrowstyle="-|>", mutation_scale=12, linewidth=1.1,
            linestyle="--", color="#444444", zorder=2,
        )
        ax.add_patch(conn)

    # ----------------------------------------------------------------------
    # Source Text brace (below words, tip down)
    # ----------------------------------------------------------------------
    src_x0 = col_x[0] - sq_w
    src_x1 = col_x[n_src - 1] + sq_w
    _, tip_y = curly_brace(ax, src_x0, src_x1, word_y - 0.35, depth=0.3, direction="down")
    ax.text((src_x0 + src_x1) / 2, tip_y - 0.3, "Source Text", fontsize=14,
            ha="center", va="center", fontweight="bold")

    # ----------------------------------------------------------------------
    # Target Text brace (above predicted words, tip down onto them)
    # ----------------------------------------------------------------------
    tgt_x0 = col_x[n_src] - sq_w
    tgt_x1 = col_x[-1] + sq_w
    _, tip_y2 = curly_brace(ax, tgt_x0, tgt_x1, pred_y + 0.4, depth=0.3, direction="up")
    ax.text((tgt_x0 + tgt_x1) / 2, tip_y2 + 0.3, "Target Text", fontsize=14,
            ha="center", va="center", fontweight="bold")

    # ----------------------------------------------------------------------
    # Separator label + arrow pointing at <s> column
    # ----------------------------------------------------------------------
    sep_x = col_x[n_src]
    ax.text(sep_x, word_y - 1.0, "Separator", fontsize=11, ha="center", va="center")
    arrow((sep_x, word_y - 0.85), (sep_x - 0.15, word_y - 0.15), lw=1.1, mscale=10)

    # ----------------------------------------------------------------------
    # Row labels
    # ----------------------------------------------------------------------
    row_labels = [
        ("softmax", (sm_y0 + sm_y1) / 2),
        ("hidden\nlayer(s)", (hid_y0 + hid_y1) / 2),
        ("embedding\nlayer", (emb_y0 + emb_y1) / 2),
    ]
    label_x = block_x0 - 0.6
    for text, y in row_labels:
        ax.text(label_x, y, text, fontsize=11, ha="right", va="center", zorder=5)

    # ----------------------------------------------------------------------
    # Outer border
    # ----------------------------------------------------------------------
    xlim = (label_x - 1.6, col_x[-1] + 1.3)
    ylim = (word_y - 1.6, tip_y2 + 1.0)
    ax.add_patch(Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
                            facecolor="none", edgecolor="black", linewidth=1.2, zorder=0))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()


def attention(figsize=(15, 7.5)):
    """
    attention_encoder_decoder_diagram.py

    Recreates the attention-based seq2seq diagram: encoder hidden states
    h^e_1..h^e_n, attention weights alpha_ij computed via dot product with
    the previous decoder state, combined into context vector c_i, which
    (together with the previous predicted word) drives the decoder hidden
    states h^d_{i-1}, h^d_i and their softmax predictions y_{i-1}, y_i.
    """

    BLUE_CELL = "#AAAAE8"
    BLUE_EDGE = "#5A5AC0"
    GREEN = "#7FE896"
    GREEN_EDGE = "#2E9E4A"
    GREEN_TEXT = "#1E7A34"
    RED_TEXT = "#CC2222"
    BLUE_TEXT = "#2255CC"
  

    fig, ax = plt.subplots(figsize=figsize)

    rng = np.random.default_rng(11)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def arrow(xy_from, xy_to, lw=1.5, mscale=13, color="black", zorder=3, style="-"):
        a = FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=mscale,
                            linewidth=lw, color=color, zorder=zorder,
                            linestyle=style, shrinkA=0, shrinkB=0)
        ax.add_patch(a)


    def histogram_pill(ax, xc, y0, y1, w=0.85):
        h = y1 - y0
        box = FancyBboxPatch(
            (xc - w / 2, y0), w, h,
            boxstyle="round,pad=0.01,rounding_size=" + str(h / 2),
            linewidth=1.0, edgecolor="black", facecolor=WHITE, zorder=3,
        )
        ax.add_patch(box)
        n_bars = 6
        heights = rng.uniform(0.25, 0.85, n_bars) * (h - 0.12)
        bar_w = (w - 0.26) / n_bars
        base = y0 + 0.06
        for i, hh in enumerate(heights):
            bx = xc - w / 2 + 0.13 + i * bar_w
            ax.add_patch(Rectangle((bx, base), bar_w * 0.65, hh,
                                    facecolor="black", edgecolor="none", zorder=4))


    def curly_brace(ax, x_start, x_end, y_ref, depth=0.32, lw=1.6, direction="down"):
        sign = -1 if direction == "down" else 1
        x_mid = (x_start + x_end) / 2
        dx = (x_end - x_start) * 0.22
        verts = [
            (x_start, y_ref),
            (x_start, y_ref + sign * depth * 0.9), (x_mid - dx, y_ref + sign * depth * 0.9),
            (x_mid, y_ref + sign * depth * 1.6),
            (x_mid + dx, y_ref + sign * depth * 0.9), (x_end, y_ref + sign * depth * 0.9),
            (x_end, y_ref),
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                Path.CURVE4, Path.CURVE4, Path.LINETO]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor="none",
                                edgecolor="black", linewidth=lw, zorder=2))
        return x_mid, y_ref + sign * depth * 1.6

    # ----------------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------------
    enc_x = [1.0, 2.5, 4.0, 5.9]
    enc_labels = [r"$h^e_1$", r"$h^e_2$", r"$h^e_3$", r"$h^e_n$"]
    x_labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_n$"]
    attn_vals = [".4", ".3", ".1", ".2"]

    hid_y0, hid_y1 = 2.15, 2.85
    attn_y = 4.1
    x_y = 1.2
    sq_w = 0.65

    ci_x, ci_y0, ci_y1 = 8.0, 3.55, 4.45
    ci_w = 0.75

    dec_x = [10.3, 11.9]
    dec_labels = [r"$h^d_{i-1}$", r"$h^d_i$"]
    pred_labels = [r"$y_{i-1}$", r"$y_i$"]
    y_in_labels = [r"$y_{i-2}$", r"$y_{i-1}$"]
    sm_y0, sm_y1 = attn_y - 0.35, attn_y + 0.35
    pred_y = 5.15

    # ----------------------------------------------------------------------
    # Encoder hidden states + inputs
    # ----------------------------------------------------------------------
    for i, xc in enumerate(enc_x):
        ax.text(xc, x_y - 0.55, x_labels[i], fontsize=13, ha="center", va="center", color=BLUE_TEXT)
        arrow((xc, x_y - 0.3), (xc, hid_y0))
        ax.add_patch(Rectangle((xc - sq_w / 2, hid_y0), sq_w, hid_y1 - hid_y0,
                                facecolor=BLUE_CELL, edgecolor=BLUE_EDGE, linewidth=1.4, zorder=3))
        ax.text(xc, (hid_y0 + hid_y1) / 2, enc_labels[i], fontsize=12, ha="center", va="center", zorder=4)
        if i < len(enc_x) - 1:
            arrow((xc + sq_w / 2, (hid_y0 + hid_y1) / 2), (enc_x[i + 1] - sq_w / 2, (hid_y0 + hid_y1) / 2))

    # arrow + dots continuing right of h^e_n
    arrow((enc_x[-1] + sq_w / 2, (hid_y0 + hid_y1) / 2), (enc_x[-1] + 0.85, (hid_y0 + hid_y1) / 2))
    ax.text(enc_x[-1] + 1.05, (hid_y0 + hid_y1) / 2, "...", fontsize=14, ha="center", va="center")

    # attention-weight circles + dashed connections
    for i, xc in enumerate(enc_x):
        ax.add_patch(Circle((xc, attn_y), 0.28, facecolor="none", edgecolor=GREEN_EDGE,
                            linewidth=1.4, linestyle="--", zorder=4))
        ax.text(xc, attn_y, attn_vals[i], fontsize=11, ha="center", va="center",
                color=GREEN_TEXT, zorder=5)
        # dashed arrow down into own h^e cell
        arrow((xc, attn_y - 0.28), (xc, hid_y1 + 0.03), lw=1.1, mscale=10,
            color=GREEN_EDGE, style="--", zorder=2)
        # solid green arrow up-right into c_i
        arrow((xc + 0.22, attn_y + 0.2), (ci_x - ci_w / 2 - 0.05, ci_y0 + (ci_y1 - ci_y0) * (0.3 + 0.15 * i)),
            lw=1.3, mscale=11, color=GREEN_EDGE, zorder=2)
        # dashed green arrow fanning down-right into decoder h^d_{i-1}
        arrow((xc + 0.15, attn_y - 0.22), (dec_x[0] - sq_w / 2 - 0.05, (hid_y0 + hid_y1) / 2 + 0.05),
            lw=1.0, mscale=9, color=GREEN_EDGE, style="--", zorder=2)

    ax.text((enc_x[0] + enc_x[-1]) / 2, attn_y + 0.9,
            r"$\sum_j \alpha_{ij} h^e_j$", fontsize=15, ha="center", va="center", color=GREEN_TEXT)
    ax.text((enc_x[1] + enc_x[2]) / 2 + 1.6, attn_y - 0.75,
            r"$h^d_{i-1} \cdot h^e_j$", fontsize=11, ha="center", va="center", color="black")
    ax.text(enc_x[0] - 0.95, attn_y, "attention\nweights\n" + r"$\alpha_{ij}$",
            fontsize=10.5, ha="center", va="center", color="black")

    # ----------------------------------------------------------------------
    # Context vector c_i
    # ----------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch(
        (ci_x - ci_w / 2, ci_y0), ci_w, ci_y1 - ci_y0,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.6, edgecolor=GREEN_EDGE, facecolor=GREEN, zorder=4,
    ))
    ax.text(ci_x, (ci_y0 + ci_y1) / 2, r"$c_i$", fontsize=14, ha="center", va="center", zorder=5)

    # c_i -> decoder h^d_{i-1} (solid black elbow arrow, built manually)
    elbow_mid = (ci_x, hid_y1 + 0.35)
    ax.plot([ci_x, ci_x], [ci_y0, elbow_mid[1]], color="black", linewidth=1.4, zorder=2)
    ax.plot([ci_x, dec_x[0]], [elbow_mid[1], elbow_mid[1]], color="black", linewidth=1.4, zorder=2)
    arrow((dec_x[0], elbow_mid[1]), (dec_x[0], hid_y1), lw=1.4, mscale=12)

    # ----------------------------------------------------------------------
    # Decoder: leading dots, hidden states, softmax, predictions
    # ----------------------------------------------------------------------
    ax.text(dec_x[0] - 1.0, (hid_y0 + hid_y1) / 2, "...", fontsize=14, ha="center", va="center")
    arrow((dec_x[0] - 0.8, (hid_y0 + hid_y1) / 2), (dec_x[0] - sq_w / 2, (hid_y0 + hid_y1) / 2))

    for i, xc in enumerate(dec_x):
        ax.add_patch(Rectangle((xc - sq_w / 2, hid_y0), sq_w, hid_y1 - hid_y0,
                                facecolor=BLUE_CELL, edgecolor=BLUE_EDGE, linewidth=1.4, zorder=3))
        ax.text(xc, (hid_y0 + hid_y1) / 2, dec_labels[i], fontsize=12, ha="center", va="center", zorder=4)
        if i < len(dec_x) - 1:
            arrow((xc + sq_w / 2, (hid_y0 + hid_y1) / 2), (dec_x[i + 1] - sq_w / 2, (hid_y0 + hid_y1) / 2))

        arrow((xc, hid_y1), (xc, sm_y0))
        histogram_pill(ax, xc, sm_y0, sm_y1)
        arrow((xc, sm_y1), (xc, pred_y - 0.28))
        ax.text(xc, pred_y, pred_labels[i], fontsize=14, ha="center", va="center",
                color=RED_TEXT, fontweight="bold")

        # y input from below
        ax.text(xc, x_y - 0.55, y_in_labels[i], fontsize=13, ha="center", va="center",
                color=RED_TEXT, fontweight="bold")
        arrow((xc, x_y - 0.3), (xc, hid_y0))

    # c_{i-1} / c_i small labels feeding decoder cells from lower-left
    ax.text(dec_x[0] - 0.55, x_y - 0.05, r"$c_{i-1}$", fontsize=10.5, ha="center", va="center")
    ax.text(dec_x[1] - 0.55, x_y - 0.05, r"$c_i$", fontsize=10.5, ha="center", va="center")

    # arrow + dots continuing right of h^d_i
    arrow((dec_x[-1] + sq_w / 2, (hid_y0 + hid_y1) / 2), (dec_x[-1] + 0.85, (hid_y0 + hid_y1) / 2))
    ax.text(dec_x[-1] + 1.05, (hid_y0 + hid_y1) / 2, "...", fontsize=14, ha="center", va="center")

    # dashed vertical separator before the shown decoder cells
    sep_x = dec_x[0] - 0.85
    ax.plot([sep_x, sep_x], [x_y - 1.0, pred_y + 0.55], linestyle="--", color="#888888",
            linewidth=1.3, zorder=1)

    # curved dashed feedback arrow (autoregressive), bottom of decoder
    fb = FancyArrowPatch(
        (dec_x[0], x_y - 0.75), (dec_x[1], x_y - 0.75),
        connectionstyle="arc3,rad=-0.4",
        arrowstyle="-|>", mutation_scale=11, linewidth=1.1,
        linestyle="--", color="#888888", zorder=2,
    )
    ax.add_patch(fb)

    # ----------------------------------------------------------------------
    # Encoder / Decoder braces + labels
    # ----------------------------------------------------------------------
    _, tip_y = curly_brace(ax, enc_x[0] - sq_w, enc_x[-1] + sq_w, x_y - 0.9, depth=0.28, direction="down")
    ax.text((enc_x[0] + enc_x[-1]) / 2, tip_y - 0.3, "Encoder", fontsize=15,
            ha="center", va="center", fontweight="bold")

    _, tip_y2 = curly_brace(ax, sep_x, dec_x[-1] + sq_w, pred_y + 0.55, depth=0.25, direction="up")
    ax.text((sep_x + dec_x[-1]) / 2, tip_y2 + 0.3, "Decoder", fontsize=15,
            ha="center", va="center", fontweight="bold")

    # ----------------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------------
    ax.set_xlim(enc_x[0] - 2.2, dec_x[-1] + 2.0)
    ax.set_ylim(x_y - 1.6, tip_y2 + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()


def rnn_vs_lstm_1():
    """
    rnn_colah_style_diagram.py

    Recreates the classic "unrolled RNN" diagram: three repeated cells (A),
    each taking x_t as input and producing h_t as output, with the middle
    cell expanded to show the internal routing through a tanh nonlinearity.
    """
    GREEN = "#DDEEC0"
    GREEN_EDGE = "#6B8E4E"
    BLUE = "#A9D6E8"
    BLUE_EDGE = "black"
    PURPLE = "#E3B8E8"
    PURPLE_EDGE = "black"
    YELLOW = "#F5DE9A"
    YELLOW_EDGE = "black"
    LINE = "black"

    fig, ax = plt.subplots(figsize=(13, 6))

    # ----------------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------------
    box_w, box_h = 2.6, 2.3
    gap = 0.55
    box_x0 = [0.0, box_w + gap, 2 * (box_w + gap)]
    box_y0, box_y1 = 0.0, box_h
    mid_y = (box_y0 + box_y1) / 2

    circle_r = 0.42
    x_circle_y = box_y0 - 1.15
    h_circle_y = box_y1 + 1.15

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def rounded_path_patch(ax, points, radius=0.18, lw=3.2, color=LINE, alpha=1.0, zorder=3):
        verts = [points[0]]
        codes = [Path.MOVETO]
        n = len(points)
        for i in range(1, n - 1):
            p_prev = np.array(points[i - 1], dtype=float)
            p_curr = np.array(points[i], dtype=float)
            p_next = np.array(points[i + 1], dtype=float)
            d1 = p_curr - p_prev
            d2 = p_next - p_curr
            len1, len2 = np.linalg.norm(d1), np.linalg.norm(d2)
            r = min(radius, len1 / 2, len2 / 2) if len1 > 0 and len2 > 0 else 0
            p_in = p_curr - d1 / len1 * r if len1 > 0 else p_curr
            p_out = p_curr + d2 / len2 * r if len2 > 0 else p_curr
            verts += [tuple(p_in), tuple(p_curr), tuple(p_out)]
            codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
        verts.append(points[-1])
        codes.append(Path.LINETO)
        patch = PathPatch(Path(verts, codes), facecolor="none", edgecolor=color,
                        linewidth=lw, alpha=alpha, capstyle="round", joinstyle="round", zorder=zorder)
        ax.add_patch(patch)


    def arrow_marker(ax, tip, direction, size=0.16, color=LINE, zorder=4):
        """direction: 'up', 'down', 'left', 'right'"""
        angle = {"up": 0, "right": -90, "down": 180, "left": 90}[direction]
        ax.plot(*tip, marker=(3, 0, angle), markersize=size * 90, color=color, zorder=zorder)


    def straight(ax, p0, p1, lw=3.2, color=LINE, zorder=3):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw,
                solid_capstyle="round", zorder=zorder)

    # ----------------------------------------------------------------------
    # Faint decorative background curves inside each box (purely cosmetic)
    # ----------------------------------------------------------------------
    def decorative_lines(ax, x0):
        pts_list = [
            [(x0, box_y0 + box_h * 0.62), (x0 + box_w * 0.18, box_y0 + box_h * 0.62),
            (x0 + box_w * 0.18, box_y0 + box_h * 0.78), (x0 + box_w * 0.75, box_y0 + box_h * 0.78)],
            [(x0, box_y0 + box_h * 0.30), (x0 + box_w * 0.28, box_y0 + box_h * 0.30),
            (x0 + box_w * 0.28, box_y0 + box_h * 0.15), (x0 + box_w * 0.85, box_y0 + box_h * 0.15)],
        ]
        for pts in pts_list:
            rounded_path_patch(ax, pts, radius=0.15, lw=1.8, color="#BBBBBB", alpha=0.35, zorder=2)

    # ----------------------------------------------------------------------
    # Draw the three green cells
    # ----------------------------------------------------------------------
    for x0 in box_x0:
        ax.add_patch(FancyBboxPatch(
            (x0, box_y0), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.28",
            linewidth=2.0, edgecolor=GREEN_EDGE, facecolor=GREEN, zorder=1,
        ))
        decorative_lines(ax, x0)

    # ----------------------------------------------------------------------
    # Left + right (collapsed) cells: "A" label, straight in/out lines
    # ----------------------------------------------------------------------
    for x0 in (box_x0[0], box_x0[2]):
        xc = x0 + box_w / 2
        ax.text(xc, mid_y, "A", fontsize=34, ha="center", va="center", fontweight="bold", zorder=3)

        # x_t input (straight, no arrowhead)
        ax.add_patch(Circle((xc, x_circle_y), circle_r, facecolor=BLUE, edgecolor=BLUE_EDGE,
                            linewidth=2.2, zorder=4))
        straight(ax, (xc, x_circle_y + circle_r), (xc, box_y0))

        # h_t output (straight, with arrowhead)
        ax.add_patch(Circle((xc, h_circle_y), circle_r, facecolor=PURPLE, edgecolor=PURPLE_EDGE,
                            linewidth=2.2, zorder=4))
        straight(ax, (xc, box_y1), (xc, h_circle_y - circle_r - 0.15))
        arrow_marker(ax, (xc, h_circle_y - circle_r - 0.02), "up")

    # horizontal pass-through arrows through/between all three boxes
    h_line_pts = [box_x0[0] - 0.6] + [x for x0 in box_x0 for x in (x0, x0 + box_w)] + [box_x0[2] + box_w + 0.6]
    # draw as separate straight segments with arrowheads just before each box entry
    straight(ax, (box_x0[0] - 0.6, mid_y), (box_x0[0], mid_y))
    for i in range(3):
        x0 = box_x0[i]
        x1 = x0 + box_w
        if i < 2:
            x_next = box_x0[i + 1]
            straight(ax, (x1, mid_y), (x_next, mid_y))
            arrow_marker(ax, (x_next - 0.02, mid_y), "right")
        else:
            straight(ax, (x1, mid_y), (x1 + 0.6, mid_y))
            arrow_marker(ax, (x1 + 0.58, mid_y), "right")

    # labels for x_{t-1}, h_{t-1}, x_{t+1}, h_{t+1}
    xc_left = box_x0[0] + box_w / 2
    xc_right = box_x0[2] + box_w / 2
    ax.text(xc_left, x_circle_y, r"$x_{t-1}$", fontsize=15, ha="center", va="center", zorder=5)
    ax.text(xc_left, h_circle_y, r"$h_{t-1}$", fontsize=15, ha="center", va="center", zorder=5)
    ax.text(xc_right, x_circle_y, r"$x_{t+1}$", fontsize=15, ha="center", va="center", zorder=5)
    ax.text(xc_right, h_circle_y, r"$h_{t+1}$", fontsize=15, ha="center", va="center", zorder=5)

    # ----------------------------------------------------------------------
    # Middle cell: detailed internal routing through tanh
    # ----------------------------------------------------------------------
    x0m = box_x0[1]
    xcm = x0m + box_w / 2
    xt_x = x0m + box_w * 0.28

    tanh_w, tanh_h = 0.62, 0.36
    tanh_cx = xcm + 0.05
    tanh_y0 = box_y0 + box_h * 0.30
    tanh_y1 = tanh_y0 + tanh_h

    # x_t circle + input line rising then curving right into tanh (from below-left)
    ax.add_patch(Circle((xt_x, x_circle_y), circle_r, facecolor=BLUE, edgecolor=BLUE_EDGE,
                        linewidth=2.2, zorder=4))
    ax.text(xt_x, x_circle_y, r"$x_t$", fontsize=15, ha="center", va="center", zorder=5)
    xt_path = [
        (xt_x, x_circle_y + circle_r),
        (xt_x, tanh_y0 - 0.55),
        (tanh_cx - 0.11, tanh_y0 - 0.55),
        (tanh_cx - 0.11, tanh_y0 - 0.02),
    ]
    rounded_path_patch(ax, xt_path, radius=0.18)
    arrow_marker(ax, (tanh_cx - 0.11, tanh_y0), "up")

    # horizontal pass-through line entering box2, curving down into tanh (from above-left)
    h_in_path = [
        (x0m, mid_y),
        (x0m + 0.55, mid_y),
        (x0m + 0.55, tanh_y0 - 0.3),
        (tanh_cx + 0.11, tanh_y0 - 0.3),
        (tanh_cx + 0.11, tanh_y0 - 0.02),
    ]
    rounded_path_patch(ax, h_in_path, radius=0.18)
    arrow_marker(ax, (tanh_cx + 0.11, tanh_y0), "up")

    # tanh box
    ax.add_patch(FancyBboxPatch(
        (tanh_cx - tanh_w / 2, tanh_y0), tanh_w, tanh_h,
        boxstyle="round,pad=0.01,rounding_size=0.06",
        linewidth=2.0, edgecolor=YELLOW_EDGE, facecolor=YELLOW, zorder=4,
    ))
    ax.text(tanh_cx, tanh_y0 + tanh_h / 2, "tanh", fontsize=13, ha="center", va="center", zorder=5)

    # output from tanh top -> junction on the mid_y spine
    out_path = [(tanh_cx, tanh_y1), (tanh_cx, mid_y)]
    rounded_path_patch(ax, out_path, radius=0.18)

    # from junction: up to h_t
    straight(ax, (tanh_cx, mid_y), (tanh_cx, h_circle_y - circle_r - 0.15))
    arrow_marker(ax, (tanh_cx, h_circle_y - circle_r - 0.02), "up")
    ax.add_patch(Circle((tanh_cx, h_circle_y), circle_r, facecolor=PURPLE, edgecolor=PURPLE_EDGE,
                        linewidth=2.2, zorder=4))
    ax.text(tanh_cx, h_circle_y, r"$h_t$", fontsize=15, ha="center", va="center", zorder=5)

    # ----------------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------------
    ax.set_xlim(box_x0[0] - 1.6, box_x0[2] + box_w + 1.4)
    ax.set_ylim(x_circle_y - circle_r - 0.4, h_circle_y + circle_r + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
