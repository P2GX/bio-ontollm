from matplotlib import patches, path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.path import Path


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
  