from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np




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
