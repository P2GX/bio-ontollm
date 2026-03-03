import matplotlib.pyplot as plt
import numpy as np

def plot_neural_net():
    fig, ax = plt.subplots(figsize=(12, 7.8))
    left, right   = 0.06, 0.94
    bottom, top   = 0.12, 0.88
    layer_sizes = [3, 6, 4, 1]
    layer_labels = ["Input\nLayer", "Hidden 1", "Hidden 2", "Output\nLayer"]
    box_width_factor=5.2
    neuron_radius=0.03
    box_width_factor=5.2
    edge_lw=1.6
    
    n_layers = len(layer_sizes)
    h_spacing = (right - left) / max(1, n_layers - 1)
    
    max_neurons = max(layer_sizes)
    v_spacing = (top - bottom) / (max_neurons + 3)   # more breathing room
    
    neuron_positions = []
    
    for i, size in enumerate(layer_sizes):
        layer_x = left + i * h_spacing
        total_height = (size - 1) * v_spacing
        y_start = (bottom + top) / 2 - total_height / 2
        layer_y = np.linspace(y_start, y_start + total_height, size)
        neuron_positions.append(list(zip([layer_x] * size, layer_y)))
    
    # ─── Connections ────────────────────────────────────────────
    for i in range(n_layers - 1):
        for x1, y1 in neuron_positions[i]:
            for x2, y2 in neuron_positions[i + 1]:
                ax.plot([x1 + neuron_radius, x2 - neuron_radius], [y1, y2],
                        color='gray', lw=0.9, alpha=0.6, zorder=1)
    
    # ─── Neurons ────────────────────────────────────────────────
    for layer in neuron_positions:
        for x, y in layer:
            circle = plt.Circle((x, y), neuron_radius, # type: ignore
                               color='white', ec='black', lw=1.2, zorder=3)
            ax.add_patch(circle)
    
    # ─── Boxes + Labels ─────────────────────────────────────────
    for i, (size, pos_list) in enumerate(zip(layer_sizes, neuron_positions)):
        x_center = pos_list[0][0]
        ys = [p[1] for p in pos_list]
        y_min, y_max = min(ys), max(ys)
        
        margin_v = neuron_radius * 2.2
        box_height = (y_max - y_min) + 2 * margin_v
        box_width  = neuron_radius * box_width_factor   

        if i == 0:
            facecolor = 'lightblue'
            edgecolor = 'navy'
        elif i == n_layers -1:
            facecolor = '#e0f7e0'         # very light green
            edgecolor = 'darkgreen'
        else:                             # Hidden layers
            facecolor = '#ffe5cc'         # soft peach / light orange
            edgecolor = 'darkorange'
        
        rect = plt.Rectangle(
            (x_center - box_width/2, y_min - margin_v),
            width=box_width,
            height=box_height,
            fill=True, fc=facecolor, alpha=0.14,
            ec=edgecolor, lw=edge_lw, zorder=2   # thicker border
        )
        ax.add_patch(rect)
        
        # Label above
        label = layer_labels[i] if layer_labels and i < len(layer_labels) else f"Layer {i}"
        ax.text(x_center, y_max + margin_v * 1.9,
                label, ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='navy')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis('off')
    plt.show()
        

def plot_neuron():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Coordinates
    inputs = [1, 2, 3, 4]
    sum_node = (5, 2.5)
    act_node = (7, 2.5)
    out_node = (9, 2.5)
    bias_pos = (5, 0.5) # Directly under summation

    # 1. Input Nodes (x1...x4)
    for i, y in enumerate(inputs):
        ax.add_artist(plt.Circle((2, y), 0.3, color='skyblue', ec='black', zorder=3))
        ax.text(1.3, y, f'$x_{i+1}$', fontsize=12, va='center')
        ax.annotate('', xy=(sum_node[0]-0.5, sum_node[1]), xytext=(2.3, y), 
                    arrowprops=dict(arrowstyle="->", color='gray'))
        ax.text(3.3, y + (2.5-y)*0.3, f'$w_{i+1}$', fontsize=10)

    # 2. Bias Node (Under Summation)
    ax.add_artist(plt.Circle(bias_pos, 0.3, color='lightgreen', ec='black', zorder=3))
    ax.text(bias_pos[0] - 0.6, bias_pos[1], '$b$', fontsize=12, va='center')
    ax.annotate('', xy=(sum_node[0], sum_node[1]-0.5), xytext=(bias_pos[0], bias_pos[1]+0.3), 
                arrowprops=dict(arrowstyle="->", color='darkgreen', lw=1.5))

    # 3. Summation Node (Σ)
    ax.add_artist(plt.Circle(sum_node, 0.5, color='orange', ec='black', zorder=3))
    ax.text(sum_node[0], sum_node[1], r'$\sum$', fontsize=20, ha='center', va='center')

    # 4. Activation Node (f) with Sigmoid Curve
    ax.add_artist(plt.Circle(act_node, 0.5, color='salmon', ec='black', zorder=3))
    
    # --- DRAW MINI SIGMOID ---
    # Create local coordinates within the circle
    sx = np.linspace(act_node[0]-0.3, act_node[0]+0.3, 20)
    # Sigmoid formula: 1 / (1 + exp(-x)) scaled to fit the circle
    sy = act_node[1] + (1 / (1 + np.exp(-15 * (sx - act_node[0])))) - 0.5
    ax.plot(sx, sy, color='white', lw=2, zorder=4)
    ax.text(act_node[0], act_node[1] + 0.6, '$\sigma(z)$', fontsize=12, ha='center')

    # 5. Connections and Output
    ax.annotate('', xy=(act_node[0]-0.5, act_node[1]), xytext=(sum_node[0]+0.5, sum_node[1]), 
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate('', xy=out_node, xytext=(act_node[0]+0.5, act_node[1]), 
                arrowprops=dict(arrowstyle="->", lw=2, color='red'))
    ax.text(9.2, 2.5, 'Output ($y$)', fontsize=12, va='center')

    # Formatting
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_maximum_chaos(resolution=32, seed=7):
    np.random.seed(seed)
    
    # 1. Setup Grid
    x = np.linspace(-4, 4, resolution)
    y = np.linspace(-4, 4, resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.c_[X.ravel(), Y.ravel()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # --- PANEL 1: LINEAR BOUNDARY ---
    # A simple diagonal line
    linear_mask = grid_points[:, 1] > (grid_points[:, 0] * 0.7 - 0.5)
    colors_lin = ['#1F51E5' if m else '#EF3324' for m in linear_mask]
    
    ax1.scatter(grid_points[:, 0], grid_points[:, 1], c=colors_lin, s=30, edgecolors='none')
    ax1.plot([-4, 4], [-4*0.7-0.5, 4*0.7-0.5], color='#58B042', lw=4)
    ax1.set_title("Linear boundary", fontsize=20, pad=15)

    # --- PANEL 2: HIGH-CHAOS NON-LINEAR ---
    # This function combines multiple frequencies to ensure "pockets" and "jaggedness"
    def get_chaos_val(px, py):
        # Base trend
        z = py - px * 0.2 
        # Layer 1: Medium waves
        z += 1.8 * np.sin(px * 1.5) * np.cos(py * 1.2)
        # Layer 2: High-frequency "jitters" (the craziness)
        z += 1.2 * np.cos(px * 3.5 + py * 2.1)
        # Layer 3: Small isolated "islands"
        z += 0.7 * np.sin(px * 5.0) * np.sin(py * 5.0)
        return z

    z_dots = get_chaos_val(grid_points[:, 0], grid_points[:, 1])
    nonlinear_mask = z_dots > 0
    colors_nonlin = ['#1F51E5' if m else '#EF3324' for m in nonlinear_mask]

    ax2.scatter(grid_points[:, 0], grid_points[:, 1], c=colors_nonlin, s=30, edgecolors='none')
    
    # Generate a very high-res contour for a smooth "wiggly" line
    xf = np.linspace(-4.2, 4.2, 300)
    yf = np.linspace(-4.2, 4.2, 300)
    Xf, Yf = np.meshgrid(xf, yf)
    Zf = get_chaos_val(Xf, Yf)
    
    # Plot the green boundary line where z=0
    ax2.contour(Xf, Yf, Zf, levels=[0], colors='#58B042', linewidths=4)
    ax2.set_title("Non-linear boundary", fontsize=20, pad=15)

    # Styling for Quarto/Notebooks
    for ax in [ax1, ax2]:
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-4.2, 4.2)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(-4, 5, 1))
        ax.set_yticks(np.arange(-4, 5, 1))
        # Keep it clean
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

