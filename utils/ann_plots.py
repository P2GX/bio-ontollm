from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np

def plot_neural_net(figsize=(12, 7.8)):
    fig, ax = plt.subplots(figsize=figsize)
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
        
        rect = patches.Rectangle(
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
    
        

def plot_neuron(figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Coordinates
    inputs = [1, 2, 3, 4]
    sum_node = (5, 2.5)
    act_node = (7, 2.5)
    out_node = (9, 2.5)
    bias_pos = (5, 0.5) # Directly under summation

    # 1. Input Nodes (x1...x4)
    for i, y in enumerate(inputs):
        ax.add_artist(patches.Circle((2, y), 0.3, color='skyblue', ec='black', zorder=3))
        ax.text(1.3, y, f'$x_{i+1}$', fontsize=12, va='center')
        ax.annotate('', xy=(sum_node[0]-0.5, sum_node[1]), xytext=(2.3, y), 
                    arrowprops=dict(arrowstyle="->", color='gray'))
        ax.text(3.3, y + (2.5-y)*0.3, f'$w_{i+1}$', fontsize=10)

    # 2. Bias Node (Under Summation)
    ax.add_artist(patches.Circle(bias_pos, 0.3, color='lightgreen', ec='black', zorder=3))
    ax.text(bias_pos[0] - 0.6, bias_pos[1], '$b$', fontsize=12, va='center')
    ax.annotate('', xy=(sum_node[0], sum_node[1]-0.5), xytext=(bias_pos[0], bias_pos[1]+0.3), 
                arrowprops=dict(arrowstyle="->", color='darkgreen', lw=1.5))

    # 3. Summation Node (Σ)
    ax.add_artist(patches.Circle(sum_node, 0.5, color='orange', ec='black', zorder=3))
    ax.text(sum_node[0], sum_node[1], r'$\sum$', fontsize=20, ha='center', va='center')

    # 4. Activation Node (f) with Sigmoid Curve
    ax.add_artist(patches.Circle(act_node, 0.5, color='salmon', ec='black', zorder=3))
    
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



def plot_maximum_chaos(figsize=(8, 4)):
    resolution=32
    seed=7
    np.random.seed(seed)
    
    # 1. Setup Grid
    x = np.linspace(-4, 4, resolution)
    y = np.linspace(-4, 4, resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.c_[X.ravel(), Y.ravel()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

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




def draw_nn_forward_pass(figsize=(12,7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Colors (matching the image)
    c_input = '#fde68a'   # Yellow
    c_hidden = '#93c5fd'  # Blue
    c_output = '#f87171'  # Red
    c_actual = '#bbf7d0'  # Green
    c_step = '#fb923c'    # Orange/Red for numbers

    # Neuron Positions (x, y)
    layer_1 = [(2, 4.5), (2, 3), (2, 1.5)]  # Input
    layer_2 = [(4.5, 4.5), (4.5, 3), (4.5, 1.5)]  # Hidden
    layer_3 = [(7, 3)]  # Output
    results = [(8.5, 3.5), (8.5, 2.5)] # Pred vs Actual

    # 1. Draw Connections (Arrows)
    # Input to Hidden
    for l1 in layer_1:
        for l2 in layer_2:
            ax.annotate("", xy=l2, xytext=l1, 
                         arrowprops=dict(
                        arrowstyle="->", 
                        color="black", 
                        lw=1,
                        shrinkA=15, # Shrinks the tail away from the hidden neuron
                        shrinkB=28  # Shrinks the head away from the output neuron
                    ))
    
    # Hidden to Output
    for l2 in layer_2:
        for l3 in layer_3:
            ax.annotate("", xy=l3, xytext=l2, 
                    arrowprops=dict(
                        arrowstyle="->", 
                        color="black", 
                        lw=1,
                        shrinkA=15, # Shrinks the tail away from the hidden neuron
                        shrinkB=28  # Shrinks the head away from the output neuron
                    ))

    # Inputs coming from left
    for i, l1 in enumerate(layer_1):
        ax.annotate("", xy=l1, xytext=(0.8, l1[1]), 
                    arrowprops=dict(arrowstyle="->", color="black", lw=1, shrinkA=15, shrinkB=28))
        
        ax.text(1.1, l1[1], "Inputs", ha='center', va='bottom', fontsize=10)

    # 2. Draw Neurons
    radius = 0.35
    for i, (x, y) in enumerate(layer_1):
        circle = patches.Circle((x, y), radius, color=c_input, ec='black', zorder=3)
        ax.add_artist(circle)
        ax.text(x, y, f'$x_{i+1}$', ha='center', va='center', fontsize=12, fontweight='bold', color='blue')

    for (x, y) in layer_2:
        circle = patches.Circle((x, y), radius, color=c_hidden, ec='black', zorder=3)
        ax.add_artist(circle)
        ax.text(x, y, 'w', ha='center', va='center', fontsize=14, fontweight='bold')

    for (x, y) in layer_3:
        circle = patches.Circle((x, y), radius, color=c_output, ec='black', zorder=3)
        ax.add_artist(circle)
        ax.text(x, y, r'$\hat{y}$', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Prediction and Actual nodes
    ax.annotate("", xy=(8.1, 3), xytext=(7.4, 3), arrowprops=dict(arrowstyle="->", color="black"))
    
    p_pred = patches.Circle(results[0], 0.25, color=c_actual, ec='black', zorder=3)
    p_actual = patches.Circle(results[1], 0.25, color=c_actual, ec='black', zorder=3)
    ax.add_artist(p_pred)
    ax.add_artist(p_actual)
    ax.text(results[0][0], results[0][1], r'$\hat{y}$', ha='center', va='center', fontsize=12)
    ax.text(results[1][0], results[1][1], 'y', ha='center', va='center', fontsize=12)

    # Labels for the end nodes
    ax.text(8.5, 4.0, "Predicted\noutput", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(8.5, 2.0, "Actual\nOutput", ha='center', va='top', fontsize=10, fontweight='bold')

    # 3. Text Labels and Steps
    #ax.text(5, 5.5, "Feed-Forward Neural Network", ha='center', fontsize=24, fontweight='bold')
    
    # Layer Titles
    ax.text(2, 0.5, "Input Layer", ha='center', fontsize=12, fontweight='bold')
    ax.text(4.5, 0.5, "Hidden Layer", ha='center', fontsize=12, fontweight='bold')
    ax.text(7, 0.5, "Output Layer", ha='center', fontsize=12, fontweight='bold')

    # Step annotations (Circles with numbers)
    steps = [
        (0.8, 5.2, "1", "Inputs enter\nthe input layer"),
        (3.5, 5.2, "2", "Weighted sum of inputs\nNonlinear activation"),
        (6.4, 4.6, "3", "Output prediction ($\hat{y}$)\nis generated"),
        (8.4, 1.4, "4", "Error")
    ]

    for x, y, num, txt in steps:
        # 1. Draw the "Open" Circle
        step_circ = patches.Circle(
            (x, y), 0.18, 
            fill=False,          # Makes it "open"
            edgecolor='#f87171', # The salmon/red color for the border
            lw=2,                # Thicker line so it's visible
            zorder=5
        )
        ax.add_artist(step_circ)
        
        # 2. Put the number inside (centered in the circle)
        ax.text(x, y, num, color='#f87171', ha='center', va='center', fontweight='bold', fontsize=10)
        
        # 3. Place text to the RIGHT of the circle
        # We add a small offset (0.3) to the x-coordinate
        ax.text(x + 0.3, y, txt, ha='left', va='center', fontsize=9, linespacing=1.2)

    # Braces for Error (simplified with a bracket)
    ax.plot([8.8, 8.9, 8.9, 8.8], [3.5, 3.5, 2.5, 2.5], color='blue', lw=1, ls='--')

    plt.tight_layout()


def draw_nn_backpropagation(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0,5.5)
    ax.axis('off')

    # 1. Configuration & Colors
    # --- Match colors from the source image ---
    c_input = '#fde68a'   # Yellow
    c_hidden = '#93c5fd'  # Blue
    c_output = '#f87171'  # Red/Salmon
    c_backprop = '#3b82f6' # A bold Blue for the backward flow
    c_step_border = '#f87171' # Red/Salmon for step rings

    # --- Define node positions on a 10x7 grid ---
    layer_1 = [(2, 4.5), (2, 3), (2, 1.5)] # Input (x1, x2, x3)
    layer_2 = [(4.5, 4.5), (4.5, 3), (4.5, 1.5)] # Hidden (w, w, w)
    layer_3 = [(7, 3)] # Output (y-hat)

    # 3. Layer Labels
    font_labels = {'fontsize': 12, 'fontweight': 'bold', 'ha': 'center'}
    ax.text(layer_1[0][0], 0.6, "Input Layer", **font_labels)
    ax.text(layer_2[0][0], 0.6, "Hidden Layer", **font_labels)
    ax.text(layer_3[0][0], 0.6, "Output Layer", **font_labels)

    # 4. Draw All Connections (Feed-Forward Arrows)
    # Important: shrinkA/shrinkB ensure arrowheads aren't hidden inside the nodes.
    # Input to Hidden
    for l1 in layer_1:
        for l2 in layer_2:
            ax.annotate("", xy=l2, xytext=l1, 
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, shrinkA=18, shrinkB=28, mutation_scale=15))
    
    # Hidden to Output
    for l2 in layer_2:
        for l3 in layer_3:
            ax.annotate("", xy=l3, xytext=l2, 
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, shrinkA=18, shrinkB=28, mutation_scale=15))

    # --- Initial inputs and final output arrows ---
    for l1 in layer_1:
        ax.annotate("", xy=l1, xytext=(1.1, l1[1]), 
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, shrinkA=0, shrinkB=28, mutation_scale=15))
    
    out_arrow_start = (layer_3[0][0] + 0.5, layer_3[0][1])
    out_arrow_end = (8.5, 3.0)
    ax.annotate("", xy=out_arrow_end, xytext=out_arrow_start, 
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, shrinkA=5, shrinkB=0, mutation_scale=15))

    # --- Final Output Labels (Text only) ---
    out_font = {'fontsize': 11, 'ha': 'left', 'va': 'center'}
    ax.text(out_arrow_end[0] + 0.1, out_arrow_end[1] + 0.25, "Outputs", **out_font)
    ax.text(out_arrow_end[0] + 0.1, out_arrow_end[1] - 0.25, "Predicted\noutput", **out_font)

    # 5. Draw Neurons (Circles with math labels)
    radius = 0.35
    for i, (x, y) in enumerate(layer_1):
        circle = patches.Circle((x, y), radius, color=c_input, ec='black', zorder=3)
        ax.add_artist(circle)
        ax.text(x, y, f'$x_{i+1}$', ha='center', va='center', fontsize=12, fontweight='bold', color='blue')

    for (x, y) in layer_2:
        circle = patches.Circle((x, y), radius, color=c_hidden, ec='black', zorder=3)
        ax.add_artist(circle)
        # Using '\sum \sigma' to be technically accurate (weighted sum + activation)
        ax.text(x, y, r'$\sum \sigma$', ha='center', va='center', fontsize=12, fontweight='bold')

    for (x, y) in layer_3:
        circle = patches.Circle((x, y), radius, color=c_output, ec='black', zorder=3)
        ax.add_artist(circle)
        # Using hat symbol for prediction
        ax.text(x, y, r'$\hat{y}$', ha='center', va='center', fontsize=16, fontweight='bold')

    # 6. Add The Backpropagation (Curved Blue Arrow)
    # ConnectionStyle="arc3,rad=.2" creates that smooth curve.
    backprop_start = (8.4, 3.3)
    backprop_end = (4.7, 4.6)
    
    arrow_props_bp = dict(
        arrowstyle="-|>", 
        color=c_backprop, 
        lw=2.5, 
        shrinkA=15, 
        shrinkB=15, 
        mutation_scale=25, # Makes the blue arrowhead larger
        connectionstyle="arc3,rad=-0.15" # Creates the elegant curve
    )
    ax.annotate("", xy=backprop_end, xytext=backprop_start, arrowprops=arrow_props_bp)

    # 7. Add Numbered Steps (Refined style)
    # x, y, number, description text
    steps = [
        (7.8, 3.8, "1", "Error - difference\nbetween predicted\noutput and actual\noutput"),
        (6.1, 4.6, "2", "Error is sent back to\neach neuron in backward\ndirection"),
        (3.8, 5.2, "3", "Gradient of error is\ncalculated with respect to\neach weight"),
    ]

    for x, y, num, txt in steps:
        # Draw the "Open" Circle (The Ring)
        step_circ = patches.Circle(
            (x, y), 0.22, 
            fill=False,          # Makes it hollow
            edgecolor=c_step_border, 
            lw=2.5,              # Stronger line
            zorder=5
        )
        ax.add_artist(step_circ)
        
        # Number inside the ring (matched to border color)
        ax.text(x, y, num, color=c_step_border, ha='center', va='center', fontweight='bold', fontsize=11)
        
        # Text to the RIGHT of the ring (or adjusted for Step 3)
        # We need a small x-offset so the text doesn't crash into the circle
        if num == "30": # Special handling for top text
            ax.text(x + 0.4, y, txt, ha='center', va='bottom', fontsize=10, linespacing=1.2)
        else: # Standard right-side placement
            ax.text(x + 0.4, y, txt, ha='left', va='center', fontsize=10, linespacing=1.2)

    plt.tight_layout()




def draw_forward_block_diagram(fig_size = (9, 4)):
    block_color = '#F2DAB3' # Light beige/tan color matching image_2.png
    box_lw = 1.0 # Box line width
    arrow_lw = 1.5 # Arrow line width
    arrow_color = 'black'
    text_font_size = 28
    title_font_size = 28
    text_params = {'ha': 'center', 'va': 'center', 'fontsize': text_font_size}
    # Setup Figure and Axes
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # --- 1. Draw Title ---
    title_text='forward'
    box_text=r'$f(\mathbf{x}_\text{in}, \theta)$'
    ax.text(5, 4.8, title_text, ha='center', fontsize=title_font_size, fontfamily='monospace')

    # --- 2. Draw The Generic Block (Box) ---
    # Center the box (4.0 to 6.0)
    box_width = 2.0
    box_height = 2.0
    box_x_start = 5.0 - (box_width / 2.0)
    box_y_start = 2.5 - (box_height / 2.0)
    
    rect = patches.Rectangle(
        (box_x_start, box_y_start), 
        box_width, 
        box_height, 
        facecolor=block_color, 
        edgecolor='black', 
        linewidth=box_lw,
        zorder=2 # Ensure box is below text
    )
    ax.add_patch(rect)
    
    # Text inside the box
    ax.text(5.0, 2.5, box_text, **text_params, zorder=3)


    # --- 3. Draw Input Flows ---
    # Position nodes for flows
    p_x_in_node = (1.5, 2.5)
    p_box_left_upper = (box_x_start, 2.5) # x_in input
    p_box_left_lower = (box_x_start, 2.0) # theta input (lower)
    p_theta_node = (1.5, 1.3) # Starting point below x_in

    # a. x_in to Box (Straight Arrow)
    arrow_props_x_in = dict(arrowstyle="-|>", color=arrow_color, lw=arrow_lw, mutation_scale=20)
    ax.annotate(r'$\mathbf{x}_\text{in}$', xy=p_box_left_upper, xytext=p_x_in_node, arrowprops=arrow_props_x_in, **text_params)

    # b. theta to Box (L-shaped angled Arrow)
    # The crucial part is connectionstyle="angle"
    # angleA=-90 (start point vertical)
    # angleB=0 (end point horizontal)
    # rad=15 (corner sharpness)
    arrow_props_theta = dict(
        arrowstyle="-|>", 
        color=arrow_color, 
        lw=arrow_lw, 
        mutation_scale=20,
        connectionstyle="angle,angleA=-90,angleB=0,rad=0.1" # Angled arrow
    )
    # The xytext is where the 'theta' label sits, which is below x_in.
    ax.annotate(r'$\theta$', xy=p_box_left_lower, xytext=p_theta_node, arrowprops=arrow_props_theta, **text_params)


  # --- 4. Draw Output Flow ---
    p_box_right_center = (box_x_start + box_width, 2.5) # Arrow starts at box edge
    arrow_tip = (7.5, 2.5)                             # Arrow ends here
    p_x_out_text = (8.0, 2.5)                          # Text sits to the right

    # 1. Draw the arrow (No text inside annotate this time)
    arrow_props_x_out = dict(arrowstyle="-|>", color=arrow_color, lw=arrow_lw, mutation_scale=20)
    ax.annotate("", xy=arrow_tip, xytext=p_box_right_center, arrowprops=arrow_props_x_out)

    # 2. Place the x_out text to the right of the tip
    ax.text(p_x_out_text[0], p_x_out_text[1], r'$\mathbf{x}_\text{out}$', **text_params)

    plt.tight_layout()


def draw_toy_network(figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 1. Colors & Settings
    c_input, c_hidden, c_output = '#fde68a', '#93c5fd', '#f87171'
    edge_props = dict(arrowstyle="-|>", color="black", lw=1.2, shrinkA=18, shrinkB=18, mutation_scale=15)
    
    # Node positions
    l1_pos = [(2, 5), (2, 2)]  # Input (x0, x1)
    l2_pos = [(5, 5), (5, 2)]  # Hidden (a0, a1)
    l3_pos = [(8, 3.5)]       # Output (a2)
    
    # 2. Draw Connections and Weights (No Superscripts)
    w_font = {'fontsize': 11, 'fontweight': 'bold', 'ha': 'center'}
    
    # Layer 1 Connections (Input -> Hidden)
    for i, p1 in enumerate(l1_pos):
        for j, p2 in enumerate(l2_pos):
            ax.annotate("", xy=p2, xytext=p1, arrowprops=edge_props)
            
            # --- Anti-Overlap Logic ---
            # For straight lines (0->0, 1->1), use the midpoint.
            # For diagonal lines (0->1, 1->0), move label closer to the destination node.
            if i == j:
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                offset = 0.25 if i == 0 else -0.4
            else:
                # Move 70% of the way toward the hidden layer to avoid the center cross
                mx = p1[0] + 0.7 * (p2[0] - p1[0])
                my = p1[1] + 0.7 * (p2[1] - p1[1])
                offset = 0.2 if i == 0 else -0.4 # Shift slightly so it's not ON the line
            if i==0 and j==0:
                idx = 1
            elif i==0 and j==1:
                idx = 2
            elif i==1 and j==0:
                idx = 3
            else:
                idx = 4
            ax.text(mx, my + offset, f'$w_{idx}$', **w_font)

    # Layer 2 Connections (Hidden -> Output)
    for j, p2 in enumerate(l2_pos):
        ax.annotate("", xy=l3_pos[0], xytext=p2, arrowprops=edge_props)
        # Midpoint is fine here as there are only two lines going to one point
        ax.text((p2[0]+8)/2, (p2[1]+3.5)/2 + 0.3, f'$w_{j+5}$', **w_font)

    # 3. Draw Neurons
    node_font = {'fontsize': 14, 'fontweight': 'bold', 'ha': 'center', 'va': 'center'}
    for i, (x, y) in enumerate(l1_pos):
        ax.add_patch(patches.Circle((x, y), 0.4, color=c_input, ec='black', zorder=3))
        ax.text(x, y, f'$x_{i+1}$', **node_font)

    for i, (x, y) in enumerate(l2_pos):
        ax.add_patch(patches.Circle((x, y), 0.4, color=c_hidden, ec='black', zorder=3))
        ax.text(x, y, f'$a_{i+1}$', **node_font)

    ax.add_patch(patches.Circle(l3_pos[0], 0.4, color=c_output, ec='black', zorder=3))
    ax.text(8, 3.5, '$a_3$', **node_font)

    # 4. Draw Biases
    bias_props = dict(arrowstyle="-|>", color="gray", lw=1.0, shrinkB=18, mutation_scale=12, ls='--')
    for i, p2 in enumerate(l2_pos):
        ax.annotate("", xy=p2, xytext=(p2[0]-0.6, p2[1]+1), arrowprops=bias_props)
        ax.text(p2[0]-0.6, p2[1]+1.1, f'$b_{i+1}$', color='gray', fontweight='bold')
    
    ax.annotate("", xy=l3_pos[0], xytext=(7.4, 4.5), arrowprops=bias_props)
    ax.text(7.4, 4.6, '$b_3$', color='gray', fontweight='bold')

  


def plot_3d_spiral(figsize=(8, 6)):
    # Explicitly creating the figure and axis with 3D projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Define the parameter t
    t = np.linspace(0, 10 * np.pi, 500)

    # Define the components of f(t)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = t

    # Use .plot and .scatter (they support 3D automatically on a 3D axis)
    ax.plot(x, y, z, color='gray', alpha=0.5, label='Path')
    
    # Color the points by t to show the flow of the function
    scatter = ax.scatter(x, y, z, c=t, cmap='viridis', s=10) # type: ignore

    # Labeling using LaTeX

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Style adjustments for clear slides
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.colorbar(scatter, label='Time ($t$)', pad=0.1, shrink=0.5)




def plot_gd(figsize=(9, 6)):

    def cost_func(w):
        return w**2


    def derivative_func(w):
        return 2 * w

    # --- 2. Generate Data for Plotting ---
    # Smooth curve for the parabola
    w_values = np.linspace(-2.5, 2.5, 300)
    c_values = cost_func(w_values)

    # --- 3. Set up the Figure and Main Curve ---
    # Using a slightly taller figure for clear labels
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the smooth parabolic cost curve in coral color, like the image
    ax.plot(w_values, c_values, color='#F08080', linewidth=3.5, label='Derivative of Cost (Curve)')

    # --- 4. Plot Gradient Descent Steps (The Iterations) ---
    # Define the starting point and simulated path
    w_path = np.array([2.0, 1.5, 1.1, 0.8, 0.55, 0.35, 0.2])
    c_path = cost_func(w_path)

    # 4a. Draw the arrows connecting the steps
    for i in range(len(w_path)-1):
        dx = w_path[i+1] - w_path[i]
        dy = c_path[i+1] - c_path[i]
        ax.arrow(w_path[i], c_path[i], dx, dy, 
                color='blue', head_width=0.1, head_length=0.08, 
                length_includes_head=True, zorder=5)

    # 4b. Draw the path points (dots)
    ax.scatter(w_path, c_path, color='black', s=40, edgecolors='none', zorder=6)

    # --- 5. Draw Annotations (Labels and Lines) ---


    # Initial Weight (The large black dot)
    w_start = w_path[0]
    c_start = c_path[0]
    ax.scatter(w_start, c_start, color='black', s=180, zorder=7) # Bigger dot

    # Arrow/Text for 'Initial Weight'
    ax.annotate('Initial\nWeight', 
                xy=(w_start, c_start + 0.3),    # Arrow points slightly above dot
                xytext=(w_start - 0.5, c_start + 1.5), # Text position
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                ha='center', fontsize=12, fontweight='bold')

    # Text and arrow for 'Incremental Step' (pointing to an intermediate step)
    ax.annotate('Incremental\nStep', 
                xy=(w_path[3], c_path[3]), 
                xytext=(w_path[3] - 1.2, c_path[3] + 1.5),
                arrowprops=dict(arrowstyle="->", color='black', linewidth=1),
                ha='center', fontsize=11, fontweight='normal')

    # Define and Plot the Tangent Line (for the gradient at start)
    tangent_slope = derivative_func(w_start)
    tangent_x = np.array([w_start - 0.4, w_start + 0.4])
    # Tangent equation: y - y1 = m(x - x1) -> y = m(x - x1) + y1
    tangent_y = tangent_slope * (tangent_x - w_start) + c_start

    # Dotted tangent line
    ax.plot(tangent_x, tangent_y, color='black', linestyle='--', linewidth=1.5, zorder=2)

    # Annotation for 'Gradient' (tangent line)
    ax.annotate('Gradient', 
                xy=(tangent_x[-1], tangent_y[-1] + 0.1), 
                xytext=(tangent_x[-1] + 1.2, tangent_y[-1] + 0.5),
                arrowprops=dict(arrowstyle="->", color='black', linewidth=1),
                ha='center', va='center', fontsize=11, fontweight='normal')

    # Minimum Cost (Bottom of the parabola)
    w_min = 0
    c_min = 0
    # Small horizontal line for 'minimum' point
    ax.hlines(y=c_min, xmin=-0.1, xmax=0.1, color='black', linewidth=1.5)

    # Text and arrow for 'Minimum Cost'
    ax.annotate('Minimum Cost', 
                xy=(w_min, c_min), 
                xytext=(w_min + 1.5, c_min + 0.8),
                arrowprops=dict(arrowstyle="->", color='black', linewidth=1),
                ha='center', fontsize=12, fontweight='bold', color='darkred')

    # Label for the Main Curve itself
    # We put this label in an empty space
    ax.text(-1.8, 1.5, 'Derivative of Cost', fontsize=12, color='#D35400', fontweight='bold', ha='center', va='center')


    # --- 6. Set Plot Limits, Axes, and Formatting ---
    ax.set_xlim(-2.5, 3.0) # Expand right for minimum label
    ax.set_ylim(-0.5, 7.0)

    # Clean, simple axis spine locations (bottom and left)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Set Labels (Positioning them near the arrow tips)
    ax.set_xlabel('Weight', fontsize=14, loc='right')
    ax.set_ylabel('Cost', fontsize=14, loc='top', rotation=0, labelpad=20)

    # Hide grid for a diagram style
    ax.grid(False)

    # Optional: Tight layout to prevent text cutoff
    plt.tight_layout()

    # Quarto will automatically collect this plot
    plt.show()

