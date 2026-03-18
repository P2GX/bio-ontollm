import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import binom
import numpy as np

def dag(figsize=(8, 6)):
    G = nx.DiGraph()

    # 2. Define the edges based on your image
    # We'll use a coordinate-like naming or simple IDs
    edges = [
        ('black', 'blue_top'), ('black', 'magenta'), ('black', 'cyan'), 
        ('black', 'yellow_bot'), ('black', 'green'),
        ('blue_top', 'yellow_top'), ('magenta', 'yellow_top'), ('magenta', 'orange'),
        ('cyan', 'orange'), ('yellow_bot', 'blue_bot'), ('green', 'blue_bot'),
        ('yellow_top', 'grey'), ('orange', 'grey'), ('blue_bot', 'grey')
    ]
    G.add_edges_from(edges)

    # 3. Define the positions to match your layout
    pos = {
        'black':      [0, 2],
        'blue_top':   [1, 4],
        'magenta':    [2, 3.2],
        'cyan':       [2.2, 2],
        'yellow_bot': [2, 0.8],
        'green':      [1, 0],
        'yellow_top': [4, 3.8],
        'orange':     [4.5, 1.8],
        'blue_bot':   [4, 0.2],
        'grey':       [6, 1.8]
    }

    # 4. Define colors for each node
    node_colors = [
        'black', 'cornflowerblue', 'magenta', 'cyan', 'yellow', 
        'lime', 'yellow', 'orange', 'cornflowerblue', 'grey'
    ]

    # 5. Identify the red path edges
    red_edges = [('black', 'yellow_bot'), ('yellow_bot', 'blue_bot'), ('blue_bot', 'grey')]
    black_edges = [e for e in G.edges() if e not in red_edges]

    # 6. Draw the plot
    plt.figure(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, edgecolors='black')

    # Draw standard black edges
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, edge_color='black', 
                        arrowstyle='->', arrowsize=20)

    # Draw the specific red path edges
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='red', 
                        arrowstyle='->', arrowsize=20, width=2)

    plt.axis('off')


def plot_binom():
    # Parameters 
    n = 250
    p = 0.06
    observed_k = 30
    k_range = np.arange(0, 41)

    # Calculate PMF
    pmf = binom.pmf(k_range, n, p)

    # Determine the 95% threshold for shading
    # ppf gives the smallest k such that P(X <= k) >= 0.95
    threshold_k = binom.ppf(0.95, n, p)

    # Create the plot
    plt.figure(figsize=(4, 4))
    ax = plt.gca()

    # 1. Plot the smoothed line (interpolated for a cleaner look)
    from scipy.interpolate import make_interp_spline
    k_smooth = np.linspace(k_range.min(), k_range.max(), 300)
    spl = make_interp_spline(k_range, pmf, k=3)
    pmf_smooth = spl(k_smooth)
    plt.plot(k_smooth, pmf_smooth, color='black', linewidth=1.5)

    # 2. Fill the 95% region in gray
    # We shade from 0 up to the threshold
    shade_x = k_smooth[k_smooth <= threshold_k]
    shade_y = pmf_smooth[k_smooth <= threshold_k]
    plt.fill_between(shade_x, 0, shade_y, color='lightgray', edgecolor='black', alpha=0.8)

    # 3. Add the vertical line at the threshold
    plt.vlines(threshold_k, 0, binom.pmf(threshold_k, n, p), color='black', linewidth=1)

    # 4. Add the arrow and label for k=30
    plt.annotate(f'$k = {observed_k}$', xy=(observed_k, 0.01), xytext=(observed_k, 0.07),
                arrowprops=dict(arrowstyle='->', color='black'),
                ha='center', fontsize=16)

    # 5. Formatting to match the original style
    plt.xlabel('$k$', fontsize=18)
    plt.ylabel('', fontsize=18) # Leave y-label empty as per original
    plt.xticks([0, 10, 20, 30, 40], fontsize=14)
    plt.yticks([0.00, 0.05, 0.10, 0.15], fontsize=14)
    plt.xlim(-2, 42)
    plt.ylim(-0.005, 0.16)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add the horizontal base line
    plt.axhline(0, color='black', linewidth=0.8)

    plt.tight_layout()