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

def plot_gsea(figsize=(8, 6)):
    # ----------------------------------------------------------------------
    # 1. Simulate a ranked gene list
    # ----------------------------------------------------------------------
    rng = np.random.default_rng(seed=7)

    N = 200          # total number of genes
    N_H = 20         # number of genes in the gene set S
    p = 1.0          # weighting exponent (p=1 is the GSEA default)

    # Ranking statistic r_j: e.g. a signal-to-noise ratio or t-statistic,
    # sorted from most positive (top of list) to most negative (bottom).
    # We simulate a smoothly decaying score plus noise.
    base_signal = np.linspace(3.0, -3.0, N)
    noise = rng.normal(scale=0.4, size=N)
    r = base_signal + noise
    r = np.sort(r)[::-1]  # ensure monotonic decreasing rank order

    ranks = np.arange(1, N + 1)  # rank positions 1..N (1 = top of list)

    # ----------------------------------------------------------------------
    # 2. Build an "enriched" gene set S
    # ----------------------------------------------------------------------
    # Bias membership probability toward the top of the ranked list so the
    # set is clearly (but not perfectly) enriched near rank 1.
    top_bias_weights = np.exp(-ranks / 40.0)  # decays with rank -> favors top
    top_bias_weights /= top_bias_weights.sum()

    in_set_idx = rng.choice(N, size=N_H, replace=False, p=top_bias_weights)
    in_set = np.zeros(N, dtype=bool)
    in_set[in_set_idx] = True

    # ----------------------------------------------------------------------
    # 3. Compute the running-sum statistic
    # ----------------------------------------------------------------------
    abs_r_weighted = np.abs(r) ** p
    N_R = abs_r_weighted[in_set].sum()          # normalization for hits
    N_miss_total = N - N_H                       # normalization for misses

    P_hit = np.zeros(N)
    P_miss = np.zeros(N)

    hit_step = np.where(in_set, abs_r_weighted / N_R, 0.0)
    miss_step = np.where(~in_set, 1.0 / N_miss_total, 0.0)

    P_hit = np.cumsum(hit_step)
    P_miss = np.cumsum(miss_step)

    ES_curve = P_hit - P_miss

    # ----------------------------------------------------------------------
    # 4. Find D = ES(S): the signed maximum deviation
    # ----------------------------------------------------------------------
    max_idx = np.argmax(ES_curve)
    min_idx = np.argmin(ES_curve)

    if abs(ES_curve[max_idx]) >= abs(ES_curve[min_idx]):
        D_idx, D_val = max_idx, ES_curve[max_idx]
    else:
        D_idx, D_val = min_idx, ES_curve[min_idx]

    # Leading-edge subset: hits occurring at or before the position of D
    leading_edge = np.sum(in_set[: D_idx + 1])

    # ----------------------------------------------------------------------
    # 5. Plot
    # ----------------------------------------------------------------------
    fig, (ax_es, ax_ticks, ax_rank) = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 0.6, 1.2], "hspace": 0.08},
    )

    # --- Top panel: P_hit, P_miss, and the ES running-sum curve ---
    ax_es.plot(ranks, P_hit, color="#1f77b4", lw=1.8, label=r"$P_{hit}(S,i)$")
    ax_es.plot(ranks, P_miss, color="#7f7f7f", lw=1.4, ls="--",
            label=r"$P_{miss}(S,i)$")
    ax_es.plot(ranks, ES_curve, color="#2ca02c", lw=2.2,
            label=r"$ES(i) = P_{hit}-P_{miss}$")

    ax_es.axhline(0, color="black", lw=0.8, alpha=0.5)

    # Mark D, the maximum deviation
    ax_es.plot([ranks[D_idx]], [D_val], "o", color="crimson", ms=8, zorder=5)
    ax_es.vlines(ranks[D_idx], 0, D_val, color="crimson", lw=1.2, ls=":")
    ax_es.annotate(
        rf"$D = ES(S) = {D_val:.3f}$" + "\n" + rf"(rank {ranks[D_idx]})",
        xy=(ranks[D_idx], D_val),
        xytext=(ranks[D_idx] + N * 0.12, D_val * 0.65 if D_val > 0 else D_val * 0.65 - 0.05),
        fontsize=9.5,
        color="crimson",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0),
    )

    ax_es.set_ylabel("Running sum")
    ax_es.set_title(
        "GSEA weighted running-sum (KS-like) statistic\n"
        f"N={N} genes, |S|={N_H}, weighting exponent p={p:g}, "
        f"leading edge = {leading_edge} genes"
    )
    ax_es.legend(loc="upper right", frameon=False, fontsize=9)

    # --- Middle panel: hit/miss "barcode" ticks along the ranked list ---
    hit_ranks = ranks[in_set]
    ax_ticks.vlines(hit_ranks, 0, 1, color="#1f77b4", lw=0.9)
    ax_ticks.set_ylim(0, 1)
    ax_ticks.set_yticks([])
    ax_ticks.set_ylabel("Hits", rotation=0, ha="right", va="center", fontsize=9)

    # --- Bottom panel: ranking metric r_j across the list ---
    ax_rank.fill_between(ranks, r, 0, color="#c7c7c7", step="mid")
    ax_rank.axhline(0, color="black", lw=0.6)
    ax_rank.set_ylabel(r"$r_j$")
    ax_rank.set_xlabel("Rank in ordered gene list (i)")

    for ax in (ax_es, ax_ticks, ax_rank):
        ax.set_xlim(1, N)

    plt.tight_layout()
