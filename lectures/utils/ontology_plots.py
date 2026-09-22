import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import networkx as nx
from scipy.stats import binom
import numpy as np


def dag(figsize=(8, 6)):
    G = nx.DiGraph()

    # 2. Define the edges based on your image
    # We'll use a coordinate-like naming or simple IDs
    edges = [
        ("black", "blue_top"),
        ("black", "magenta"),
        ("black", "cyan"),
        ("black", "yellow_bot"),
        ("black", "green"),
        ("blue_top", "yellow_top"),
        ("magenta", "yellow_top"),
        ("magenta", "orange"),
        ("cyan", "orange"),
        ("yellow_bot", "blue_bot"),
        ("green", "blue_bot"),
        ("yellow_top", "grey"),
        ("orange", "grey"),
        ("blue_bot", "grey"),
    ]
    G.add_edges_from(edges)

    # 3. Define the positions to match your layout
    pos = {
        "black": [0, 2],
        "blue_top": [1, 4],
        "magenta": [2, 3.2],
        "cyan": [2.2, 2],
        "yellow_bot": [2, 0.8],
        "green": [1, 0],
        "yellow_top": [4, 3.8],
        "orange": [4.5, 1.8],
        "blue_bot": [4, 0.2],
        "grey": [6, 1.8],
    }

    # 4. Define colors for each node
    node_colors = [
        "black",
        "cornflowerblue",
        "magenta",
        "cyan",
        "yellow",
        "lime",
        "yellow",
        "orange",
        "cornflowerblue",
        "grey",
    ]

    # 5. Identify the red path edges
    red_edges = [
        ("black", "yellow_bot"),
        ("yellow_bot", "blue_bot"),
        ("blue_bot", "grey"),
    ]
    black_edges = [e for e in G.edges() if e not in red_edges]

    # 6. Draw the plot
    plt.figure(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=800, edgecolors="black"
    )

    # Draw standard black edges
    nx.draw_networkx_edges(
        G, pos, edgelist=black_edges, edge_color="black", arrowstyle="->", arrowsize=20
    )

    # Draw the specific red path edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=red_edges,
        edge_color="red",
        arrowstyle="->",
        arrowsize=20,
        width=2,
    )

    plt.axis("off")


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
    plt.plot(k_smooth, pmf_smooth, color="black", linewidth=1.5)

    # 2. Fill the 95% region in gray
    # We shade from 0 up to the threshold
    shade_x = k_smooth[k_smooth <= threshold_k]
    shade_y = pmf_smooth[k_smooth <= threshold_k]
    plt.fill_between(
        shade_x, 0, shade_y, color="lightgray", edgecolor="black", alpha=0.8
    )

    # 3. Add the vertical line at the threshold
    plt.vlines(threshold_k, 0, binom.pmf(threshold_k, n, p), color="black", linewidth=1)

    # 4. Add the arrow and label for k=30
    plt.annotate(
        f"$k = {observed_k}$",
        xy=(observed_k, 0.01),
        xytext=(observed_k, 0.07),
        arrowprops=dict(arrowstyle="->", color="black"),
        ha="center",
        fontsize=16,
    )

    # 5. Formatting to match the original style
    plt.xlabel("$k$", fontsize=18)
    plt.ylabel("", fontsize=18)  # Leave y-label empty as per original
    plt.xticks([0, 10, 20, 30, 40], fontsize=14)
    plt.yticks([0.00, 0.05, 0.10, 0.15], fontsize=14)
    plt.xlim(-2, 42)
    plt.ylim(-0.005, 0.16)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add the horizontal base line
    plt.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()


def plot_gsea(figsize=(8, 6)):
    # ----------------------------------------------------------------------
    # 1. Simulate a ranked gene list
    # ----------------------------------------------------------------------
    rng = np.random.default_rng(seed=7)

    N = 200  # total number of genes
    N_H = 20  # number of genes in the gene set S
    p = 1.0  # weighting exponent (p=1 is the GSEA default)

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
    N_R = abs_r_weighted[in_set].sum()  # normalization for hits
    N_miss_total = N - N_H  # normalization for misses

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
        3,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 0.6, 1.2], "hspace": 0.08},
    )

    # --- Top panel: P_hit, P_miss, and the ES running-sum curve ---
    ax_es.plot(ranks, P_hit, color="#1f77b4", lw=1.8, label=r"$P_{hit}(S,i)$")
    ax_es.plot(
        ranks, P_miss, color="#7f7f7f", lw=1.4, ls="--", label=r"$P_{miss}(S,i)$"
    )
    ax_es.plot(
        ranks, ES_curve, color="#2ca02c", lw=2.2, label=r"$ES(i) = P_{hit}-P_{miss}$"
    )

    ax_es.axhline(0, color="black", lw=0.8, alpha=0.5)

    # Mark D, the maximum deviation
    ax_es.plot([ranks[D_idx]], [D_val], "o", color="crimson", ms=8, zorder=5)
    ax_es.vlines(ranks[D_idx], 0, D_val, color="crimson", lw=1.2, ls=":")
    ax_es.annotate(
        rf"$D = ES(S) = {D_val:.3f}$" + "\n" + rf"(rank {ranks[D_idx]})",
        xy=(ranks[D_idx], D_val),
        xytext=(
            ranks[D_idx] + N * 0.12,
            D_val * 0.65 if D_val > 0 else D_val * 0.65 - 0.05,
        ),
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


def binomial_explanation(figsize=(9, 7.5)):
    """
    Illustrates the binomial coefficient C(10,3) = 10!/(3!7!):
    """
    WHITE = "#f5f0e6"
    BLACK = "#2b2b2b"
    EDGE = "#1a1a1a"
    GRAY_EDGE = "#555555"

    # Positions (0-indexed) that are "black" in the fixed color pattern
    BLACK_POS = {1, 4, 7}
    N = 10

    # distinguishable colors for the "all balls different" row
    cmap = plt.get_cmap("tab10")
    DISTINCT_COLORS = [cmap(i % 10) for i in range(N)]

    def draw_row(
        ax,
        y,
        labels,
        colors,
        r=0.32,
        gap=0.85,
        label_color="white",
        fontsize=13,
        edge_colors=None,
    ):
        """Draw a row of N circles at height y, each with a number label."""
        x0 = 0.5
        for i in range(N):
            cx = x0 + i * gap
            ec = edge_colors[i] if edge_colors is not None else EDGE
            circ = patches.Circle(
                (cx, y), r, facecolor=colors[i], edgecolor=ec, linewidth=1.6, zorder=2
            )
            ax.add_patch(circ)
            if labels is not None:
                ax.text(
                    cx,
                    y,
                    str(labels[i]),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontweight="bold",
                    color=(
                        label_color[i] if isinstance(label_color, list) else label_color
                    ),
                    zorder=3,
                )
        return x0, gap

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 10.2)
    ax.set_aspect("equal")
    ax.axis("off")

    gap = 0.85

    # ---- Row A: all 10 balls fully distinguishable ----
    y_a = 9.3
    draw_row(
        ax,
        y_a,
        labels=list(range(N)),
        colors=DISTINCT_COLORS,
        label_color="white",
        gap=gap,
    )
    ax.text(
        4.75,
        y_a + 0.65,
        r"10 distinguishable balls $\rightarrow$ 10! orderings",
        ha="center",
        va="center",
        fontsize=13,
    )

    # ---- Arrow down ----
    ax.annotate(
        "",
        xy=(4.75, 8.15),
        xytext=(4.75, 8.7),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2),
    )
    ax.text(
        5.6,
        8.42,
        'fix which 3\nare "black"',
        fontsize=10.5,
        color="#444444",
        ha="left",
        va="center",
    )

    # ---- Rows B1-B3: same color pattern, different number labels ----
    # black positions keep 3 distinct numbers permuted among themselves;
    # white positions keep 7 distinct numbers permuted among themselves
    rng = np.random.default_rng(3)
    black_labels_pool = [1, 4, 7]
    white_labels_pool = [0, 2, 3, 5, 6, 8, 9]

    label_rows = []
    bl = black_labels_pool.copy()
    wl = white_labels_pool.copy()
    label_rows.append((bl.copy(), wl.copy()))
    rng.shuffle(bl)
    rng.shuffle(wl)
    label_rows.append((bl.copy(), wl.copy()))
    rng.shuffle(bl)
    rng.shuffle(wl)
    label_rows.append((bl.copy(), wl.copy()))

    y_positions = [7.3, 6.35, 5.4]
    for row_i, y in enumerate(y_positions):
        bl_row, wl_row = label_rows[row_i]
        bi, wi = 0, 0
        labels = []
        colors = []
        label_colors = []
        edge_colors = []
        for pos in range(N):
            if pos in BLACK_POS:
                labels.append(bl_row[bi])
                bi += 1
                colors.append(BLACK)
                label_colors.append("white")
                edge_colors.append(EDGE)
            else:
                labels.append(wl_row[wi])
                wi += 1
                colors.append(WHITE)
                label_colors.append("#1a1a1a")
                edge_colors.append(EDGE)
        draw_row(
            ax,
            y,
            labels=labels,
            colors=colors,
            label_color=label_colors,
            gap=gap,
            edge_colors=edge_colors,
        )

    ax.text(
        9.2,
        (7.3 + 5.4) / 2,
        r"$3! \times 7!$" + "\nlabelings,\nsame pattern",
        ha="left",
        va="center",
        fontsize=11.5,
        color="#444444",
    )
    # brace-like bracket on the right of the three rows
    brace_x = 8.85
    ax.plot(
        [brace_x, brace_x + 0.1, brace_x + 0.1, brace_x],
        [5.05, 5.05, 7.6, 7.6],
        color="#888888",
        lw=1.3,
    )

    ax.text(4.75, 4.75, "...", ha="center", va="center", fontsize=18, color="#888888")

    # ---- Arrow down to final color-only pattern ----
    ax.annotate(
        "",
        xy=(4.75, 3.55),
        xytext=(4.75, 4.35),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2),
    )
    ax.text(
        5.6,
        3.95,
        "discard labels,\nkeep only color",
        fontsize=10.5,
        color="#444444",
        ha="left",
        va="center",
    )

    # ---- Row C: final color-only pattern (no numbers) ----
    y_c = 2.75
    colors_c = [BLACK if pos in BLACK_POS else WHITE for pos in range(N)]
    draw_row(ax, y_c, labels=None, colors=colors_c, gap=gap)

    # ---- Formula ----
    ax.text(
        4.75,
        1.35,
        r"$\binom{10}{3} = \frac{10!}{3! \, 7!} = 120$" + "\ndistinct color patterns",
        ha="center",
        va="center",
        fontsize=17,
    )

    plt.tight_layout()


def analysis_path(figsize=(3.0, 7.2)):
    """
    Condensed GO-analysis pathway for a narrow slide column: fewer
    steps, no subtitles, larger relative font so it stays legible
    when embedded at ~30-40% slide width.
    """
    STEP1 = "#3a6ea5"  # RNA-seq / input
    STEP2 = "#2f8fd6"  # preprocessing
    STEP4 = "#f2a71b"  # DE testing
    STEP5 = "#7fb069"  # gene lists
    STEP6 = "#e0632b"  # annotated GO terms
    STEP7 = "#4d9078"  # enrichment analysis
    STEP8 = "#800080"  # Overrepresented GO terms
    TEXT_LIGHT = "#ffffff"
    CONNECTOR = "#8a97a8"

    boxes = [
        dict(label="RNA-seq:\nCondition A vs. B", color=STEP1),
        dict(label="Preprocessing\n(counts, filtering,\nID mapping)", color=STEP2),
        dict(label="DESeq2 / limma", color=STEP4),
        dict(label="Gene lists", color=STEP5),
        dict(label="Annotated\nGO\nterms", color=STEP6),
        dict(label="Enrichment\nanalysis", color=STEP7),
        dict(label="Overrepresented\nGO\nterms", color=STEP8),
    ]

    fig, ax = plt.subplots(figsize=figsize)
    n = len(boxes)
    gap = 0.35
    h = (10 - gap * (n - 1)) / n  # equal-height boxes filling a 0-10 y-range

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 10)
    ax.axis("off")

    cx = 2.0
    box_w = 3.5
    y = 10 - h / 2
    positions = []

    for i, b in enumerate(boxes):
        positions.append(y)

        shadow = FancyBboxPatch(
            (cx - box_w / 2 + 0.05, y - h / 2 - 0.05),
            box_w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.14",
            facecolor="#000000",
            edgecolor="none",
            alpha=0.15,
            zorder=1,
        )
        ax.add_patch(shadow)

        box = FancyBboxPatch(
            (cx - box_w / 2, y - h / 2),
            box_w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.14",
            facecolor=b["color"],
            edgecolor="white",
            linewidth=1.6,
            zorder=2,
        )
        ax.add_patch(box)

        ax.text(
            cx,
            y,
            b["label"],
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=TEXT_LIGHT,
            zorder=3,
            linespacing=1.3,
        )

        if i > 0:
            prev_y = positions[i - 1]
            ax.annotate(
                "",
                xy=(cx, y + h / 2),
                xytext=(cx, prev_y - h / 2),
                arrowprops=dict(
                    arrowstyle="-|>", color=CONNECTOR, lw=2.2, shrinkA=0, shrinkB=0
                ),
                zorder=1,
            )

        y -= h + gap

    plt.tight_layout()
    return fig


def hyperg_vs_binom(figsize=(8, 7)):
    """
    Binomial (coin, sampling WITH replacement) vs.
    Hypergeometric/Fisher (urn, sampling WITHOUT replacement).

    Two rows: before first sample / after first sample.
    Two columns: coin / urn.
    Minimal text: just the probability expression below each figure.
    """
    rng = np.random.default_rng(7)

    WHITE = "#f5f0e6"
    BLACK = "#2b2b2b"
    GOLD = "#c9a227"
    GOLD_DARK = "#8a6d1a"
    EDGE = "#1a1a1a"

    def draw_coin(ax, cx, cy, r, face=None, faded=False):
        """Draw a coin. face=None -> resting/unflipped ('?').
        face='H' or 'T' -> shows that face after a flip."""
        alpha = 0.35 if faded else 1.0
        circle = patches.Circle(
            (cx, cy),
            r,
            facecolor=GOLD,
            edgecolor=GOLD_DARK,
            linewidth=2.5,
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(circle)
        inner = patches.Circle(
            (cx, cy),
            r * 0.8,
            facecolor="none",
            edgecolor=GOLD_DARK,
            linewidth=1,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(inner)

        label = "T" if face is None else ("H" if face == "H" else "T")

        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=r * 55,
            fontweight="bold",
            color=EDGE,
            alpha=alpha,
            zorder=4,
            family="sans-serif",
        )

    def draw_urn(
        ax,
        x0,
        y0,
        n_white,
        n_black,
        cols=4,
        ball_r=0.18,
        removed_ball=None,
        removed_pos=None,
    ):
        """Draw an urn (rounded container) filled with white/black balls."""
        total = n_white + n_black
        rows = int(np.ceil(total / cols)) if total > 0 else 1
        width = cols * (2 * ball_r + 0.11)
        height = rows * (2 * ball_r + 0.08) + 0.3

        urn = patches.FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.12,rounding_size=0.15",
            edgecolor="#6b4a2b",
            facecolor="#faf6ee",
            linewidth=2.5,
            zorder=1,
        )
        ax.add_patch(urn)

        colors = ["white"] * n_white + ["black"] * n_black
        rng.shuffle(colors)

        for i, color in enumerate(colors):
            row, col = divmod(i, cols)
            cx = x0 + 0.15 + col * (2 * ball_r + 0.08) + ball_r
            cy = y0 + height - 0.25 - row * (2 * ball_r + 0.08) - ball_r
            face = WHITE if color == "white" else BLACK
            ball = patches.Circle(
                (cx, cy),
                ball_r,
                facecolor=face,
                edgecolor=EDGE,
                linewidth=1.2,
                zorder=2,
            )
            ax.add_patch(ball)

        return width, height

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax in axes.flat:
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 2.6)
        ax.set_aspect("equal")
        ax.axis("off")

    # ---- Row 1: BEFORE first sample ----
    ax = axes[0, 0]
    draw_coin(ax, 1.5, 1.4, 0.55, face=None)
    ax.text(1.5, 0.45, r"$P(H) = 0.5$", ha="center", va="center", fontsize=15)

    ax = axes[0, 1]
    draw_urn(ax, 0.6, 0.6, n_white=6, n_black=6, cols=4, ball_r=0.2)
    ax.text(1.5, 0.25, r"$P(B) = \frac{6}{12}$", ha="center", va="center", fontsize=15)

    # ---- Row 2: AFTER first sample ----
    ax = axes[1, 0]
    draw_coin(ax, 1.5, 1.4, 0.55, face="H")
    ax.text(1.5, 0.45, r"$P(H) = 0.5$", ha="center", va="center", fontsize=15)

    ax = axes[1, 1]
    draw_urn(
        ax,
        0.6,
        0.6,
        n_white=6,
        n_black=5,
        cols=4,
        ball_r=0.2,
        removed_ball="black",
        removed_pos=(2.55, 2.25),
    )
    ax.text(1.5, 0.25, r"$P(B) = \frac{5}{12}$", ha="center", va="center", fontsize=15)

    plt.tight_layout()
    plt.show()


def gora_shapes():
    """
    GO overrepresentation analysis general sets
    """
    NAVY = "#1b2a4a"
    GO_COLOR = "#f2a71b"  # GO-term-annotated genes
    DE_COLOR = "#4d9078"  # differentially expressed genes
    POP_FILL = "#eef1f5"
    POP_EDGE = "#8a97a8"
    TEXT_DARK = "#132436"
    LEADER = "#6b7686"

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Population rectangle (background) ----
    population = FancyBboxPatch(
        (0.4, 0.4),
        9.2,
        6.8,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        facecolor=POP_FILL,
        edgecolor=POP_EDGE,
        linewidth=2.2,
        zorder=1,
    )
    ax.add_patch(population)

    ax.text(
        5,
        6.85,
        "All genes in genome / background set (N)",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=TEXT_DARK,
        zorder=5,
    )

    # ---- Two overlapping circles ----
    r1, r2 = 2.0, 1.7  # GO-term circle, DE circle
    c1 = (4.1, 3.3)  # GO-term circle center
    c2 = (5.9, 3.3)  # DE circle center

    circle_go = Circle(
        c1,
        r1,
        facecolor=GO_COLOR,
        edgecolor="white",
        linewidth=2.5,
        alpha=0.85,
        zorder=2,
    )
    circle_de = Circle(
        c2,
        r2,
        facecolor=DE_COLOR,
        edgecolor="white",
        linewidth=2.5,
        alpha=0.85,
        zorder=3,
    )
    ax.add_patch(circle_go)
    ax.add_patch(circle_de)

    # ---- Overlap label, small, sits inside the lens region ----
    ax.text(
        (c1[0] + c2[0]) / 2 + 0.15,
        c1[1],
        "k",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="white",
        zorder=4,
    )

    # ---- External labels with leader lines ----
    def label_with_leader(ax, point, text_xy, text, ha="center", color=TEXT_DARK):
        ax.annotate(
            text,
            xy=point,
            xytext=text_xy,
            ha=ha,
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="-", color=LEADER, lw=1.4, shrinkA=2, shrinkB=6),
            zorder=6,
        )

    label_with_leader(
        ax,
        point=(c1[0] - 1.1, c1[1] + 0.9),
        text_xy=(1.3, 6.0),
        text="GO-term annotated\ngenes (K)",
        ha="left",
    )
    label_with_leader(
        ax,
        point=(c2[0] + 1.05, c2[1] + 0.7),
        text_xy=(8.7, 6.0),
        text="Differentially\nexpressed genes (n)",
        ha="right",
    )

    plt.tight_layout()
