"""
Recreates a 2-state Markov chain transition diagram (Sunny <-> Cloudy),
with self-loop and cross-transition probabilities.

Run:  python markov_chain.py
Output: markov_chain.png in the same folder.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
from scipy import stats


# ---------- palette ----------
SUNNY_COLOR = "#E8C51B"
RAINY_COLOR = "#3E6DA8"
EDGE_COLOR  = "#1a1a1a"
ARROW_COLOR = "#1a1a1a"
TEXT_COLOR  = "#1a1a1a"


def mcmc_sunny_cloudy(figsize=(8, 5)):
    """
    Recreates a 2-state Markov chain transition diagram (Sunny <-> Rainy),
    with self-loop and cross-transition probabilities.
    """
    _fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- node positions ----
    sunny_xy = (2.6, 3.0)
    rainy_xy = (7.4, 3.0)
    node_r = 1.0

    # ---- state circles ----
    ax.add_patch(Circle(sunny_xy, node_r, facecolor=SUNNY_COLOR, edgecolor=EDGE_COLOR, linewidth=1.8, zorder=3))
    ax.add_patch(Circle(rainy_xy, node_r, facecolor=RAINY_COLOR, edgecolor=EDGE_COLOR, linewidth=1.8, zorder=3))

    ax.text(*sunny_xy, "Sunny", fontsize=15, fontweight="bold", ha="center", va="center",
            color="black", zorder=4)
    ax.text(*rainy_xy, "Cloudy", fontsize=15, fontweight="bold", ha="center", va="center",
            color="white", zorder=4)


    def curved_arrow(ax, start, end, bulge, color=ARROW_COLOR, linewidth=1.8,
                    head_size=22, n=100, zorder=2):
        """
        Draw a quadratic-Bezier curved arrow from start to end with a given
        perpendicular 'bulge' (positive = bulges to the left of start->end
        direction), and an arrowhead whose rotation is computed analytically
        from the curve's true tangent at the endpoint.

        This avoids FancyArrowPatch's arc3 connectionstyle, whose automatic
        arrowhead can end up visibly misaligned with the curve for some
        bulge signs (a real matplotlib quirk, not a parameter you can fix
        by tuning shrinkA/shrinkB).
        """
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        mid = (start + end) / 2
        d = end - start
        perp = np.array([-d[1], d[0]])
        perp = perp / np.linalg.norm(perp)
        ctrl = mid + bulge * perp

        t = np.linspace(0, 1, n)[:, None]
        pts = (1 - t) ** 2 * start + 2 * (1 - t) * t * ctrl + t ** 2 * end
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth, zorder=zorder)

        # analytic tangent at t=1 of a quadratic Bezier: direction = end - ctrl
        tangent = end - ctrl
        angle = np.degrees(np.arctan2(tangent[1], tangent[0]))
        # matplotlib's (numsides, style, angle) triangle marker points "up" at angle=0
        ax.plot(*end, marker=(3, 0, angle - 90), markersize=head_size / 1.4,
                color=color, zorder=zorder + 1, markeredgewidth=0)

        return ctrl  # in case caller wants to place a label near the peak


    # ---- cross-transition arrows ----
    # Sunny -> Rainy (top arc, bulges upward, away from the circles)
    ctrl_top = curved_arrow(
        ax,
        start=(sunny_xy[0] + 0.55, sunny_xy[1] + 0.85),
        end=(rainy_xy[0] - 0.55, rainy_xy[1] + 0.85),
        bulge=1.7,
    )
    ax.text((sunny_xy[0] + rainy_xy[0]) / 2, 5.05, "70%", fontsize=14, ha="center", va="center", color=TEXT_COLOR)

    # Rainy -> Sunny (bottom arc, bulges downward, away from the circles)
    ctrl_bottom = curved_arrow(
        ax,
        start=(rainy_xy[0] - 0.55, rainy_xy[1] - 0.85),
        end=(sunny_xy[0] + 0.55, sunny_xy[1] - 0.85),
        bulge=1.7,
    )
    ax.text((sunny_xy[0] + rainy_xy[0]) / 2, 0.95, "50%", fontsize=14, ha="center", va="center", color=TEXT_COLOR)

    # ---- self-loop arrows ----
    def _loop_tangent_angle(t_end, loop_r, squash, sign):
        """
        Analytic tangent direction (degrees) of the self-loop curve at t_end.

        The curve is always parameterized with t running from 200deg down to
        -20deg (i.e. t strictly DEcreasing), so velocity along the path is
        -d/dt of the parametric position -- that's a fixed fact of how the
        curve is built below, not something to infer per call.
        """
        dxdt = -loop_r * np.sin(t_end)
        dydt = sign * loop_r * squash * np.cos(t_end)
        return np.degrees(np.arctan2(-dydt, -dxdt))


    def self_loop(ax, center, node_r, side="left", color=ARROW_COLOR, loop_r=0.55, squash=0.65):
        """
        Draw a small self-loop arrow attached to a node.

        side="left"  -> loop sits above-left of the node.
        side="right" -> the exact vertical mirror of "left" (same t-range;
                        y-component and x-offset both negated via `sign`),
                        sits below-right of the node.
        """
        cx, cy = center
        sign = 1 if side == "left" else -1

        # Loop's nearest edge should sit flush against the node, not floating above it.
        # (loop_r*squash is the loop's actual radius in that direction.)
        gap_offset = node_r + loop_r * squash - 0.05  # small overlap so it visually attaches
        loop_center = (cx - 0.05 * sign, cy + sign * gap_offset)

        t = np.linspace(np.radians(200), np.radians(-20), 100)
        xs = loop_center[0] + loop_r * np.cos(t)
        ys = loop_center[1] + sign * loop_r * np.sin(t) * squash
        ax.plot(xs, ys, color=color, linewidth=1.8, zorder=2)

        # Arrowhead angle computed analytically (see _loop_tangent_angle), rather
        # than from a FancyArrowPatch built off the last two sample points: those
        # are only ~0.015 units apart, and FancyArrowPatch's default shrinkA/
        # shrinkB (a *fixed pixel* amount) eats a segment that tiny almost
        # entirely, producing an erratic head direction.
        angle = _loop_tangent_angle(t[-1], loop_r, squash, sign)
        ax.plot(xs[-1], ys[-1], marker=(3, 0, angle - 90), markersize=13,
                color=color, zorder=3, markeredgewidth=0)

        return loop_center

    loop_c1 = self_loop(ax, sunny_xy, node_r, side="left")
    ax.text(loop_c1[0] - 0.05, loop_c1[1] + 0.62, "30%", fontsize=14, ha="center", va="center", color=TEXT_COLOR)

    loop_c2 = self_loop(ax, rainy_xy, node_r, side="right")
    ax.text(loop_c2[0] + 0.05, loop_c2[1] - 0.62, "50%", fontsize=14, ha="center", va="center", color=TEXT_COLOR)

    plt.tight_layout()


def burn_in(figsize=(13, 4)):
    """
    Draws a Markov chain diagram: a horizontal sequence of states x_0, x_1, ...,
    connected by transition arrows, with the first `n_burnin` states bracketed
    and labeled "burn-in", and each state after that labeled p(x) to indicate
    it's (approximately) a draw from the target distribution.
    """

   

    # ---------- config ----------
    N_TOTAL = 10       # total number of states shown, x_0 ... x_{N_TOTAL-1}
    N_BURNIN = 5        # first N_BURNIN states are the burn-in period

    NODE_R = 0.42
    SPACING = 1.7
    Y = 0.0

    NODE_FACE = "#EAF1FB"
    NODE_EDGE = "#1a1a1a"
    ARROW_COLOR = "#1a1a1a"
    BRACKET_COLOR = "#8a2f2f"
    BURNIN_TEXT_COLOR = "#8a2f2f"
    TARGET_TEXT_COLOR = "#1f5c3c"

    _fig, ax = plt.subplots(figsize=figsize)

    xs = [i * SPACING for i in range(N_TOTAL)]

    # ---- nodes ----
    for i, x in enumerate(xs):
        ax.add_patch(Circle((x, Y), NODE_R, facecolor=NODE_FACE, edgecolor=NODE_EDGE,
                            linewidth=1.6, zorder=3))
        ax.text(x, Y, f"$x_{{{i}}}$", fontsize=15, ha="center", va="center", zorder=4)

    # ---- transition arrows between consecutive nodes ----
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch(
            (x0 + NODE_R, Y), (x1 - NODE_R, Y),
            arrowstyle="-|>", mutation_scale=16, linewidth=1.6,
            color=ARROW_COLOR, zorder=2, shrinkA=0, shrinkB=0,
        ))

    # ---- burn-in bracket + label (below the first N_BURNIN nodes) ----
    bracket_y = Y - NODE_R - 0.30
    tick_h = 0.10
    left = xs[0] - NODE_R
    right = xs[N_BURNIN - 1] + NODE_R
    mid = (left + right) / 2

    ax.plot([left, right], [bracket_y, bracket_y], color=BRACKET_COLOR, linewidth=1.6, zorder=2)
    for tx in (left, right, mid):
        ax.plot([tx, tx], [bracket_y, bracket_y + tick_h], color=BRACKET_COLOR, linewidth=1.6, zorder=2)

    ax.text(mid, bracket_y - 0.28, "burn-in", fontsize=14, fontweight="bold",
            ha="center", va="top", color=BURNIN_TEXT_COLOR)

    # ---- p(x) labels under each post-burn-in node ----
    for x in xs[N_BURNIN:]:
        ax.text(x, Y - NODE_R - 0.30, "$p(x)$", fontsize=14, ha="center", va="top",
                color=TARGET_TEXT_COLOR)

    # ---- cosmetic ----
    ax.set_xlim(xs[0] - 1.0, xs[-1] + 1.0)
    ax.set_ylim(-1.6, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()


def run_metropolis_mcmc(start_val=0.0, n_steps=5000, proposal_width=1.0):
    """
    Runs a Metropolis MCMC sampler for a standard normal distribution N(0, 1)
    and plots the trace, density comparison, and QQ-plot.
    Example usage:
    run_metropolis_mcmc(start_val=2.5, n_steps=5000)
    """
    # 1. MCMC Sampling
    samples = np.empty(n_steps)
    current_x = start_val
    
    # Log pdf of standard normal (omitting constant as it cancels out in ratio)
    def log_target(x):
        return -0.5 * (x ** 2)

    accepted = 0
    for i in range(n_steps):
        # Propose a new state uniformly within current_x +/- proposal_width/2
        proposed_x = np.random.uniform(current_x - proposal_width / 2.0, 
                                       current_x + proposal_width / 2.0)
        
        # Acceptance ratio in log space (symmetric proposal means q terms cancel)
        log_h = log_target(proposed_x) - log_target(current_x)
        
        # Accept/reject step
        if np.log(np.random.uniform(0, 1)) < log_h:
            current_x = proposed_x
            accepted += 1
            
        samples[i] = current_x

    print(f"Acceptance rate: {accepted / n_steps * 100:.2f}%")

    # 2. Visualization Layout Setup
    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(2, 2)
    
    # Top Row: Trace plot spanning both columns
    ax_trace = fig.add_subplot(gs[0, :])
    ax_trace.plot(samples, color='teal', alpha=0.6, lw=0.8)
    ax_trace.axhline(0, color='red', linestyle='--', alpha=0.7, label='Target Mean')
    ax_trace.set_ylim(-3, 3)
    ax_trace.set_xlim(0, n_steps)
    ax_trace.set_title('MCMC Sampling Trace (Standard Normal Distribution)', fontsize=12)
    ax_trace.set_xlabel('Iteration Step')
    ax_trace.set_ylabel('Current Value ($x$)')
    ax_trace.legend(loc='upper right')
    
    # Bottom Left: Density Plot (True vs MCMC)
    ax_density = fig.add_subplot(gs[1, 0])
    ax_density.hist(samples, bins=50, density=True, alpha=0.5, color='teal', label='MCMC Samples')
    
    x_grid = np.linspace(-3, 3, 200)
    true_pdf = stats.norm.pdf(x_grid, 0, 1)
    ax_density.plot(x_grid, true_pdf, 'r-', lw=2, label='True $\mathcal{N}(0,1)$')
    ax_density.set_title('Density Comparison')
    ax_density.set_xlabel('$x$')
    ax_density.set_ylabel('Density')
    ax_density.legend(loc='upper right')
    
    # Bottom Right: QQ Plot
    ax_qq = fig.add_subplot(gs[1, 1])
    stats.probplot(samples, dist="norm", plot=ax_qq)
    ax_qq.set_title('Normal QQ-Plot')
    
    plt.tight_layout()
    plt.show()

