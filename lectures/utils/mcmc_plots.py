"""
Recreates a 2-state Markov chain transition diagram (Sunny <-> Rainy),
with self-loop and cross-transition probabilities.

Run:  python markov_chain.py
Output: markov_chain.png in the same folder.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

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
    ax.text(*rainy_xy, "Rainy", fontsize=15, fontweight="bold", ha="center", va="center",
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
