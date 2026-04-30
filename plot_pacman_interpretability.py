#!/usr/bin/env python3
"""
plot_pacman_interpretability.py

Composite 3-panel figure: ghost far (Chosen K=4) | ghost close (Chosen K=1) | correlation line.

Panels:
  1. PacMan screenshot: ghost far  → policy chooses K=4 (can plan safely)
  2. PacMan screenshot: ghost close → policy chooses K=1 (react immediately)
  3. Compact line+scatter: mean nearest-ghost distance by chosen K

States are constructed directly — player at fixed start, ghosts placed
at known valid (non-wall) positions from the DEFAULT_MAZE layout:
  Player P: row 23, col 13
  Ghost house: rows 13–15 (far from player, distance ≈ 8–12)
  Close ghosts: row 23 cols 11 & 15, row 20 cols 11 & 15

Coordinate convention (from jumanji viewer.py):
  player_locations: Position(x=row, y=col)
  ghost_locations[i]: [col, row]   ← note reversed order

Saves: figures/pacman_interpretability.pdf

Usage:
    python plot_pacman_interpretability.py
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# Style — sourced from plot_config.py
# ============================================================

from plot_config import (
    C_BLACK, C_WHITE, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY,
    FS_TITLE, FS_LABEL, FS_TICK, FS_BADGE, FS_ANNOT,
    C_BLUE, FILL_ALPHA, LINE_LW, MARKER_SIZE, CAPSIZE, ERR_LW,
    apply_style,
)
apply_style()

FONT_SIZE_TITLE = FS_TITLE
FONT_SIZE_LABEL = FS_LABEL
FONT_SIZE_TICK  = FS_TICK
FONT_SIZE_K_BOX = FS_BADGE
FONT_SIZE_ANNOT = FS_ANNOT
LINE_COLOR      = C_BLUE

FIG_WIDTH  = 11.0
FIG_HEIGHT = 4.2

# ============================================================
# Ghost distance by K  (placeholder — replace with real data)
# K=1 chosen when ghost is close (low distance)
# K=4 chosen when ghost is far  (high distance)
# ============================================================

GHOST_DIST_BY_K = [
    (1, 2.2, 0.4),
    (2, 3.8, 0.3),
    (3, 6.1, 0.2),
    (4, 8.2, 0.3),
]

# ============================================================
# Setup
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)



# ============================================================
# Construct states
# ============================================================

def build_states():
    """
    Return (far_state, close_state) by constructing them from a reset state.

    Far  state: reset state as-is. Ghosts in their house (rows 13–15),
                player at row 23, col 13. Min distance ≈ 8–12.

    Close state: same base state but ghost_locations replaced so all four
                 ghosts are within 2–5 cells of the player (row 23, col 13).
                 Positions chosen from open (non-wall) cells in DEFAULT_MAZE.
    """
    import jax
    import jax.numpy as jnp
    from jumanji.environments.routing.pac_man import PacMan

    env = PacMan()
    key = jax.random.PRNGKey(0)
    base_state, _ = jax.jit(env.reset)(key)

    px, py = int(base_state.player_locations.x), int(base_state.player_locations.y)
    print(f"Player: row={px}, col={py}")
    print(f"Ghost locs (reset): {np.array(base_state.ghost_locations)}")

    far_state = base_state  # ghosts already far in home

    # Close ghost positions — open cells near player (row=23, col=13)
    # ghost_locations[i] = [col, row]  ← jumanji convention
    close_ghosts = jnp.array(
        [[11, 23],   # col=11, row=23
         [15, 23],   # col=15, row=23
         [11, 20],   # col=11, row=20
         [15, 20]],  # col=15, row=20
        dtype=jnp.int32,
    )
    close_state = base_state.replace(ghost_locations=close_ghosts)

    def dist(state):
        px_ = int(state.player_locations.x)
        py_ = int(state.player_locations.y)
        ds = []
        for i in range(4):
            gc = int(state.ghost_locations[i, 0])  # col
            gr = int(state.ghost_locations[i, 1])  # row
            ds.append(abs(px_ - gr) + abs(py_ - gc))
        return ds

    print(f"Far  state distances: {dist(far_state)}")
    print(f"Close state distances: {dist(close_state)}")

    return far_state, close_state


# ============================================================
# Render state → numpy RGB
# ============================================================

def state_to_rgb(state):
    from jumanji.environments.routing.pac_man.viewer import create_grid_image
    return np.array(create_grid_image(state))


# ============================================================
# Figure
# ============================================================

def plot(far_state, close_state):
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1.0, 1.0, 0.75],
        wspace=0.22,
        left=0.03, right=0.96,
        top=0.88, bottom=0.14,
    )
    ax_far   = fig.add_subplot(gs[0])
    ax_close = fig.add_subplot(gs[1])
    ax_line  = fig.add_subplot(gs[2])

    # ---- game screenshots ----
    # Ghost far → policy plans deeply (K=4); ghost close → react immediately (K=1)
    panels = [
        (ax_far,   far_state,   "Chosen K = 4", "Ghost far"),
        (ax_close, close_state, "Chosen K = 1", "Ghost close"),
    ]
    for ax, state, k_txt, title_txt in panels:
        ax.set_title(title_txt, fontsize=FONT_SIZE_TITLE, pad=6)
        ax.set_axis_off()
        rgb = state_to_rgb(state)
        ax.imshow(rgb, interpolation="nearest")
        ax.text(
            0.5, -0.05,
            k_txt,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=FONT_SIZE_K_BOX,
            color=C_BLACK,
        )

    # ---- compact line+scatter: ghost distance by chosen K ----
    ks    = np.array([d[0] for d in GHOST_DIST_BY_K])
    means = np.array([d[1] for d in GHOST_DIST_BY_K])
    ses   = np.array([d[2] for d in GHOST_DIST_BY_K])

    ax_line.fill_between(ks, means - ses, means + ses,
                         color=LINE_COLOR, alpha=FILL_ALPHA)
    ax_line.plot(ks, means, "o-",
                 color=LINE_COLOR, linewidth=LINE_LW,
                 markersize=MARKER_SIZE, markeredgewidth=0)
    ax_line.errorbar(ks, means, yerr=ses,
                     fmt="none", ecolor=LINE_COLOR,
                     elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW,
                     alpha=0.6)

    ax_line.set_xticks(ks)
    ax_line.set_xticklabels([f"K={k}" for k in ks], fontsize=FONT_SIZE_TICK)
    ax_line.set_ylabel("Nearest ghost distance", fontsize=FONT_SIZE_LABEL)
    ax_line.set_title("Threat proximity\nby chosen K", fontsize=FONT_SIZE_TITLE)
    ax_line.set_ylim(bottom=0)
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)
    ax_line.grid(False)

    ax_line.text(
        0.97, 0.05, "placeholder data",
        transform=ax_line.transAxes, ha="right", va="bottom",
        fontsize=FONT_SIZE_ANNOT, color=C_MID_GRAY, style="italic",
    )

    out = os.path.join(FIGS, "pacman_interpretability.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ============================================================

if __name__ == "__main__":
    print("Building PacMan states...")
    far_state, close_state = build_states()
    print("Composing figure...")
    plot(far_state, close_state)
