#!/usr/bin/env python3
"""
plot_interpretability.py

Combined interpretability figure:
Top row: PacMan subplots
Bottom row: Tetris subplots
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
    FS_TITLE, FS_LABEL, FS_TICK, FS_BADGE, FS_ANNOT,
    C_BLUE, FILL_ALPHA, BAR_ALPHA, LINE_LW, MARKER_SIZE, CAPSIZE, ERR_LW,
    apply_style,
)
apply_style()

FONT_SIZE_TITLE = FS_TITLE
FONT_SIZE_LABEL = FS_LABEL
FONT_SIZE_TICK  = FS_TICK
FONT_SIZE_K_BOX = FS_BADGE
FONT_SIZE_ANNOT = FS_ANNOT
LINE_COLOR      = C_BLUE
BAR_COLOR       = C_BLUE

# PacMan data
GHOST_DIST_BY_K = [
    (1, 2.2, 0.4),
    (2, 3.8, 0.3),
    (3, 6.1, 0.2),
    (4, 8.2, 0.3),
]

# Tetris data
BOARD_FILL_BY_K = [
    (1, 0.049, 0.001),
    (2, 0.322, 0.005),
    (4, 0.250, 0.002),
]
PIECE_MEAN_K = [
    ("J", 2.66, 0.04),
    ("S", 2.74, 0.04),
    ("I", 2.82, 0.04),
    ("L", 2.84, 0.04),
    ("T", 2.88, 0.04),
    ("O", 2.88, 0.04),
    ("Z", 2.98, 0.04),
]

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

# ============================================================
# Construct states (PacMan)
# ============================================================

def build_pacman_states():
    import jax
    import jax.numpy as jnp
    from jumanji.environments.routing.pac_man import PacMan

    env = PacMan()
    key = jax.random.PRNGKey(0)
    base_state, _ = jax.jit(env.reset)(key)

    far_state = base_state
    close_ghosts = jnp.array(
        [[11, 23],
         [15, 23],
         [11, 20],
         [15, 20]],
        dtype=jnp.int32,
    )
    close_state = base_state.replace(ghost_locations=close_ghosts)
    return far_state, close_state

def state_to_rgb(state):
    from jumanji.environments.routing.pac_man.viewer import create_grid_image
    return np.array(create_grid_image(state))

# ============================================================
# Construct states (Tetris)
# ============================================================

def build_tetris_states():
    import jax
    import jax.numpy as jnp
    from jumanji.environments.packing.tetris import Tetris

    env = Tetris()
    key = jax.random.PRNGKey(0)
    base_state, _ = jax.jit(env.reset)(key)

    sparse_state = base_state
    nr, nc = env.num_rows, env.num_cols
    dense_grid = np.array(base_state.grid_padded)

    col_heights = [7, 5, 8, 6, 4, 8, 7, 5, 7, 6]
    for col in range(nc):
        for offset in range(col_heights[col]):
            row = nr - 1 - offset
            dense_grid[row, col] = (col // 2 + (row // 2) * 2) % 7 + 1

    dense_grid_jnp = jnp.array(dense_grid, dtype=base_state.grid_padded.dtype)
    dense_state = base_state.replace(grid_padded=dense_grid_jnp)

    return sparse_state, dense_state, env

def draw_tetris(state, ax, viewer):
    grid = viewer._create_rendering_grid(state)
    viewer._add_grid_image(ax, grid)
    ax.invert_yaxis()

# ============================================================
# Figure
# ============================================================

def plot(far_state, close_state, sparse_state, dense_state, tetris_env):
    from jumanji.environments.packing.tetris.viewer import TetrisViewer
    viewer = TetrisViewer(
        num_rows=tetris_env.num_rows, num_cols=tetris_env.num_cols, render_mode="rgb_array"
    )

    fig = plt.figure(figsize=(13.0, 8.5))
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=0.1)

    # --- Top Row: PacMan ---
    # Width ratios from original: 1.0, 1.0, 0.75
    # Let's add some empty space to the right if we want to match Tetris width (13.0 instead of 11.0)
    # Actually, we can just use 1.0, 1.0, 0.75, and 0.5 (blank) or similar, or just adjust the right margin.
    gs_pacman = gridspec.GridSpec(
        1, 3, figure=subfigs[0],
        width_ratios=[1.0, 1.0, 0.75],
        wspace=0.22,
        left=0.03, right=0.96,
        top=0.88, bottom=0.14,
    )
    ax_far   = subfigs[0].add_subplot(gs_pacman[0])
    ax_close = subfigs[0].add_subplot(gs_pacman[1])
    ax_line  = subfigs[0].add_subplot(gs_pacman[2])

    panels_pacman = [
        (ax_far,   far_state,   "Chosen K = 4", "Ghost far"),
        (ax_close, close_state, "Chosen K = 1", "Ghost close"),
    ]
    for ax, state, k_txt, title_txt in panels_pacman:
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
            color="black",
        )

    ks    = np.array([d[0] for d in GHOST_DIST_BY_K])
    means = np.array([d[1] for d in GHOST_DIST_BY_K])
    ses   = np.array([d[2] for d in GHOST_DIST_BY_K])

    ax_line.fill_between(ks, means - ses, means + ses, color=LINE_COLOR, alpha=FILL_ALPHA)
    ax_line.plot(ks, means, "o-", color=LINE_COLOR, linewidth=LINE_LW, markersize=MARKER_SIZE, markeredgewidth=0)
    ax_line.errorbar(ks, means, yerr=ses, fmt="none", ecolor=LINE_COLOR, elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW, alpha=0.6)

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
        fontsize=FONT_SIZE_ANNOT, color="gray", style="italic",
    )

    # --- Bottom Row: Tetris ---
    gs_tetris = gridspec.GridSpec(
        1, 4, figure=subfigs[1],
        width_ratios=[1.0, 1.0, 0.80, 1.05],
        wspace=0.28,
        left=0.03, right=0.97,
        top=0.88, bottom=0.14,
    )
    ax_sparse = subfigs[1].add_subplot(gs_tetris[0])
    ax_dense  = subfigs[1].add_subplot(gs_tetris[1])
    ax_fill   = subfigs[1].add_subplot(gs_tetris[2])
    ax_piece  = subfigs[1].add_subplot(gs_tetris[3])

    panels_tetris = [
        (ax_sparse, sparse_state, "Chosen K = 1", "Sparse board"),
        (ax_dense,  dense_state,  "Chosen K = 4", "Dense board"),
    ]
    for ax, state, k_txt, title_txt in panels_tetris:
        ax.set_title(title_txt, fontsize=FONT_SIZE_TITLE, pad=6)
        draw_tetris(state, ax, viewer)
        ax.text(
            0.5, -0.05,
            k_txt,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=FONT_SIZE_K_BOX,
            color="black",
        )

    k_vals = np.array([d[0] for d in BOARD_FILL_BY_K])
    fills  = np.array([d[1] for d in BOARD_FILL_BY_K])
    f_ses  = np.array([d[2] for d in BOARD_FILL_BY_K])

    ax_fill.fill_between(k_vals, fills - f_ses, fills + f_ses, color=LINE_COLOR, alpha=FILL_ALPHA)
    ax_fill.plot(k_vals, fills, "o-", color=LINE_COLOR, linewidth=LINE_LW, markersize=MARKER_SIZE, markeredgewidth=0)
    ax_fill.errorbar(k_vals, fills, yerr=f_ses, fmt="none", ecolor=LINE_COLOR, elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW, alpha=0.6)

    ax_fill.set_xticks(k_vals)
    ax_fill.set_xticklabels([f"K={int(k)}" for k in k_vals], fontsize=FONT_SIZE_TICK)
    ax_fill.set_ylabel("Board fill fraction", fontsize=FONT_SIZE_LABEL)
    ax_fill.set_title("Board density\nby chosen K", fontsize=FONT_SIZE_TITLE)
    ax_fill.set_ylim(bottom=0)
    ax_fill.spines["top"].set_visible(False)
    ax_fill.spines["right"].set_visible(False)
    ax_fill.grid(False)
    ax_fill.text(
        0.97, 0.05, "K=3 never chosen",
        transform=ax_fill.transAxes, ha="right", va="bottom",
        fontsize=FONT_SIZE_ANNOT, color="gray", style="italic",
    )

    pieces = [d[0] for d in PIECE_MEAN_K]
    p_means = np.array([d[1] for d in PIECE_MEAN_K])
    p_ses   = np.array([d[2] for d in PIECE_MEAN_K])
    y = np.arange(len(pieces))

    BAR_HEIGHT = 0.55
    ax_piece.barh(y, p_means, height=BAR_HEIGHT, color=BAR_COLOR, alpha=BAR_ALPHA, linewidth=0)
    ax_piece.errorbar(p_means, y, xerr=p_ses, fmt="none", ecolor="black", elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)

    ax_piece.set_yticks(y)
    ax_piece.set_yticklabels(pieces, fontsize=FONT_SIZE_TICK)
    ax_piece.set_xlabel("Mean chosen K", fontsize=FONT_SIZE_LABEL)
    ax_piece.set_title("Piece complexity\nvs. deliberation", fontsize=FONT_SIZE_TITLE)
    ax_piece.spines["top"].set_visible(False)
    ax_piece.spines["right"].set_visible(False)
    ax_piece.grid(False)
    ax_piece.set_xlim(left=2.5, right=3.1)

    out = os.path.join(FIGS, "interpretability_combined.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

if __name__ == "__main__":
    print("Building PacMan states...")
    far_state, close_state = build_pacman_states()
    print("Building Tetris states...")
    sparse_state, dense_state, tetris_env = build_tetris_states()
    print("Composing figure...")
    plot(far_state, close_state, sparse_state, dense_state, tetris_env)
