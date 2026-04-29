#!/usr/bin/env python3
"""
plot_tetris_interpretability.py

Composite 4-panel figure for Tetris RT interpretability.

Panels:
  1. Sparse board  → Chosen K = 1  (react immediately on easy boards)
  2. Dense board   → Chosen K = 4  (plan deeply on critical boards)
  3. Compact line: board fill fraction by chosen K  (K=1, K=2, K=4; K=3 never selected)
  4. Compact horizontal bar: mean chosen K by piece type (sorted low→high)

Board construction:
  Sparse state: jumanji Tetris reset state (empty 10×10 board)
  Dense state:  reset state with bottom 7 rows filled using varied piece-color IDs

All SE values are from real RT_KT gating-policy rollouts.

Saves: figures/tetris_interpretability.pdf

Usage:
    python plot_tetris_interpretability.py
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ============================================================
# Style controls  ← change these
# ============================================================

FONT_SIZE_TITLE  = 14       # subplot titles
FONT_SIZE_LABEL  = 12       # axis labels
FONT_SIZE_TICK   = 10       # tick labels
FONT_SIZE_K_BOX  = 13       # "Chosen K = N" badge on screenshots
FONT_SIZE_ANNOT  = 9        # small italic notes

FIG_WIDTH        = 13.0
FIG_HEIGHT       = 4.4

LINE_COLOR       = "#2c6fad"   # color for board-density line
LINE_LW          = 2.0
MARKER_SIZE      = 7
FILL_ALPHA       = 0.18
CAPSIZE          = 3
ERR_LW           = 0.9

BAR_COLOR        = "#2c6fad"   # color for piece-type bars
BAR_ALPHA        = 0.85
BAR_HEIGHT       = 0.55

# ============================================================
# Real data from RT_KT gating-policy rollouts
# ============================================================

# Board fill by chosen K  (K=3 omitted — never selected)
BOARD_FILL_BY_K = [
    (1, 0.049, 0.001),
    (2, 0.322, 0.005),
    (4, 0.250, 0.002),
]

# Mean chosen K by piece type (sorted low → high)
PIECE_MEAN_K = [
    ("J", 2.66, 0.04),
    ("S", 2.74, 0.04),
    ("I", 2.82, 0.04),
    ("L", 2.84, 0.04),
    ("T", 2.88, 0.04),
    ("O", 2.88, 0.04),
    ("Z", 2.98, 0.04),
]

# ============================================================
# Setup
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

sns.set_theme(style="white", font_scale=1.0)
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.weight":       "normal",
    "axes.titleweight":  "normal",
    "axes.labelweight":  "normal",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
})


# ============================================================
# Construct Tetris states
# ============================================================

def build_states():
    import jax
    import jax.numpy as jnp
    from jumanji.environments.packing.tetris import Tetris

    env = Tetris()
    key = jax.random.PRNGKey(0)
    base_state, _ = jax.jit(env.reset)(key)

    sparse_state = base_state  # empty board

    # Dense state: fill bottom 7 of 10 rows with varying piece-color IDs (1–7)
    # Mimics a late-game critical board with stacked pieces and one-gap holes
    nr, nc = env.num_rows, env.num_cols
    dense_grid = np.array(base_state.grid_padded)

    # Row patterns (0 = empty cell, 1–7 = piece-color IDs)
    # Rows 3–9 are filled; rows 0–2 left empty (piece falls from top)
    patterns = {
        9: [1, 2, 3, 4, 5, 6, 7, 1, 2, 3],   # nearly full bottom
        8: [1, 0, 3, 4, 5, 6, 7, 1, 2, 3],
        7: [0, 2, 3, 0, 5, 6, 7, 1, 0, 3],
        6: [1, 2, 0, 4, 5, 0, 7, 1, 2, 0],
        5: [1, 2, 3, 4, 0, 6, 0, 1, 2, 3],
        4: [0, 2, 3, 4, 5, 6, 7, 0, 2, 0],
        3: [1, 0, 3, 0, 5, 0, 7, 1, 0, 3],
    }
    for row_idx, pattern in patterns.items():
        dense_grid[row_idx, :nc] = pattern

    dense_grid_jnp = jnp.array(dense_grid, dtype=base_state.grid_padded.dtype)
    dense_state = base_state.replace(grid_padded=dense_grid_jnp)

    fill_sparse = float((np.array(sparse_state.grid_padded)[:nr, :nc] > 0).sum()) / (nr * nc)
    fill_dense  = float((dense_grid[:nr, :nc] > 0).sum()) / (nr * nc)
    print(f"Sparse board fill: {fill_sparse:.3f}")
    print(f"Dense  board fill: {fill_dense:.3f}")

    return sparse_state, dense_state, env


# ============================================================
# Draw a Tetris state into an existing axes
# ============================================================

def draw_tetris(state, ax, viewer):
    """Render a Tetris state into ax using the viewer's drawing logic."""
    grid = viewer._create_rendering_grid(state)
    viewer._add_grid_image(ax, grid)
    ax.invert_yaxis()


# ============================================================
# Figure
# ============================================================

def plot(sparse_state, dense_state, env):
    from jumanji.environments.packing.tetris.viewer import TetrisViewer

    viewer = TetrisViewer(
        num_rows=env.num_rows, num_cols=env.num_cols, render_mode="rgb_array"
    )

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = gridspec.GridSpec(
        1, 4,
        width_ratios=[1.0, 1.0, 0.80, 1.05],
        wspace=0.28,
        left=0.03, right=0.97,
        top=0.88, bottom=0.14,
    )
    ax_sparse = fig.add_subplot(gs[0])
    ax_dense  = fig.add_subplot(gs[1])
    ax_fill   = fig.add_subplot(gs[2])
    ax_piece  = fig.add_subplot(gs[3])

    # ---- Tetris board screenshots ----
    panels = [
        (ax_sparse, sparse_state, "Chosen K = 1", "Sparse board"),
        (ax_dense,  dense_state,  "Chosen K = 4", "Dense board"),
    ]
    for ax, state, k_txt, title_txt in panels:
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

    # ---- Line+scatter: board fill by chosen K ----
    k_vals = np.array([d[0] for d in BOARD_FILL_BY_K])
    fills  = np.array([d[1] for d in BOARD_FILL_BY_K])
    f_ses  = np.array([d[2] for d in BOARD_FILL_BY_K])

    ax_fill.fill_between(k_vals, fills - f_ses, fills + f_ses,
                         color=LINE_COLOR, alpha=FILL_ALPHA)
    ax_fill.plot(k_vals, fills, "o-",
                 color=LINE_COLOR, linewidth=LINE_LW,
                 markersize=MARKER_SIZE, markeredgewidth=0)
    ax_fill.errorbar(k_vals, fills, yerr=f_ses,
                     fmt="none", ecolor=LINE_COLOR,
                     elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW,
                     alpha=0.6)

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

    # ---- Horizontal bar: mean K by piece type ----
    pieces = [d[0] for d in PIECE_MEAN_K]
    p_means = np.array([d[1] for d in PIECE_MEAN_K])
    p_ses   = np.array([d[2] for d in PIECE_MEAN_K])
    y = np.arange(len(pieces))

    ax_piece.barh(y, p_means, height=BAR_HEIGHT,
                  color=BAR_COLOR, alpha=BAR_ALPHA, linewidth=0)
    ax_piece.errorbar(p_means, y, xerr=p_ses,
                      fmt="none", ecolor="black",
                      elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)

    ax_piece.set_yticks(y)
    ax_piece.set_yticklabels(pieces, fontsize=FONT_SIZE_TICK)
    ax_piece.set_xlabel("Mean chosen K", fontsize=FONT_SIZE_LABEL)
    ax_piece.set_title("Piece complexity\nvs. deliberation", fontsize=FONT_SIZE_TITLE)
    ax_piece.spines["top"].set_visible(False)
    ax_piece.spines["right"].set_visible(False)
    ax_piece.grid(False)
    # tight x range around the data
    ax_piece.set_xlim(left=2.5, right=3.1)

    out = os.path.join(FIGS, "tetris_interpretability.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ============================================================

if __name__ == "__main__":
    print("Building Tetris states...")
    sparse_state, dense_state, env = build_states()
    print("Composing figure...")
    plot(sparse_state, dense_state, env)
