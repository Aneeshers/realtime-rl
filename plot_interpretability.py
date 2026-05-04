#!/usr/bin/env python3
"""
plot_interpretability.py

Combined interpretability figure:
  Top row:    PacMan subplots
  Middle row: Tetris subplots
  Bottom row: Snake subplots (pure data — no board screenshots)
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Style — sourced from plot_config.py
# ============================================================

from plot_config import (
    C_BLACK, C_WHITE, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY,
    FS_TITLE, FS_LABEL, FS_TICK, FS_BADGE, FS_ANNOT,
    C_BLUE, FILL_ALPHA, BAR_ALPHA, LINE_LW, MARKER_SIZE, CAPSIZE, ERR_LW,
    K_COLORS,
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

# ── PacMan data ───────────────────────────────────────────────────────────────
PACMAN_CROP_LEFT = dict(
    top=44,
    bottom=86,
    left=25,
    right=65,
)

PACMAN_CROP_RIGHT = dict(
    top=36,
    bottom=78,
    left=20,
    right=60,
)

GHOST_DIST_BY_K = [
    (1, 6.83, 0.05),
    (2, 8.40, 0.07),
    (3, np.nan, np.nan),
    (4, 11.59, 0.14),
]

PELLET_FRAC_BY_K = [
    (1, 0.611, 0.002),
    (2, 0.752, 0.002),
    (3, np.nan, np.nan),
    (4, 0.985, 0.001),
]

# ── Tetris data ───────────────────────────────────────────────────────────────
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

# ── Snake data ────────────────────────────────────────────────────────────────
SNAKE_REACH_BY_K = [
    (1, 134.73, 0.08),
    (2, 134.43, 0.26),
    (3, 120.41, 0.46),
    (4, np.nan,   np.nan),
]

SNAKE_DENSITY_BY_K = [
    (1, 0.2685, 0.0008),
    (2, 0.2352, 0.0023),
    (3, 0.3359, 0.0038),
    (4, np.nan,   np.nan),
]

SNAKE_K_OVERALL  = [81.6, 15.2,  3.2, 0.0]
SNAKE_K_POST_EAT = [70.8, 15.7, 13.5, 0.0]
SNAKE_POST_EAT_N = 1546

# ─────────────────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)


def _spine_clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


# ── PacMan state builders ─────────────────────────────────────────────────────

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


def state_to_rgb(state, crop):
    from jumanji.environments.routing.pac_man.viewer import create_grid_image
    img = np.array(create_grid_image(state))
    return img[crop["top"]:crop["bottom"], crop["left"]:crop["right"]]


# ── Tetris state builders ─────────────────────────────────────────────────────

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


# ── Snake state builders ──────────────────────────────────────────────────────

def build_snake_states():
    import jax
    import jax.numpy as jnp
    from jumanji.environments.routing.snake import Snake
    from jumanji.environments.routing.snake.types import Position

    env = Snake()
    key = jax.random.PRNGKey(42)
    base_state, _ = jax.jit(env.reset)(key)

    # State 1: Short snake (K=2 trigger) — open board, fruit far away
    body_s1 = jnp.zeros((12, 12), bool)
    bs_s1 = jnp.zeros((12, 12), jnp.int32)
    path_short = [
        (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 4), (2, 3), (2, 2), (2, 1),
    ]
    for i, (r, c) in enumerate(path_short):
        body_s1 = body_s1.at[r, c].set(True)
        bs_s1 = bs_s1.at[r, c].set(i + 1)
    hr, hc = path_short[-1]
    state_short = base_state.replace(
        body=body_s1,
        body_state=bs_s1,
        head_position=Position(row=jnp.int32(hr), col=jnp.int32(hc)),
        tail=(bs_s1 == 1),
        fruit_position=Position(row=jnp.int32(9), col=jnp.int32(9)),
        length=jnp.int32(len(path_short)),
    )

    # State 2: Long snake (K=3 trigger) — crowded board, body creating corridors
    body_s2 = jnp.zeros((12, 12), bool)
    bs_s2 = jnp.zeros((12, 12), jnp.int32)
    path_long = [
        (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
        (4, 9),
        (5, 9), (5, 8), (5, 7), (5, 6), (5, 5), (5, 4), (5, 3), (5, 2),
        (6, 2),
        (7, 2), (7, 3),
    ]
    for i, (r, c) in enumerate(path_long):
        body_s2 = body_s2.at[r, c].set(True)
        bs_s2 = bs_s2.at[r, c].set(i + 1)
    hr2, hc2 = path_long[-1]
    state_long = base_state.replace(
        body=body_s2,
        body_state=bs_s2,
        head_position=Position(row=jnp.int32(hr2), col=jnp.int32(hc2)),
        tail=(bs_s2 == 1),
        fruit_position=Position(row=jnp.int32(8), col=jnp.int32(7)),
        length=jnp.int32(len(path_long)),
    )

    return state_short, state_long


def draw_snake(state, ax):
    from jumanji.environments.routing.snake.viewer import SnakeViewer
    viewer = SnakeViewer(render_mode="rgb_array")
    viewer._draw_board(ax, state)
    for patch in viewer._create_entities(state):
        ax.add_patch(patch)
    ax.set_aspect("equal")


# ── Figure ────────────────────────────────────────────────────────────────────

def plot(far_state, close_state, sparse_state, dense_state, tetris_env,
         short_snake, long_snake):
    from jumanji.environments.packing.tetris.viewer import TetrisViewer

    viewer = TetrisViewer(
        num_rows=tetris_env.num_rows,
        num_cols=tetris_env.num_cols,
        render_mode="rgb_array",
    )

    # Wider and shorter canvas.
    fig = plt.figure(figsize=(16.5, 9.2))
    subfigs = fig.subfigures(
        3, 1,
        height_ratios=[0.74, 0.74, 0.74],
        hspace=0.08,
    )

    # Common row layout:
    #   [ board block | plot block ]
    # Inside the board block, the two state panels have their own tiny wspace.
    # Inside the plot block, the chart spacing remains normal.
    outer_width_ratios = [2.95, 1.55]
    boards_wspace = 0.02
    plots_wspace = 0.25

    # ── Top Row: PacMan ───────────────────────────────────────────────────────
    gs_outer = subfigs[0].add_gridspec(
        1, 2,
        width_ratios=outer_width_ratios,
        wspace=0.06,
        left=0.02,
        right=0.985,
        top=0.96,
        bottom=0.12,
    )
    gs_boards = gs_outer[0, 0].subgridspec(
        1, 2,
        width_ratios=[1.0, 1.0],
        wspace=boards_wspace,
    )
    gs_plots = gs_outer[0, 1].subgridspec(
        1, 2,
        width_ratios=[0.82, 1.0],
        wspace=plots_wspace,
    )

    ax_close  = subfigs[0].add_subplot(gs_boards[0, 0])
    ax_far    = subfigs[0].add_subplot(gs_boards[0, 1])
    ax_line   = subfigs[0].add_subplot(gs_plots[0, 0])
    ax_pellet = subfigs[0].add_subplot(gs_plots[0, 1])

    for ax, state, k_txt, title_txt, crop in [
        (ax_close, close_state, "Chosen K = 1", "Ghost close", PACMAN_CROP_LEFT),
        (ax_far,   far_state,   "Chosen K = 4", "Ghost far",   PACMAN_CROP_RIGHT),
    ]:
        ax.set_title(title_txt, fontsize=FONT_SIZE_TITLE, pad=6)
        ax.set_axis_off()
        ax.imshow(state_to_rgb(state, crop), interpolation="nearest")
        ax.text(
            0.5, -0.05, k_txt,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=FONT_SIZE_K_BOX,
            color=C_BLACK,
        )

    ks = np.array([d[0] for d in GHOST_DIST_BY_K])
    means = np.array([d[1] for d in GHOST_DIST_BY_K])
    ses = np.array([d[2] for d in GHOST_DIST_BY_K])
    xs = np.arange(4)
    display_means = [v if not np.isnan(v) else 0.0 for v in means]
    display_ses = [v if not np.isnan(v) else 0.0 for v in ses]
    bar_alphas = [BAR_ALPHA if not np.isnan(m) else 0.25 for m in means]

    bars = ax_line.bar(xs, display_means, color=[K_COLORS[i] for i in range(4)], linewidth=0)
    for bar, alpha in zip(bars, bar_alphas):
        bar.set_alpha(alpha)

    valid_err = [i for i in range(4) if not np.isnan(means[i])]
    ax_line.errorbar(
        [xs[i] for i in valid_err],
        [display_means[i] for i in valid_err],
        yerr=[display_ses[i] for i in valid_err],
        fmt="none",
        ecolor=C_BLACK,
        elinewidth=ERR_LW,
        capsize=CAPSIZE,
        capthick=ERR_LW,
    )
    ax_line.set_xticks(xs)
    ax_line.set_xticklabels([f"K={k}" for k in ks], fontsize=FONT_SIZE_TICK)
    ax_line.set_ylabel("Nearest ghost distance", fontsize=FONT_SIZE_LABEL)
    ax_line.set_title("Threat proximity\nby chosen K", fontsize=FONT_SIZE_TITLE)
    ax_line.set_ylim(bottom=0)
    _spine_clean(ax_line)

    p_ks = np.array([d[0] for d in PELLET_FRAC_BY_K])
    p_means = np.array([d[1] for d in PELLET_FRAC_BY_K])
    p_ses = np.array([d[2] for d in PELLET_FRAC_BY_K])
    p_display = [v if not np.isnan(v) else 0.0 for v in p_means]
    p_disp_se = [v if not np.isnan(v) else 0.0 for v in p_ses]
    p_alphas = [BAR_ALPHA if not np.isnan(m) else 0.25 for m in p_means]

    bars_p = ax_pellet.bar(xs, p_display, color=[K_COLORS[i] for i in range(4)], linewidth=0)
    for bar, alpha in zip(bars_p, p_alphas):
        bar.set_alpha(alpha)

    p_valid = [i for i in range(4) if not np.isnan(p_means[i])]
    ax_pellet.errorbar(
        [xs[i] for i in p_valid],
        [p_display[i] for i in p_valid],
        yerr=[p_disp_se[i] for i in p_valid],
        fmt="none",
        ecolor=C_BLACK,
        elinewidth=ERR_LW,
        capsize=CAPSIZE,
        capthick=ERR_LW,
    )
    ax_pellet.set_xticks(xs)
    ax_pellet.set_xticklabels([f"K={k}" for k in p_ks], fontsize=FONT_SIZE_TICK)
    ax_pellet.set_ylabel("Pellet fraction", fontsize=FONT_SIZE_LABEL)
    ax_pellet.set_title("Pellet fraction\nby chosen K", fontsize=FONT_SIZE_TITLE)
    ax_pellet.set_ylim(0, 1.1)
    _spine_clean(ax_pellet)

    # ── Middle Row: Tetris ────────────────────────────────────────────────────
    gs_outer = subfigs[1].add_gridspec(
        1, 2,
        width_ratios=outer_width_ratios,
        wspace=0.06,
        left=0.02,
        right=0.985,
        top=0.96,
        bottom=0.12,
    )
    gs_boards = gs_outer[0, 0].subgridspec(
        1, 2,
        width_ratios=[1.0, 1.0],
        wspace=boards_wspace,
    )
    gs_plots = gs_outer[0, 1].subgridspec(
        1, 2,
        width_ratios=[0.82, 1.0],
        wspace=plots_wspace,
    )

    ax_sparse = subfigs[1].add_subplot(gs_boards[0, 0])
    ax_dense  = subfigs[1].add_subplot(gs_boards[0, 1])
    ax_fill   = subfigs[1].add_subplot(gs_plots[0, 0])
    ax_piece  = subfigs[1].add_subplot(gs_plots[0, 1])

    for ax, state, k_txt, title_txt in [
        (ax_sparse, sparse_state, "Chosen K = 1", "Sparse board"),
        (ax_dense,  dense_state,  "Chosen K = 4", "Dense board"),
    ]:
        ax.set_title(title_txt, fontsize=FONT_SIZE_TITLE, pad=6)
        draw_tetris(state, ax, viewer)
        ax.text(
            0.5, -0.05, k_txt,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=FONT_SIZE_K_BOX,
            color=C_BLACK,
        )

    k_vals = np.array([d[0] for d in BOARD_FILL_BY_K])
    fills = np.array([d[1] for d in BOARD_FILL_BY_K])
    f_ses = np.array([d[2] for d in BOARD_FILL_BY_K])
    ax_fill.fill_between(k_vals, fills - f_ses, fills + f_ses, color=LINE_COLOR, alpha=FILL_ALPHA)
    ax_fill.plot(
        k_vals, fills, "o-",
        color=LINE_COLOR,
        linewidth=LINE_LW,
        markersize=MARKER_SIZE,
        markeredgewidth=0,
    )
    ax_fill.errorbar(
        k_vals, fills, yerr=f_ses,
        fmt="none", ecolor=LINE_COLOR,
        elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW, alpha=0.6,
    )
    ax_fill.set_xticks(k_vals)
    ax_fill.set_xticklabels([f"K={int(k)}" for k in k_vals], fontsize=FONT_SIZE_TICK)
    ax_fill.set_ylabel("Board fill fraction", fontsize=FONT_SIZE_LABEL)
    ax_fill.set_title("Board density\nby chosen K", fontsize=FONT_SIZE_TITLE)
    ax_fill.set_ylim(bottom=0)
    _spine_clean(ax_fill)

    pieces = [d[0] for d in PIECE_MEAN_K]
    p_means = np.array([d[1] for d in PIECE_MEAN_K])
    p_ses = np.array([d[2] for d in PIECE_MEAN_K])
    y = np.arange(len(pieces))
    ax_piece.barh(y, p_means, height=0.55, color=BAR_COLOR, alpha=BAR_ALPHA, linewidth=0)
    ax_piece.errorbar(
        p_means, y, xerr=p_ses,
        fmt="none", ecolor=C_BLACK,
        elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW,
    )
    ax_piece.set_yticks(y)
    ax_piece.set_yticklabels(pieces, fontsize=FONT_SIZE_TICK)
    ax_piece.set_xlabel("Mean chosen K", fontsize=FONT_SIZE_LABEL)
    ax_piece.set_title("Piece complexity\nvs. deliberation", fontsize=FONT_SIZE_TITLE)
    _spine_clean(ax_piece)
    ax_piece.set_xlim(left=2.5, right=3.1)

    # ── Bottom Row: Snake ─────────────────────────────────────────────────────
    gs_outer = subfigs[2].add_gridspec(
        1, 2,
        width_ratios=outer_width_ratios,
        wspace=0.06,
        left=0.02,
        right=0.985,
        top=0.96,
        bottom=0.12,
    )
    gs_boards = gs_outer[0, 0].subgridspec(
        1, 2,
        width_ratios=[1.0, 1.0],
        wspace=boards_wspace,
    )
    gs_plots = gs_outer[0, 1].subgridspec(
        1, 2,
        width_ratios=[0.82, 1.0],
        wspace=plots_wspace,
    )

    ax_short   = subfigs[2].add_subplot(gs_boards[0, 0])
    ax_long    = subfigs[2].add_subplot(gs_boards[0, 1])
    ax_reach   = subfigs[2].add_subplot(gs_plots[0, 0])
    ax_posteat = subfigs[2].add_subplot(gs_plots[0, 1])

    for ax, state, k_txt, title_txt in [
        (ax_short, short_snake, "Chosen K = 2", "Short snake"),
        (ax_long,  long_snake,  "Chosen K = 3", "Long snake"),
    ]:
        draw_snake(state, ax)
        ax.set_title(title_txt, fontsize=FONT_SIZE_TITLE, pad=6)
        ax.text(
            0.5, -0.05, k_txt,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=FONT_SIZE_K_BOX,
            color=C_BLACK,
        )

    k_idx = [0, 1, 2, 3]
    reach_means = [SNAKE_REACH_BY_K[k][1] for k in k_idx]
    reach_ses = [SNAKE_REACH_BY_K[k][2] for k in k_idx]
    bar_colors = [K_COLORS[k] for k in k_idx]
    bar_alphas = [BAR_ALPHA if not np.isnan(reach_means[k]) else 0.25 for k in k_idx]
    display_means = [v if not np.isnan(v) else 0.0 for v in reach_means]
    display_ses = [v if not np.isnan(v) else 0.0 for v in reach_ses]
    xs = np.arange(4)

    bars = ax_reach.bar(xs, display_means, color=bar_colors, linewidth=0)
    for bar, alpha in zip(bars, bar_alphas):
        bar.set_alpha(alpha)

    valid_err = [i for i in range(4) if not np.isnan(reach_means[i])]
    ax_reach.errorbar(
        [xs[i] for i in valid_err],
        [display_means[i] for i in valid_err],
        yerr=[display_ses[i] for i in valid_err],
        fmt="none",
        ecolor=C_BLACK,
        elinewidth=ERR_LW,
        capsize=CAPSIZE,
        capthick=ERR_LW,
    )
    ax_reach.set_xticks(xs)
    ax_reach.set_xticklabels(["K=1", "K=2", "K=3", "K=4"], fontsize=FONT_SIZE_TICK)
    ax_reach.set_ylabel("Reachable cells", fontsize=FONT_SIZE_LABEL)
    ax_reach.set_title("Reachability\nby chosen K", fontsize=FONT_SIZE_TITLE)
    ax_reach.set_ylim(0, 148)
    ax_reach.axhline(144, color=C_MID_GRAY, linestyle="--", linewidth=0.8)
    _spine_clean(ax_reach)

    x_k = np.arange(4)
    width = 0.36
    ax_posteat.bar(
        x_k - width / 2,
        SNAKE_K_OVERALL,
        width=width,
        label="Overall",
        linewidth=0,
        color=[K_COLORS[k] for k in range(4)],
        alpha=0.55,
    )
    ax_posteat.bar(
        x_k + width / 2,
        SNAKE_K_POST_EAT,
        width=width,
        label="Post-eating",
        linewidth=0,
        color=[K_COLORS[k] for k in range(4)],
        alpha=BAR_ALPHA,
    )
    ax_posteat.set_xticks(x_k)
    ax_posteat.set_xticklabels(["K=1", "K=2", "K=3", "K=4"], fontsize=FONT_SIZE_TICK)
    ax_posteat.set_ylabel("% of steps", fontsize=FONT_SIZE_LABEL)
    ax_posteat.set_title("Overall vs. \npost-eating", fontsize=FONT_SIZE_TITLE)
    ax_posteat.set_ylim(0, 95)
    ax_posteat.annotate(
        "4.2x",
        xy=(2 + width / 2, SNAKE_K_POST_EAT[2]),
        xytext=(2 + width / 2 + 0.05, SNAKE_K_POST_EAT[2] + 8),
        fontsize=FONT_SIZE_ANNOT,
        color=K_COLORS[2],
        arrowprops=dict(arrowstyle="-", color=K_COLORS[2], lw=1.2),
        ha="left",
    )
    ax_posteat.legend(fontsize=FONT_SIZE_ANNOT, frameon=False, loc="upper right")
    _spine_clean(ax_posteat)

    out = os.path.join(FIGS, "interpretability_combined.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Building PacMan states...")
    far_state, close_state = build_pacman_states()
    print("Building Tetris states...")
    sparse_state, dense_state, tetris_env = build_tetris_states()
    print("Building Snake states...")
    short_snake, long_snake = build_snake_states()
    print("Composing figure...")
    plot(
        far_state, close_state,
        sparse_state, dense_state, tetris_env,
        short_snake, long_snake,
    )