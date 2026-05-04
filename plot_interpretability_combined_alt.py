#!/usr/bin/env python3
"""
plot_interpretability_combined_alt.py

Alternative combined interpretability figure with:
  - Snake moved to the middle row.
  - Tetris board-density panel shown as a bar chart.
  - K shown on the y-axis wherever it is the conditioned variable.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_config import (
    C_BLACK, C_WHITE, C_MID_GRAY,
    FS_TITLE, FS_LABEL, FS_TICK, FS_BADGE, FS_ANNOT,
    C_BLUE, BAR_ALPHA, LINE_LW, CAPSIZE, ERR_LW,
    K_COLORS,
    apply_style,
)

apply_style()

FONT_SIZE_TITLE = FS_TITLE - 1
FONT_SIZE_LABEL = FS_LABEL - 1
FONT_SIZE_TICK = FS_TICK - 1
FONT_SIZE_K_BOX = FS_BADGE - 1
FONT_SIZE_ANNOT = FS_ANNOT - 1
BAR_COLOR = C_BLUE

PACMAN_CROP_LEFT = dict(top=44, bottom=86, left=25, right=65)
PACMAN_CROP_RIGHT = dict(top=36, bottom=78, left=20, right=60)

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

SNAKE_REACH_BY_K = [
    (1, 134.73, 0.08),
    (2, 134.43, 0.26),
    (3, 120.41, 0.46),
    (4, np.nan, np.nan),
]

SNAKE_K_OVERALL = [81.6, 15.2, 3.2, 0.0]
SNAKE_K_POST_EAT = [70.8, 15.7, 13.5, 0.0]

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)


def _spine_clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _set_k_axis(ax):
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels([f"K={k}" for k in [1, 2, 3, 4]], fontsize=FONT_SIZE_TICK)
    ax.set_ylim(0.5, 4.5)


def _plot_k_horizontal_bars(ax, data, xlabel, title, xlim=None, refline=None):
    ks = np.array([d[0] for d in data])
    means = np.array([d[1] for d in data], dtype=float)
    ses = np.array([d[2] for d in data], dtype=float)

    display_means = np.where(np.isnan(means), 0.0, means)
    display_ses = np.where(np.isnan(ses), 0.0, ses)
    alphas = [BAR_ALPHA if not np.isnan(m) else 0.25 for m in means]
    colors = [K_COLORS[k - 1] for k in ks]

    bars = ax.barh(ks, display_means, color=colors, linewidth=0, height=0.6)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    valid = ~np.isnan(means)
    ax.errorbar(
        display_means[valid],
        ks[valid],
        xerr=display_ses[valid],
        fmt="none",
        ecolor=C_BLACK,
        elinewidth=ERR_LW,
        capsize=CAPSIZE,
        capthick=ERR_LW,
    )
    if refline is not None:
        ax.axvline(refline, color=C_MID_GRAY, linestyle="--", linewidth=0.8)
    _set_k_axis(ax)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    if xlim is not None:
        ax.set_xlim(*xlim)
    _spine_clean(ax)


def build_pacman_states():
    import jax
    import jax.numpy as jnp
    from jumanji.environments.routing.pac_man import PacMan

    env = PacMan()
    key = jax.random.PRNGKey(0)
    base_state, _ = jax.jit(env.reset)(key)

    far_state = base_state
    close_ghosts = jnp.array([[11, 23], [15, 23], [11, 20], [15, 20]], dtype=jnp.int32)
    close_state = base_state.replace(ghost_locations=close_ghosts)
    return far_state, close_state


def state_to_rgb(state, crop):
    from jumanji.environments.routing.pac_man.viewer import create_grid_image

    img = np.array(create_grid_image(state))
    return img[crop["top"]:crop["bottom"], crop["left"]:crop["right"]]


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

    dense_state = base_state.replace(
        grid_padded=jnp.array(dense_grid, dtype=base_state.grid_padded.dtype)
    )
    return sparse_state, dense_state, env


def draw_tetris(state, ax, viewer):
    grid = viewer._create_rendering_grid(state)
    viewer._add_grid_image(ax, grid)
    ax.invert_yaxis()


def build_snake_states():
    import jax
    import jax.numpy as jnp
    from jumanji.environments.routing.snake import Snake
    from jumanji.environments.routing.snake.types import Position

    env = Snake()
    key = jax.random.PRNGKey(42)
    base_state, _ = jax.jit(env.reset)(key)

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


def plot(far_state, close_state, sparse_state, dense_state, tetris_env, short_snake, long_snake):
    from jumanji.environments.packing.tetris.viewer import TetrisViewer

    viewer = TetrisViewer(
        num_rows=tetris_env.num_rows,
        num_cols=tetris_env.num_cols,
        render_mode="rgb_array",
    )

    fig = plt.figure(figsize=(16.0, 10.4))
    subfigs = fig.subfigures(3, 1, hspace=0.06)
    board_to_plot_ratio = [1.0, 0.95]
    main_wspace = 0.06
    board_wspace = 0.01

    gs_top = subfigs[0].add_gridspec(1, 2, width_ratios=board_to_plot_ratio, wspace=main_wspace)
    gs_top_boards = gs_top[0].subgridspec(1, 2, wspace=board_wspace)
    gs_top_plots = gs_top[1].subgridspec(1, 2, wspace=0.30)

    ax_close = subfigs[0].add_subplot(gs_top_boards[0])
    ax_far = subfigs[0].add_subplot(gs_top_boards[1])
    ax_ghost = subfigs[0].add_subplot(gs_top_plots[0])
    ax_pellet = subfigs[0].add_subplot(gs_top_plots[1])

    for ax, state, k_txt, title_txt, crop in [
        (ax_close, close_state, "Chosen K = 1", "Ghost close", PACMAN_CROP_LEFT),
        (ax_far, far_state, "Chosen K = 4", "Ghost far", PACMAN_CROP_RIGHT),
    ]:
        ax.set_title(title_txt, fontsize=FONT_SIZE_TITLE, pad=6)
        ax.set_axis_off()
        ax.imshow(state_to_rgb(state, crop), interpolation="nearest")
        ax.text(0.5, -0.05, k_txt, transform=ax.transAxes, ha="center", va="top",
                fontsize=FONT_SIZE_K_BOX, color=C_BLACK)

    _plot_k_horizontal_bars(
        ax_ghost, GHOST_DIST_BY_K, "Nearest ghost distance", "Threat proximity", xlim=(0, 12.5)
    )
    _plot_k_horizontal_bars(
        ax_pellet, PELLET_FRAC_BY_K, "Pellet fraction", "Pellet fraction", xlim=(0, 1.05)
    )

    gs_mid = subfigs[1].add_gridspec(1, 2, width_ratios=board_to_plot_ratio, wspace=main_wspace)
    gs_mid_boards = gs_mid[0].subgridspec(1, 2, wspace=board_wspace)
    gs_mid_plots = gs_mid[1].subgridspec(1, 2, wspace=0.30)

    ax_short = subfigs[1].add_subplot(gs_mid_boards[0])
    ax_long = subfigs[1].add_subplot(gs_mid_boards[1])
    ax_reach = subfigs[1].add_subplot(gs_mid_plots[0])
    ax_posteat = subfigs[1].add_subplot(gs_mid_plots[1])

    for ax, state, k_txt, title_txt in [
        (ax_short, short_snake, "Chosen K = 2", "Short snake"),
        (ax_long, long_snake, "Chosen K = 3", "Long snake"),
    ]:
        draw_snake(state, ax)
        ax.set_title(title_txt, fontsize=FONT_SIZE_TITLE, pad=6)
        ax.text(0.5, -0.05, k_txt, transform=ax.transAxes, ha="center", va="top",
                fontsize=FONT_SIZE_K_BOX, color=C_BLACK)

    _plot_k_horizontal_bars(
        ax_reach, SNAKE_REACH_BY_K, "Reachable cells", "Reachability", xlim=(0, 148), refline=144
    )

    ks = np.array([1, 2, 3, 4])
    width = 0.18
    ax_posteat.barh(
        ks - width / 2,
        SNAKE_K_OVERALL,
        height=width,
        label="Overall",
        linewidth=0,
        color=[K_COLORS[k - 1] for k in ks],
        alpha=0.55,
    )
    ax_posteat.barh(
        ks + width / 2,
        SNAKE_K_POST_EAT,
        height=width,
        label="Post-eating",
        linewidth=0,
        color=[K_COLORS[k - 1] for k in ks],
        alpha=BAR_ALPHA,
    )
    _set_k_axis(ax_posteat)
    ax_posteat.set_xlabel("% of steps", fontsize=FONT_SIZE_LABEL)
    ax_posteat.set_title("Overall / post-eating", fontsize=FONT_SIZE_TITLE)
    ax_posteat.set_xlim(0, 95)
    ax_posteat.annotate(
        "4.2x",
        xy=(SNAKE_K_POST_EAT[2], 3 + width / 2),
        xytext=(SNAKE_K_POST_EAT[2] + 7, 3.28),
        fontsize=FONT_SIZE_ANNOT,
        color=K_COLORS[2],
        arrowprops=dict(arrowstyle="-", color=K_COLORS[2], lw=1.2),
        ha="left",
        va="center",
    )
    ax_posteat.legend(
        fontsize=FONT_SIZE_ANNOT,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.01, 0.98),
    )
    _spine_clean(ax_posteat)

    gs_bot = subfigs[2].add_gridspec(1, 2, width_ratios=board_to_plot_ratio, wspace=main_wspace)
    gs_bot_boards = gs_bot[0].subgridspec(1, 2, wspace=board_wspace)
    gs_bot_plots = gs_bot[1].subgridspec(1, 2, wspace=0.32)

    ax_sparse = subfigs[2].add_subplot(gs_bot_boards[0])
    ax_dense = subfigs[2].add_subplot(gs_bot_boards[1])
    ax_fill = subfigs[2].add_subplot(gs_bot_plots[0])
    ax_piece = subfigs[2].add_subplot(gs_bot_plots[1])

    for ax, state, k_txt, title_txt in [
        (ax_sparse, sparse_state, "Chosen K = 1", "Sparse board"),
        (ax_dense, dense_state, "Chosen K = 4", "Dense board"),
    ]:
        ax.set_title(title_txt, fontsize=FONT_SIZE_TITLE, pad=6)
        draw_tetris(state, ax, viewer)
        ax.text(0.5, -0.05, k_txt, transform=ax.transAxes, ha="center", va="top",
                fontsize=FONT_SIZE_K_BOX, color=C_BLACK)

    _plot_k_horizontal_bars(
        ax_fill, BOARD_FILL_BY_K, "Board fill fraction", "Board density", xlim=(0, 0.36)
    )

    pieces = [d[0] for d in PIECE_MEAN_K]
    means = np.array([d[1] for d in PIECE_MEAN_K])
    ses = np.array([d[2] for d in PIECE_MEAN_K])
    x = np.arange(len(pieces))
    ax_piece.bar(x, means, color=BAR_COLOR, alpha=BAR_ALPHA, linewidth=0, width=0.68)
    ax_piece.errorbar(
        x, means, yerr=ses, fmt="none", ecolor=C_BLACK,
        elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW,
    )
    ax_piece.set_xticks(x)
    ax_piece.set_xticklabels(pieces, fontsize=FONT_SIZE_TICK)
    ax_piece.set_ylabel("Mean chosen K", fontsize=FONT_SIZE_LABEL)
    ax_piece.set_xlabel("Piece type", fontsize=FONT_SIZE_LABEL)
    ax_piece.set_title("Piece complexity", fontsize=FONT_SIZE_TITLE)
    ax_piece.set_ylim(2.5, 3.1)
    _spine_clean(ax_piece)

    out = os.path.join(FIGS, "interpretability_combined_alt.pdf")
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
        far_state,
        close_state,
        sparse_state,
        dense_state,
        tetris_env,
        short_snake,
        long_snake,
    )
