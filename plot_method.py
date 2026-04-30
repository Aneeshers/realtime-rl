#!/usr/bin/env python3
"""
plot_method.py

Method figure: 3 panels explaining the committed-action planning mechanism.

  Panel A — Tetris dense board (seaborn heatmap, Blues) + gating policy P(K) bar chart
  Panel B — Execution timeline: K=1 (react immediately) vs K=4 (plan deeply)
  Panel C — MCTS search tree, nodes value-colored with Reds colormap, selected path in red

Saves: figures/method.pdf

Usage:
    python plot_method.py
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle
import seaborn as sns

# ============================================================
# Style controls  ← change these
# ============================================================

from plot_config import (
    C_BLACK, C_WHITE, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY,
    C_BLACK, C_WHITE, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY,
    FS_TITLE, FS_LABEL, FS_TICK, FS_BADGE, FS_LEGEND, FS_ANNOT,
    C_RED, C_NAVY, C_STATE, C_COMMIT, C_BLUE,
    apply_style,
)

GATING_COLOR   = C_RED
BADGE_COLOR    = C_NAVY
STATE_COLOR    = C_STATE
COMMIT_COLOR   = C_COMMIT
PLAN_WIN_COLOR = C_BLUE

FONT_SIZE_TITLE = FS_TITLE
FONT_SIZE_LABEL = FS_LABEL
FONT_SIZE_TICK  = FS_TICK
FONT_SIZE_BADGE = FS_BADGE
FONT_SIZE_BOX   = FS_LEGEND
FONT_SIZE_ANNOT = FS_ANNOT

FIG_WIDTH   = 13.5
FIG_HEIGHT  = 4.3

# ============================================================
# Setup
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

apply_style()


# ============================================================
# Build Tetris dense board (reused from plot_tetris_interpretability)
# ============================================================

def build_dense_board():
    import jax
    import jax.numpy as jnp
    from jumanji.environments.packing.tetris import Tetris

    env = Tetris()
    key = jax.random.PRNGKey(0)
    base_state, _ = jax.jit(env.reset)(key)
    nr, nc = env.num_rows, env.num_cols

    dense_grid = np.array(base_state.grid_padded)
    col_heights = [7, 5, 8, 6, 4, 8, 7, 5, 7, 6]
    for col in range(nc):
        for offset in range(col_heights[col]):
            row = nr - 1 - offset
            dense_grid[row, col] = 1

    return (dense_grid[:nr, :nc] > 0).astype(float)


# ============================================================
# Panel A: Tetris heatmap + P(K) bar chart
# ============================================================

def draw_panel_a(ax_top, ax_bot):
    board = build_dense_board()

    # -- Tetris heatmap --
    sns.heatmap(
        board,
        ax=ax_top,
        cmap="Blues",
        vmin=0, vmax=1,
        linewidths=0.4,
        linecolor=C_LIGHT_GRAY,
        cbar=False,
        square=True,
    )
    ax_top.set_axis_off()
    ax_top.set_title("State $s_t$", fontsize=FONT_SIZE_TITLE, pad=4)

    # -- P(K) bar chart (schematic gating output for a dense board) --
    probs     = [0.04, 0.08, 0.08, 0.80]
    blues_pal = sns.color_palette("Blues", n_colors=6)[3:]
    bar_cols  = list(blues_pal[:3]) + [GATING_COLOR]
    y = np.arange(4)

    ax_bot.barh(y, probs, height=0.52,
                color=bar_cols, alpha=0.88, linewidth=0)
    ax_bot.set_yticks(y)
    ax_bot.set_yticklabels(["K=1", "K=2", "K=3", "K=4"],
                            fontsize=FONT_SIZE_TICK)
    ax_bot.set_xlim(0, 1.0)
    ax_bot.set_xlabel("P(K)", fontsize=FONT_SIZE_LABEL)
    ax_bot.set_title("Gating policy", fontsize=FONT_SIZE_TITLE, pad=4)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.tick_params(labelsize=FONT_SIZE_TICK)


# ============================================================
# Panel B: Execution timeline helpers
# ============================================================

def _box(ax, x, y, w, h, fc, ec=C_DARK_GRAY, alpha=1.0, lw=0.8,
         label="", lsize=8.5, lcolor=C_BLACK):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=fc, edgecolor=ec,
                          alpha=alpha, linewidth=lw, zorder=3)
    ax.add_patch(rect)
    if label:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center",
                fontsize=lsize, color=lcolor, zorder=4)


def _arrow(ax, x0, y, x1, color=C_DARK_GRAY, lw=1.1):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=9),
                zorder=3)


def _badge(ax, x, y, text, fsize=10):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fsize, color=C_WHITE,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=BADGE_COLOR,
                      edgecolor="none", alpha=0.93),
            zorder=5)


def draw_panel_b(ax):
    ax.set_xlim(-0.8, 9.8)
    ax.set_ylim(-0.6, 8.2)
    ax.set_axis_off()
    ax.set_title("Execution timeline", fontsize=FONT_SIZE_TITLE, pad=8)

    bh  = 0.75   # standard box height
    bw  = 1.25   # state box width
    abw = 0.88   # action / committed box width
    mbw = 1.10   # MCTS box width (wider)
    mbh = 0.90   # MCTS box height (taller)
    gap = 0.30   # gap between boxes (before arrow starts)
    aw  = 0.22   # arrow length

    # ======================================================
    # K=1 strip  (y_center = 6.0)
    # ======================================================
    y1c = 6.0
    y1  = y1c - bh / 2

    _badge(ax, -0.5, y1c, "K = 1", fsize=FONT_SIZE_BADGE)

    x = 0.3
    _box(ax, x, y1, bw, bh, STATE_COLOR, label="$s_t$", lsize=FONT_SIZE_BOX)
    x += bw + gap;  _arrow(ax, x - gap, y1c, x)
    _box(ax, x, y1, abw, bh, GATING_COLOR, ec=GATING_COLOR, alpha=0.88,
         label="act", lsize=FONT_SIZE_BOX, lcolor=C_WHITE)
    x += abw + gap;  _arrow(ax, x - gap, y1c, x)
    _box(ax, x, y1, bw, bh, STATE_COLOR, label="$s_{t+1}$", lsize=FONT_SIZE_BOX)

    ax.text(0.3 + bw / 2, y1c + bh / 2 + 0.18,
            "react immediately",
            ha="center", va="bottom", fontsize=FONT_SIZE_ANNOT,
            color=C_MID_GRAY, style="italic")

    # divider
    ax.axhline(4.25, color=C_LIGHT_GRAY, linewidth=0.8, zorder=0)

    # ======================================================
    # K=4 strip  (y_center = 2.8)
    # ======================================================
    y4c = 2.8
    y4  = y4c - bh / 2
    y4m = y4c - mbh / 2   # MCTS box top

    _badge(ax, -0.5, y4c, "K = 4", fsize=FONT_SIZE_BADGE)

    x = 0.3
    _box(ax, x, y4, bw, bh, STATE_COLOR, label="$s_t$", lsize=FONT_SIZE_BOX)

    # planning window spans from first committed box to end of MCTS box
    pw_x0 = x + bw + gap
    x += bw + gap;  _arrow(ax, x - gap, y4c, x)

    # 3 committed-action boxes
    for _ in range(3):
        _box(ax, x, y4, abw, bh, COMMIT_COLOR, ec=C_MID_GRAY, alpha=0.75,
             label=r"$\pi_{reflex}$", lsize=FONT_SIZE_BOX - 1, lcolor=C_DARK_GRAY)
        x += abw + gap;  _arrow(ax, x - gap, y4c, x)

    # MCTS action box (larger, red)
    _box(ax, x, y4m, mbw, mbh, GATING_COLOR, ec=GATING_COLOR, alpha=0.88,
         label="MCTS", lsize=FONT_SIZE_BOX, lcolor=C_WHITE)
    pw_x1 = x + mbw
    x += mbw + gap;  _arrow(ax, x - gap, y4c, x)

    # final state
    _box(ax, x, y4, bw, bh, STATE_COLOR, label="$s_{t+4}$", lsize=FONT_SIZE_BOX)

    # Planning window shading
    pw_pad = 0.12
    pw_rect = FancyBboxPatch(
        (pw_x0 - pw_pad, y4m - pw_pad),
        (pw_x1 - pw_x0) + 2 * pw_pad,
        mbh + 2 * pw_pad,
        boxstyle="round,pad=0.05",
        facecolor=PLAN_WIN_COLOR, edgecolor="none",
        alpha=0.10, zorder=1,
    )
    ax.add_patch(pw_rect)

    # Bracket above planning window
    bkt_y = y4c + mbh / 2 + 0.40
    bkt_x0, bkt_x1 = pw_x0 - pw_pad, pw_x1 + pw_pad
    tick_h = 0.16
    ax.plot([bkt_x0, bkt_x1], [bkt_y, bkt_y],
            color=PLAN_WIN_COLOR, lw=1.0, zorder=2)
    ax.plot([bkt_x0, bkt_x0], [bkt_y - tick_h, bkt_y],
            color=PLAN_WIN_COLOR, lw=1.0, zorder=2)
    ax.plot([bkt_x1, bkt_x1], [bkt_y - tick_h, bkt_y],
            color=PLAN_WIN_COLOR, lw=1.0, zorder=2)
    ax.text((bkt_x0 + bkt_x1) / 2, bkt_y + 0.12,
            "MCTS planning  (128 sims)",
            ha="center", va="bottom",
            fontsize=FONT_SIZE_ANNOT, color=PLAN_WIN_COLOR)

    ax.text(0.3 + bw / 2, y4 - 0.18,
            "plan deeply",
            ha="center", va="top", fontsize=FONT_SIZE_ANNOT,
            color=C_MID_GRAY, style="italic")


# ============================================================
# Panel C: MCTS tree (schematic)
# ============================================================

def draw_panel_c(ax):
    ax.set_xlim(-0.15, 2.25)
    ax.set_ylim(0.9, 4.5)
    ax.set_axis_off()
    ax.set_title("MCTS search", fontsize=FONT_SIZE_TITLE, pad=8)
    ax.set_aspect("equal")

    cmap   = plt.cm.Reds
    R      = 0.17    # leaf / child node radius
    R_root = 0.21    # root node radius

    # (x, y, value ∈ [0,1])
    nodes = {
        0: (1.05, 3.90, 0.72),   # root
        1: (0.35, 2.85, 0.82),   # depth-1  ← best: selected path
        2: (1.05, 2.85, 0.65),
        3: (1.75, 2.85, 0.41),
        4: (0.15, 1.75, 0.88),   # depth-2  ← best leaf: selected
        5: (0.55, 1.75, 0.71),
        6: (0.85, 1.75, 0.60),
        7: (1.25, 1.75, 0.48),
        8: (1.55, 1.75, 0.33),
        9: (1.95, 1.75, 0.21),
    }
    edges = [
        (0,1),(0,2),(0,3),
        (1,4),(1,5),
        (2,6),(2,7),
        (3,8),(3,9),
    ]
    sel_edges = {(0,1),(1,4)}
    sel_nodes = {0, 1, 4}

    # Draw edges first
    for (p, c) in edges:
        px, py, _ = nodes[p]
        cx, cy, _ = nodes[c]
        rp = R_root if p == 0 else R
        is_sel = (p, c) in sel_edges
        ax.plot([px, cx], [py - rp, cy + R],
                color=GATING_COLOR if is_sel else "#d0d0d0",
                linewidth=2.2 if is_sel else 0.9,
                zorder=2 if is_sel else 1,
                solid_capstyle="round")

    # Draw nodes
    for nid, (nx, ny, val) in nodes.items():
        r  = R_root if nid == 0 else R
        fc = cmap(0.25 + val * 0.60)
        is_sel = nid in sel_nodes
        ec = GATING_COLOR if is_sel else "#aaaaaa"
        ew = 1.8 if is_sel else 0.7
        circle = Circle((nx, ny), r,
                        facecolor=fc, edgecolor=ec,
                        linewidth=ew, zorder=3)
        ax.add_patch(circle)
        txt_color = C_WHITE if val > 0.55 else "#333333"
        ax.text(nx, ny, f"{val:.2f}",
                ha="center", va="center",
                fontsize=6.0, color=txt_color, zorder=4)

    # Selected-path annotation
    ax.text(2.18, 1.75, "selected\npath",
            ha="right", va="center",
            fontsize=FONT_SIZE_ANNOT - 1, color=GATING_COLOR,
            style="italic")


# ============================================================
# Compose figure
# ============================================================

def plot():
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1.0, 1.75, 0.80],
        wspace=0.30,
        left=0.03, right=0.97,
        top=0.90, bottom=0.08,
    )

    # Panel A: two stacked sub-panels
    gs_a = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=gs[0],
        height_ratios=[1.5, 1.0],
        hspace=0.50,
    )
    ax_a_top = fig.add_subplot(gs_a[0])
    ax_a_bot = fig.add_subplot(gs_a[1])
    draw_panel_a(ax_a_top, ax_a_bot)

    # Panel B
    ax_b = fig.add_subplot(gs[1])
    draw_panel_b(ax_b)

    # Panel C
    ax_c = fig.add_subplot(gs[2])
    draw_panel_c(ax_c)

    out = os.path.join(FIGS, "method.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Building method figure...")
    plot()
