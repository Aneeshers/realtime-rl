#!/usr/bin/env python3
"""
plot_tetris_committed.py  (v5)

5-frame committed-action strip using jumanji TetrisViewer rendering.
- Background filled cells: uniform semi-transparent gray
- Time label above each board: "$t{+}1$ (left)" — action in parentheses
- Action label below each board: "spawn" / "commit: π≈" / "MCTS action"
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import ConnectionPatch

# ============================================================
# Paths & style
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

sns.set_theme(style="white", font_scale=1.0)
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.weight":      "normal",
    "axes.titleweight": "normal",
    "axes.grid":        False,
})

# ============================================================
# Font sizes  ← adjust these
# ============================================================
BASE = 7
FS_TIME     = 16 + BASE   # time label above board ("$t{+}1$ (left)")
FS_BOT      = 15 + BASE   # action label below board ("commit: π≈", "MCTS action")
FS_ARRLBL   =  18 + BASE   # ← + ↓ labels on arrows
FS_BADGE    = 15 + BASE   # K = 4 badge
FS_BRACKET  = 16 + BASE   # MCTS planning bracket text
FS_GAP      =  15 + BASE    # "gap" annotation

# ============================================================
# Colors & geometry
# ============================================================

GATING_COLOR  = "#C94040"
BADGE_COLOR   = "#1a3a6b"
BG_FILL_RGBA  = (0.50, 0.50, 0.50, 0.65)   # semi-transparent gray

FIG_WIDTH  = 16.0
FIG_HEIGHT = 5.5

NROWS   = 8
NCOLS   = 10
RG_ROWS = 4 + NROWS   # 12 rows: 4 spawn + 8 board

# Cell values → jumanji color_id = value % 10 + 1
#   CV_BG     = 3  → id 4 → overridden to BG_FILL_RGBA (gray)
#   CV_COMMIT = 6  → id 7 → HSV(0.6) blue
#   CV_MCTS   = 10 → id 1 → HSV(0.0) red
CV_BG     = 3
CV_COMMIT = 6
CV_MCTS   = 10

# ============================================================
# Rendering grids
# ============================================================

def _base_rg() -> np.ndarray:
    rg = np.zeros((RG_ROWS, NCOLS), dtype=int)
    rg[8,  :6]  = CV_BG   # board row 4, cols 0-5 (gap at 6-9)
    rg[9:12, :] = CV_BG   # board rows 5-7, all cols
    return rg


# time_lbl: shown above board (includes parenthetical action)
# bot_lbl:  shown below board (descriptive)
FRAMES = [
    dict(
        rg_piece=[(2, 3), (2, 4), (2, 5), (2, 6)],
        cv=CV_COMMIT,
        time_lbl="$t$",
        bot_lbl="spawn",
        arr_lbl=None,
    ),
    dict(
        rg_piece=[(4, 2), (4, 3), (4, 4), (4, 5)],
        cv=CV_COMMIT,
        time_lbl="$t{+}1$  (left)",
        bot_lbl=r"commit: $\pi_{\approx}$",
        arr_lbl="← + ↓",
    ),
    dict(
        rg_piece=[(5, 1), (5, 2), (5, 3), (5, 4)],
        cv=CV_COMMIT,
        time_lbl="$t{+}2$  (left)",
        bot_lbl=r"commit: $\pi_{\approx}$",
        arr_lbl="← + ↓",
    ),
    dict(
        rg_piece=[(6, 0), (6, 1), (6, 2), (6, 3)],
        cv=CV_COMMIT,
        time_lbl="$t{+}3$  (left)",
        bot_lbl=r"commit: $\pi_{\approx}$",
        arr_lbl="← + ↓",
    ),
    dict(
        rg_piece=[(8, 6), (8, 7), (8, 8), (8, 9)],
        cv=CV_MCTS,
        time_lbl="$t{+}4$  (place)",
        bot_lbl="MCTS action",
        arr_lbl=None,
    ),
]


def build_rg(frame: dict) -> np.ndarray:
    rg = _base_rg()
    for (r, c) in frame["rg_piece"]:
        rg[r, c] = frame["cv"]
    return rg


def draw_board(ax, rg, viewer):
    viewer._draw_grid(rg, ax)
    ax.set_axis_off()
    ax.set_aspect(1)
    ax.relim()
    ax.autoscale_view()
    ax.invert_yaxis()


# ============================================================
# Main figure
# ============================================================

def plot():
    from jumanji.environments.packing.tetris.viewer import TetrisViewer

    viewer = TetrisViewer(num_rows=NROWS, num_cols=NCOLS, render_mode="rgb_array")
    # Override background cell color to semi-transparent gray
    bg_cid = CV_BG % (len(viewer.colors) - 1) + 1
    viewer.colors[bg_cid] = BG_FILL_RGBA

    n = len(FRAMES)
    fig, axes = plt.subplots(
        1, n,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        gridspec_kw={"wspace": 0.22},   # equal widths; extra wspace for arrows
    )
    # Vertical budget:
    #   0.18–0.72 : board axes
    #   above 0.72: time labels (via ax.text, clip_on=False) + MCTS bracket
    #   below 0.18: action labels ("commit: π≈" etc.)
    fig.subplots_adjust(top=0.72, bottom=0.18, left=0.08, right=0.97)

    # ── Draw boards + labels ───────────────────────────────────
    for i, (ax, frame) in enumerate(zip(axes, FRAMES)):
        rg = build_rg(frame)
        draw_board(ax, rg, viewer)

        is_mcts = frame["cv"] == CV_MCTS

        # Time label ABOVE board — single line, includes "(left)" for commit frames
        ax.text(0.5, 1.03, frame["time_lbl"],
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=FS_TIME,
                color=GATING_COLOR if is_mcts else "#222222",
                clip_on=False)

        # Action label BELOW board
        ax.text(0.5, -0.06, frame["bot_lbl"],
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=FS_BOT,
                color=GATING_COLOR if is_mcts else "black",
                style="normal" if i == 0 else "italic",
                clip_on=False)

    # ── Arrows between boards ──────────────────────────────────
    # Removed as requested


    # ── MCTS planning bracket ──────────────────────────────────
    # Bracket spans axes 1–4; placed in the band between time labels and fig top
    ax1_l = fig.transFigure.inverted().transform(
        axes[1].transAxes.transform((0.0, 1.0)))
    ax4_r = fig.transFigure.inverted().transform(
        axes[4].transAxes.transform((1.0, 1.0)))
    # Place bracket at ~0.81, an intermediate height
    bkt_y  = 0.81
    tick_h = 0.016

    fig.lines.append(plt.Line2D(
        [ax1_l[0], ax4_r[0]], [bkt_y, bkt_y],
        transform=fig.transFigure, color=GATING_COLOR, lw=1.3, zorder=5))
    for xb in [ax1_l[0], ax4_r[0]]:
        fig.lines.append(plt.Line2D(
            [xb, xb], [bkt_y - tick_h, bkt_y],
            transform=fig.transFigure, color=GATING_COLOR, lw=1.3, zorder=5))
    fig.text(0.5 * (ax1_l[0] + ax4_r[0]), bkt_y + 0.008,
             "MCTS planning  (128 simulations)",
             ha="center", va="bottom",
             fontsize=FS_BRACKET, color=GATING_COLOR,
             transform=fig.transFigure)

    # ── "K = 4" badge left of first board ──────────────────────
    ax0_mid = fig.transFigure.inverted().transform(
        axes[0].transAxes.transform((0.0, 0.5)))
    fig.text(ax0_mid[0] - 0.034, ax0_mid[1], "K = 4",
             ha="center", va="center",
             fontsize=FS_BADGE, color="black",
             transform=fig.transFigure, zorder=6)

    # ── "gap" annotation on last board ─────────────────────────
    # Removed as requested

    out = os.path.join(FIGS, "tetris_committed.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
