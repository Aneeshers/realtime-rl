#!/usr/bin/env python3
"""Animated side-by-side option-timeline for blog release.

Top row: K=1 (snap decision).  Bottom row: K=4 (deeper plan).
Both rows start their "gate picks K" beat simultaneously so the viewer sees
K=1 finish its planned action while K=4 is still mid-MCTS.

Saves:
  figures/option_timeline.gif

Usage:
  python animate_option_timeline.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import seaborn as sns

from plot_config import (
    C_BLACK, C_BLUE, C_DARK_GRAY, C_MID_GRAY,
    C_NAVY, C_RED, C_STATE,
    FS_ANNOT, FS_BADGE,
    apply_style,
)

apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)
OUT_GIF = os.path.join(FIGS, "option_timeline.gif")

FPS = 30
T_TOTAL = 9.0
N_FRAMES = int(FPS * T_TOTAL)

FIG_W = 13.6
FIG_H = 4.8

STATE_W = 1.15
STATE_H = 0.62

PLUS_FONT = 5
FONT_STATE  = FS_ANNOT + 1 + PLUS_FONT
FONT_LABEL  = FS_ANNOT + 1 + PLUS_FONT
FONT_SMALL  = FS_ANNOT - 1 + PLUS_FONT
FONT_GATE   = FS_BADGE - 3 + PLUS_FONT
FONT_ACTION = FS_ANNOT + 2 + PLUS_FONT

FINAL_STATE_COLOR = sns.color_palette("crest", 7)[4]
OPTION_COLOR      = sns.color_palette("crest", 7)[5]

X_STATES_K4 = [0.0, 2.0, 4.0, 6.0, 8.6]
X_STATES_K1 = [0.0, 2.6]
X_RANGE = (-2.7, 10.2)
Y_RANGE = (-0.92, 1.18)

GATE_LABEL_X = -1.75
GATE_LABEL_Y = 0.15


def _ease(t, start, end):
    if end <= start:
        return 1.0 if t >= end else 0.0
    u = (t - start) / (end - start)
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return u * u * (3.0 - 2.0 * u)


def _make_state_box(ax, x, y, label, facecolor, edgecolor):
    rect = FancyBboxPatch(
        (x - STATE_W / 2, y - STATE_H / 2),
        STATE_W, STATE_H,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=1.6, zorder=3,
    )
    ax.add_patch(rect)
    txt = ax.text(x, y, label, ha="center", va="center",
                  fontsize=FONT_STATE, color=C_BLACK, zorder=4)
    rect.set_alpha(0.0)
    txt.set_alpha(0.0)
    return rect, txt


def _make_arrow(ax, x0, x1, y, color, lw=2.1):
    src = (x0 + STATE_W / 2 - 0.05, y)
    dst = (x1 - STATE_W / 2 + 0.05, y)
    arr = FancyArrowPatch(
        posA=src, posB=src,
        arrowstyle="-|>", color=color, lw=lw,
        mutation_scale=10, shrinkA=0, shrinkB=0, zorder=2,
    )
    ax.add_patch(arr)
    arr.set_alpha(0.0)
    return arr, src, dst


def _make_arrow_label(ax, x0, x1, y, label, color, dy=0.28, fsize=None):
    if fsize is None:
        fsize = FONT_SMALL
    txt = ax.text(0.5 * (x0 + x1), y + dy, label,
                  ha="center", va="bottom", fontsize=fsize, color=color)
    txt.set_alpha(0.0)
    return txt


def _make_mcts_bar(ax, x0, x1, y, text):
    line = Line2D([x0, x0], [y, y], color=C_RED, lw=3.0,
                  solid_capstyle="round", zorder=1)
    ax.add_line(line)
    line.set_alpha(0.0)
    cap = ax.text(0.5 * (x0 + x1), y - 0.13, text,
                  ha="center", va="top", fontsize=FONT_SMALL, color=C_RED)
    cap.set_alpha(0.0)
    return line, cap, x0, x1


def _make_top_bracket(ax, x0, x1, y, text):
    top = Line2D([x0, x0], [y, y], color=OPTION_COLOR, lw=1.8, zorder=1)
    left = Line2D([x0, x0], [y, y - 0.09], color=OPTION_COLOR, lw=1.8, zorder=1)
    right = Line2D([x0, x0], [y, y - 0.09], color=OPTION_COLOR, lw=1.8, zorder=1)
    for ln in (top, left, right):
        ax.add_line(ln)
        ln.set_alpha(0.0)
    txt = ax.text(0.5 * (x0 + x1), y + 0.08, text,
                  ha="center", va="bottom",
                  fontsize=FONT_LABEL, color=OPTION_COLOR)
    txt.set_alpha(0.0)
    return top, left, right, txt, x0, x1


def _make_gate(ax, y, K):
    label = ax.text(GATE_LABEL_X, y + GATE_LABEL_Y, f"gate\npicks $K={K}$",
                    ha="center", va="center",
                    fontsize=FONT_GATE, color=C_NAVY)
    label.set_alpha(0.0)
    return label


def _make_meta_caption(ax, x, y, label):
    txt = ax.text(x, y - 0.69, label,
                  ha="center", va="top",
                  fontsize=FONT_SMALL, color=C_DARK_GRAY)
    txt.set_alpha(0.0)
    return txt


def _interp_arrow(arr, src, dst, progress):
    progress = 0.0 if progress < 0.0 else (1.0 if progress > 1.0 else progress)
    end_x = src[0] + progress * (dst[0] - src[0])
    end_y = src[1] + progress * (dst[1] - src[1])
    arr.set_positions(src, (end_x, end_y))


def build_row_k1(ax):
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.axis("off")
    A = {}
    A["gate_label"] = _make_gate(ax, 0.0, 1)
    A["state_t"]  = _make_state_box(ax, X_STATES_K1[0], 0.0, r"$s_t$", C_STATE, C_BLUE)
    A["state_t1"] = _make_state_box(ax, X_STATES_K1[1], 0.0, r"$s_{t+1}$", "#eef7ea", FINAL_STATE_COLOR)
    A["plan_arr"], A["plan_src"], A["plan_dst"] = _make_arrow(
        ax, X_STATES_K1[0], X_STATES_K1[1], 0.0, C_RED, lw=2.4)
    A["plan_lbl"] = _make_arrow_label(
        ax, X_STATES_K1[0], X_STATES_K1[1], 0.0,
        r"$\pi_{\mathrm{plan}}^{(K)}$", C_RED, dy=0.22, fsize=FONT_ACTION)
    A["bracket"]  = _make_top_bracket(
        ax, X_STATES_K1[0], X_STATES_K1[1], 0.68, r"option $o_K$  ($K{=}1$)")
    A["caption"]  = _make_meta_caption(ax, X_STATES_K1[1], 0.0, r"meta level at $s_{t+1}$")
    return A


def build_row_k4(ax):
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.axis("off")
    A = {}
    A["gate_label"] = _make_gate(ax, 0.0, 4)
    A["state_t"]  = _make_state_box(ax, X_STATES_K4[0], 0.0, r"$s_t$",       C_STATE, C_BLUE)
    A["state_t1"] = _make_state_box(ax, X_STATES_K4[1], 0.0, r"$s_{t+1}$",   C_STATE, C_BLUE)
    A["state_t2"] = _make_state_box(ax, X_STATES_K4[2], 0.0, r"$\cdots$",    C_STATE, C_BLUE)
    A["state_t3"] = _make_state_box(ax, X_STATES_K4[3], 0.0, r"$s_{t+K-1}$", C_STATE, C_BLUE)
    A["state_t4"] = _make_state_box(ax, X_STATES_K4[4], 0.0, r"$s_{t+K}$",   "#eef7ea", FINAL_STATE_COLOR)

    A["refl1"] = _make_arrow(ax, X_STATES_K4[0], X_STATES_K4[1], 0.0, C_MID_GRAY)
    A["refl1_lbl"] = _make_arrow_label(ax, X_STATES_K4[0], X_STATES_K4[1], 0.0,
                                       r"$\pi_{\mathrm{reflex}}$", C_MID_GRAY, fsize=FONT_ACTION)
    A["refl2"] = _make_arrow(ax, X_STATES_K4[1], X_STATES_K4[2], 0.0, C_MID_GRAY)
    A["refl2_lbl"] = _make_arrow_label(ax, X_STATES_K4[1], X_STATES_K4[2], 0.0,
                                       r"$\pi_{\mathrm{reflex}}$", C_MID_GRAY, fsize=FONT_ACTION)
    A["refl3"] = _make_arrow(ax, X_STATES_K4[2], X_STATES_K4[3], 0.0, C_MID_GRAY)
    A["refl3_lbl"] = _make_arrow_label(ax, X_STATES_K4[2], X_STATES_K4[3], 0.0,
                                       r"$\pi_{\mathrm{reflex}}$", C_MID_GRAY, fsize=FONT_ACTION)
    A["plan"] = _make_arrow(ax, X_STATES_K4[3], X_STATES_K4[4], 0.0, C_RED, lw=2.4)
    A["plan_lbl"] = _make_arrow_label(ax, X_STATES_K4[3], X_STATES_K4[4], 0.0,
                                      r"$\pi_{\mathrm{plan}}^{(K)}$", C_RED, dy=0.22, fsize=FONT_ACTION)
    A["mcts"]    = _make_mcts_bar(ax, X_STATES_K4[0], X_STATES_K4[3], -0.58, r"MCTS runs ($K$ frames)")
    A["bracket"] = _make_top_bracket(ax, X_STATES_K4[0], X_STATES_K4[4], 0.68,
                                     r"option $o_K$ (holding time $K$)")
    A["caption"] = _make_meta_caption(ax, X_STATES_K4[4], 0.0, r"meta level at $s_{t+K}$")
    return A


def _set_box_alpha(state_pair, a):
    rect, txt = state_pair
    rect.set_alpha(a)
    txt.set_alpha(a)


def _step_arrow(row, arrow_key, state_key, label_key, t, t0, t1):
    a = _ease(t, t0, t1)
    arr, src, dst = row[arrow_key]
    arr.set_alpha(a)
    _interp_arrow(arr, src, dst, a)
    row[label_key].set_alpha(a)
    _set_box_alpha(row[state_key], a)


def update(frame_idx, k1, k4):
    t = frame_idx / FPS

    # Both rows: gate label + s_t  (0.0–1.1)
    a_gate = _ease(t, 0.0, 1.1)
    for row in (k1, k4):
        row["gate_label"].set_alpha(a_gate)
        _set_box_alpha(row["state_t"], a_gate)

    # K=1: planned arrow + final state (1.4–2.2)
    a = _ease(t, 1.4, 2.2)
    k1["plan_arr"].set_alpha(a)
    _interp_arrow(k1["plan_arr"], k1["plan_src"], k1["plan_dst"], a)
    k1["plan_lbl"].set_alpha(a)
    _set_box_alpha(k1["state_t1"], a)

    # K=4: MCTS bar fades in fast, grows over 1.4–4.4
    line, cap, x0, x1 = k4["mcts"]
    line.set_alpha(_ease(t, 1.4, 1.7))
    cap.set_alpha(_ease(t, 1.4, 1.7))
    prog = _ease(t, 1.4, 4.4)
    line.set_xdata([x0, x0 + prog * (x1 - x0)])

    # K=4 reflex steps
    _step_arrow(k4, "refl1", "state_t1", "refl1_lbl", t, 1.4, 2.1)
    _step_arrow(k4, "refl2", "state_t2", "refl2_lbl", t, 2.1, 2.8)
    _step_arrow(k4, "refl3", "state_t3", "refl3_lbl", t, 2.8, 3.5)

    # K=4 planned arrow + final state (3.5–4.5)
    a = _ease(t, 3.5, 4.5)
    arr, src, dst = k4["plan"]
    arr.set_alpha(a)
    _interp_arrow(arr, src, dst, a)
    k4["plan_lbl"].set_alpha(a)
    _set_box_alpha(k4["state_t4"], a)

    # Top brackets + meta captions (4.7–5.5)
    a_brk = _ease(t, 4.7, 5.5)
    for row in (k1, k4):
        top_line, left, right, lbl_txt, bx0, bx1 = row["bracket"]
        top_line.set_alpha(a_brk)
        left.set_alpha(a_brk)
        right.set_alpha(a_brk)
        lbl_txt.set_alpha(a_brk)
        right_x = bx0 + a_brk * (bx1 - bx0)
        top_line.set_xdata([bx0, right_x])
        right.set_xdata([right_x, right_x])
        row["caption"].set_alpha(a_brk)

    return []


def main():
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.02, hspace=0.05)
    k1 = build_row_k1(ax_top)
    k4 = build_row_k4(ax_bot)

    anim = FuncAnimation(
        fig, update, frames=N_FRAMES,
        fargs=(k1, k4),
        interval=1000.0 / FPS, blit=False,
    )
    anim.save(OUT_GIF, writer=PillowWriter(fps=FPS), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_GIF}")


if __name__ == "__main__":
    main()
