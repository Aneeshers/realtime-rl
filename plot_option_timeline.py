#!/usr/bin/env python3
"""
plot_option_timeline.py

Compact horizontal timeline for the budgeted-option formalism in Section 3.1.

Saves:
  figures/option_timeline.pdf
  figures/option_timeline.png

Usage:
  python plot_option_timeline.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from plot_config import (
    C_BLACK,
    C_BLUE,
    C_COMMIT,
    C_DARK_GRAY,
    C_LIGHT_GRAY,
    C_MID_GRAY,
    C_NAVY,
    C_RED,
    C_STATE,
    C_WHITE,
    FS_ANNOT,
    FS_BADGE,
    FS_LABEL,
    apply_style,
)

apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

OUT_PDF = os.path.join(FIGS, "option_timeline.pdf")
OUT_PNG = os.path.join(FIGS, "option_timeline.png")

FIG_W = 13.0
FIG_H = 2.65

STATE_W = 1.15
STATE_H = 0.62
Y_STATE = 0.0
X_STATES = [0.0, 2.0, 4.0, 6.0, 8.6]

FONT_STATE = FS_ANNOT + 1
FONT_LABEL = FS_ANNOT + 1
FONT_SMALL = FS_ANNOT - 1
FONT_GATE = FS_BADGE - 3
FONT_REFLEX = FS_ANNOT + 1

FINAL_STATE_COLOR = sns.color_palette("crest", 7)[4]
OPTION_COLOR = sns.color_palette("crest", 7)[5]


def _state_box(ax, x, y, label, facecolor, edgecolor, textcolor=C_BLACK):
    rect = FancyBboxPatch(
        (x - STATE_W / 2, y - STATE_H / 2),
        STATE_W,
        STATE_H,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.6,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=FONT_STATE,
        color=textcolor,
        zorder=4,
    )


def _arrow(ax, x0, x1, y, label, color, dy=0.28, lw=2.1, fsize=None):
    if fsize is None:
        fsize = FONT_SMALL
    ax.annotate(
        "",
        xy=(x1 - STATE_W / 2 + 0.05, y),
        xytext=(x0 + STATE_W / 2 - 0.05, y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=10,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=2,
    )
    ax.text(
        0.5 * (x0 + x1),
        y + dy,
        label,
        ha="center",
        va="bottom",
        fontsize=fsize,
        color=color,
    )


def _top_bracket(ax, x0, x1, y, text):
    ax.plot([x0, x1], [y, y], color=OPTION_COLOR, lw=1.8, zorder=1)
    ax.plot([x0, x0], [y, y - 0.12], color=OPTION_COLOR, lw=1.8, zorder=1)
    ax.plot([x1, x1], [y, y - 0.12], color=OPTION_COLOR, lw=1.8, zorder=1)
    ax.text(
        0.5 * (x0 + x1),
        y + 0.12,
        text,
        ha="center",
        va="bottom",
        fontsize=FONT_LABEL,
        color=OPTION_COLOR,
    )


def _mcts_bar(ax, x0, x1, y, text):
    ax.plot([x0, x1], [y, y], color=C_RED, lw=3.0, solid_capstyle="round", zorder=1)
    ax.text(
        0.5 * (x0 + x1),
        y - 0.18,
        text,
        ha="center",
        va="top",
        fontsize=FONT_SMALL,
        color=C_RED,
    )


def draw():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(-2.1, 10.2)
    ax.set_ylim(-1.18, 1.25)
    ax.axis("off")

    state_labels = [
        r"$s_t$",
        r"$s_{t+1}$",
        r"$\cdots$",
        r"$s_{t+K-1}$",
        r"$s_{t+K}$",
    ]

    for i, (x, label) in enumerate(zip(X_STATES, state_labels)):
        if i == len(X_STATES) - 1:
            _state_box(ax, x, Y_STATE, label, facecolor="#eef7ea", edgecolor=FINAL_STATE_COLOR)
        else:
            _state_box(ax, x, Y_STATE, label, facecolor=C_STATE, edgecolor=C_BLUE)

    _arrow(ax, X_STATES[0], X_STATES[1], Y_STATE, r"$\pi_{\mathrm{reflex}}$", C_MID_GRAY, fsize=FONT_REFLEX)
    _arrow(ax, X_STATES[1], X_STATES[2], Y_STATE, r"$\pi_{\mathrm{reflex}}$", C_MID_GRAY, fsize=FONT_REFLEX)
    _arrow(ax, X_STATES[2], X_STATES[3], Y_STATE, r"$\pi_{\mathrm{reflex}}$", C_MID_GRAY, fsize=FONT_REFLEX)
    _arrow(ax, X_STATES[3], X_STATES[4], Y_STATE, r"$\pi_{\mathrm{plan}}^{(K)}$", C_RED, lw=2.4)

    _mcts_bar(
        ax,
        X_STATES[0],
        X_STATES[3],
        -0.85,
        r"MCTS runs ($K$ frames)",
    )
    _top_bracket(
        ax,
        X_STATES[0],
        X_STATES[4],
        0.8,
        r"option $o_K$ (holding time $K$)",
    )

    ax.text(
        -1.55,
        0.1,
        "gate\npicks $K$",
        ha="center",
        va="center",
        fontsize=FONT_GATE,
        color=C_NAVY,
    )
    ax.annotate(
        "",
        xy=(X_STATES[0] - STATE_W / 2 + 0.03, Y_STATE),
        xytext=(-0.9, Y_STATE),
        arrowprops=dict(arrowstyle="-|>", color=C_NAVY, lw=2.0, mutation_scale=10),
        zorder=2,
    )

    ax.text(
        X_STATES[4],
        -0.97,
        r"return to meta-level at $s_{t+K}$",
        ha="center",
        va="top",
        fontsize=FONT_SMALL,
        color=C_DARK_GRAY,
    )

    fig.tight_layout(pad=0.15)
    fig.savefig(OUT_PDF, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    draw()
