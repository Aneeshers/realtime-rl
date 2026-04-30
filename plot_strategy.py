#!/usr/bin/env python3
"""
plot_strategy.py

Planning-depth allocation frequency over normalized episode time.
4 panels: Pac-Man | Tetris RT | Speed Hex | Sokoban
All data is placeholder — replace coordinates once real rollout data is available.

Saves: figures/strategy.pdf

Usage:
    python plot_strategy.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Style — sourced from plot_config.py
# ============================================================

from plot_config import (
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    K_COLORS, LINE_LW,
    apply_style,
)
apply_style()

FONT_SIZE_TITLE  = FS_TITLE
FONT_SIZE_LABEL  = FS_LABEL
FONT_SIZE_TICK   = FS_TICK
FONT_SIZE_LEGEND = FS_LEGEND
FONT_SIZE_ANNOT  = FS_ANNOT
LINE_COLORS      = K_COLORS

FIG_WIDTH    = 12.0
FIG_HEIGHT   = 3.0
LINE_WIDTH   = LINE_LW
MARKER_EVERY = 3
MARKER_SIZE  = 4
LINE_STYLES  = ["-", "-", "-", "--"]
K3_ALPHA     = 0.30

# ============================================================
# Placeholder data
# (Replace with real per-bin rollout frequencies)
# ============================================================

T = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

ENVS = [
    {
        "title":    "Pac-Man",
        "xlabel":   "Episode progress",
        "legend":   ["K=1", "K=2", "K=3", "K=4"],
        "show_legend": True,
        "placeholder": True,
        "series": [
            np.array([0.68, 0.65, 0.62, 0.55, 0.50, 0.48, 0.45, 0.42, 0.40, 0.38]),  # K=1
            np.array([0.30, 0.32, 0.35, 0.42, 0.46, 0.49, 0.50, 0.52, 0.53, 0.55]),  # K=2
            np.zeros(10),                                                               # K=3
            np.array([0.02, 0.03, 0.03, 0.03, 0.04, 0.03, 0.05, 0.06, 0.07, 0.07]),  # K=4
        ],
    },
    {
        "title":    "Real-Time Tetris",
        "xlabel":   "Episode progress",
        "legend":   ["K=1", "K=2", "K=3", "K=4"],
        "show_legend": False,
        "placeholder": True,
        "series": [
            np.array([0.58, 0.52, 0.44, 0.36, 0.28, 0.20, 0.16, 0.13, 0.10, 0.08]),
            np.array([0.28, 0.27, 0.25, 0.23, 0.21, 0.19, 0.18, 0.17, 0.16, 0.15]),
            np.zeros(10),
            np.array([0.14, 0.21, 0.31, 0.41, 0.51, 0.61, 0.66, 0.70, 0.74, 0.77]),
        ],
    },
    {
        "title":    "Speed Hex",
        "xlabel":   "Move fraction",
        "legend":   ["2 sims", "8 sims", "32 sims", "128 sims"],
        "show_legend": False,
        "placeholder": True,
        "series": [
            np.array([0.55, 0.50, 0.40, 0.32, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10]),
            np.array([0.28, 0.30, 0.33, 0.35, 0.37, 0.38, 0.37, 0.36, 0.35, 0.33]),
            np.array([0.12, 0.14, 0.18, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38]),
            np.array([0.05, 0.06, 0.09, 0.11, 0.13, 0.14, 0.15, 0.17, 0.18, 0.19]),
        ],
    },
    {
        "title":    "Sokoban",
        "xlabel":   "Episode progress",
        "legend":   ["K=1", "K=2", "K=3", "K=4"],
        "show_legend": False,
        "placeholder": True,
        "series": [
            np.array([0.40, 0.38, 0.35, 0.30, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10]),
            np.array([0.30, 0.32, 0.34, 0.36, 0.38, 0.37, 0.36, 0.35, 0.33, 0.32]),
            np.array([0.22, 0.22, 0.24, 0.26, 0.28, 0.32, 0.35, 0.38, 0.42, 0.45]),
            np.array([0.08, 0.08, 0.07, 0.08, 0.09, 0.11, 0.11, 0.12, 0.13, 0.13]),
        ],
    },
]

# ============================================================
# Setup
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)



def _apply_spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


# ============================================================
# Plot
# ============================================================

def plot_strategy():
    fig, axes = plt.subplots(1, len(ENVS), figsize=(FIG_WIDTH, FIG_HEIGHT))

    for ax, env in zip(axes, ENVS):
        for i, (series, label) in enumerate(zip(env["series"], env["legend"])):
            is_k3 = (series == 0).all()
            alpha = K3_ALPHA if is_k3 else 1.0
            markevery = None if is_k3 else MARKER_EVERY
            ax.plot(
                T, series,
                color=LINE_COLORS[i],
                linestyle=LINE_STYLES[i],
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=MARKER_SIZE,
                markevery=markevery,
                alpha=alpha,
                label=label,
                markeredgewidth=0,
            )

        ax.set_xlim(0.05, 1.05)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(env["xlabel"], fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Frequency" if ax is axes[0] else "", fontsize=FONT_SIZE_LABEL)
        ax.tick_params(labelsize=FONT_SIZE_TICK)

        title = env["title"] + (" †" if env["placeholder"] else "")
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)

        _apply_spine_style(ax)

        if env["show_legend"]:
            ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False,
                      loc="upper right", handlelength=1.5)

        if env["placeholder"]:
            ax.text(
                0.97, 0.97, "placeholder",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_SIZE_ANNOT, color="gray", style="italic",
            )

    fig.tight_layout(pad=0.8)
    out = os.path.join(FIGS, "strategy.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_strategy()
