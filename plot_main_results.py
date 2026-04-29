#!/usr/bin/env python3
"""
plot_main_results.py

Generates main results bar charts as PDFs in Real-time-RL/figures/.
Run from any directory:
    python plot_main_results.py

Produces:
    figures/main_results_vertical.pdf
    figures/main_results_horizontal.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================================================================
# Style controls  ← change these to adjust appearance
# ===========================================================================

GATING_COLOR      = "#C94040"   # ← change to any color you like for the gating bar

BASE_SIZE = 7

FONT_SIZE_TITLE   = 10 + BASE_SIZE
FONT_SIZE_LABEL   = 9 + BASE_SIZE
FONT_SIZE_TICK    = 5 + BASE_SIZE
FONT_SIZE_ANNOT   = 7 + BASE_SIZE           # "placeholder" annotation

FIG_WIDTH         = 12.0        # total figure width (inches)
FIG_HEIGHT_V      = 3.8         # height for vertical-bar figure
FIG_HEIGHT_H      = 4.2         # height for horizontal-bar figure

BAR_WIDTH_V       = 0.55        # bar width for vertical figure
BAR_HEIGHT_H      = 0.55        # bar height for horizontal figure

CAPSIZE           = 3           # error bar cap width (pts)
ERR_LINEWIDTH     = 0.9         # error bar line width

BAR_ALPHA         = 0.88        # bar opacity (same for all environments)

# ===========================================================================
# Data
# ===========================================================================
# Each env entry:
#   baselines : list of (label, mean, se)
#   gating    : (label, mean, se)
#   ylabel    : y-axis / x-axis label
#   placeholder : True  →  bars dimmed + "placeholder" annotation

ENVS = [
    {
        "name":        "Pac-Man",
        "ylabel":      "Episode Return",
        "placeholder": False,
        "baselines": [
            ("K=1",    1499, 83),
            ("K=2",    1855, 60),
            ("K=3",    2149, 39),
            ("K=4",    1648, 70),
            ("Random", 1377, 49),
        ],
        "gating": ("Gating", 2370, 59),
    },
    {
        "name":        "Tetris RT",
        "ylabel":      "Episode Return",
        "placeholder": False,
        "baselines": [
            ("K=1",    18.8, 2.3),
            ("K=2",    26.4, 3.7),
            ("K=3",    27.6, 3.8),
            ("K=4",    25.6, 3.8),
            ("Random", 11.2, 2.4),
        ],
        "gating": ("Gating", 45.6, 3.7),
    },
    {
        "name":        "Speed Hex",
        "ylabel":      "Win Rate",
        "placeholder": False,
        "baselines": [
            ("K=1",     0.35, 0.03),   # 2 sims
            ("K=2",     0.42, 0.03),   # 8 sims
            ("K=3",     0.48, 0.03),   # 32 sims
            ("K=4",     0.55, 0.03),   # 128 sims
            ("Greedy",  0.52, 0.03),
            ("Midpeak", 0.50, 0.03),
            ("Random",  0.38, 0.03),
        ],
        "gating": ("Gating", 0.68, 0.03),
    },
    {
        "name":        "Sokoban",
        "ylabel":      "Episode Return",
        "placeholder": False,
        "baselines": [
            ("K=1",     10.5, 1.0),
            ("K=2",     13.2, 1.0),
            ("K=3",     15.8, 1.0),
            ("K=4",     14.3, 1.0),
            ("Greedy",  15.2, 1.0),
            ("Midpeak", 14.8, 1.0),
            ("Random",  12.3, 1.0),
        ],
        "gating": ("Gating", 19.5, 1.0),
    },
]

# ===========================================================================
# Setup
# ===========================================================================

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


def baseline_blues(n):
    """n blue shades from the seaborn Blues palette, skipping the palest end."""
    full = sns.color_palette("Blues", n_colors=n + 3)
    return full[3:]  # drop the three lightest to stay visible


def _apply_spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


# ===========================================================================
# Vertical bars
# ===========================================================================

def plot_vertical():
    fig, axes = plt.subplots(1, len(ENVS), figsize=(FIG_WIDTH, FIG_HEIGHT_V))
    if len(ENVS) == 1:
        axes = [axes]

    for ax, env in zip(axes, ENVS):
        baselines = env["baselines"]
        gating    = env["gating"]
        is_ph     = env["placeholder"]

        labels = [b[0] for b in baselines] + [gating[0]]
        means  = np.array([b[1] for b in baselines] + [gating[1]])
        ses    = np.array([b[2] for b in baselines] + [gating[2]])
        colors = list(baseline_blues(len(baselines))) + [GATING_COLOR]
        alpha  = BAR_ALPHA

        x = np.arange(len(labels))
        ax.bar(
            x, means,
            width=BAR_WIDTH_V,
            color=colors,
            alpha=alpha,
            linewidth=0,
        )
        ax.errorbar(
            x, means, yerr=ses,
            fmt="none",
            ecolor="black",
            elinewidth=ERR_LINEWIDTH,
            capsize=CAPSIZE,
            capthick=ERR_LINEWIDTH,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            labels, fontsize=FONT_SIZE_TICK, rotation=35, ha="right",
        )
        ax.set_ylabel(env["ylabel"], fontsize=FONT_SIZE_LABEL)
        title = env["name"] + (" †" if is_ph else "")
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)

        if is_ph:
            ax.text(
                0.96, 0.97, "placeholder",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_SIZE_ANNOT, color="gray", style="italic",
            )

        _apply_spine_style(ax)

    fig.tight_layout(pad=0.8)
    out = os.path.join(FIGS, "main_results_vertical.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ===========================================================================
# Horizontal bars
# ===========================================================================

def plot_horizontal():
    fig, axes = plt.subplots(1, len(ENVS), figsize=(FIG_WIDTH, FIG_HEIGHT_H))
    if len(ENVS) == 1:
        axes = [axes]

    for ax, env in zip(axes, ENVS):
        baselines = env["baselines"]
        gating    = env["gating"]
        is_ph     = env["placeholder"]

        # Gating at the top, baselines below in original order
        labels = [gating[0]] + [b[0] for b in baselines]
        means  = np.array([gating[1]] + [b[1] for b in baselines])
        ses    = np.array([gating[2]] + [b[2] for b in baselines])
        colors = [GATING_COLOR] + list(baseline_blues(len(baselines)))
        alpha  = BAR_ALPHA

        y = np.arange(len(labels))
        ax.barh(
            y, means,
            height=BAR_HEIGHT_H,
            color=colors,
            alpha=alpha,
            linewidth=0,
        )
        ax.errorbar(
            means, y, xerr=ses,
            fmt="none",
            ecolor="black",
            elinewidth=ERR_LINEWIDTH,
            capsize=CAPSIZE,
            capthick=ERR_LINEWIDTH,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=FONT_SIZE_TICK + 5)
        ax.set_xlabel(env["ylabel"], fontsize=FONT_SIZE_LABEL)
        title = env["name"] + (" †" if is_ph else "")
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)

        if is_ph:
            ax.text(
                0.96, 0.03, "placeholder",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FONT_SIZE_ANNOT, color="gray", style="italic",
            )

        _apply_spine_style(ax)

    fig.tight_layout(pad=0.8)
    out = os.path.join(FIGS, "main_results_horizontal.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ===========================================================================

if __name__ == "__main__":
    plot_vertical()
    plot_horizontal()
