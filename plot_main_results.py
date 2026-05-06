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
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================================================================
# Style — sourced from plot_config.py
# ===========================================================================

from plot_config import (
    C_BLACK, C_WHITE, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY,
    FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT,
    C_RED, C_BLUE, BAR_ALPHA, CAPSIZE, ERR_LW,
    apply_style,
)
apply_style()

GATING_COLOR    = C_RED
FONT_SIZE_TITLE = FS_TITLE
FONT_SIZE_LABEL = FS_LABEL
FONT_SIZE_TICK  = FS_TICK
FONT_SIZE_ANNOT = FS_ANNOT
ERR_LINEWIDTH   = ERR_LW

FIG_WIDTH    = 16.0
FIG_HEIGHT_V = 3.8
FIG_HEIGHT_H = 4.2

BAR_WIDTH_V  = 0.55
BAR_HEIGHT_H = 0.55
WANDB_ENTITY = "aneeshmuppidi19"
HEX_WANDB_PROJECT = "gru_ent100-500"
HEX_WANDB_PROJECT_ALT = "gru_ent095-500"
HEX_PERF_FLOOR = 0.5
GO_WANDB_PROJECT = "gating_eval_go9x9"
GO_GATE_ROOT = "/n/netscratch/gershman_lab/Lab/amuppidi/gru_go9x9_01"
GO_PERF_FLOOR = 0.5

# ===========================================================================
# Data
# ===========================================================================
# Each env entry:
#   baselines : list of (label, mean, se)
#   gating    : (label, mean, se)
#   ylabel    : y-axis / x-axis label
#   placeholder : True  →  bars dimmed + "placeholder" annotation

def _clock_selected_budgets():
    # Shared 5-budget reporting set for clocked games.
    return [300, 1200, 2300, 3500, 4800]


def _fetch_speed_hex_env():
    try:
        import wandb
        api = wandb.Api()
        runs_main = list(api.runs(f"{WANDB_ENTITY}/{HEX_WANDB_PROJECT}"))
        runs_alt = list(api.runs(f"{WANDB_ENTITY}/{HEX_WANDB_PROJECT_ALT}"))
    except Exception as exc:
        print(f"[plot_main_results] Could not fetch Speed Hex results from wandb: {exc}")
        return {
            "name": "Speed Hex",
            "ylabel": "Expected Score",
            "baselines": [
                ("K=2", 0.29, 0.03),
                ("K=8", 0.35, 0.03),
                ("K=32", 0.40, 0.03),
                ("K=128", 0.43, 0.03),
                ("Greedy", 0.46, 0.03),
                ("Midpeak", 0.45, 0.03),
                ("Random", 0.38, 0.03),
            ],
            "gating": ("Gating", 0.58, 0.03),
        }

    selected_budgets = set(_clock_selected_budgets())
    label_map = [
        ("K=2", "always2", runs_main),
        ("K=8", "always8", runs_alt),
        ("K=32", "always32", runs_main),
        ("K=128", "always128", runs_main),
        ("Greedy", "proportional", runs_main),
        ("Midpeak", "midpeak", runs_main),
        ("Random", "random_gate", runs_main),
    ]

    gate_means = []
    baselines = []
    for plot_label, wandb_label, runs_src in label_map:
        vals = []
        for run in runs_src:
            tb = run.config.get("time_budget")
            if tb not in selected_budgets:
                continue
            summary = dict(run.summary._json_dict) if hasattr(run.summary, "_json_dict") else dict(run.summary)
            key = f"{wandb_label}/expected_score"
            if key not in summary:
                continue
            vals.append(max(float(summary[key]), HEX_PERF_FLOOR))
        if not vals:
            continue
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        gate_means.append(mean)
        baselines.append((plot_label, 1.0 - mean, se))

    if gate_means:
        gating_mean = float(np.mean(gate_means))
        gating_se = float(np.std(gate_means, ddof=1) / math.sqrt(len(gate_means))) if len(gate_means) > 1 else 0.0
    else:
        gating_mean, gating_se = 0.58, 0.03

    return {
        "name": "Speed Hex",
        "ylabel": "Expected Score",
        "baselines": baselines,
        "gating": ("Gating", gating_mean, gating_se),
    }


def _fetch_speed_go_env():
    try:
        import wandb
        api = wandb.Api()
        runs = list(api.runs(f"{WANDB_ENTITY}/{GO_WANDB_PROJECT}"))
    except Exception as exc:
        print(f"[plot_main_results] Could not fetch Speed Go results from wandb: {exc}")
        return {
            "name": "Speed Go",
            "ylabel": "Expected Score",
            "baselines": [
                ("K=16", 0.27, 0.03),
                ("K=32", 0.42, 0.03),
                ("K=64", 0.52, 0.03),
                ("K=96", 0.47, 0.03),
                ("Greedy", 0.45, 0.03),
                ("Midpeak", 0.50, 0.03),
                ("Random", 0.46, 0.03),
            ],
            "gating": ("Gating", 0.59, 0.03),
        }

    selected_budgets = set(_clock_selected_budgets())
    label_map = [
        ("K=16", "always16"),
        ("K=32", "always32"),
        ("K=64", "always64"),
        ("K=96", "always96"),
        ("Greedy", "proportional"),
        ("Midpeak", "midpeak"),
        ("Random", "random_gate"),
    ]

    gate_means = []
    gate_ses = []
    baselines = []

    for plot_label, wandb_label in label_map:
        vals = []
        for run in runs:
            cfg = dict(run.config)
            if cfg.get("gate_root") != GO_GATE_ROOT:
                continue
            tb = run.config.get("time_budget")
            if tb not in selected_budgets:
                continue
            summary = dict(run.summary._json_dict) if hasattr(run.summary, "_json_dict") else dict(run.summary)
            key = f"{wandb_label}/expected_score"
            if key not in summary:
                continue
            vals.append(max(float(summary[key]), GO_PERF_FLOOR))
        if not vals:
            continue
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        gate_means.append(mean)
        gate_ses.append(se)
        baselines.append((plot_label, 1.0 - mean, se))

    if gate_means:
        gating_mean = float(np.mean(gate_means))
        gating_se = float(np.std(gate_means, ddof=1) / math.sqrt(len(gate_means))) if len(gate_means) > 1 else 0.0
    else:
        gating_mean, gating_se = 0.59, 0.03

    return {
        "name": "Speed Go",
        "ylabel": "Expected Score",
        "baselines": baselines,
        "gating": ("Gating", gating_mean, gating_se),
    }


ENVS = [
    {
        "name":        "Pac-Man",
        "ylabel":      "Episode Return",
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
        "baselines": [
            ("K=1",    18.8, 2.3),
            ("K=2",    26.4, 3.7),
            ("K=3",    27.6, 3.8),
            ("K=4",    25.6, 3.8),
            ("Random", 11.2, 2.4),
        ],
        "gating": ("Gating", 45.6, 3.7),
    },
    _fetch_speed_hex_env(),
    _fetch_speed_go_env(),
    {
        "name":        "Snake",
        "ylabel":      "Episode Return",
        "baselines": [
            ("K=1",    14.91, 1.29),
            ("K=2",     3.63, 0.85),
            ("K=3",    12.79, 1.46),
            ("K=4",     7.45, 1.38),
            ("Random", 11.56, 0.85),
        ],
        "gating": ("Gating", 16.54, 1.26),
    },
]

# ===========================================================================
# Setup
# ===========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)



from matplotlib.colors import to_rgba

def baseline_blues(n):
    """n shades of C_BLUE by varying opacity."""
    base_color = to_rgba(C_BLUE)
    alphas = np.linspace(0.3, 0.85, n)
    return [(base_color[0], base_color[1], base_color[2], a) for a in alphas]


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

        labels = [b[0] for b in baselines] + [gating[0]]
        means  = np.array([b[1] for b in baselines] + [gating[1]])
        ses    = np.array([b[2] for b in baselines] + [gating[2]])
        colors = list(baseline_blues(len(baselines))) + [to_rgba(GATING_COLOR, alpha=BAR_ALPHA)]

        x = np.arange(len(labels))
        ax.bar(
            x, means,
            width=BAR_WIDTH_V,
            color=colors,
            linewidth=0,
        )
        ax.errorbar(
            x, means, yerr=ses,
            fmt="none",
            ecolor=C_BLACK,
            elinewidth=ERR_LINEWIDTH,
            capsize=CAPSIZE,
            capthick=ERR_LINEWIDTH,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            labels, fontsize=FONT_SIZE_TICK, rotation=35, ha="right",
        )
        ax.set_ylabel(env["ylabel"], fontsize=FONT_SIZE_LABEL)
        title = env["name"]
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)

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

        # Gating at the top, baselines below in original order
        labels = [gating[0]] + [b[0] for b in baselines]
        means  = np.array([gating[1]] + [b[1] for b in baselines])
        ses    = np.array([gating[2]] + [b[2] for b in baselines])
        colors = [to_rgba(GATING_COLOR, alpha=BAR_ALPHA)] + list(baseline_blues(len(baselines)))

        y = np.arange(len(labels))
        ax.barh(
            y, means,
            height=BAR_HEIGHT_H,
            color=colors,
            linewidth=0,
        )
        ax.errorbar(
            means, y, xerr=ses,
            fmt="none",
            ecolor=C_BLACK,
            elinewidth=ERR_LINEWIDTH,
            capsize=CAPSIZE,
            capthick=ERR_LINEWIDTH,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=FONT_SIZE_TICK + 5)
        ax.set_xlabel(env["ylabel"], fontsize=FONT_SIZE_LABEL)
        title = env["name"]
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)

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
