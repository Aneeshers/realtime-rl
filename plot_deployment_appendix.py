#!/usr/bin/env python3
"""
plot_deployment_appendix.py

Creates a larger appendix figure with the full deployment breakdown across
environment, GPU, and FPS.

Produces:
    figures/deployment_appendix.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_config import (
    C_BLACK, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY,
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND,
    C_BLUE, C_RED, BAR_ALPHA, ERR_LW, CAPSIZE,
    apply_style,
)
apply_style()

from plot_deployment import (
    GAMES, GPUS, FPS_LIST, GPU_LABELS, GAME_LABELS, GPU_COLORS,
    _load_summary_rows,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)


def _apply_spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _rows_by(game, gpu, rows):
    return [r for r in rows if r["game"] == game and r["gpu"] == gpu]


def plot_deployment_appendix():
    rows = _load_summary_rows()
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.0), sharex=True)

    # Row 1: return vs FPS
    for col, game in enumerate(GAMES):
        ax = axes[0, col]
        for gpu in GPUS:
            subset = _rows_by(game, gpu, rows)
            vals = [next(r["ret_mean"] for r in subset if r["fps"] == fps) for fps in FPS_LIST]
            ses = [next(r["ret_se"] for r in subset if r["fps"] == fps) for fps in FPS_LIST]
            ax.plot(FPS_LIST, vals, color=GPU_COLORS[gpu], marker="o", linewidth=1.8, label=GPU_LABELS[gpu])
            ax.errorbar(FPS_LIST, vals, yerr=ses, fmt="none", ecolor=GPU_COLORS[gpu],
                        elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)
        ax.set_title(GAME_LABELS[game], fontsize=FS_TITLE)
        if col == 0:
            ax.set_ylabel("Return", fontsize=FS_LABEL)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        _apply_spine_style(ax)

    # Row 2: miss rate vs FPS
    for col, game in enumerate(GAMES):
        ax = axes[1, col]
        for gpu in GPUS:
            subset = _rows_by(game, gpu, rows)
            vals = [max(next(r["miss_pct"] for r in subset if r["fps"] == fps), 1e-3) for fps in FPS_LIST]
            ax.plot(FPS_LIST, vals, color=GPU_COLORS[gpu], marker="o", linewidth=1.8)
        if col == 0:
            ax.set_ylabel("Miss Rate (%)", fontsize=FS_LABEL)
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=FS_TICK)
        _apply_spine_style(ax)

    # Row 3: p95 slack vs FPS
    for col, game in enumerate(GAMES):
        ax = axes[2, col]
        ax.axhline(0.0, color=C_DARK_GRAY, linestyle="--", linewidth=1.0)
        for gpu in GPUS:
            subset = _rows_by(game, gpu, rows)
            vals = [next(r["slack_p95"] for r in subset if r["fps"] == fps) for fps in FPS_LIST]
            ax.plot(FPS_LIST, vals, color=GPU_COLORS[gpu], marker="o", linewidth=1.8)
        if col == 0:
            ax.set_ylabel("p95 Slack (ms)", fontsize=FS_LABEL)
        ax.set_xlabel("FPS", fontsize=FS_LABEL)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        _apply_spine_style(ax)

    handles = [plt.Line2D([0], [0], color=GPU_COLORS[g], marker="o", linewidth=1.8) for g in GPUS]
    fig.legend(handles, [GPU_LABELS[g] for g in GPUS], loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False, fontsize=FS_LEGEND)
    fig.tight_layout(rect=[0, 0.05, 1, 1], pad=1.0, w_pad=1.1, h_pad=1.0)
    out = os.path.join(FIGS, "deployment_appendix.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_deployment_appendix()
