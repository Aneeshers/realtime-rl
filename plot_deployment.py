#!/usr/bin/env python3
"""
plot_deployment.py

Generates fig:deployment — a 3-panel dashboard validating the 2-GPU real-time setup.
  Panel A: Timeline (Gantt chart) of asynchronous execution.
  Panel B: Hardware latency distributions (violin plots) for H100, A100, V100.
  Panel C: Zero-shot Sim-to-Real transfer performance (Tetris RT).
  Panel D: Zero-shot Sim-to-Real transfer performance (Speed Hex).

All data is placeholder — replace with actual deployment logs.

Produces:
    figures/deployment.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===========================================================================
# Style — sourced from plot_config.py
# ===========================================================================

from plot_config import (
    C_BLACK, C_WHITE, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY,
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    C_BLUE, C_RED, BAR_ALPHA, ERR_LW, CAPSIZE,
    apply_style,
)
apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

# Colors specific to this plot
GPU1_COLOR = C_BLUE  # Env + Fast Policy
GPU2_COLOR = C_RED   # MCTS

# ===========================================================================
# Placeholder Data
# ===========================================================================

# Panel A: Timeline (GPU1 vs GPU2)
# Format: (start_time, duration) in milliseconds
TIMELINE_LIMIT = 50.0  # Show 50ms window

# GPU1 executes many fast steps
GPU1_BLOCKS = [
    (0.0, 3.2), (4.0, 3.0), (8.0, 3.3), 
    (12.0, 3.1), (16.0, 3.4), (20.0, 3.2),
    (24.0, 3.0), (28.0, 3.1), (32.0, 3.3),
    (36.0, 3.1), (40.0, 3.0), (44.0, 3.2), (48.0, 3.1)
]

# GPU2 executes slower MCTS blocks asynchronously
GPU2_BLOCKS = [
    (0.5, 12.0), (13.5, 11.5), (26.0, 12.5), (39.5, 11.8)
]

# Panel B: Hardware Latency (Violin)
# Generate synthetic distributions
np.random.seed(42)
LATS_H100 = np.random.normal(loc=12.0, scale=1.5, size=500)
LATS_A100 = np.random.normal(loc=16.5, scale=2.8, size=500)
LATS_V100 = np.random.normal(loc=23.0, scale=4.5, size=500)
DEADLINE_MS = 25.0  # Hypothetical frame deadline

# Panel C: Sim-to-Real Transfer (Tetris RT)
SIM_SCORE_TETRIS, SIM_SE_TETRIS = 45.6, 3.7
REAL_SCORE_TETRIS, REAL_SE_TETRIS = 44.8, 4.1

# Panel D: Sim-to-Real Transfer (Speed Hex)
SIM_SCORE_HEX, SIM_SE_HEX = 0.55, 0.03
REAL_SCORE_HEX, REAL_SE_HEX = 0.53, 0.04

# ===========================================================================
# Plotting Helpers
# ===========================================================================

def _apply_spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

# ===========================================================================
# Main Plot
# ===========================================================================

def plot_deployment():
    # Width ratios give the timeline a bit more horizontal space
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.5), gridspec_kw={'width_ratios': [1.5, 1, 0.8, 0.8]})

    # -----------------------------------------------------------
    # Panel A: Asynchronous Timeline
    # -----------------------------------------------------------
    ax = axes[0]
    
    # broken_barh takes a list of (start, width) tuples and a (ymin, height) tuple
    ax.broken_barh(GPU1_BLOCKS, (10, 8), facecolors=GPU1_COLOR, alpha=BAR_ALPHA, edgecolor="none")
    ax.broken_barh(GPU2_BLOCKS, (20, 8), facecolors=GPU2_COLOR, alpha=BAR_ALPHA, edgecolor="none")
    
    ax.set_ylim(5, 35)
    ax.set_xlim(0, TIMELINE_LIMIT)
    ax.set_xlabel("Time (ms)", fontsize=FS_LABEL)
    
    # Custom y-ticks for GPU labels
    ax.set_yticks([14, 24])
    ax.set_yticklabels(["GPU 1\n(Env + $\pi_{reflex}$)", "GPU 2\n(MCTS)"], fontsize=FS_TICK)
    ax.tick_params(axis='x', labelsize=FS_TICK)
    
    ax.set_title("Asynchronous Execution\nTimeline", fontsize=FS_TITLE)
    _apply_spine_style(ax)

    # -----------------------------------------------------------
    # Panel B: Hardware Latency Distributions
    # -----------------------------------------------------------
    ax = axes[1]
    data = [LATS_H100, LATS_A100, LATS_V100]
    labels = ["H100", "A100", "V100"]
    
    parts = ax.violinplot(data, positions=[1, 2, 3], showmeans=True, showextrema=False)
    
    # Style the violins
    for pc in parts['bodies']:
        pc.set_facecolor(GPU2_COLOR)
        pc.set_alpha(0.6)
    parts['cmeans'].set_color(C_BLACK)
    
    # Add deadline line
    ax.axhline(DEADLINE_MS, color=C_DARK_GRAY, linestyle="--", linewidth=1.5, zorder=0)
    ax.text(2.5, DEADLINE_MS + 0.5, "Frame Deadline", color=C_DARK_GRAY, fontsize=FS_TICK, va="bottom", ha="right")
    
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel("MCTS Latency (ms)", fontsize=FS_LABEL)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    
    ax.set_title("Hardware Inference\nVariance", fontsize=FS_TITLE)
    _apply_spine_style(ax)

    # -----------------------------------------------------------
    # Panel C: Sim-to-Real Transfer (Tetris RT)
    # -----------------------------------------------------------
    ax = axes[2]
    
    labels = ["Sim\n(Train)", "Real\n2-GPU"]
    means = [SIM_SCORE_TETRIS, REAL_SCORE_TETRIS]
    ses = [SIM_SE_TETRIS, REAL_SE_TETRIS]
    
    # We can use C_MID_GRAY for Sim and C_BLUE for Real
    colors = [C_MID_GRAY, GPU1_COLOR]
    
    x = np.arange(len(labels))
    ax.bar(x, means, width=0.6, color=colors, alpha=BAR_ALPHA, linewidth=0)
    ax.errorbar(x, means, yerr=ses, fmt="none", ecolor=C_BLACK, 
                elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel("Episode Return", fontsize=FS_LABEL)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    
    ax.set_title("Sim-to-Real\n(Tetris RT)", fontsize=FS_TITLE)
    _apply_spine_style(ax)

    # -----------------------------------------------------------
    # Panel D: Sim-to-Real Transfer (Speed Hex)
    # -----------------------------------------------------------
    ax = axes[3]
    
    labels = ["Sim\n(Train)", "Real\n2-GPU"]
    means = [SIM_SCORE_HEX, REAL_SCORE_HEX]
    ses = [SIM_SE_HEX, REAL_SE_HEX]
    
    colors = [C_MID_GRAY, GPU1_COLOR]
    
    x = np.arange(len(labels))
    ax.bar(x, means, width=0.6, color=colors, alpha=BAR_ALPHA, linewidth=0)
    ax.errorbar(x, means, yerr=ses, fmt="none", ecolor=C_BLACK, 
                elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel("Win Rate", fontsize=FS_LABEL)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    
    ax.set_title("Sim-to-Real\n(Speed Hex)", fontsize=FS_TITLE)
    _apply_spine_style(ax)

    # -----------------------------------------------------------
    # Layout and Save
    # -----------------------------------------------------------
    fig.tight_layout(pad=1.5)
    out = os.path.join(FIGS, "deployment.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

if __name__ == "__main__":
    plot_deployment()
