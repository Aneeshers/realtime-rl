#!/usr/bin/env python3
"""
Create a temporary Go + Hex calibration figure for the appendix.

Left panel:
  Real Speed Go inference-tournament summary.

Right panel:
  Mock Speed Hex calibration curve illustrating the selected
  2 / 8 / 32 / 128 simulation options on a log-scaled budget axis.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_config import C_BLUE, C_RED, FS_LABEL, FS_TICK, LINE_LW, MARKER_SIZE, CAPSIZE, apply_style

apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)


def main() -> None:
    go_budgets = np.array([2, 4, 8, 16, 32, 64, 96, 128], dtype=float)
    go_scores = np.array([0.2580, 0.3050, 0.2999, 0.3136, 0.5064, 0.7044, 0.7820, 0.8307], dtype=float)

    # Mock Hex curve: monotone improvement with log-spaced gains aligned to
    # the selected options 2 / 8 / 32 / 128.
    hex_budgets = np.array([2, 4, 8, 16, 32, 64, 128], dtype=float)
    hex_scores = np.array([0.34, 0.39, 0.47, 0.54, 0.63, 0.72, 0.81], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)

    panels = [
        (axes[0], "Speed Go", go_budgets, go_scores, [16, 32, 64, 96]),
        (axes[1], "Speed Hex", hex_budgets, hex_scores, [2, 8, 32, 128]),
    ]

    for ax, title, budgets, scores, highlights in panels:
        x = np.arange(len(budgets))
        ax.plot(
            x, scores,
            color=C_BLUE,
            marker="o",
            linewidth=LINE_LW,
            markersize=MARKER_SIZE,
        )
        idx = {int(b): i for i, b in enumerate(budgets)}
        hx = [idx[b] for b in highlights if b in idx]
        hy = [scores[idx[b]] for b in highlights if b in idx]
        ax.scatter(
            hx, hy,
            s=180,
            facecolors="none",
            edgecolors=C_RED,
            linewidths=2.2,
            zorder=5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(b)) for b in budgets], fontsize=FS_TICK)
        ax.set_title(title, fontsize=FS_LABEL)
        ax.set_xlabel("Inference budget", fontsize=FS_LABEL)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_ylim(0.2, 0.9)

    axes[0].set_ylabel("Expected score vs all budgets", fontsize=FS_LABEL)

    out = os.path.join(FIGS, "clock_budget_calibration_go_hex_mock.pdf")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
