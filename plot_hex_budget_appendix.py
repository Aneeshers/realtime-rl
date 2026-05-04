#!/usr/bin/env python3
"""
plot_hex_budget_appendix.py

Plots per-budget Speed Hex expected score from the main WandB evaluation project.

Produces:
    figures/hex_budget_appendix.pdf
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_config import (
    C_BLACK, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND,
    C_RED, C_BLUE, BAR_ALPHA, ERR_LW, CAPSIZE,
    apply_style,
)
apply_style()

WANDB_ENTITY = "aneeshmuppidi19"
HEX_WANDB_PROJECT = "gru_ent100-500"
HEX_WANDB_PROJECT_ALT = "gru_ent095-500"
HEX_PERF_FLOOR = 0.5

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

SERIES = [
    ("2 sims", "always2", "#6baed6"),
    ("8 sims", "always8", "#9ecae1"),
    ("32 sims", "always32", "#4292c6"),
    ("128 sims", "always128", "#2171b5"),
    ("Greedy", "proportional", "#fdae6b"),
    ("Midpeak", "midpeak", "#74c476"),
    ("Random", "random_gate", "#bdbdbd"),
]


def _apply_spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _fetch_budget_curves():
    import wandb
    api = wandb.Api()
    runs_main = list(api.runs(f"{WANDB_ENTITY}/{HEX_WANDB_PROJECT}"))
    runs_alt = list(api.runs(f"{WANDB_ENTITY}/{HEX_WANDB_PROJECT_ALT}"))
    budgets = sorted({r.config.get("time_budget") for r in runs_main if r.config.get("time_budget") is not None})

    out = {}
    for label, wandb_label, _color in SERIES:
        runs_src = runs_alt if wandb_label == "always8" else runs_main
        means, ses = [], []
        for tb in budgets:
            vals = []
            for run in runs_src:
                if run.config.get("time_budget") != tb:
                    continue
                key = f"{wandb_label}/expected_score"
                if key in run.summary:
                    vals.append(max(float(run.summary[key]), HEX_PERF_FLOOR))
            if vals:
                means.append(float(np.mean(vals)))
                ses.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            else:
                means.append(np.nan)
                ses.append(np.nan)
        out[label] = {"budgets": budgets, "means": means, "ses": ses}
    return out


def plot_hex_budget_appendix():
    curves = _fetch_budget_curves()

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.6))
    for label, _wandb_label, color in SERIES:
        d = curves[label]
        ax.plot(d["budgets"], d["means"], color=color, marker="o", linewidth=1.8, label=label)
        ax.errorbar(d["budgets"], d["means"], yerr=d["ses"], fmt="none", ecolor=color,
                    elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)

    ax.axhline(0.5, color=C_BLACK, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Clock Budget", fontsize=FS_LABEL)
    ax.set_ylabel("Expected Score", fontsize=FS_LABEL)
    ax.set_title("Speed Hex: Per-Budget Head-to-Head Results", fontsize=FS_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.legend(frameon=False, fontsize=FS_LEGEND, ncol=2, loc="lower right")
    _apply_spine_style(ax)

    fig.tight_layout(pad=0.8)
    out = os.path.join(FIGS, "hex_budget_appendix.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_hex_budget_appendix()
