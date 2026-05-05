#!/usr/bin/env python3
"""
plot_timeout_hex_appendix.py

Plots strict-timeout Speed Hex unique expected score vs clock budget from WandB
eval runs and prints a compact textual summary for the paper appendix.

Produces:
    figures/hex_timeout_appendix.pdf
    figures/hex_timeout_appendix.png

Usage:
    python plot_timeout_hex_appendix.py
    python plot_timeout_hex_appendix.py --project timeout_hex_eval
"""

import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_config import (
    C_BLACK,
    FS_TITLE,
    FS_LABEL,
    FS_TICK,
    FS_LEGEND,
    ERR_LW,
    CAPSIZE,
    apply_style,
)

apply_style()

WANDB_ENTITY_DEFAULT = "aneeshmuppidi19"
WANDB_PROJECT_DEFAULT = "timeout_hex_eval"

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

SERIES = [
    ("Policy-only", "always0", "#4d4d4d"),
    ("2 sims", "always2", "#9ecae1"),
    ("8 sims", "always8", "#6baed6"),
    ("32 sims", "always32", "#4292c6"),
    ("128 sims", "always128", "#2171b5"),
    ("Random", "random_gate", "#bdbdbd"),
    ("Midpeak", "midpeak", "#74c476"),
    ("Greedy", "proportional", "#fdae6b"),
]


def _apply_spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _fetch_eval_grid(project: str, entity: str):
    import wandb

    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}"))
    budgets = sorted({
        int(run.config.get("time_budget"))
        for run in runs
        if run.config.get("time_budget") is not None and run.config.get("mode") != "eval_summary"
    })
    seeds = sorted({
        int(run.config.get("seed"))
        for run in runs
        if run.config.get("time_budget") is not None and run.config.get("seed") is not None
    })

    curves: Dict[str, Dict[str, List[float]]] = {}
    for display_label, wandb_label, _color in SERIES:
        means, ses = [], []
        for tb in budgets:
            vals = []
            for run in runs:
                if run.config.get("mode") == "eval_summary":
                    continue
                if int(run.config.get("time_budget", -1)) != tb:
                    continue
                key = f"{wandb_label}/expected_score"
                if key in run.summary:
                    vals.append(float(run.summary[key]))
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                means.append(float(arr.mean()))
                ses.append(float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0)
            else:
                means.append(float("nan"))
                ses.append(float("nan"))
        curves[display_label] = {
            "wandb_label": wandb_label,
            "budgets": budgets,
            "means": means,
            "ses": ses,
        }
    return curves, budgets, seeds


def _print_summary(curves: Dict[str, Dict[str, List[float]]], budgets: List[int], seeds: List[int]):
    print("=== Timeout Hex Unique-Game Summary ===")
    print(f"Budgets: {budgets}")
    print(f"Seeds used: {seeds}")
    print("")

    overall_rows: List[Tuple[float, str]] = []
    for display_label, _wandb_label, _color in SERIES:
        d = curves[display_label]
        means = np.asarray(d["means"], dtype=np.float64)
        valid = ~np.isnan(means)
        if not valid.any():
            print(f"{display_label:>12}: no data")
            continue
        valid_budgets = np.asarray(budgets)[valid]
        valid_means = means[valid]
        avg = float(valid_means.mean())
        min_idx = int(np.argmin(valid_means))
        max_idx = int(np.argmax(valid_means))
        min_b = int(valid_budgets[min_idx])
        max_b = int(valid_budgets[max_idx])
        min_v = float(valid_means[min_idx])
        max_v = float(valid_means[max_idx])
        slope = max_v - min_v
        above_parity = bool(np.all(valid_means > 0.5))
        overall_rows.append((avg, display_label))
        print(
            f"{display_label:>12}: unique_avg={avg:.3f} | min={min_v:.3f} @ T={min_b} | "
            f"max={max_v:.3f} @ T={max_b} | range={slope:.3f} | above_parity_all={above_parity}"
        )

    print("")
    overall_rows.sort(reverse=True)
    print("Opponent ranking by mean unique expected score:")
    for rank, (avg, label) in enumerate(overall_rows, start=1):
        print(f"  {rank}. {label}: {avg:.3f}")

    print("")
    print("Per-budget spread:")
    for i, tb in enumerate(budgets):
        budget_rows = []
        for display_label, _wandb_label, _color in SERIES:
            val = curves[display_label]["means"][i]
            if not math.isnan(val):
                budget_rows.append((float(val), display_label))
        budget_rows.sort(reverse=True)
        if not budget_rows:
            continue
        top_val, top_label = budget_rows[0]
        bot_val, bot_label = budget_rows[-1]
        print(
            f"  T={tb}: best={top_label} ({top_val:.3f}), "
            f"worst={bot_label} ({bot_val:.3f}), spread={top_val - bot_val:.3f}"
        )


def plot_timeout_hex_appendix(project: str, entity: str, output_stem: str):
    curves, budgets, seeds = _fetch_eval_grid(project=project, entity=entity)
    _print_summary(curves, budgets, seeds)

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.8))
    for display_label, _wandb_label, color in SERIES:
        d = curves[display_label]
        x = np.asarray(d["budgets"], dtype=np.float64)
        y = np.asarray(d["means"], dtype=np.float64)
        se = np.asarray(d["ses"], dtype=np.float64)
        valid = ~np.isnan(y)
        if not valid.any():
            continue
        ax.plot(
            x[valid],
            y[valid],
            color=color,
            marker="o",
            linewidth=1.9,
            markersize=4.5,
            label=display_label,
        )
        ax.errorbar(
            x[valid],
            y[valid],
            yerr=se[valid],
            fmt="none",
            ecolor=color,
            elinewidth=ERR_LW,
            capsize=CAPSIZE,
            capthick=ERR_LW,
        )

    ax.axhline(0.5, color=C_BLACK, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Clock budget", fontsize=FS_LABEL)
    ax.set_ylabel("Unique expected score", fontsize=FS_LABEL)
    ax.set_title(
        "Strict-timeout Speed Hex is easy to solve by avoiding flag-falls",
        fontsize=FS_TITLE - 4,
        pad=8,
    )
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=FS_LEGEND - 1, ncol=2, loc="lower right")
    _apply_spine_style(ax)

    fig.tight_layout(pad=0.8)
    pdf_out = os.path.join(FIGS, f"{output_stem}.pdf")
    png_out = os.path.join(FIGS, f"{output_stem}.png")
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, dpi=220, bbox_inches="tight")
    print(f"Saved: {pdf_out}")
    print(f"Saved: {png_out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot strict-timeout Speed Hex appendix figure from WandB")
    parser.add_argument("--project", type=str, default=WANDB_PROJECT_DEFAULT)
    parser.add_argument("--entity", type=str, default=WANDB_ENTITY_DEFAULT)
    parser.add_argument("--output_stem", type=str, default="hex_timeout_appendix")
    args = parser.parse_args()
    plot_timeout_hex_appendix(project=args.project, entity=args.entity, output_stem=args.output_stem)


if __name__ == "__main__":
    main()
