#!/usr/bin/env python3
"""
Analyze Speed Go gating-policy evaluation runs from W&B.

Outputs:
  - One figure with expected score vs clock budget for each opponent plus the
    average over opponents.
  - A printed summary using the paper's fixed 5-budget subset.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

from plot_config import (
    FS_LABEL, FS_LEGEND, FS_TICK, FS_TITLE, LINE_LW, MARKER_SIZE,
    C_BLUE, C_RED, C_NAVY, K_COLORS, apply_style,
)

apply_style()

ENTITY = "aneeshmuppidi19"
PROJECT = "gating_eval_go9x902"
GATE_ROOT = "/n/netscratch/gershman_lab/Lab/amuppidi/gru_go9x9_01"
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)
OUT = os.path.join(FIGS, "go_gate_eval_compare_16_32_48_64.pdf")
TITLE = "Speed Go"
SELECTED_BUDGETS = [300, 1200, 2300, 3500, 4800]


def _canonical_opp(opp: str) -> str:
    if opp in {"0", "always0"}:
        return "policy-only"
    if opp == "random":
        return "random_gate"
    return opp


def _opp_sort_key(opp: str):
    if opp.startswith("always") and opp[6:].isdigit():
        return (0, int(opp[6:]), opp)
    order = {"policy-only": 0, "proportional": 1, "midpeak": 2, "random_gate": 3}
    return (1, order.get(opp, 99), opp)


def _summary_dict(run):
    s = run.summary
    if hasattr(s, "_json_dict"):
        return dict(s._json_dict)
    return dict(s)


def _mean_se(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0
    se = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    return mean, se


def fetch_run_data():
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    data = defaultdict(lambda: defaultdict(list))

    for run in runs:
        cfg = dict(run.config)
        time_budget = cfg.get("time_budget")
        if cfg.get("gate_root") != GATE_ROOT or time_budget is None:
            continue
        summ = _summary_dict(run)
        for key, val in summ.items():
            if not key.endswith("/expected_score"):
                continue
            opp = _canonical_opp(key[:-len("/expected_score")])
            if opp.startswith("trend/"):
                continue
            if val is None:
                continue
            data[opp][int(time_budget)].append(float(val))
    return data


def aggregate_group(group_data):
    opponents = sorted(group_data.keys(), key=_opp_sort_key)
    budgets = sorted({b for opp in opponents for b in group_data[opp].keys()})
    mean = {opp: [] for opp in opponents}
    se = {opp: [] for opp in opponents}
    avg_mean = []
    avg_se = []

    for b in budgets:
        per_opp_means = []
        per_opp_ses = []
        for opp in opponents:
            m, s = _mean_se(group_data[opp].get(b, []))
            mean[opp].append(m)
            se[opp].append(s)
            if np.isfinite(m):
                per_opp_means.append(m)
                per_opp_ses.append(s if np.isfinite(s) else 0.0)
        if per_opp_means:
            avg_mean.append(float(np.mean(per_opp_means)))
            avg_se.append(float(np.sqrt(np.sum(np.square(per_opp_ses))) / max(len(per_opp_ses), 1)))
        else:
            avg_mean.append(float("nan"))
            avg_se.append(float("nan"))

    return budgets, opponents, mean, se, np.array(avg_mean), np.array(avg_se)


def plot_group(label, payload):
    budgets, opponents, mean, se, avg_mean, avg_se = payload
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colors = [C_BLUE, C_RED, C_NAVY] + K_COLORS + ["#cc7a00", "#008b8b", "#7f7f7f"]
    for i, opp in enumerate(opponents):
        y = np.asarray(mean[opp], dtype=float)
        yerr = np.asarray(se[opp], dtype=float)
        ax.errorbar(
            budgets, y, yerr=yerr, marker="o", markersize=MARKER_SIZE - 1,
            linewidth=LINE_LW, capsize=3, color=colors[i % len(colors)], label=opp
        )

    ax.plot(budgets, avg_mean, color="black", linewidth=2.5, linestyle="--", label="avg")
    ax.fill_between(budgets, avg_mean - avg_se, avg_mean + avg_se, color="black", alpha=0.12)
    ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1.2)
    ax.set_title(label, fontsize=FS_TITLE)
    ax.set_xlabel("Clock budget", fontsize=FS_LABEL)
    ax.set_ylabel("Expected score", fontsize=FS_LABEL)
    ax.set_xticks(budgets)
    ax.tick_params(axis="x", rotation=35, labelsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(True, alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=FS_LEGEND)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(OUT, dpi=220)
    print(f"Saved: {OUT}")
    plt.close(fig)


def main():
    raw = fetch_run_data()
    if not raw:
        raise RuntimeError(f"No matching runs found in {ENTITY}/{PROJECT} for gate_root={GATE_ROOT}")

    payload = aggregate_group(raw)
    budgets, opponents, mean, se, avg_mean, avg_se = payload

    idx_map = {b: i for i, b in enumerate(budgets)}
    missing = [b for b in SELECTED_BUDGETS if b not in idx_map]
    if missing:
        raise RuntimeError(f"Selected budgets missing from data: {missing}")

    score = float(np.mean([avg_mean[idx_map[b]] for b in SELECTED_BUDGETS]))
    score_se = float(np.sqrt(np.sum([avg_se[idx_map[b]] ** 2 for b in SELECTED_BUDGETS])) / len(SELECTED_BUDGETS))
    print(
        f"{TITLE}: selected budgets = {SELECTED_BUDGETS} | "
        f"avg expected score={score:.4f} +/- {score_se:.4f}"
    )
    for opp in opponents:
        vals = [mean[opp][idx_map[b]] for b in SELECTED_BUDGETS]
        ses = [se[opp][idx_map[b]] if np.isfinite(se[opp][idx_map[b]]) else 0.0 for b in SELECTED_BUDGETS]
        opp_mean = float(np.mean(vals))
        opp_se = float(np.sqrt(np.sum(np.square(ses))) / len(SELECTED_BUDGETS))
        print(f"  {opp:<12} avg over {SELECTED_BUDGETS} = {opp_mean:.4f} +/- {opp_se:.4f}")

    plot_group(TITLE, payload)


if __name__ == "__main__":
    main()
