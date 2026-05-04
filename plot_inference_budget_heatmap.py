#!/usr/bin/env python3
"""
plot_inference_budget_heatmap.py

Fetch pairwise inference-budget tournament results from Weights & Biases and
render an expected-score heatmap.

By default this is configured for the Go 9x9 inference-budget tournament.

Usage:
    python plot_inference_budget_heatmap.py
    python plot_inference_budget_heatmap.py --project go9x9_inference_tournament
    python plot_inference_budget_heatmap.py --project hex_inference_tournament --nsim nsim_32
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from plot_config import C_BLUE, C_RED, FS_LABEL, FS_TICK, FS_ANNOT, LINE_LW, MARKER_SIZE, CAPSIZE, apply_style

apply_style()

ENTITY = "aneeshmuppidi19"
DEFAULT_PROJECT = "go9x9_inference_tournament"
DEFAULT_NSIM = "nsim_16"
DEFAULT_BUDGETS = [2, 4, 8, 16, 32, 64, 96, 128]

PAIR_KEY_RE = re.compile(
    r"^pair/(nsim_\d+)_seed(\d+)_infb(-?\d+)_vs_infb(-?\d+)/(.*)$"
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Plot inference-budget expected-score heatmap from W&B.")
    p.add_argument("--entity", type=str, default=ENTITY)
    p.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    p.add_argument("--nsim", type=str, default=DEFAULT_NSIM)
    p.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--title", type=str, default=None)
    return p.parse_args()


def get_summary_dict(run: wandb.apis.public.Run) -> Dict[str, object]:
    s = run.summary
    if hasattr(s, "_json_dict"):
        return dict(s._json_dict)
    try:
        return dict(s)
    except Exception:
        return {}


def fetch_pair_records(entity: str, project: str, nsim_tag: str):
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}"))
    records = {}

    for run in runs:
        summary = get_summary_dict(run)
        for key, val in summary.items():
            if val is None:
                continue
            m = PAIR_KEY_RE.match(key)
            if not m:
                continue
            nsim_k, seed_s, b0_s, b1_s, metric = m.groups()
            if nsim_k != nsim_tag:
                continue
            seed = int(seed_s)
            b0 = int(b0_s)
            b1 = int(b1_s)
            low, high = (b0, b1) if b0 <= b1 else (b1, b0)
            rec_key = (seed, low, high)
            records.setdefault(rec_key, {})[metric] = float(val)
    return records


def expected_score(wins: float, draws: float, games: float) -> float:
    if not np.isfinite(games) or games <= 0:
        return np.nan
    return (wins + 0.5 * draws) / games


def score_lookup(records, seed: int, i: int, j: int) -> float:
    low, high = (i, j) if i <= j else (j, i)
    rec = records.get((seed, low, high))
    if rec is None:
        return np.nan

    games = rec.get("games", np.nan)
    draws = rec.get("draws", np.nan)
    wins_low = rec.get(f"infb{low}_wins", np.nan)
    wins_high = rec.get(f"infb{high}_wins", np.nan) if low != high else np.nan

    if i == low:
        return expected_score(wins_low, draws, games)
    if np.isfinite(wins_high):
        return expected_score(wins_high, draws, games)
    s_low = expected_score(wins_low, draws, games)
    return 1.0 - s_low


def build_score_tensor(records, budgets: List[int]) -> Tuple[np.ndarray, List[int]]:
    seeds = sorted({seed for (seed, _, _) in records.keys()})
    S = np.full((len(seeds), len(budgets), len(budgets)), np.nan, dtype=float)
    for si, seed in enumerate(seeds):
        for i, bi in enumerate(budgets):
            for j, bj in enumerate(budgets):
                S[si, i, j] = score_lookup(records, seed, bi, bj)
    return S, seeds


def nansem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = np.sum(~np.isnan(x), axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return sd / np.sqrt(n)


def aggregate_vs_all_opponents(score_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_seeds, k, _ = score_tensor.shape
    per_seed = np.full((num_seeds, k), np.nan, dtype=float)
    for si in range(num_seeds):
        for i in range(k):
            vals = [score_tensor[si, i, j] for j in range(k) if j != i]
            per_seed[si, i] = np.nanmean(vals) if vals else np.nan
    return per_seed, np.nanmean(per_seed, axis=0), nansem(per_seed, axis=0)


def games_lookup(records, seed: int, i: int, j: int) -> float:
    low, high = (i, j) if i <= j else (j, i)
    rec = records.get((seed, low, high))
    if rec is None:
        return np.nan
    return float(rec.get("games", np.nan))


@dataclass
class PairObs:
    i: int
    j: int
    n: float
    s: float


def fit_elo_bt(pairs: List[PairObs], budgets: List[int], anchor_budget: int,
               max_iter: int = 200, tol: float = 1e-10, ridge: float = 1e-3) -> np.ndarray:
    idx = {b: k for k, b in enumerate(budgets)}
    m = len(budgets)
    if anchor_budget not in idx:
        anchor_budget = min(budgets)
    anchor_idx = idx[anchor_budget]
    free = [k for k in range(m) if k != anchor_idx]

    c = math.log(10.0) / 400.0
    r = np.zeros(m, dtype=float)

    for _ in range(max_iter):
        g = np.zeros(m, dtype=float)
        h = np.zeros((m, m), dtype=float)
        for obs in pairs:
            a = idx[obs.i]
            b = idx[obs.j]
            n = float(obs.n)
            s = float(obs.s)
            d = r[a] - r[b]
            p = 1.0 / (1.0 + math.exp(-c * d))
            y = s * n
            e = y - n * p
            g[a] += c * e
            g[b] -= c * e
            w = (c * c) * n * p * (1.0 - p)
            h[a, a] -= w
            h[b, b] -= w
            h[a, b] += w
            h[b, a] += w

        hf = h[np.ix_(free, free)]
        gf = g[free]
        a = (-hf) + ridge * np.eye(len(free))
        try:
            delta = np.linalg.solve(a, gf)
        except np.linalg.LinAlgError:
            break
        r[free] += delta
        if np.max(np.abs(delta)) < tol:
            break

    r[anchor_idx] = 0.0
    return r


def elo_ratings_per_seed(records, budgets: List[int], seeds: List[int], anchor_budget: int,
                         smooth_alpha: float = 0.5) -> np.ndarray:
    out = np.full((len(seeds), len(budgets)), np.nan, dtype=float)
    for si, seed in enumerate(seeds):
        pairs = []
        for i_idx, i in enumerate(budgets):
            for j in budgets[i_idx + 1:]:
                s = score_lookup(records, seed, i, j)
                n = games_lookup(records, seed, i, j)
                if not (np.isfinite(s) and np.isfinite(n) and n > 0):
                    continue
                s_smooth = (s * n + smooth_alpha) / (n + 2.0 * smooth_alpha)
                pairs.append(PairObs(i=i, j=j, n=n, s=s_smooth))
        if pairs:
            out[si] = fit_elo_bt(pairs, budgets, anchor_budget)
    return out


def plot_heatmap(mean_mat: np.ndarray, sem_mat: np.ndarray, budgets: List[int], title: str, out: str):
    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    im = ax.imshow(mean_mat, vmin=0.0, vmax=1.0, cmap="viridis")

    ax.set_xticks(range(len(budgets)))
    ax.set_yticks(range(len(budgets)))
    ax.set_xticklabels([str(b) for b in budgets], fontsize=FS_TICK)
    ax.set_yticklabels([str(b) for b in budgets], fontsize=FS_TICK)
    ax.set_xlabel("Column budget", fontsize=FS_LABEL)
    ax.set_ylabel("Row budget", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_LABEL)

    for i in range(len(budgets)):
        for j in range(len(budgets)):
            m = mean_mat[i, j]
            e = sem_mat[i, j]
            if not np.isfinite(m):
                continue
            color = "white" if m < 0.55 else "black"
            label = f"{m:.2f}"
            if np.isfinite(e):
                label = f"{m:.2f}\n±{e:.2f}"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=FS_ANNOT)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Expected score", fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)

    fig.tight_layout()
    fig.savefig(out, dpi=220)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_line(budgets: List[int], mean_vec: np.ndarray, sem_vec: np.ndarray,
              title: str, ylabel: str, out: str, color: str):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(budgets))
    ax.errorbar(
        x, mean_vec, yerr=sem_vec, color=color, marker="o",
        linewidth=LINE_LW, markersize=MARKER_SIZE, capsize=CAPSIZE
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets], fontsize=FS_TICK)
    ax.set_xlabel("Inference budget", fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_LABEL)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=FS_TICK)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    args = parse_args()
    records = fetch_pair_records(args.entity, args.project, args.nsim)
    if not records:
        raise RuntimeError(
            f"No pair records found in {args.entity}/{args.project} for {args.nsim}."
        )

    score_tensor, seeds = build_score_tensor(records, args.budgets)
    mean_mat = np.nanmean(score_tensor, axis=0)
    if len(seeds) > 1:
        sem_mat = nansem(score_tensor, axis=0)
    else:
        sem_mat = np.full_like(mean_mat, np.nan)

    title = args.title or f"Inference-budget expected score | {args.project} | {args.nsim}"
    out = args.out or os.path.join(FIGS, f"{args.project}_{args.nsim}_expected_score_heatmap.pdf")
    plot_heatmap(mean_mat, sem_mat, args.budgets, title, out)

    _, exp_mean, exp_sem = aggregate_vs_all_opponents(score_tensor)
    exp_out = os.path.join(FIGS, f"{args.project}_{args.nsim}_expected_score_line.pdf")
    plot_line(
        args.budgets,
        exp_mean,
        exp_sem,
        title=f"Expected score vs all budgets | {args.project} | {args.nsim}",
        ylabel="Expected score",
        out=exp_out,
        color=C_BLUE,
    )

    anchor_budget = min(args.budgets)
    elo_seed = elo_ratings_per_seed(records, args.budgets, seeds, anchor_budget=anchor_budget)
    elo_mean = np.nanmean(elo_seed, axis=0)
    elo_sem = nansem(elo_seed, axis=0)
    elo_out = os.path.join(FIGS, f"{args.project}_{args.nsim}_elo_line.pdf")
    plot_line(
        args.budgets,
        elo_mean,
        elo_sem,
        title=f"Elo-style budget rating | {args.project} | {args.nsim}",
        ylabel=f"Elo vs budget {anchor_budget}",
        out=elo_out,
        color=C_RED,
    )


if __name__ == "__main__":
    main()
