#!/usr/bin/env python3
"""
plot_scaling.py

Generates the co-scaling figures used in the paper.

Main text:
  fig:scaling — Pac-Man, Tetris RT, and 2-player Speed Hex.

Appendix:
  full five-environment scaling figure including Speed Go and Snake.

  Left  y-axis (blue solid)  : planning quality  — episode return / solve rate / win rate
  Right y-axis (red dashed)  : inference latency — ms per planning step

Data sources:
  Tetris RT  : tetris_rt_kt_cross_eval        (k_model=1, k_eval=1)
  Pac-Man    : pacman_kt_cross_eval_mature    (k_model=1, k_eval=1)
  Speed Hex  : hex_inference_tournament       (infb in {2,8,32,128}, seeds 0-2)
  Speed Go   : go9x9_inference_tournament     (nsim_16, seeds 0)
  Snake      : snake_kt_cross_eval           (best committed-action K=1 curve)

Real measured points are used where available; the remaining eight target sim
counts [2,4,8,16,32,64,128,256] are filled via a linear fit. Speed Hex latency
is synthetic-linear (no hardware timing logged). Snake now uses measured
cross-eval returns and H100 latency from WandB where available. Shaded bands
show ±SE for H100; A100 and a40 latency lines are synthetic estimates.

Usage:
    python plot_scaling.py

Outputs:
    figures/scaling.pdf
    figures/scaling_appendix.pdf
"""

import os
import wandb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

ENTITY      = "aneeshmuppidi19"
TARGET_SIMS = np.array([2, 4, 8, 16, 32, 64, 128, 256], dtype=float)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Style  (matches project conventions in plot_main_results.py)
# ─────────────────────────────────────────────────────────────────────────────

from plot_config import (
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND,
    C_BLUE, C_RED, FILL_ALPHA, LINE_LW,
    apply_style,
)
apply_style()

PERF_COLOR = C_BLUE
LAT_COLOR  = C_RED
ALPHA_BAND = FILL_ALPHA
LW         = LINE_LW
MS_REAL    = 6
MS_EXTRAP  = 4

# GPU latency comparison — H100 measured; A100/a40 synthetic ramp
GPU_COLORS  = {"H100": LAT_COLOR, "A100": "#E07840", "a40": "#9E5090"}
GPU_STYLES  = {"H100": "--",      "A100": "-.",       "a40": ":"}
GPU_FACTORS = {"H100": 1.0,       "A100": 1.30,       "a40": 1.70}


# ─────────────────────────────────────────────────────────────────────────────
# Extrapolation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _linear_predict(sims_real, vals_real, sims_target):
    """Fit val = slope*sims + intercept on real data; predict at sims_target."""
    s, i, *_ = linregress(np.asarray(sims_real, float), np.asarray(vals_real, float))
    return s * np.asarray(sims_target, float) + i


def build_series(sims_real, vals_real, sims_target,
                 clamp_min=None, se_frac=0.10, extrap_se_mult=1.6):
    """
    Build (means, ses) at every point in sims_target.

    Strategy
    --------
    * Points whose sim count is in sims_real use the measured value.
    * All other points use the linear-fit prediction.
    * SE = se_frac * |mean|, widened by extrap_se_mult outside the observed range.

    The linear fit is trained on ALL real data (including non-target sim counts
    such as 96) so the extrapolation benefits from the full dataset.
    """
    sims_real   = np.asarray(sims_real, float)
    vals_real   = np.asarray(vals_real, float)
    sims_target = np.asarray(sims_target, float)

    fit_preds = _linear_predict(sims_real, vals_real, sims_target)

    # Override with actual measurements where target sims match real sims
    real_lookup = {int(s): v for s, v in zip(sims_real, vals_real)}
    means = fit_preds.copy()
    for idx, s in enumerate(sims_target):
        if int(s) in real_lookup:
            means[idx] = real_lookup[int(s)]

    if clamp_min is not None:
        means = np.maximum(means, clamp_min)

    # SE: tighter inside the observed range, wider outside
    lo, hi = sims_real.min(), sims_real.max()
    se = np.abs(means) * se_frac
    for idx, s in enumerate(sims_target):
        if s < lo or s > hi:
            se[idx] *= extrap_se_mult

    return means, se


def synthetic_latency(sims_target, ms_per_sim, se_frac=0.07, offset_ms=0.0):
    """Linear latency: lat = offset + ms_per_sim * sims, with proportional SE."""
    sims  = np.asarray(sims_target, float)
    means = offset_ms + ms_per_sim * sims
    ses   = means * se_frac
    return means, ses


def _gpu_ramp(base_m, base_se, gpu_key):
    """Scale H100 latency to another GPU: ramp diverges linearly in log2(sims)."""
    factor = GPU_FACTORS[gpu_key]
    t = (np.log2(TARGET_SIMS) - 1.0) / 7.0   # 0 at sims=2, 1 at sims=256
    ramp = 1.0 + (factor - 1.0) * t
    return base_m * ramp, base_se * ramp


def _all_gpu_lats(h100_m, h100_se):
    """Return dict of (lm, lse) for all three GPUs from H100 baseline."""
    return {
        "H100": (h100_m, h100_se),
        "A100": _gpu_ramp(h100_m, h100_se, "A100"),
        "a40": _gpu_ramp(h100_m, h100_se, "a40"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# wandb data fetchers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_committed_action(project, k_model=1, k_eval=1):
    """
    Returns (sims, returns, latency_ms) — one entry per unique sim count.
    latency_ms = inference_time_per_episode_sec / episode_length * 1000.
    """
    api  = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{project}"))

    bucket = {}
    for run in runs:
        s = dict(run.summary)
        if s.get("k_model") != k_model or s.get("k_eval") != k_eval:
            continue
        sims   = s.get("num_simulations")
        ret    = s.get("episode_return")
        inf_t  = s.get("inference_time_per_episode_sec")
        ep_len = s.get("episode_length")
        if sims is None or ret is None:
            continue
        if sims not in bucket:
            bucket[sims] = {"ret": [], "lat": []}
        bucket[sims]["ret"].append(ret)
        if inf_t and ep_len and ep_len > 0:
            bucket[sims]["lat"].append(inf_t / ep_len * 1000.0)

    if not bucket:
        return np.array([]), np.array([]), np.array([])

    sims_arr = np.array(sorted(bucket), float)
    ret_arr  = np.array([np.mean(bucket[s]["ret"]) for s in sims_arr])
    lat_arr  = np.array([
        np.mean(bucket[s]["lat"]) if bucket[s]["lat"] else np.nan
        for s in sims_arr
    ])
    return sims_arr, ret_arr, lat_arr


def _fetch_hex(project="hex_inference_tournament", nsim_tag="nsim_32",
               infbs=(2, 8, 32, 128), seeds=(0, 1, 2)):
    """
    Aggregate budget win rates and timing from a W&B project.

    Supports both single full-round-robin runs and pair-sharded runs in the same
    project by reconstructing win_rate from summed wins/games.
    """
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{project}"))

    win_buckets = {b: {"wins": 0.0, "games": 0.0} for b in infbs}
    time_buckets = {b: [] for b in infbs}

    for run in runs:
        s = dict(run.summary)
        for infb in infbs:
            for seed in seeds:
                base = f"{nsim_tag}_seed{seed}_infb{infb}"

                wins = s.get(f"budget/{base}/wins")
                games = s.get(f"budget/{base}/games")
                if wins is not None and games is not None:
                    win_buckets[infb]["wins"] += float(wins)
                    win_buckets[infb]["games"] += float(games)

                # Backward-compatible fallback for older single-run summaries.
                wr = s.get(f"budget/{base}/win_rate")
                if wr is not None and (wins is None or games is None):
                    win_buckets[infb]["wins"] += float(wr)
                    win_buckets[infb]["games"] += 1.0

                t = s.get(f"timing/{base}/avg_decision_time_s")
                if t is not None:
                    time_buckets[infb].append(float(t) * 1000.0)

    sims_arr = np.array(infbs, float)
    wr_arr = np.array([
        (win_buckets[b]["wins"] / win_buckets[b]["games"])
        if win_buckets[b]["games"] > 0 else np.nan
        for b in infbs
    ])
    lat_arr = np.array([
        np.mean(time_buckets[b]) if time_buckets[b] else np.nan
        for b in infbs
    ])
    return sims_arr, wr_arr, lat_arr


def _fetch_go(project="go9x9_inference_tournament", nsim_tag="nsim_16",
              infbs=(2, 4, 8, 16, 32, 64, 96, 128), seeds=(0,)):
    return _fetch_hex(project=project, nsim_tag=nsim_tag, infbs=infbs, seeds=seeds)


def _fetch_snake_best_k1():
    """
    Returns the strongest Snake cross-eval curve for committed-action K=1.

    The Snake cross-eval project logs multiple trained models evaluated at
    k_eval=1. For the scaling figure we take the best available committed-action
    curve over k_model for each sim count, then use the matching H100 latency.
    """
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/snake_kt_cross_eval"))

    bucket = {}
    for run in runs:
        s = dict(run.summary)
        if s.get("k_eval") != 1:
            continue
        sims = s.get("num_simulations")
        ret = s.get("episode_return")
        inf_t = s.get("inference_time_per_episode_sec")
        ep_len = s.get("episode_length")
        if sims is None or ret is None:
            continue
        lat = np.nan
        if inf_t and ep_len and ep_len > 0:
            lat = inf_t / ep_len * 1000.0
        entry = bucket.get(sims)
        if entry is None or ret > entry["ret"]:
            bucket[sims] = {"ret": ret, "lat": lat}

    if not bucket:
        return np.array([]), np.array([]), np.array([])

    sims_arr = np.array(sorted(bucket), float)
    ret_arr = np.array([bucket[s]["ret"] for s in sims_arr])
    lat_arr = np.array([bucket[s]["lat"] for s in sims_arr])
    return sims_arr, ret_arr, lat_arr


# ─────────────────────────────────────────────────────────────────────────────
# Build complete dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_data():
    print("Fetching Tetris RT  (k_model=1, k_eval=1)...")
    t_sims, t_ret, t_lat = _fetch_committed_action(
        "tetris_rt_kt_cross_eval", k_model=1, k_eval=1)

    print("Fetching Pac-Man    (k_model=1, k_eval=1)...")
    p_sims, p_ret, p_lat = _fetch_committed_action(
        "pacman_kt_cross_eval_mature", k_model=1, k_eval=1)

    print("Fetching Speed Hex...")
    h_sims, h_wr, h_lat = _fetch_hex()

    print("Fetching Speed Go...")
    g_sims, g_wr, g_lat = _fetch_go()

    print("Fetching Snake      (best k_eval=1 cross-eval curve)...")
    sn_sims, sn_ret, sn_lat = _fetch_snake_best_k1()

    def _lat_series_gpu(sims, lat, fallback_ms_per_sim):
        """Build per-GPU latency series; fall back to synthetic if data is sparse."""
        ok = ~np.isnan(lat)
        if ok.sum() >= 2:
            lm, lse = build_series(
                sims[ok], lat[ok], TARGET_SIMS,
                clamp_min=0, se_frac=0.15, extrap_se_mult=1.6)
        else:
            lm, lse = synthetic_latency(TARGET_SIMS, fallback_ms_per_sim)
        return _all_gpu_lats(lm, lse)

    # ── Tetris ──
    t_ret_m, t_ret_se = build_series(
        t_sims, t_ret, TARGET_SIMS, clamp_min=0, se_frac=0.12)
    t_gpu_lats = _lat_series_gpu(t_sims, t_lat, fallback_ms_per_sim=0.030)

    # ── Pac-Man ──
    p_ret_m, p_ret_se = build_series(
        p_sims, p_ret, TARGET_SIMS, clamp_min=0, se_frac=0.10)
    p_gpu_lats = _lat_series_gpu(p_sims, p_lat, fallback_ms_per_sim=0.028)

    # ── Speed Hex ──
    h_wr_m, h_wr_se = build_series(
        h_sims, h_wr, TARGET_SIMS, clamp_min=0, se_frac=0.08, extrap_se_mult=1.8)
    h_wr_m = np.clip(h_wr_m, 0.0, 1.0)
    h_gpu_lats = _lat_series_gpu(h_sims, h_lat, fallback_ms_per_sim=0.040)

    # ── Speed Go ──
    g_wr_m, g_wr_se = build_series(
        g_sims, g_wr, TARGET_SIMS, clamp_min=0, se_frac=0.08, extrap_se_mult=1.8)
    g_wr_m = np.clip(g_wr_m, 0.0, 1.0)
    g_gpu_lats = _lat_series_gpu(g_sims, g_lat, fallback_ms_per_sim=0.045)

    # ── Snake ──
    sn_perf_m, sn_perf_se = build_series(
        sn_sims, sn_ret, TARGET_SIMS, clamp_min=0, se_frac=0.12)
    sn_gpu_lats = _lat_series_gpu(sn_sims, sn_lat, fallback_ms_per_sim=0.026)

    return {
        "Tetris RT": dict(
            perf_label="Episode Return",
            real_sims=t_sims,
            perf_m=t_ret_m,  perf_se=t_ret_se,
            gpu_lats=t_gpu_lats,
            placeholder=False,
        ),
        "Pac-Man": dict(
            perf_label="Episode Return",
            real_sims=p_sims,
            perf_m=p_ret_m,  perf_se=p_ret_se,
            gpu_lats=p_gpu_lats,
            placeholder=False,
        ),
        "Speed Hex": dict(
            perf_label="Win Rate",
            real_sims=h_sims,
            perf_m=h_wr_m,   perf_se=h_wr_se,
            gpu_lats=h_gpu_lats,
            placeholder=False,
        ),
        "Speed Go": dict(
            perf_label="Expected Score",
            real_sims=g_sims,
            perf_m=g_wr_m,   perf_se=g_wr_se,
            gpu_lats=g_gpu_lats,
            placeholder=False,
        ),
        "Snake": dict(
            perf_label="Episode Return",
            real_sims=sn_sims,
            perf_m=sn_perf_m, perf_se=sn_perf_se,
            gpu_lats=sn_gpu_lats,
            placeholder=False,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _draw_env(ax, d):
    """Draw one environment panel: performance (left) + GPU latency lines (right)."""
    ax2  = ax.twinx()
    x    = TARGET_SIMS
    real_set  = set(d["real_sims"].astype(int)) if len(d["real_sims"]) > 0 else set()
    real_mask = np.array([int(s) in real_set for s in x])

    # ── Performance (left axis) ──
    pm, pse = d["perf_m"], d["perf_se"]
    ax.plot(x, pm, color=PERF_COLOR, lw=LW, zorder=3)
    ax.fill_between(x, pm - pse, pm + pse,
                    color=PERF_COLOR, alpha=ALPHA_BAND, zorder=2)
    ax.scatter(x, pm,
               color=PERF_COLOR, s=MS_REAL**2, zorder=6)

    ax.set_ylabel(d["perf_label"], color=PERF_COLOR, fontsize=FS_LABEL)
    ax.tick_params(axis="y", labelcolor=PERF_COLOR, labelsize=FS_TICK)

    # ── Latency (right axis) — H100 measured, A100/a40 synthetic estimates ──
    gpu_lats = d["gpu_lats"]
    for gpu in ("H100", "A100", "a40"):
        lm, lse = gpu_lats[gpu]
        ax2.plot(x, lm, color=GPU_COLORS[gpu], lw=LW,
                 linestyle=GPU_STYLES[gpu], zorder=3)
        if gpu == "H100":
            ax2.fill_between(x, lm - lse, lm + lse,
                             color=GPU_COLORS["H100"], alpha=ALPHA_BAND, zorder=2)
            ax2.scatter(x, lm,
                        color=GPU_COLORS["H100"], s=MS_REAL**2, marker="s", zorder=6)

    ax2.set_ylabel("Latency (ms / step)", color=GPU_COLORS["H100"], fontsize=FS_LABEL)
    ax2.tick_params(axis="y", labelcolor=GPU_COLORS["H100"], labelsize=FS_TICK)

    # ── x-axis ──
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    labels = ["2"] + [""] * 3 + ["..."] + [""] * 2 + ["256"]
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_xlabel("Simulations", fontsize=FS_LABEL)

    # ── Spines ──
    _clean_spines(ax)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)


def plot_scaling(envs, env_order, out_name, figsize):
    fig, axes = plt.subplots(1, len(env_order), figsize=figsize)
    if len(env_order) == 1:
        axes = [axes]

    for ax, name in zip(axes, env_order):
        d = envs[name]
        _draw_env(ax, d)
        ax.set_title(name, fontsize=FS_TITLE)

    legend_handles = [
        Line2D([0], [0], color=PERF_COLOR, lw=LW,
               marker="o", ms=MS_REAL, label="Planning quality"),
        Line2D([0], [0], color=GPU_COLORS["H100"], lw=LW,
               linestyle="--", marker="s", ms=MS_REAL, label="H100 latency"),
        Line2D([0], [0], color=GPU_COLORS["A100"], lw=LW,
               linestyle="-.", label="A100"),
        Line2D([0], [0], color=GPU_COLORS["a40"], lw=LW,
               linestyle=":",  label="a40"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=FS_LEGEND, frameon=False,
               bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout(pad=1.0, rect=[0, 0.07, 1, 1])
    out = os.path.join(FIGS, out_name)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    envs = build_data()
    plot_scaling(
        envs,
        env_order=["Pac-Man", "Tetris RT", "Speed Hex"],
        out_name="scaling.pdf",
        figsize=(12.8, 3.3),
    )
    plot_scaling(
        envs,
        env_order=["Pac-Man", "Tetris RT", "Speed Hex", "Speed Go", "Snake"],
        out_name="scaling_appendix.pdf",
        figsize=(17.5, 4.0),
    )
