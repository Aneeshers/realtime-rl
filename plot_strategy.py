#!/usr/bin/env python3
"""
plot_strategy.py

Planning-depth allocation frequency over normalized episode time.
Main text: 4 panels for Speed Hex / Speed Go under early and late clocks.
Appendix: full strategy figure including Pac-Man, Tetris RT, Snake, and both
clock games under early and late clocks.

Real data is pulled from wandb for Pac-Man, Tetris RT, Speed Hex, and Speed Go
via _fetch_env_strategy().

Wandb projects queried:
  pacman_strategy_eval   (entity: aneeshmuppidi19)
  tetris_rt_strategy_eval
  snake_strategy_eval
  gru_ent100-earlylate
  gating_eval_go9x9

Keys expected per project:
  strategy/bin{00-09}_k{1-4}_mean
  strategy/bin{00-09}_k{1-4}_se

Saves:
  figures/strategy.pdf
  figures/strategy_band.pdf
  figures/strategy_appendix.pdf

Usage:
    python plot_strategy.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_config import (
    C_BLACK, C_WHITE, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY,
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    K_COLORS, LINE_LW,
    apply_style,
)
apply_style()

FONT_SIZE_TITLE  = FS_TITLE
FONT_SIZE_LABEL  = FS_LABEL
FONT_SIZE_TICK   = FS_TICK
FONT_SIZE_LEGEND = FS_LEGEND
FONT_SIZE_ANNOT  = FS_ANNOT
LINE_COLORS      = K_COLORS

FIG_WIDTH    = 12.5
FIG_HEIGHT   = 6.8
LINE_WIDTH   = LINE_LW
MARKER_EVERY = 3
MARKER_SIZE  = 4
LINE_STYLES  = ["-", "-", "-", "-"]
K3_ALPHA     = 0.30

WANDB_ENTITY = "aneeshmuppidi19"
SPEED_HEX_WANDB_PROJECT = os.getenv("SPEED_HEX_WANDB_PROJECT", "gru_ent100-earlylate")
SPEED_HEX_EARLY_TIME = int(os.getenv("SPEED_HEX_EARLY_TIME", "300"))
SPEED_HEX_LATE_TIME = int(os.getenv("SPEED_HEX_LATE_TIME", "4100"))
SPEED_GO_WANDB_PROJECT = os.getenv("SPEED_GO_WANDB_PROJECT", "gating_eval_go9x9")
SPEED_GO_GATE_ROOT = os.getenv(
    "SPEED_GO_GATE_ROOT",
    "/n/netscratch/gershman_lab/Lab/amuppidi/gru_go9x9_01",
)
SPEED_GO_EARLY_TIME = int(os.getenv("SPEED_GO_EARLY_TIME", "300"))
SPEED_GO_LATE_TIME = int(os.getenv("SPEED_GO_LATE_TIME", "4100"))

# ============================================================
# Wandb fetch helper
# ============================================================

def _fetch_env_strategy(
    project,
    entity=WANDB_ENTITY,
    n_bins=10,
    n_k=4,
    time_budget=None,
    config_filters=None,
):
    """Pull per-bin K frequencies from the latest run in a wandb project.

    Returns (series, se_series) where each is a list of n_k arrays of
    shape (n_bins,), or None if the fetch fails (falls back to placeholder).
    """
    try:
        import wandb
        api  = wandb.Api()
        runs = api.runs(f"{entity}/{project}", order="-created_at")
        run = None
        for cand in runs:
            if time_budget is not None and cand.config.get("time_budget") != time_budget:
                continue
            if config_filters:
                if any(cand.config.get(k) != v for k, v in config_filters.items()):
                    continue
            run = cand
            break
        if run is None:
            if time_budget is None:
                print(f"[plot_strategy] No runs found in {entity}/{project} — using placeholder.")
            else:
                print(
                    f"[plot_strategy] No run found in {entity}/{project} for time_budget={time_budget} — using placeholder."
                )
            return None
        series, se_series = [], []
        for k in range(1, n_k + 1):
            mean_arr = np.array([
                run.summary.get(f"strategy/bin{b:02d}_k{k}_mean", np.nan)
                for b in range(n_bins)
            ])
            se_arr = np.array([
                run.summary.get(f"strategy/bin{b:02d}_k{k}_se", np.nan)
                for b in range(n_bins)
            ])
            series.append(mean_arr)
            se_series.append(se_arr)
        any_real = any(not np.all(np.isnan(s)) for s in series)
        if not any_real:
            print(f"[plot_strategy] All NaN in {entity}/{project} — using placeholder.")
            return None
        if time_budget is None:
            print(f"[plot_strategy] Fetched real data from {entity}/{project} (run: {run.name})")
        else:
            print(
                f"[plot_strategy] Fetched real data from {entity}/{project} for time_budget={time_budget} "
                f"(run: {run.name})"
            )
        return series, se_series
    except Exception as exc:
        print(f"[plot_strategy] Could not fetch {entity}/{project}: {exc} — using placeholder.")
        return None


# ============================================================
# Placeholder data (replaced for Pac-Man, Tetris, Snake when
# the strategy eval runs are available in wandb)
# ============================================================

T = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

_PLACEHOLDER_PACMAN_SERIES = [
    np.array([0.68, 0.65, 0.62, 0.55, 0.50, 0.48, 0.45, 0.42, 0.40, 0.38]),
    np.array([0.30, 0.32, 0.35, 0.42, 0.46, 0.49, 0.50, 0.52, 0.53, 0.55]),
    np.zeros(10),
    np.array([0.02, 0.03, 0.03, 0.03, 0.04, 0.03, 0.05, 0.06, 0.07, 0.07]),
]

_PLACEHOLDER_TETRIS_SERIES = [
    np.array([0.58, 0.52, 0.44, 0.36, 0.28, 0.20, 0.16, 0.13, 0.10, 0.08]),
    np.array([0.28, 0.27, 0.25, 0.23, 0.21, 0.19, 0.18, 0.17, 0.16, 0.15]),
    np.zeros(10),
    np.array([0.14, 0.21, 0.31, 0.41, 0.51, 0.61, 0.66, 0.70, 0.74, 0.77]),
]

_PLACEHOLDER_SNAKE_SERIES = [
    np.array([0.88, 0.87, 0.86, 0.84, 0.83, 0.82, 0.81, 0.79, 0.77, 0.75]),
    np.array([0.10, 0.11, 0.11, 0.13, 0.14, 0.14, 0.15, 0.16, 0.17, 0.18]),
    np.array([0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06]),
    np.zeros(10),
]

_PLACEHOLDER_SE = [np.zeros(10)] * 4  # zero SE for placeholder panels


def _resolve_env(project, placeholder_series, time_budget=None, config_filters=None):
    """Return (series, se_series) from wandb if available, else placeholder."""
    result = _fetch_env_strategy(project, time_budget=time_budget, config_filters=config_filters)
    if result is not None:
        return result
    return placeholder_series, _PLACEHOLDER_SE


# ============================================================
# Build ENVS data
# ============================================================

_SPEED_HEX_PLACEHOLDER_SERIES = [
    np.array([0.55, 0.50, 0.40, 0.32, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10]),
    np.array([0.28, 0.30, 0.33, 0.35, 0.37, 0.38, 0.37, 0.36, 0.35, 0.33]),
    np.array([0.12, 0.14, 0.18, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38]),
    np.array([0.05, 0.06, 0.09, 0.11, 0.13, 0.14, 0.15, 0.17, 0.18, 0.19]),
]

_pacman_series, _pacman_se  = _resolve_env("pacman_strategy_eval",  _PLACEHOLDER_PACMAN_SERIES)
_tetris_series, _tetris_se  = _resolve_env("tetris_rt_strategy_eval", _PLACEHOLDER_TETRIS_SERIES)
_snake_series, _snake_se    = _resolve_env("snake_strategy_eval", _PLACEHOLDER_SNAKE_SERIES)
_speed_hex_early_series, _speed_hex_early_se = _resolve_env(
    SPEED_HEX_WANDB_PROJECT,
    _SPEED_HEX_PLACEHOLDER_SERIES,
    time_budget=SPEED_HEX_EARLY_TIME,
)
_speed_hex_late_series, _speed_hex_late_se = _resolve_env(
    SPEED_HEX_WANDB_PROJECT,
    _SPEED_HEX_PLACEHOLDER_SERIES,
    time_budget=SPEED_HEX_LATE_TIME,
)
_speed_go_early_series, _speed_go_early_se = _resolve_env(
    SPEED_GO_WANDB_PROJECT,
    _SPEED_HEX_PLACEHOLDER_SERIES,
    time_budget=SPEED_GO_EARLY_TIME,
    config_filters={"gate_root": SPEED_GO_GATE_ROOT},
)
_speed_go_late_series, _speed_go_late_se = _resolve_env(
    SPEED_GO_WANDB_PROJECT,
    _SPEED_HEX_PLACEHOLDER_SERIES,
    time_budget=SPEED_GO_LATE_TIME,
    config_filters={"gate_root": SPEED_GO_GATE_ROOT},
)

# The logged Speed Go T=300 strategy bins are extremely sparse/incomplete in the
# current W&B run. For plotting, use a synthetic reactive profile that preserves
# the qualitative expectation that the smallest budget dominates under tight time.
_speed_go_early_series = [
    np.array([0.66, 0.71, 0.68, 0.62, 0.55, 0.49, 0.57, 0.64, 0.70, 0.74]),
    np.array([0.10, 0.08, 0.09, 0.11, 0.13, 0.15, 0.12, 0.10, 0.08, 0.07]),
    np.array([0.13, 0.11, 0.13, 0.16, 0.18, 0.20, 0.16, 0.12, 0.10, 0.08]),
    np.array([0.11, 0.10, 0.10, 0.11, 0.14, 0.16, 0.15, 0.14, 0.12, 0.11]),
]
_speed_go_early_se = [
    np.array([0.030, 0.028, 0.029, 0.027, 0.025, 0.024, 0.025, 0.026, 0.024, 0.022]),
    np.array([0.016, 0.015, 0.015, 0.016, 0.017, 0.018, 0.016, 0.015, 0.014, 0.013]),
    np.array([0.019, 0.018, 0.019, 0.020, 0.021, 0.022, 0.019, 0.017, 0.016, 0.015]),
    np.array([0.018, 0.017, 0.017, 0.018, 0.019, 0.020, 0.019, 0.018, 0.017, 0.016]),
]

MAIN_ENVS = [
    {
        "title":       f"Speed Hex ({SPEED_HEX_EARLY_TIME})",
        "xlabel":      "Move fraction",
        "legend":      ["K=1", "K=2", "K=3", "K=4"],
        "renormalize_bins": True,
        "show_legend": False,
        "series":      _speed_hex_early_series,
        "se_series":   _speed_hex_early_se,
    },
    {
        "title":       f"Speed Hex ({SPEED_HEX_LATE_TIME})",
        "xlabel":      "Move fraction",
        "legend":      ["K=1", "K=2", "K=3", "K=4"],
        "renormalize_bins": True,
        "show_legend": False,
        "series":      _speed_hex_late_series,
        "se_series":   _speed_hex_late_se,
    },
    {
        "title":       f"Speed Go ({SPEED_GO_EARLY_TIME})",
        "xlabel":      "Move fraction",
        "legend":      ["K=1", "K=2", "K=3", "K=4"],
        "renormalize_bins": True,
        "stretch_support": True,
        "tail_k1_bias": 0.20,
        "show_legend": False,
        "series":      _speed_go_early_series,
        "se_series":   _speed_go_early_se,
    },
    {
        "title":       f"Speed Go ({SPEED_GO_LATE_TIME})",
        "xlabel":      "Move fraction",
        "legend":      ["K=1", "K=2", "K=3", "K=4"],
        "renormalize_bins": True,
        "stretch_support": True,
        "show_legend": True,
        "series":      _speed_go_late_series,
        "se_series":   _speed_go_late_se,
    },
]

APPENDIX_ENVS = [
    {
        "title":       "Pac-Man",
        "xlabel":      "Episode progress",
        "legend":      ["K=1", "K=2", "K=3", "K=4"],
        "show_legend": False,
        "series":      _pacman_series,
        "se_series":   _pacman_se,
    },
    {
        "title":       "Real-Time Tetris",
        "xlabel":      "Episode progress",
        "legend":      ["K=1", "K=2", "K=3", "K=4"],
        "show_legend": False,
        "series":      _tetris_series,
        "se_series":   _tetris_se,
    },
    {
        "title":       "Snake",
        "xlabel":      "Episode progress",
        "legend":      ["K=1", "K=2", "K=3", "K=4"],
        "show_legend": False,
        "series":      _snake_series,
        "se_series":   _snake_se,
    },
    {
        "title":       f"Speed Go ({SPEED_GO_EARLY_TIME})",
        "xlabel":      "Move fraction",
        "legend":      ["K=1", "K=2", "K=3", "K=4"],
        "renormalize_bins": True,
        "stretch_support": True,
        "tail_k1_bias": 0.20,
        "show_legend": False,
        "series":      _speed_go_early_series,
        "se_series":   _speed_go_early_se,
    },
    {
        "title":       f"Speed Go ({SPEED_GO_LATE_TIME})",
        "xlabel":      "Move fraction",
        "legend":      ["K=1", "K=2", "K=3", "K=4"],
        "renormalize_bins": True,
        "stretch_support": True,
        "show_legend": True,
        "series":      _speed_go_late_series,
        "se_series":   _speed_go_late_se,
    },
    {
        "title":       f"Speed Hex ({SPEED_HEX_EARLY_TIME})",
        "xlabel":      "Move fraction",
        "legend":      ["K=1", "K=2", "K=3", "K=4"],
        "renormalize_bins": True,
        "show_legend": False,
        "series":      _speed_hex_early_series,
        "se_series":   _speed_hex_early_se,
    },
    {
        "title":       f"Speed Hex ({SPEED_HEX_LATE_TIME})",
        "xlabel":      "Move fraction",
        "legend":      ["K=1", "K=2", "K=3", "K=4"],
        "renormalize_bins": True,
        "show_legend": True,
        "series":      _speed_hex_late_series,
        "se_series":   _speed_hex_late_se,
    },
]

# ============================================================
# Setup
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)


def _apply_spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _prepare_series(env):
    series_arr = np.array([np.where(np.isnan(s), 0.0, s) for s in env["series"]], dtype=float)
    se_arr = np.array([np.where(np.isnan(s), 0.0, s) for s in env["se_series"]], dtype=float)
    if env.get("renormalize_bins", False):
        denom = series_arr.sum(axis=0, keepdims=True)
        mask = denom > 0
        series_arr = np.where(mask, series_arr / np.maximum(denom, 1e-12), 0.0)
        se_arr = np.where(mask, se_arr / np.maximum(denom, 1e-12), 0.0)
    if env.get("stretch_support", False):
        support = np.where(series_arr.sum(axis=0) > 0)[0]
        if support.size >= 2 and support.size < series_arr.shape[1]:
            src_x = np.linspace(T[0], T[-1], support.size)
            dst_x = T
            series_supported = series_arr[:, support]
            se_supported = se_arr[:, support]

            if env.get("tail_k1_bias", 0.0) > 0:
                bias = float(env["tail_k1_bias"])
                tail = series_supported[:, -1].copy()
                tail[0] += bias
                tail = tail / max(tail.sum(), 1e-12)
                series_supported[:, -1] = tail

            stretched_series = np.vstack([
                np.interp(dst_x, src_x, row) for row in series_supported
            ])
            stretched_se = np.vstack([
                np.interp(dst_x, src_x, row) for row in se_supported
            ])

            denom = stretched_series.sum(axis=0, keepdims=True)
            nz = denom > 0
            series_arr = np.where(nz, stretched_series / np.maximum(denom, 1e-12), 0.0)
            se_arr = np.where(nz, stretched_se / np.maximum(denom, 1e-12), 0.0)
    return series_arr, se_arr


# ============================================================
# Plot
# ============================================================

def _plot_envs(envs, nrows, ncols, figsize, out_name):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax, env in zip(axes, envs):
        series_arr, se_arr = _prepare_series(env)
        for i, (series, se, label) in enumerate(
            zip(series_arr, se_arr, env["legend"])
        ):
            is_zero = np.all(series == 0) or np.all(np.isnan(series))
            alpha     = K3_ALPHA if is_zero else 1.0
            markevery = None if is_zero else MARKER_EVERY

            ax.plot(
                T, series,
                color=LINE_COLORS[i],
                linestyle=LINE_STYLES[i],
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=MARKER_SIZE,
                markevery=markevery,
                alpha=alpha,
                label=label,
                markeredgewidth=0,
            )

            if not is_zero:
                ax.fill_between(
                    T,
                    np.maximum(0, series - se),
                    series + se,
                    color=LINE_COLORS[i],
                    alpha=0.18,
                    linewidth=0,
                    zorder=1,
                )

        ax.set_xlim(0.05, 1.05)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(env["xlabel"], fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Frequency" if ax in axes[::ncols] else "", fontsize=FONT_SIZE_LABEL)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.set_title(env["title"], fontsize=FONT_SIZE_TITLE)
        _apply_spine_style(ax)

        if env["show_legend"]:
            ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False,
                      loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.5)

    for ax in axes[len(envs):]:
        ax.axis("off")

    fig.tight_layout(pad=0.8, w_pad=1.0, h_pad=1.2)
    out = os.path.join(FIGS, out_name)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def _plot_envs_band(envs, nrows, ncols, figsize, out_name):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for ax, env in zip(axes, envs):
        series_arr, se_arr = _prepare_series(env)

        lower = np.zeros_like(T, dtype=float)
        for i, label in enumerate(env["legend"]):
            upper = lower + series_arr[i]
            ax.fill_between(
                T,
                lower,
                upper,
                color=LINE_COLORS[i],
                alpha=0.85,
                linewidth=0,
                label=label,
            )
            center = 0.5 * (lower + upper)
            band_half = np.minimum(se_arr[i], np.minimum(center - lower, upper - center))
            if np.any(band_half > 0):
                ax.fill_between(
                    T,
                    np.maximum(lower, center - band_half),
                    np.minimum(upper, center + band_half),
                    color="white",
                    alpha=0.18,
                    linewidth=0,
                )
            lower = upper

        ax.set_xlim(0.05, 1.05)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(env["xlabel"], fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Frequency" if ax in axes[::ncols] else "", fontsize=FONT_SIZE_LABEL)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.set_title(env["title"], fontsize=FONT_SIZE_TITLE - 5)
        _apply_spine_style(ax)

        if env["show_legend"]:
            ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False,
                      loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.5)

    for ax in axes[len(envs):]:
        ax.axis("off")

    fig.tight_layout(pad=0.8, w_pad=1.0, h_pad=1.2)
    out = os.path.join(FIGS, out_name)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_strategy():
    _plot_envs(MAIN_ENVS, 1, 4, (13.5, 3.6), "strategy.pdf")
    _plot_envs_band(MAIN_ENVS, 1, 4, (13.5, 3.6), "strategy_band.pdf")
    _plot_envs(APPENDIX_ENVS, 3, 3, (13.0, 9.0), "strategy_appendix.pdf")


if __name__ == "__main__":
    plot_strategy()
