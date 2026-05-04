#!/usr/bin/env python3
"""
plot_strategy.py

Planning-depth allocation frequency over normalized episode time.
5 panels: Pac-Man | Tetris RT | Speed Hex | Sokoban | Snake

Real data is pulled from wandb for Pac-Man, Tetris RT, and Snake via
_fetch_env_strategy().  Speed Hex and Sokoban remain placeholder until
those strategy evals are run.

Wandb projects queried:
  pacman_strategy_eval   (entity: aneeshmuppidi19)
  tetris_rt_strategy_eval
  snake_strategy_eval

Keys expected per project (logged by eval_{pacman,tetris,snake}_strategy.py):
  strategy/bin{00-09}_k{1-4}_mean
  strategy/bin{00-09}_k{1-4}_se

Saves: figures/strategy.pdf

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

FIG_WIDTH    = 21.0
FIG_HEIGHT   = 3.0
LINE_WIDTH   = LINE_LW
MARKER_EVERY = 3
MARKER_SIZE  = 4
LINE_STYLES  = ["-", "-", "-", "-"]
K3_ALPHA     = 0.30

WANDB_ENTITY = "aneeshmuppidi19"
SPEED_HEX_WANDB_PROJECT = os.getenv("SPEED_HEX_WANDB_PROJECT", "gru_ent100-earlylate")
SPEED_HEX_EARLY_TIME = int(os.getenv("SPEED_HEX_EARLY_TIME", "300"))
SPEED_HEX_LATE_TIME = int(os.getenv("SPEED_HEX_LATE_TIME", "4100"))

# ============================================================
# Wandb fetch helper
# ============================================================

def _fetch_env_strategy(project, entity=WANDB_ENTITY, n_bins=10, n_k=4, time_budget=None):
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


def _resolve_env(project, placeholder_series, time_budget=None):
    """Return (series, se_series) from wandb if available, else placeholder."""
    result = _fetch_env_strategy(project, time_budget=time_budget)
    if result is not None:
        return result
    return placeholder_series, _PLACEHOLDER_SE


# ============================================================
# Build ENVS data (fetches wandb for 3 real envs)
# ============================================================

_SPEED_HEX_PLACEHOLDER_SERIES = [
    np.array([0.55, 0.50, 0.40, 0.32, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10]),
    np.array([0.28, 0.30, 0.33, 0.35, 0.37, 0.38, 0.37, 0.36, 0.35, 0.33]),
    np.array([0.12, 0.14, 0.18, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38]),
    np.array([0.05, 0.06, 0.09, 0.11, 0.13, 0.14, 0.15, 0.17, 0.18, 0.19]),
]

_pacman_series, _pacman_se  = _resolve_env("pacman_strategy_eval",  _PLACEHOLDER_PACMAN_SERIES)
_tetris_series, _tetris_se  = _resolve_env("tetris_rt_strategy_eval", _PLACEHOLDER_TETRIS_SERIES)
_snake_series,  _snake_se   = _resolve_env("snake_strategy_eval",   _PLACEHOLDER_SNAKE_SERIES)
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

ENVS = [
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
        "title":       f"Speed Hex ({SPEED_HEX_EARLY_TIME})",
        "xlabel":      "Move fraction",
        "legend":      ["2 sims", "8 sims", "32 sims", "128 sims"],
        "show_legend": False,
        "series":      _speed_hex_early_series,
        "se_series":   _speed_hex_early_se,
    },
    {
        "title":       f"Speed Hex ({SPEED_HEX_LATE_TIME})",
        "xlabel":      "Move fraction",
        "legend":      ["2 sims", "8 sims", "32 sims", "128 sims"],
        "show_legend": False,
        "series":      _speed_hex_late_series,
        "se_series":   _speed_hex_late_se,
    },
    {
        "title":       "Sokoban",
        "xlabel":      "Episode progress",
        "legend":      ["K=1", "K=2", "K=3", "K=4"],
        "show_legend": False,
        "series": [
            np.array([0.40, 0.38, 0.35, 0.30, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10]),
            np.array([0.30, 0.32, 0.34, 0.36, 0.38, 0.37, 0.36, 0.35, 0.33, 0.32]),
            np.array([0.22, 0.22, 0.24, 0.26, 0.28, 0.32, 0.35, 0.38, 0.42, 0.45]),
            np.array([0.08, 0.08, 0.07, 0.08, 0.09, 0.11, 0.11, 0.12, 0.13, 0.13]),
        ],
        "se_series": _PLACEHOLDER_SE,
    },
    {
        "title":       "Snake",
        "xlabel":      "Episode progress",
        "legend":      ["K=1", "K=2", "K=3", "K=4"],
        "show_legend": True,
        "series":      _snake_series,
        "se_series":   _snake_se,
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


# ============================================================
# Plot
# ============================================================

def plot_strategy():
    fig, axes = plt.subplots(1, len(ENVS), figsize=(FIG_WIDTH, FIG_HEIGHT))

    for ax, env in zip(axes, ENVS):
        for i, (series, se, label) in enumerate(
            zip(env["series"], env["se_series"], env["legend"])
        ):
            is_zero = np.all(series == 0) or np.all(np.isnan(series))
            alpha     = K3_ALPHA if is_zero else 1.0
            markevery = None if is_zero else MARKER_EVERY

            plot_series = np.where(np.isnan(series), 0.0, series)

            ax.plot(
                T, plot_series,
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
                se_clean = np.where(np.isnan(se), 0.0, se)
                ax.fill_between(
                    T,
                    np.maximum(0, plot_series - se_clean),
                    plot_series + se_clean,
                    color=LINE_COLORS[i],
                    alpha=0.18,
                    linewidth=0,
                    zorder=1,
                )

        ax.set_xlim(0.05, 1.05)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(env["xlabel"], fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Frequency" if ax is axes[0] else "", fontsize=FONT_SIZE_LABEL)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.set_title(env["title"], fontsize=FONT_SIZE_TITLE)
        _apply_spine_style(ax)

        if env["show_legend"]:
            ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False,
                      loc="center left", bbox_to_anchor=(1.05, 0.5), handlelength=1.5)

    fig.tight_layout(pad=0.8)
    out = os.path.join(FIGS, "strategy.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_strategy()
