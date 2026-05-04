#!/usr/bin/env python3
"""
plot_deployment.py

Generates fig:deployment — a one-row dashboard validating the 2-GPU real-time setup.
  Panel A: Timeline (Gantt chart) of asynchronous execution at 9 FPS.
  Panel B: Violin plots of measured MCTS latency across environments and GPUs.
  Panel C: Average deadline miss rate bars with SE across environments.
  Panel D: Sim-vs-real normalized return bars using main-results simulation values.

Produces:
    figures/deployment.pdf
"""

import json
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
    C_BLUE, C_RED, BAR_ALPHA, ERR_LW, CAPSIZE, K_COLORS,
    apply_style,
)
apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

# Colors
GPU1_COLOR = C_BLUE   # Env + Fast Policy (GPU 0)
GPU2_COLOR = K_COLORS[3]    # MCTS (GPU 1) purple
GPU_COLORS = {"h100": C_BLUE, "a100": C_RED, "a40": "#2E8B57"}
GAME_COLORS = {"tetris": C_BLUE, "pacman": C_RED, "snake": C_BLACK}
GPU_LINESTYLES = {"h100": "-", "a100": "--", "a40": ":"}
GPU_MARKERS = {"h100": "o", "a100": "s", "a40": "^"}
TITLE_FS = max(FS_TITLE - 4, 8)
SMALL_LABEL_FS = max(FS_LABEL - 2, 8)
SMALL_LEGEND_FS = max(FS_LEGEND - 2, 7)

GAMES = ["tetris", "pacman", "snake"]
GPUS = ["h100", "a100", "a40"]
FPS_LIST = [8, 9, 10, 11, 12]
GPU_LABELS = {"h100": "H100", "a100": "A100", "a40": "A40"}
GAME_LABELS = {"tetris": "Tetris", "pacman": "Pacman", "snake": "Snake"}
SIM_RESULTS = {
    "pacman": {"mean": 2370.0, "se": 59.0},
    "tetris": {"mean": 45.6, "se": 3.7},
    "snake": {"mean": 16.54, "se": 1.26},
}

# ===========================================================================
# Deploy result paths
# ===========================================================================

DEPLOY_ROOT = "/n/netscratch/gershman_lab/Lab/amuppidi/work"

# Legacy single-FPS (fps=9) Tetris paths — kept for backward compat.
_LEGACY_PATHS = {
    "H100": os.path.join(DEPLOY_ROOT, "tetris_rt_deploy_out",      "deploy_results.json"),
    "A100": os.path.join(DEPLOY_ROOT, "tetris_rt_deploy_out_a100", "deploy_results.json"),
    "A40":  os.path.join(DEPLOY_ROOT, "tetris_rt_deploy_out_a40",  "deploy_results.json"),
}

DEADLINE_MS = 444.0   # K=4 budget: 4 frames × 111ms at 9 FPS

# ===========================================================================
# Panel A: 9 FPS async timeline
# ===========================================================================

FPS = 9.0
FRAME_MS = 1000.0 / FPS
TIMELINE_LIMIT = 680.0

GPU1_BLOCKS = [(FRAME_MS * k, 2.5) for k in range(7)]
GPU2_BLOCKS = [
    (5.0, 200.0),
    (FRAME_MS * 4 + 5, 18.0),
    (FRAME_MS * 5 + 5, 200.0),
]


def _new_path(game, gpu_type, fps):
    """Path written by deploy_rt_realtime_{h100,a100,a40}.sh array jobs."""
    fps_key = int(fps) if fps == int(fps) else fps
    return os.path.join(DEPLOY_ROOT, "rt_deploy_out", game, gpu_type, f"fps_{fps_key}", "deploy_results.json")


def _load_result(path):
    """Load deploy_results.json; return (latencies_array, full_dict) or (None, None)."""
    if os.path.exists(path):
        d = json.load(open(path))
        lats = np.array(d["mcts_latency_ms"]["all"])
        return lats, d
    return None, None


def load_all_results(game="tetris", fps=9):
    """Return {gpu_type: (latencies, result_dict)} for all available GPUs at given game/fps."""
    results = {}
    for gpu_type in GPUS:
        gpu_label = GPU_LABELS[gpu_type]
        lats, d = _load_result(_new_path(game, gpu_type, fps))
        if lats is not None:
            print(f"  {gpu_label} ({game} fps={fps}): {len(lats)} samples (new path)")
            results[gpu_type] = (lats, d)
        elif game == "tetris" and fps == 9:
            lats, d = _load_result(_LEGACY_PATHS[gpu_label])
            if lats is not None:
                print(f"  {gpu_label} ({game} fps={fps}): {len(lats)} samples (legacy path)")
                results[gpu_type] = (lats, d)
    return results


def _load_latencies(gpu_label):
    """Loader used by the 9-FPS tetris violin (Panel B)."""
    legacy = _LEGACY_PATHS[gpu_label]
    new    = _new_path("tetris", gpu_label.lower(), 9)
    for p in (new, legacy):
        lats, _ = _load_result(p)
        if lats is not None:
            print(f"  {gpu_label}: loaded {len(lats)} measured latency samples from {p}")
            return lats, True
    return None, False


def _apply_spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _load_summary_rows():
    rows = []
    for game in GAMES:
        for gpu in GPUS:
            for fps in FPS_LIST:
                p = _new_path(game, gpu, fps)
                lats, d = _load_result(p)
                if lats is None and game == "tetris" and fps == 9:
                    lats, d = _load_result(_LEGACY_PATHS[gpu.upper()])
                if lats is None:
                    continue

                budget_ms = 4 * (1000.0 / fps)
                p95_lat = np.percentile(lats, 95)
                rows.append({
                    "game": game,
                    "gpu": gpu,
                    "fps": fps,
                    "ret_mean": d.get("mean_return", float("nan")),
                    "ret_se": d.get("se_return", float("nan")),
                    "miss_pct": 100.0 * d.get("deadline_misses", 0) / max(d.get("total_meta_steps", len(lats)), 1),
                    "slack_p95": budget_ms - p95_lat,
                })
    return rows


def _grouped_latency_samples():
    grouped = {(game, gpu): [] for game in GAMES for gpu in GPUS}
    for game in GAMES:
        for gpu in GPUS:
            for fps in FPS_LIST:
                p = _new_path(game, gpu, fps)
                lats, _ = _load_result(p)
                if lats is None and game == "tetris" and fps == 9:
                    lats, _ = _load_result(_LEGACY_PATHS[gpu.upper()])
                if lats is not None:
                    grouped[(game, gpu)].append(lats)
    return {
        key: np.concatenate(val) for key, val in grouped.items() if val
    }


def _plot_timeline(ax):
    ax.broken_barh(GPU1_BLOCKS, (10, 8), facecolors=GPU1_COLOR, alpha=BAR_ALPHA, edgecolor="none")
    ax.broken_barh(GPU2_BLOCKS, (20, 8), facecolors=GPU2_COLOR, alpha=BAR_ALPHA, edgecolor="none")

    for k in range(1, 7):
        ax.axvline(FRAME_MS * k, color=C_LIGHT_GRAY, linestyle="--", linewidth=0.8, zorder=0)

    ax.annotate("", xy=(FRAME_MS * 4, 7), xytext=(0, 7),
                arrowprops=dict(arrowstyle="<->", color=C_DARK_GRAY, lw=1.0))
    ax.text(FRAME_MS * 2, 5.5, "K=4 meta-step", color=C_DARK_GRAY,
            fontsize=FS_ANNOT, ha="center", va="top")

    ax.annotate("", xy=(FRAME_MS * 5, 7), xytext=(FRAME_MS * 4, 7),
                arrowprops=dict(arrowstyle="<->", color=C_DARK_GRAY, lw=1.0))
    ax.text(FRAME_MS * 4.5, 5.5, "K=1", color=C_DARK_GRAY,
            fontsize=FS_ANNOT, ha="center", va="top")

    ax.set_ylim(3, 35)
    ax.set_xlim(0, TIMELINE_LIMIT)
    ax.set_xlabel("Time (ms)", fontsize=FS_LABEL)
    ax.set_yticks([14, 24])
    ax.set_yticklabels(["GPU 0\n(Env + $\\pi_{reflex}$)", "GPU 1\n(MCTS)"], fontsize=FS_TICK)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_title("Asynchronous Execution\nTimeline (9 FPS)", fontsize=TITLE_FS)
    _apply_spine_style(ax)


def _plot_latency_violins(ax, grouped_lats):
    positions = []
    datasets = []
    colors = []
    centers = []
    pos = 1.0
    offsets = [-0.24, 0.0, 0.24]

    for game in GAMES:
        group_positions = []
        for offset, gpu in zip(offsets, GPUS):
            key = (game, gpu)
            if key not in grouped_lats:
                continue
            positions.append(pos + offset)
            group_positions.append(pos + offset)
            datasets.append(grouped_lats[key])
            colors.append(GPU_COLORS[gpu])
        centers.append(pos)
        pos += 1.15

    parts = ax.violinplot(datasets, positions=positions, widths=0.18, showmeans=True, showextrema=False)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_alpha(0.55)
    parts["cmeans"].set_color(C_BLACK)

    ax.set_xticks(centers)
    ax.set_xticklabels([GAME_LABELS[g] for g in GAMES], fontsize=FS_TICK)
    ax.set_ylabel("MCTS Latency (ms)", fontsize=FS_LABEL)
    ax.set_ylim(0, 450)
    ax.set_title("Latency Violins\n(all FPS combined)", fontsize=TITLE_FS)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    _apply_spine_style(ax)


def _plot_miss_rate_bars(ax, rows):
    means = []
    ses = []
    for gpu in GPUS:
        per_env_means = []
        for game in GAMES:
            subset = [r["miss_pct"] for r in rows if r["game"] == game and r["gpu"] == gpu]
            if subset:
                per_env_means.append(np.mean(np.asarray(subset, float)))
        arr = np.asarray(per_env_means, float)
        means.append(np.mean(arr))
        ses.append(np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0)

    x = np.arange(len(GPUS))
    ax.bar(x, means, color=[GPU_COLORS[g] for g in GPUS], alpha=BAR_ALPHA, linewidth=0)
    ax.errorbar(x, means, yerr=ses, fmt="none", ecolor=C_BLACK,
                elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)
    ax.set_xticks(x)
    ax.set_xticklabels([GPU_LABELS[g] for g in GPUS], fontsize=FS_TICK)
    ax.set_yscale("log")
    ax.set_ylabel("Deadline Miss %", fontsize=FS_LABEL)
    ax.set_title("Average Miss Rate\n(across envs +/- SE)", fontsize=TITLE_FS)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    _apply_spine_style(ax)


def _plot_transfer_bars(ax, rows):
    x = np.arange(len(GAMES))
    width = 0.19

    sim_means = [100.0 for _ in GAMES]
    sim_ses = [100.0 * (SIM_RESULTS[g]["se"] / SIM_RESULTS[g]["mean"]) for g in GAMES]
    ax.bar(x - 1.5 * width, sim_means, width=width, color=C_MID_GRAY, alpha=BAR_ALPHA, linewidth=0, label="Sim")
    ax.errorbar(x - 1.5 * width, sim_means, yerr=sim_ses, fmt="none", ecolor=C_BLACK,
                elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)

    for idx, gpu in enumerate(GPUS):
        means = []
        ses = []
        for game in GAMES:
            real_mean = next(r["ret_mean"] for r in rows if r["game"] == game and r["gpu"] == gpu and r["fps"] == 9)
            real_se = next(r["ret_se"] for r in rows if r["game"] == game and r["gpu"] == gpu and r["fps"] == 9)
            sim_mean = SIM_RESULTS[game]["mean"]
            sim_se = SIM_RESULTS[game]["se"]
            means.append(100.0 * real_mean / sim_mean)
            ses.append(100.0 * np.sqrt((real_se / sim_mean) ** 2 + ((real_mean * sim_se) / (sim_mean ** 2)) ** 2))
        xpos = x + (-0.5 + idx) * width
        ax.bar(xpos, means, width=width, color=GPU_COLORS[gpu], alpha=BAR_ALPHA, linewidth=0, label=GPU_LABELS[gpu])
        ax.errorbar(xpos, means, yerr=ses, fmt="none", ecolor=C_BLACK,
                    elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)

    ax.set_xticks(x)
    ax.set_xticklabels([GAME_LABELS[g] for g in GAMES], fontsize=FS_TICK)
    ax.set_ylabel("Return Relative to Sim (%)", fontsize=SMALL_LABEL_FS)
    ax.set_title("Sim vs Real Return\n(normalized to sim = 100%)", fontsize=TITLE_FS)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.axhline(100.0, color=C_DARK_GRAY, linestyle="--", linewidth=1.0, zorder=0)
    _apply_spine_style(ax)


# ===========================================================================
# Main
# ===========================================================================


def plot_deployment():
    fig, axes = plt.subplots(
        2, 2, figsize=(10.5, 8.5),
    )
    summary_rows = _load_summary_rows()
    grouped_lats = _grouped_latency_samples()
    _plot_timeline(axes[0, 0])
    _plot_latency_violins(axes[0, 1], grouped_lats)
    _plot_miss_rate_bars(axes[1, 0], summary_rows)
    _plot_transfer_bars(axes[1, 1], summary_rows)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_MID_GRAY, alpha=BAR_ALPHA),
        plt.Rectangle((0, 0), 1, 1, color=GPU_COLORS["h100"], alpha=BAR_ALPHA),
        plt.Rectangle((0, 0), 1, 1, color=GPU_COLORS["a100"], alpha=BAR_ALPHA),
        plt.Rectangle((0, 0), 1, 1, color=GPU_COLORS["a40"], alpha=BAR_ALPHA),
    ]
    fig.legend(
        legend_handles,
        ["Sim", "H100", "A100", "A40"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        fontsize=SMALL_LEGEND_FS,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    # -----------------------------------------------------------
    # Layout and save
    # -----------------------------------------------------------
    fig.tight_layout(pad=1.0, w_pad=1.5, h_pad=2.0, rect=[0, 0.05, 1, 1])
    out = os.path.join(FIGS, "deployment.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

def _plot_combined_latency_miss(ax, rows, grouped_lats):
    """Combined plot: latency violins with miss rate as color intensity or overlay."""
    
    # Create a twin x-axis for the miss rate
    ax2 = ax.twinx()
    
    # Plot latency violins on primary axis
    positions = []
    datasets = []
    colors = []
    centers = []
    pos = 1.0
    offsets = [-0.24, 0.0, 0.24]
    
    # Calculate miss rates for each GPU and environment
    miss_rates = {gpu: [] for gpu in GPUS}
    for gpu in GPUS:
        for game in GAMES:
            subset = [r["miss_pct"] for r in rows if r["game"] == game and r["gpu"] == gpu]
            if subset:
                miss_rates[gpu].append(np.mean(subset))
    
    for game in GAMES:
        group_positions = []
        for offset, gpu in zip(offsets, GPUS):
            key = (game, gpu)
            if key not in grouped_lats:
                continue
            positions.append(pos + offset)
            group_positions.append(pos + offset)
            datasets.append(grouped_lats[key])
            # Color intensity based on miss rate
            avg_miss = np.mean(miss_rates[gpu]) if miss_rates[gpu] else 0
            # Darker = higher miss rate, but keep base color
            colors.append(GPU_COLORS[gpu])
        centers.append(pos)
        pos += 1.15
    
    # Plot violins
    parts = ax.violinplot(datasets, positions=positions, widths=0.18, 
                         showmeans=True, showextrema=False)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_alpha(0.55)
    parts["cmeans"].set_color(C_BLACK)
    
    # Add miss rate as bar plot on secondary axis
    x_positions = centers  # Center positions for each game
    bar_width = 0.8
    
    for idx, gpu in enumerate(GPUS):
        miss_values = [np.mean([r["miss_pct"] for r in rows 
                               if r["game"] == game and r["gpu"] == gpu]) 
                      for game in GAMES]
        # Offset bars slightly
        bar_x = [x + (idx - 1) * bar_width/3 for x in x_positions]
        bars = ax2.bar(bar_x, miss_values, width=bar_width/3, 
                      color=GPU_COLORS[gpu], alpha=0.3, 
                      label=f"{GPU_LABELS[gpu]} miss %")
    
    # Styling
    ax.set_xticks(centers)
    ax.set_xticklabels([GAME_LABELS[g] for g in GAMES], fontsize=FS_TICK)
    ax.set_ylabel("MCTS Latency (ms)", fontsize=FS_LABEL, color=C_BLACK)
    ax.set_ylim(0, 450)
    ax.tick_params(axis="y", labelcolor=C_BLACK, labelsize=FS_TICK)
    
    ax2.set_ylabel("Deadline Miss Rate (%)", fontsize=FS_LABEL, color=C_RED)
    ax2.tick_params(axis="y", labelcolor=C_RED, labelsize=FS_TICK)
    ax2.set_ylim(0, 100)  # Assuming miss rate is percentage
    
    ax.set_title("Latency & Miss Rate\n(colors: GPU, bars: miss %)", fontsize=TITLE_FS)
    _apply_spine_style(ax)
    
    # Add legend for miss rates
    ax2.legend(loc='upper right', fontsize=SMALL_LEGEND_FS, frameon=False)

def plot_deployment_compact():
    """Compact version with 3 panels instead of 4."""
    fig, axes = plt.subplots(
        1, 3, figsize=(15.0, 4.4),
        gridspec_kw={"width_ratios": [1.25, 1.8, 1.1]},  # Give more space to combined plot
    )
    summary_rows = _load_summary_rows()
    grouped_lats = _grouped_latency_samples()
    
    _plot_timeline(axes[0])
    _plot_combined_latency_miss(axes[1], summary_rows, grouped_lats)
    _plot_transfer_bars(axes[2], summary_rows)
    
    # Simplified legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_MID_GRAY, alpha=BAR_ALPHA),
        plt.Rectangle((0, 0), 1, 1, color=GPU_COLORS["h100"], alpha=BAR_ALPHA),
        plt.Rectangle((0, 0), 1, 1, color=GPU_COLORS["a100"], alpha=BAR_ALPHA),
        plt.Rectangle((0, 0), 1, 1, color=GPU_COLORS["a40"], alpha=BAR_ALPHA),
    ]
    fig.legend(
        legend_handles,
        ["Sim", "H100", "A100", "A40"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=4,
        frameon=False,
        fontsize=SMALL_LEGEND_FS,
    )
    
    fig.tight_layout(pad=1.0, w_pad=1.1, rect=[0, 0.08, 1, 1])
    out = os.path.join(FIGS, "deployment_compact.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

def print_summary():
    """Print a detailed text summary of all available deployment results."""
    GAMES    = ["tetris", "pacman", "snake"]
    GPUS     = ["h100", "a100", "a40"]
    FPS_LIST = [8, 9, 10, 11, 12]

    SEP  = "=" * 100
    SEP2 = "-" * 100

    # ── 1. Coverage matrix ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("COVERAGE MATRIX  (✓ = results available)")
    print(SEP)
    header = f"{'game/gpu':<14}" + "".join(f"  fps={f}" for f in FPS_LIST)
    print(header)
    print(SEP2)
    for game in GAMES:
        for gpu in GPUS:
            row = f"{game}/{gpu:<8}"
            for fps in FPS_LIST:
                # check new path first, then legacy for tetris@9
                p = _new_path(game, gpu, fps)
                if os.path.exists(p):
                    row += "     ✓  "
                elif game == "tetris" and fps == 9 and os.path.exists(_LEGACY_PATHS[gpu.upper()]):
                    row += "    (L) "   # legacy path
                else:
                    row += "     ·  "
            print(row)
    print(f"  (L) = loaded from legacy path (tetris fps=9 pre-array-job runs)")

    # ── 2. Per-run detailed table ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("PER-RUN RESULTS")
    print(SEP)

    col_w = 12
    hdr = (f"{'game':<8} {'gpu':<6} {'fps':>4}  "
           f"{'n_ep':>5} {'steps':>6}  "
           f"{'return':>10} {'±SE':>7}  "
           f"{'lat_mean':>9} {'lat_p50':>8} {'lat_p95':>8} {'lat_p99':>8} {'lat_max':>8}  "
           f"{'budget':>7} {'slack_p95':>10}  "
           f"{'misses':>7} {'miss%':>7}  "
           f"K1%  K2%  K3%  K4%")
    print(hdr)
    print(SEP2)

    all_rows = []   # collect for cross-env/fps analysis later

    for game in GAMES:
        for gpu in GPUS:
            for fps in FPS_LIST:
                # try new path, then legacy
                p = _new_path(game, gpu, fps)
                lats, d = _load_result(p)
                if lats is None and game == "tetris" and fps == 9:
                    lats, d = _load_result(_LEGACY_PATHS[gpu.upper()])
                if lats is None:
                    continue

                budget_ms  = 4 * (1000.0 / fps)   # K=4 budget
                mean_lat   = np.mean(lats)
                p50_lat    = np.percentile(lats, 50)
                p95_lat    = np.percentile(lats, 95)
                p99_lat    = np.percentile(lats, 99)
                max_lat    = np.max(lats)
                slack_p95  = budget_ms - p95_lat

                misses     = d.get("deadline_misses", 0)
                steps      = d.get("total_meta_steps", len(lats))
                miss_pct   = 100.0 * misses / max(steps, 1)

                ret_mean   = d.get("mean_return", float("nan"))
                ret_se     = d.get("se_return",   float("nan"))
                n_eps      = d.get("n_episodes",  "?")

                k_dist     = d.get("k_distribution", {})
                k1 = k_dist.get("k1", float("nan")) * 100
                k2 = k_dist.get("k2", float("nan")) * 100
                k3 = k_dist.get("k3", float("nan")) * 100
                k4 = k_dist.get("k4", float("nan")) * 100

                row = (f"{game:<8} {gpu:<6} {fps:>4}  "
                       f"{n_eps:>5} {steps:>6}  "
                       f"{ret_mean:>10.2f} {ret_se:>7.2f}  "
                       f"{mean_lat:>9.1f} {p50_lat:>8.1f} {p95_lat:>8.1f} {p99_lat:>8.1f} {max_lat:>8.1f}  "
                       f"{budget_ms:>7.1f} {slack_p95:>10.1f}  "
                       f"{misses:>7} {miss_pct:>7.3f}  "
                       f"{k1:4.1f} {k2:4.1f} {k3:4.1f} {k4:4.1f}")
                print(row)
                all_rows.append(dict(
                    game=game, gpu=gpu, fps=fps,
                    budget_ms=budget_ms, mean_lat=mean_lat,
                    p95_lat=p95_lat, slack_p95=slack_p95,
                    misses=misses, steps=steps, miss_pct=miss_pct,
                    ret_mean=ret_mean, ret_se=ret_se,
                    k1=k1, k2=k2, k3=k3, k4=k4,
                ))
        print(SEP2)   # blank separator between games

    # ── 3. Deadline analysis: miss rate vs FPS per GPU ────────────────────────
    if all_rows:
        print(f"\n{SEP}")
        print("DEADLINE MISS RATE (%)  by  game × gpu × fps")
        print("(budget = 4 × frame_ms;  slack_p95 = budget − p95_latency)")
        print(SEP)
        miss_hdr = f"{'game':<8} {'gpu':<6}  " + "  ".join(f"fps={f}(slack)" for f in FPS_LIST)
        print(miss_hdr)
        print(SEP2)
        for game in GAMES:
            for gpu in GPUS:
                row_data = {r["fps"]: r for r in all_rows if r["game"]==game and r["gpu"]==gpu}
                if not row_data:
                    continue
                cells = []
                for fps in FPS_LIST:
                    if fps in row_data:
                        r = row_data[fps]
                        cells.append(f"{r['miss_pct']:5.3f}%({r['slack_p95']:+6.0f}ms)")
                    else:
                        cells.append("          ·         ")
                print(f"{game:<8} {gpu:<6}  " + "  ".join(cells))
            print(SEP2)

        # ── 4. K-distribution stability across FPS ────────────────────────────
        print(f"\n{SEP}")
        print("K-DISTRIBUTION STABILITY  (K1% / K2% / K3% / K4%)  across FPS")
        print(SEP)
        for game in GAMES:
            print(f"  {game.upper()}")
            for gpu in GPUS:
                row_data = {r["fps"]: r for r in all_rows if r["game"]==game and r["gpu"]==gpu}
                if not row_data:
                    continue
                cells = []
                for fps in FPS_LIST:
                    if fps in row_data:
                        r = row_data[fps]
                        cells.append(f"fps={fps}: {r['k1']:.0f}/{r['k2']:.0f}/{r['k3']:.0f}/{r['k4']:.0f}")
                    else:
                        cells.append(f"fps={fps}: ·")
                print(f"    {gpu:<6}  " + "   ".join(cells))
            print()

        # ── 5. Return vs FPS ──────────────────────────────────────────────────
        print(f"\n{SEP}")
        print("EPISODE RETURN (mean ± SE)  across FPS")
        print(SEP)
        for game in GAMES:
            print(f"  {game.upper()}")
            for gpu in GPUS:
                row_data = {r["fps"]: r for r in all_rows if r["game"]==game and r["gpu"]==gpu}
                if not row_data:
                    continue
                cells = []
                for fps in FPS_LIST:
                    if fps in row_data:
                        r = row_data[fps]
                        cells.append(f"fps={fps}: {r['ret_mean']:.1f}±{r['ret_se']:.1f}")
                    else:
                        cells.append(f"fps={fps}: ·")
                print(f"    {gpu:<6}  " + "   ".join(cells))
            print()

    print(SEP)
    print("END OF SUMMARY")
    print(SEP + "\n")


if __name__ == "__main__":
    print_summary()
    plot_deployment()
