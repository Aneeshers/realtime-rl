#!/usr/bin/env python3
"""
plot_deployment.py

Generates fig:deployment — a one-row dashboard validating the 2-GPU real-time setup.
  Panel A: Timeline (Gantt chart) of asynchronous execution at 9 FPS with deadline miss indicators.
  Panel B: Violin plots of measured MCTS latency across environments and GPUs.
  Panel C: Average deadline miss rate bars with SE across environments.
  Panel D: Sim-vs-real normalized return bars using main-results simulation values.
  Panel E: Cumulative miss distribution with deadline visualization.

Produces:
    figures/deployment.pdf
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ===========================================================================
# Style — sourced from plot_config.py
# ===========================================================================

# Define style defaults (in case plot_config is not available)
C_BLACK = "#000000"
C_WHITE = "#FFFFFF"
C_DARK_GRAY = "#333333"
C_MID_GRAY = "#808080"
C_LIGHT_GRAY = "#CCCCCC"
FS_TITLE = 12
FS_LABEL = 10
FS_TICK = 8
FS_LEGEND = 9
FS_ANNOT = 8
C_BLUE = "#1f77b4"
C_RED = "#d62728"
BAR_ALPHA = 0.7
ERR_LW = 1.0
CAPSIZE = 3

def apply_style():
    """Apply matplotlib style settings."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

try:
    from plot_config import (
        C_BLACK, C_WHITE, C_DARK_GRAY, C_MID_GRAY, C_LIGHT_GRAY,
        FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
        C_BLUE, C_RED, BAR_ALPHA, ERR_LW, CAPSIZE,
        apply_style,
    )
except ImportError:
    pass  # Use defaults defined above

apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

# Colors
GPU1_COLOR = C_BLUE   # Env + Fast Policy (GPU 0)
GPU2_COLOR = C_RED    # MCTS (GPU 1)
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
# Panel A: 9 FPS async timeline WITH DEADLINE MISS VISUALIZATION
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
                    "latencies": lats,  # Store latencies for miss analysis
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


def _plot_timeline_with_misses(ax, latency_data=None, miss_data=None):
    """
    Enhanced timeline with deadline miss indicators.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    latency_data : array-like, optional
        Latency measurements for miss detection
    miss_data : dict, optional
        Pre-computed miss information with timestamps
    """
    # Draw GPU execution blocks
    ax.broken_barh(GPU1_BLOCKS, (10, 8), facecolors=GPU1_COLOR, alpha=BAR_ALPHA, edgecolor="none")
    ax.broken_barh(GPU2_BLOCKS, (20, 8), facecolors=GPU2_COLOR, alpha=BAR_ALPHA, edgecolor="none")
    
    # Add deadline miss markers from actual data
    miss_count = 0
    if latency_data is not None and len(latency_data) > 0:
        # Simulate MCTS execution timestamps based on frame timing
        # Each MCTS occurs at start of each meta-step (approximately)
        for i, lat in enumerate(latency_data[:50]):  # Show first 50 to avoid clutter
            # Approximate timing: MCTS starts at frame boundaries
            exec_time = 5 + (i % 6) * FRAME_MS  # Cycle through frames
            if lat > DEADLINE_MS:
                ax.scatter(exec_time, 24, color='red', s=45, marker='x', 
                          zorder=5, alpha=0.8, linewidth=1.8)
                miss_count += 1
    
    # Add deadline indicator line
    ax.axhline(y=28, color='red', linestyle=':', linewidth=1.5, alpha=0.6, 
              label=f'Deadline ({DEADLINE_MS:.0f}ms)')
    
    # Add frame boundaries
    for k in range(1, 7):
        ax.axvline(FRAME_MS * k, color=C_LIGHT_GRAY, linestyle="--", linewidth=0.8, zorder=0)
    
    # Add annotations
    ax.annotate("", xy=(FRAME_MS * 4, 7), xytext=(0, 7),
                arrowprops=dict(arrowstyle="<->", color=C_DARK_GRAY, lw=1.0))
    ax.text(FRAME_MS * 2, 5.5, "K=4 meta-step", color=C_DARK_GRAY,
            fontsize=FS_ANNOT, ha="center", va="top")
    
    ax.annotate("", xy=(FRAME_MS * 5, 7), xytext=(FRAME_MS * 4, 7),
                arrowprops=dict(arrowstyle="<->", color=C_DARK_GRAY, lw=1.0))
    ax.text(FRAME_MS * 4.5, 5.5, "K=1", color=C_DARK_GRAY,
            fontsize=FS_ANNOT, ha="center", va="top")
    
    # Add legend for deadline misses
    if miss_count > 0:
        ax.scatter([], [], color='red', marker='x', s=45, linewidth=1.8,
                  label=f'Deadline Misses (showing {min(miss_count, 50)})')
    
    ax.set_ylim(3, 40)
    ax.set_xlim(0, TIMELINE_LIMIT)
    ax.set_xlabel("Time (ms)", fontsize=FS_LABEL)
    ax.set_yticks([14, 24])
    ax.set_yticklabels(["GPU 0\n(Env + $\\pi_{reflex}$)", "GPU 1\n(MCTS)"], fontsize=FS_TICK)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_title("Async Timeline (9 FPS)\n× = Deadline Miss", fontsize=TITLE_FS)
    
    # Add legend if we have misses
    if miss_count > 0:
        ax.legend(loc='upper right', fontsize=FS_LEGEND-2, frameon=True, 
                 fancybox=False, edgecolor=C_LIGHT_GRAY)
    
    _apply_spine_style(ax)
    
    # Add miss rate annotation
    if latency_data is not None and len(latency_data) > 0:
        total_misses = np.sum(latency_data > DEADLINE_MS)
        miss_rate = 100.0 * total_misses / len(latency_data)
        ax.text(0.98, 0.02, f'Miss Rate: {miss_rate:.1f}%', 
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=FS_ANNOT, style='italic',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def _plot_latency_violins(ax, grouped_lats):
    """Plot latency distributions as violin plots."""
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
    
    if not datasets:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
               ha='center', va='center', fontsize=FS_TITLE)
        return
    
    parts = ax.violinplot(datasets, positions=positions, widths=0.18, 
                         showmeans=True, showextrema=False)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_alpha(0.55)
    parts["cmeans"].set_color(C_BLACK)
    
    # Add deadline line
    ax.axhline(DEADLINE_MS, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
              label=f'Deadline ({DEADLINE_MS:.0f}ms)')
    
    ax.set_xticks(centers)
    ax.set_xticklabels([GAME_LABELS[g] for g in GAMES], fontsize=FS_TICK)
    ax.set_ylabel("MCTS Latency (ms)", fontsize=FS_LABEL)
    ax.set_ylim(0, 550)
    ax.set_title("Latency Distributions\n(all FPS combined)", fontsize=TITLE_FS)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.legend(loc='upper left', fontsize=FS_LEGEND-2, frameon=False)
    _apply_spine_style(ax)


def _plot_miss_rate_bars(ax, rows):
    """Plot deadline miss rate with error bars."""
    means = []
    ses = []
    for gpu in GPUS:
        per_env_means = []
        for game in GAMES:
            subset = [r["miss_pct"] for r in rows if r["game"] == game and r["gpu"] == gpu]
            if subset:
                per_env_means.append(np.mean(np.asarray(subset, float)))
        if per_env_means:
            arr = np.asarray(per_env_means, float)
            means.append(np.mean(arr))
            ses.append(np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0)
        else:
            means.append(0)
            ses.append(0)
    
    x = np.arange(len(GPUS))
    bars = ax.bar(x, means, color=[GPU_COLORS[g] for g in GPUS], 
                  alpha=BAR_ALPHA, linewidth=0, edgecolor='none')
    ax.errorbar(x, means, yerr=ses, fmt='none', ecolor=C_BLACK,
                elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{mean:.1f}%', ha='center', va='bottom', fontsize=FS_ANNOT)
    
    ax.set_xticks(x)
    ax.set_xticklabels([GPU_LABELS[g] for g in GPUS], fontsize=FS_TICK)
    ax.set_yscale('log')
    ax.set_ylabel("Deadline Miss %", fontsize=FS_LABEL)
    ax.set_title("Average Miss Rate\n(across envs +/- SE)", fontsize=TITLE_FS)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(True, alpha=0.3, axis='y')
    _apply_spine_style(ax)


def _plot_transfer_bars(ax, rows):
    """Plot simulation to real transfer performance."""
    x = np.arange(len(GAMES))
    width = 0.19
    
    # Simulation baseline
    sim_means = [100.0 for _ in GAMES]
    sim_ses = [100.0 * (SIM_RESULTS[g]["se"] / SIM_RESULTS[g]["mean"]) for g in GAMES]
    ax.bar(x - 1.5 * width, sim_means, width=width, color=C_MID_GRAY, 
           alpha=BAR_ALPHA, linewidth=0, label="Simulation")
    ax.errorbar(x - 1.5 * width, sim_means, yerr=sim_ses, fmt='none', ecolor=C_BLACK,
                elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)
    
    # Real deployment results
    for idx, gpu in enumerate(GPUS):
        means = []
        ses = []
        for game in GAMES:
            # Find results for this game, gpu at 9 FPS
            real_results = [r for r in rows if r["game"] == game and 
                           r["gpu"] == gpu and r["fps"] == 9]
            if real_results:
                real_mean = real_results[0]["ret_mean"]
                real_se = real_results[0]["ret_se"]
                sim_mean = SIM_RESULTS[game]["mean"]
                sim_se = SIM_RESULTS[game]["se"]
                means.append(100.0 * real_mean / sim_mean)
                ses.append(100.0 * np.sqrt((real_se / sim_mean) ** 2 + 
                                          ((real_mean * sim_se) / (sim_mean ** 2)) ** 2))
            else:
                means.append(0)
                ses.append(0)
        
        xpos = x + (-0.5 + idx) * width
        bars = ax.bar(xpos, means, width=width, color=GPU_COLORS[gpu], 
                     alpha=BAR_ALPHA, linewidth=0, label=GPU_LABELS[gpu])
        ax.errorbar(xpos, means, yerr=ses, fmt='none', ecolor=C_BLACK,
                   elinewidth=ERR_LW, capsize=CAPSIZE, capthick=ERR_LW)
    
    ax.set_xticks(x)
    ax.set_xticklabels([GAME_LABELS[g] for g in GAMES], fontsize=FS_TICK)
    ax.set_ylabel("Return Relative to Sim (%)", fontsize=SMALL_LABEL_FS)
    ax.set_title("Sim vs Real Return\n(normalized to sim = 100%)", fontsize=TITLE_FS)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.axhline(100.0, color=C_DARK_GRAY, linestyle="--", linewidth=1.0, zorder=0)
    ax.set_ylim(0, 130)
    ax.legend(loc='upper left', fontsize=FS_LEGEND-2, ncol=2, frameon=False)
    _apply_spine_style(ax)


def _plot_cumulative_misses(ax, latencies, budget_ms, gpu_name="H100"):
    """
    Plot cumulative distribution of latencies with deadline marker.
    Shows where deadline misses occur in the distribution.
    """
    if latencies is None or len(latencies) == 0:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
               ha='center', va='center', fontsize=FS_TITLE)
        return
    
    sorted_lats = np.sort(latencies)
    cum_prob = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
    
    # Plot CDF
    ax.plot(sorted_lats, cum_prob * 100, 'b-', linewidth=2.5, label='Latency CDF')
    
    # Add deadline line
    ax.axvline(budget_ms, color='red', linestyle='--', linewidth=2, alpha=0.8,
              label=f'Budget ({budget_ms:.0f}ms)')
    
    # Highlight miss region
    miss_indices = sorted_lats > budget_ms
    if np.any(miss_indices):
        ax.fill_between(sorted_lats[miss_indices], 
                        0, cum_prob[miss_indices] * 100,
                        color='red', alpha=0.3, label='Miss Region')
        
        # Add annotation for miss rate
        miss_rate = np.mean(latencies > budget_ms) * 100
        ax.annotate(f'Miss Rate: {miss_rate:.1f}%', 
                   xy=(budget_ms, 50), xytext=(budget_ms + 50, 60),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1),
                   fontsize=FS_ANNOT, color='red', weight='bold')
    
    # Add percentiles
    percentiles = [50, 90, 95, 99]
    for p in percentiles:
        p_val = np.percentile(latencies, p)
        if p_val < ax.get_xlim()[1]:
            ax.plot(p_val, p, 'ko', markersize=4)
            ax.annotate(f'P{p}', xy=(p_val, p), xytext=(p_val + 10, p + 2),
                       fontsize=FS_ANNOT-1, alpha=0.7)
    
    ax.set_xlabel('MCTS Latency (ms)', fontsize=FS_LABEL)
    ax.set_ylabel('Cumulative Probability (%)', fontsize=FS_LABEL)
    ax.set_title(f'Deadline Miss Distribution\n{GPU_LABELS[gpu_name]} @ 9 FPS', 
                fontsize=TITLE_FS)
    ax.legend(loc='lower right', fontsize=FS_LEGEND-2, frameon=False)
    ax.grid(True, alpha=0.3, axis='both')
    ax.set_xlim(0, min(800, np.percentile(latencies, 99.5)))
    ax.set_ylim(0, 105)
    _apply_spine_style(ax)


def _plot_miss_heatmap(ax, summary_rows):
    """
    Optional: Plot deadline miss rate heatmap across FPS and GPUs.
    This can replace one of the panels if needed.
    """
    # Create matrix of miss rates
    miss_matrix = []
    for gpu in GPUS:
        gpu_misses = []
        for fps in FPS_LIST:
            miss_rates = []
            for game in GAMES:
                matches = [r["miss_pct"] for r in summary_rows 
                          if r["game"] == game and r["gpu"] == gpu and r["fps"] == fps]
                if matches:
                    miss_rates.append(matches[0])
            if miss_rates:
                gpu_misses.append(np.mean(miss_rates))
            else:
                gpu_misses.append(0)
        miss_matrix.append(gpu_misses)
    
    im = ax.imshow(miss_matrix, aspect='auto', cmap='YlOrRd', 
                   vmin=0, vmax=max(20, np.max(miss_matrix)), origin='upper')
    
    ax.set_xticks(range(len(FPS_LIST)))
    ax.set_xticklabels(FPS_LIST, fontsize=FS_TICK)
    ax.set_yticks(range(len(GPUS)))
    ax.set_yticklabels([GPU_LABELS[g] for g in GPUS], fontsize=FS_TICK)
    ax.set_xlabel('Frame Rate (FPS)', fontsize=FS_LABEL)
    ax.set_ylabel('GPU', fontsize=FS_LABEL)
    ax.set_title('Deadline Miss Rate (%)\nAcross FPS and GPUs', fontsize=TITLE_FS)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add text annotations
    for i in range(len(GPUS)):
        for j in range(len(FPS_LIST)):
            text = ax.text(j, i, f'{miss_matrix[i][j]:.1f}',
                          ha="center", va="center", color="white", 
                          fontsize=FS_ANNOT-1, weight='bold')
    
    _apply_spine_style(ax)


# ===========================================================================
# Main
# ===========================================================================

def plot_deployment():
    """Generate complete deployment dashboard with deadline miss visualization."""
    # Create figure with 5 panels
    fig, axes = plt.subplots(
        1, 5, figsize=(22.0, 4.4),
        gridspec_kw={"width_ratios": [1.3, 1.1, 0.9, 1.1, 1.0]},
    )
    
    # Load data
    print("Loading deployment results...")
    summary_rows = _load_summary_rows()
    grouped_lats = _grouped_latency_samples()
    
    # Load specific latency data for timeline and CDF (Tetris on H100 as example)
    tetris_h100_path = _new_path("tetris", "h100", 9)
    tetris_h100_lats, tetris_h100_data = _load_result(tetris_h100_path)
    if tetris_h100_lats is None:
        # Try legacy path
        tetris_h100_lats, tetris_h100_data = _load_result(_LEGACY_PATHS["H100"])
    
    # Plot all panels
    print("Plotting timeline with miss indicators...")
    _plot_timeline_with_misses(axes[0], tetris_h100_lats)
    
    print("Plotting latency distributions...")
    _plot_latency_violins(axes[1], grouped_lats)
    
    print("Plotting miss rate bars...")
    _plot_miss_rate_bars(axes[2], summary_rows)
    
    print("Plotting transfer performance...")
    _plot_transfer_bars(axes[3], summary_rows)
    
    print("Plotting cumulative miss distribution...")
    _plot_cumulative_misses(axes[4], tetris_h100_lats, DEADLINE_MS, "h100")
    
    # Create main legend for the figure
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=C_MID_GRAY, alpha=BAR_ALPHA, label='Simulation'),
        Rectangle((0, 0), 1, 1, facecolor=GPU_COLORS["h100"], alpha=BAR_ALPHA, label='H100'),
        Rectangle((0, 0), 1, 1, facecolor=GPU_COLORS["a100"], alpha=BAR_ALPHA, label='A100'),
        Rectangle((0, 0), 1, 1, facecolor=GPU_COLORS["a40"], alpha=BAR_ALPHA, label='A40'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Deadline'),
        plt.Line2D([0], [0], color='red', marker='x', linestyle='none', 
                   markersize=8, linewidth=1.8, label='Deadline Miss'),
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=6,
        frameon=False,
        fontsize=SMALL_LEGEND_FS,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    
    # Layout and save
    fig.tight_layout(pad=1.0, w_pad=1.2, rect=[0, 0.06, 1, 1])
    out = os.path.join(FIGS, "deployment_with_misses.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"\nSaved: {out}")
    
    # Also save as PNG for quick viewing
    png_out = os.path.join(FIGS, "deployment_with_misses.png")
    fig.savefig(png_out, bbox_inches="tight", dpi=150)
    print(f"Saved: {png_out}")
    
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
                p = _new_path(game, gpu, fps)
                if os.path.exists(p):
                    row += "     ✓  "
                elif game == "tetris" and fps == 9 and os.path.exists(_LEGACY_PATHS[gpu.upper()]):
                    row += "    (L) "
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

    all_rows = []

    for game in GAMES:
        for gpu in GPUS:
            for fps in FPS_LIST:
                p = _new_path(game, gpu, fps)
                lats, d = _load_result(p)
                if lats is None and game == "tetris" and fps == 9:
                    lats, d = _load_result(_LEGACY_PATHS[gpu.upper()])
                if lats is None:
                    continue

                budget_ms  = 4 * (1000.0 / fps)
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
        print(SEP2)

    # ── 3. Deadline miss statistics summary ───────────────────────────────────
    if all_rows:
        print(f"\n{SEP}")
        print("DEADLINE MISS STATISTICS SUMMARY")
        print(SEP)
        
        for gpu in GPUS:
            print(f"\n{GPU_LABELS[gpu]} GPU:")
            print(f"{'FPS':>4} {'Miss Rate':>10} {'P95 Latency':>12} {'Budget':>8} {'Slack':>10}")
            print(SEP2[:45])
            for fps in FPS_LIST:
                fps_misses = [r["miss_pct"] for r in all_rows 
                             if r["gpu"] == gpu and r["fps"] == fps]
                fps_p95 = [r["p95_lat"] for r in all_rows 
                          if r["gpu"] == gpu and r["fps"] == fps]
                if fps_misses and fps_p95:
                    budget = 4 * (1000.0 / fps)
                    slack = budget - np.mean(fps_p95)
                    print(f"{fps:>4} {np.mean(fps_misses):>9.2f}% {np.mean(fps_p95):>11.1f}ms "
                          f"{budget:>7.1f}ms {slack:>9.1f}ms")

    print(f"\n{SEP}")
    print("END OF SUMMARY")
    print(f"{SEP}\n")


if __name__ == "__main__":
    print_summary()
    plot_deployment()