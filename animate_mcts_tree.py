#!/usr/bin/env python3
"""Animated MCTS tree growth for blog release.

Left: a small tree (1 root + 4 action children + 12 leaves) grows over 32
hand-scripted simulations. Each simulation flashes a red descent path,
discovers a leaf, runs a green backprop trace, and grows the visited
root-child's circle in proportion to sqrt(visits).

Right: bar chart of P(a) = N(a) / sum(N) sharpening from uniform toward
the most-visited action.

Saves: figures/mcts_tree.gif
"""

import os
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from plot_config import (
    C_BLACK, C_BLUE, C_NAVY, C_RED, C_LIGHT_GRAY, C_DARK_GRAY,
    FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT,
    K_COLORS,
    apply_style,
)

apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)
OUT_GIF_BASIC = os.path.join(FIGS, "mcts_tree.gif")
OUT_GIF_SCALING = os.path.join(FIGS, "mcts_tree_scaling.gif")

# ── Timing ────────────────────────────────────────────────────────────────────
FPS       = 30
T_INTRO   = 0.6
T_PER_SIM = 0.22
T_OUTRO   = 0.6
N_SIMS    = 32
T_SIMS    = T_PER_SIM * N_SIMS
T_TOTAL   = T_INTRO + T_SIMS + T_OUTRO
N_FRAMES  = int(FPS * T_TOTAL)

# ── Tree layout (data coords; set_aspect("equal") keeps it true) ──────────────
ROOT_X, ROOT_Y = 0.5, 0.92
ACTION_XS = [0.15, 0.38, 0.62, 0.85]
ACTION_Y  = 0.55
LEAF_DXS  = [-0.10, 0.0, 0.10]
LEAF_Y    = 0.18
LEAF_XS   = [[ax + dx for dx in LEAF_DXS] for ax in ACTION_XS]   # 4×3

ACTION_COLORS = K_COLORS                                          # 4 colors
BACKPROP_COLOR = K_COLORS[2]                                      # green

# ── Hand-scripted simulation sequence ────────────────────────────────────────
# (action_idx ∈ 0..3, leaf_idx_under_action ∈ 0..2)
# Convergent: a0 wins decisively; a1 runner-up; a2/a3 rarely visited.
SIM_SEQUENCE = [
    # exploration: spread visits across actions
    (0, 1), (1, 0), (2, 2), (3, 1), (0, 0), (1, 2),
    # mid: a0 and a1 starting to dominate
    (0, 2), (1, 1), (0, 1), (2, 0), (0, 0), (1, 0),
    (0, 2), (3, 2), (0, 1), (1, 2), (0, 0), (0, 2),
    # late: a0 dominates
    (1, 1), (0, 1), (0, 0), (2, 1), (0, 2), (0, 1),
    (1, 0), (0, 0), (0, 2), (0, 1), (1, 2), (0, 0),
    (0, 1), (0, 2),
]
assert len(SIM_SEQUENCE) == N_SIMS


def _ease(t, t0, t1):
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    u = (t - t0) / (t1 - t0)
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return u * u * (3.0 - 2.0 * u)


def build_state_history():
    """history[i] = (visits, discovered_leaves) AFTER i simulations completed."""
    visits = np.zeros(4, dtype=int)
    discovered = set()
    history = [(visits.copy(), set())]
    for (a, l) in SIM_SEQUENCE:
        visits[a] += 1
        discovered.add((a, l))
        history.append((visits.copy(), discovered.copy()))
    return history


# ── Pac-Man scaling data (from paper Fig 3 / plot_scaling.py) ────────────────
# Cached locally so we don't refetch on every render. To refresh, delete the cache.
PACMAN_CACHE = os.path.join(HERE, ".cache", "pacman_scaling.npz")
N_DENSE = 240   # interpolated points along log-x for smooth curve reveal


def load_pacman_scaling():
    """Return (sims_dense, perf_dense, perf_se_dense, lat_dense, lat_se_dense,
    sims_real, perf_real, lat_real) for the Pac-Man scaling panel.
    Fetches from W&B and caches. SE bands match paper Fig 3."""
    REQUIRED_KEYS = {"sims_dense", "perf_dense", "perf_se_dense",
                     "lat_dense", "lat_se_dense",
                     "sims_real", "perf_real", "lat_real"}
    if os.path.exists(PACMAN_CACHE):
        z = np.load(PACMAN_CACHE)
        if REQUIRED_KEYS.issubset(z.files):
            return (z["sims_dense"], z["perf_dense"], z["perf_se_dense"],
                    z["lat_dense"], z["lat_se_dense"],
                    z["sims_real"], z["perf_real"], z["lat_real"])

    print("Fetching Pac-Man scaling data from W&B (one-time cache)...")
    from plot_scaling import (
        _fetch_committed_action, build_series, synthetic_latency, TARGET_SIMS,
    )
    sims, ret, lat = _fetch_committed_action(
        "pacman_kt_cross_eval_mature", k_model=1, k_eval=1,
    )
    perf_real_means, perf_real_ses = build_series(
        sims, ret, TARGET_SIMS, clamp_min=0, se_frac=0.10,
    )
    ok = ~np.isnan(lat)
    if ok.sum() >= 2:
        lat_real_means, lat_real_ses = build_series(
            sims[ok], lat[ok], TARGET_SIMS,
            clamp_min=0, se_frac=0.15, extrap_se_mult=1.6,
        )
    else:
        lat_real_means, lat_real_ses = synthetic_latency(TARGET_SIMS, 0.028)

    log_real = np.log2(TARGET_SIMS)
    log_dense = np.linspace(log_real[0], log_real[-1], N_DENSE)
    sims_dense = 2.0 ** log_dense
    perf_dense    = np.interp(log_dense, log_real, perf_real_means)
    perf_se_dense = np.interp(log_dense, log_real, perf_real_ses)
    lat_dense     = np.interp(log_dense, log_real, lat_real_means)
    lat_se_dense  = np.interp(log_dense, log_real, lat_real_ses)

    os.makedirs(os.path.dirname(PACMAN_CACHE), exist_ok=True)
    np.savez(
        PACMAN_CACHE,
        sims_dense=sims_dense, perf_dense=perf_dense, perf_se_dense=perf_se_dense,
        lat_dense=lat_dense, lat_se_dense=lat_se_dense,
        sims_real=TARGET_SIMS, perf_real=perf_real_means, lat_real=lat_real_means,
    )
    print(f"  cached -> {PACMAN_CACHE}")
    return (sims_dense, perf_dense, perf_se_dense, lat_dense, lat_se_dense,
            TARGET_SIMS, perf_real_means, lat_real_means)


def _leaf_value(action, leaf):
    """Plausible per-leaf value tag, deterministic. Range 0.30..0.85."""
    return 0.30 + 0.55 * (((action * 3 + leaf) * 7919) % 100) / 100.0


def build_figure(with_scaling=True):
    if with_scaling:
        fig = plt.figure(figsize=(15.5, 5.2))
        gs = gridspec.GridSpec(
            1, 3, width_ratios=[1.8, 0.9, 1.15],
            wspace=0.45,
            left=0.03, right=0.94, top=0.91, bottom=0.14,
        )
        ax_tree = fig.add_subplot(gs[0])
        ax_bars = fig.add_subplot(gs[1])
        ax_scale = fig.add_subplot(gs[2])
    else:
        fig = plt.figure(figsize=(11.5, 5.2))
        gs = gridspec.GridSpec(
            1, 2, width_ratios=[1.8, 0.9],
            wspace=0.35,
            left=0.03, right=0.97, top=0.91, bottom=0.12,
        )
        ax_tree = fig.add_subplot(gs[0])
        ax_bars = fig.add_subplot(gs[1])
        ax_scale = None

    # Tight xlim matched to the tree axis display aspect so the tree fills the panel.
    ax_tree.set_xlim(-0.15, 1.15)
    ax_tree.set_ylim(0.04, 1.04)
    ax_tree.set_aspect("equal")
    ax_tree.axis("off")
    ax_tree.set_title("MCTS: tree growth via simulations", fontsize=FS_TITLE, pad=6)

    # Baseline edges (grey, always visible)
    edges = {}
    for a in range(4):
        ln = Line2D([ROOT_X, ACTION_XS[a]], [ROOT_Y, ACTION_Y],
                    color=C_LIGHT_GRAY, lw=1.1, zorder=1)
        ax_tree.add_line(ln)
        edges[("root", a)] = ln
    for a in range(4):
        for l in range(3):
            ln = Line2D([ACTION_XS[a], LEAF_XS[a][l]], [ACTION_Y, LEAF_Y],
                        color=C_LIGHT_GRAY, lw=1.1, zorder=1)
            ax_tree.add_line(ln)
            edges[(a, l)] = ln

    # Transient overlay edges for descent / backprop highlight
    ov_top = Line2D([0, 0], [0, 0], color=C_RED, lw=3.0,
                    solid_capstyle="round", zorder=4)
    ov_bot = Line2D([0, 0], [0, 0], color=C_RED, lw=3.0,
                    solid_capstyle="round", zorder=4)
    ov_top.set_alpha(0.0)
    ov_bot.set_alpha(0.0)
    ax_tree.add_line(ov_top)
    ax_tree.add_line(ov_bot)

    # Nodes — single scatter for per-point RGBA / size control.
    node_x = [ROOT_X] + ACTION_XS + [LEAF_XS[a][l] for a in range(4) for l in range(3)]
    node_y = [ROOT_Y] + [ACTION_Y] * 4 + [LEAF_Y] * 12
    base_colors = (
        [C_NAVY]
        + list(ACTION_COLORS)
        + [ACTION_COLORS[a] for a in range(4) for _ in range(3)]
    )
    sizes = np.array([900.0] + [200.0] * 4 + [80.0] * 12, dtype=float)
    alphas = np.array([1.0] + [0.75] * 4 + [0.0] * 12, dtype=float)
    rgba = np.array([list(mcolors.to_rgba(c)) for c in base_colors])
    rgba[:, 3] = alphas

    nodes = ax_tree.scatter(
        node_x, node_y,
        s=sizes, facecolors=rgba,
        edgecolors=C_BLACK, linewidths=0.7,
        zorder=5,
    )

    # Root and action-child text labels
    ax_tree.text(ROOT_X, ROOT_Y + 0.05, "root  $s_t$",
                 ha="center", va="bottom",
                 fontsize=FS_ANNOT, color=C_DARK_GRAY)
    for a in range(4):
        ax_tree.text(ACTION_XS[a], ACTION_Y + 0.05, f"$a_{a}$",
                     ha="center", va="bottom",
                     fontsize=FS_ANNOT, color=C_DARK_GRAY)

    # Live value-tag (repositioned each sim, centered under the leaf)
    v_label = ax_tree.text(0, 0, "", ha="center", va="top",
                            fontsize=FS_ANNOT, color=C_DARK_GRAY,
                            style="italic", alpha=0.0, zorder=6)

    # Simulation counter
    counter = ax_tree.text(0.5, 0.0, "Simulation 0 / 32",
                            ha="center", va="bottom",
                            fontsize=FS_LABEL, color=C_DARK_GRAY)

    # Bars
    bars = ax_bars.bar(
        range(4), [0, 0, 0, 0], color=ACTION_COLORS,
        edgecolor=C_BLACK, linewidth=0.7, width=0.72,
    )
    ax_bars.set_ylim(0, 1.05)
    ax_bars.set_xticks(range(4))
    ax_bars.set_xticklabels([f"$a_{i}$" for i in range(4)], fontsize=FS_TICK)
    ax_bars.set_ylabel(r"$P(a) \propto N(s,a)$", fontsize=FS_LABEL)
    ax_bars.set_title("Recommended action", fontsize=FS_TITLE, pad=6)
    ax_bars.tick_params(axis="y", labelsize=FS_TICK)
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)

    if ax_scale is None:
        return (fig, nodes, sizes, rgba, ov_top, ov_bot, v_label, counter, bars,
                None, None, None, None, None, None,           # perf_line..lat_dots
                None, None, None, None, None,                  # sims_dense..lat_se_dense
                None, None, None)                              # sims_real, *_rgba

    # Scaling subplot: Pac-Man planning quality (blue solid) and H100 latency
    # (red dashed) — same panel as paper Fig 3, animated to reveal progressively.
    (sims_dense, perf_dense, perf_se_dense, lat_dense, lat_se_dense,
     sims_real, perf_real, lat_real) = load_pacman_scaling()
    # Align both y-axes so each curve traverses bottom-left to top-right, filling
    # the panel. (Otherwise return starts at ~2400/4400 = 55% height while
    # latency starts near 0, leaving a big visual gap.)
    perf_lo = float(perf_dense.min())
    perf_hi = float(perf_dense.max())
    perf_pad = 0.08 * (perf_hi - perf_lo)
    lat_lo  = float(lat_dense.min())
    lat_hi  = float(lat_dense.max())
    lat_pad  = 0.08 * (lat_hi - lat_lo)
    ymin_perf = max(0.0, perf_lo - perf_pad)
    ymax_perf = perf_hi + perf_pad
    ymin_lat  = max(0.0, lat_lo - lat_pad)
    ymax_lat  = lat_hi + lat_pad

    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

    ax_scale.set_xscale("log", base=2)
    ax_scale.set_xlim(sims_real[0], sims_real[-1])
    ax_scale.set_ylim(ymin_perf, ymax_perf)
    ax_scale.set_xlabel("simulations  $\\rightarrow$", fontsize=FS_TICK,
                        color=C_DARK_GRAY)
    ax_scale.set_ylabel("quality  $\\rightarrow$", color=C_BLUE, fontsize=FS_TICK)
    ax_scale.spines["top"].set_visible(False)
    ax_scale.set_title("Quality vs. latency", fontsize=FS_TITLE, pad=4)

    ax_lat = ax_scale.twinx()  # twinx shares x; do NOT re-set xscale here
    ax_lat.set_ylim(ymin_lat, ymax_lat)
    ax_lat.set_ylabel("latency  $\\rightarrow$", color=C_RED, fontsize=FS_TICK)
    ax_lat.spines["top"].set_visible(False)

    # Strip ticks + tick labels on all three axes for an abstract look.
    for a in (ax_scale, ax_lat):
        a.set_xticks([])
        a.set_yticks([])
        a.xaxis.set_minor_locator(NullLocator())
        a.yaxis.set_minor_locator(NullLocator())

    from matplotlib.patches import Polygon

    LINE_WIDTH = 3.2
    BAND_ALPHA = 0.20

    # SE bands (Polygon patches whose vertices we'll update each frame).
    perf_band = Polygon(np.array([[sims_real[0], ymin_perf]]),
                        closed=True, facecolor=C_BLUE, alpha=BAND_ALPHA,
                        edgecolor="none", zorder=2)
    lat_band  = Polygon(np.array([[sims_real[0], ymin_lat]]),
                        closed=True, facecolor=C_RED,  alpha=BAND_ALPHA,
                        edgecolor="none", zorder=2)
    ax_scale.add_patch(perf_band)
    ax_lat.add_patch(lat_band)

    perf_line, = ax_scale.plot([], [], color=C_BLUE, lw=LINE_WIDTH, zorder=4)
    lat_line,  = ax_lat.plot([], [],   color=C_RED,  lw=LINE_WIDTH,
                              linestyle="--", zorder=4)

    # Measured-point markers — alpha-controlled per frame to "appear" when revealed.
    perf_dots = ax_scale.scatter(
        sims_real, perf_real, color=C_BLUE, s=36,
        edgecolor="white", linewidths=0.6, zorder=6,
    )
    lat_dots = ax_lat.scatter(
        sims_real, lat_real, color=C_RED, s=32, marker="s",
        edgecolor="white", linewidths=0.6, zorder=6,
    )
    # Initialize all dots hidden (alpha=0); update() will reveal them as the x-sweep passes.
    perf_dots_rgba = np.tile(np.array([*mcolors.to_rgba(C_BLUE)]), (len(sims_real), 1))
    lat_dots_rgba  = np.tile(np.array([*mcolors.to_rgba(C_RED)]),  (len(sims_real), 1))
    perf_dots_rgba[:, 3] = 0.0
    lat_dots_rgba[:, 3]  = 0.0
    perf_dots.set_facecolors(perf_dots_rgba)
    lat_dots.set_facecolors(lat_dots_rgba)

    return (fig, nodes, sizes, rgba, ov_top, ov_bot, v_label, counter, bars,
            perf_line, lat_line, perf_band, lat_band, perf_dots, lat_dots,
            sims_dense, perf_dense, perf_se_dense, lat_dense, lat_se_dense,
            sims_real, perf_dots_rgba, lat_dots_rgba)


def update(frame_idx, ctx):
    (nodes, sizes, rgba, ov_top, ov_bot, v_label, counter, bars,
     perf_line, lat_line, perf_band, lat_band, perf_dots, lat_dots,
     sims_dense, perf_dense, perf_se_dense, lat_dense, lat_se_dense, sims_real,
     perf_dots_rgba, lat_dots_rgba,
     state_history) = ctx

    t = frame_idx / FPS
    if t < T_INTRO:
        sim_done = 0
        cur_sim = None
        phase = 0.0
    elif t < T_INTRO + T_SIMS:
        elapsed = t - T_INTRO
        sim_idx = min(int(elapsed / T_PER_SIM), N_SIMS - 1)
        phase = (elapsed - sim_idx * T_PER_SIM) / T_PER_SIM
        cur_sim = sim_idx
        sim_done = sim_idx if phase < 1.0 else sim_idx + 1
    else:
        sim_done = N_SIMS
        cur_sim = None
        phase = 0.0

    visits, discovered = state_history[sim_done]

    # Persistent leaf alphas (discovered leaves fade in once)
    for a in range(4):
        for l in range(3):
            idx = 1 + 4 + (a * 3 + l)
            rgba[idx, 3] = 0.85 if (a, l) in discovered else 0.0

    # Persistent action-child sizes — grow with sqrt(visits)
    for a in range(4):
        sizes[1 + a] = 200.0 + 110.0 * math.sqrt(visits[a])

    # Transient: current simulation highlight
    if cur_sim is not None:
        action, leaf = SIM_SEQUENCE[cur_sim]
        leaf_node_idx = 1 + 4 + (action * 3 + leaf)

        # Fade leaf in during this sim (so the "discovery" is visible)
        if (action, leaf) not in discovered:
            rgba[leaf_node_idx, 3] = max(
                rgba[leaf_node_idx, 3],
                _ease(phase, 0.22, 0.42) * 0.85,
            )

        # Overlay colour: red while descending (phase < 0.5), green on backprop
        col_rgba = list(mcolors.to_rgba(C_RED if phase < 0.55 else BACKPROP_COLOR))

        a_top = _ease(phase, 0.00, 0.18) - _ease(phase, 0.85, 1.00)
        a_bot = _ease(phase, 0.12, 0.30) - _ease(phase, 0.85, 1.00)
        a_top = max(0.0, min(1.0, a_top))
        a_bot = max(0.0, min(1.0, a_bot))

        ov_top.set_data([ROOT_X, ACTION_XS[action]], [ROOT_Y, ACTION_Y])
        ov_bot.set_data([ACTION_XS[action], LEAF_XS[action][leaf]],
                        [ACTION_Y, LEAF_Y])
        ov_top.set_color(col_rgba)
        ov_bot.set_color(col_rgba)
        ov_top.set_alpha(a_top)
        ov_bot.set_alpha(a_bot)

        # Floating value tag below the leaf row
        v_alpha = _ease(phase, 0.28, 0.45) - _ease(phase, 0.78, 0.92)
        v_alpha = max(0.0, min(1.0, v_alpha))
        v_label.set_position((LEAF_XS[action][leaf], LEAF_Y - 0.07))
        v_label.set_text(f"V = {_leaf_value(action, leaf):.2f}")
        v_label.set_alpha(v_alpha)
    else:
        ov_top.set_alpha(0.0)
        ov_bot.set_alpha(0.0)
        v_label.set_alpha(0.0)

    nodes.set_sizes(sizes)
    nodes.set_facecolors(rgba)
    counter.set_text(f"Simulation {sim_done} / {N_SIMS}")

    total = int(visits.sum())
    p = (visits / total) if total > 0 else np.zeros(4)
    for bar, h in zip(bars, p):
        bar.set_height(h)

    # Pac-Man scaling: reveal curves up to a sweep position that progresses linearly
    # in log-x from sims_real[0] to sims_real[-1] across the animation's sim phase.
    if perf_line is None:
        return []   # 2-panel variant — no scaling subplot to update

    if t < T_INTRO:
        reveal_frac = 0.0
    elif t < T_INTRO + T_SIMS:
        reveal_frac = (t - T_INTRO) / T_SIMS
    else:
        reveal_frac = 1.0
    n_visible = int(round(reveal_frac * len(sims_dense)))
    n_visible = max(0, min(len(sims_dense), n_visible))
    perf_line.set_data(sims_dense[:n_visible], perf_dense[:n_visible])
    lat_line.set_data(sims_dense[:n_visible], lat_dense[:n_visible])

    # Grow the SE bands as Polygon vertices: top edge (mean+se) left→right,
    # then bottom edge (mean-se) right→left to close the polygon.
    if n_visible >= 2:
        x_seg = sims_dense[:n_visible]
        p_hi = perf_dense[:n_visible] + perf_se_dense[:n_visible]
        p_lo = perf_dense[:n_visible] - perf_se_dense[:n_visible]
        perf_band.set_xy(np.vstack([
            np.column_stack([x_seg, p_hi]),
            np.column_stack([x_seg[::-1], p_lo[::-1]]),
        ]))
        l_hi = lat_dense[:n_visible] + lat_se_dense[:n_visible]
        l_lo = lat_dense[:n_visible] - lat_se_dense[:n_visible]
        lat_band.set_xy(np.vstack([
            np.column_stack([x_seg, l_hi]),
            np.column_stack([x_seg[::-1], l_lo[::-1]]),
        ]))
    else:
        perf_band.set_xy(np.array([[sims_dense[0], perf_dense[0]]]))
        lat_band.set_xy(np.array([[sims_dense[0], lat_dense[0]]]))

    # Measured dots fade in once the sweep passes their x position.
    if n_visible > 0:
        x_sweep = sims_dense[n_visible - 1]
    else:
        x_sweep = sims_real[0]
    for i, sx in enumerate(sims_real):
        a = 1.0 if sx <= x_sweep + 1e-6 else 0.0
        perf_dots_rgba[i, 3] = a
        lat_dots_rgba[i, 3]  = a
    perf_dots.set_facecolors(perf_dots_rgba)
    lat_dots.set_facecolors(lat_dots_rgba)

    return []


def _render(with_scaling, out_path):
    state_history = build_state_history()
    (fig, nodes, sizes, rgba, ov_top, ov_bot, v_label, counter, bars,
     perf_line, lat_line, perf_band, lat_band, perf_dots, lat_dots,
     sims_dense, perf_dense, perf_se_dense, lat_dense, lat_se_dense,
     sims_real, perf_dots_rgba, lat_dots_rgba) = build_figure(with_scaling=with_scaling)
    ctx = (nodes, sizes, rgba, ov_top, ov_bot, v_label, counter, bars,
           perf_line, lat_line, perf_band, lat_band, perf_dots, lat_dots,
           sims_dense, perf_dense, perf_se_dense, lat_dense, lat_se_dense,
           sims_real, perf_dots_rgba, lat_dots_rgba,
           state_history)
    anim = FuncAnimation(
        fig, update, frames=N_FRAMES,
        fargs=(ctx,),
        interval=1000.0 / FPS, blit=False,
    )
    print(f"Rendering {N_FRAMES} frames at {FPS} fps -> {out_path}")
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    _render(with_scaling=False, out_path=OUT_GIF_BASIC)
    _render(with_scaling=True,  out_path=OUT_GIF_SCALING)


if __name__ == "__main__":
    main()
