#!/usr/bin/env python3
"""Animated Pacman ghost-approach with adaptive gate P(K) bar chart.

Four ghosts converge on the player, then retreat to the ghost house. As the
nearest-ghost distance shrinks, the gate's P(K) shifts from K=4-dominant
(plan deeply) to K=1-dominant (react). Approach-then-retreat makes the loop
seamless. P(K) uses a Gaussian profile centered on the paper's empirical
nearest-ghost distances per K (Section "Pac-Man" interpretability).

Saves: figures/pacman_gate.gif
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

from plot_config import (
    C_BLACK, C_WHITE, C_NAVY, C_DARK_GRAY,
    FS_TITLE, FS_LABEL, FS_TICK, FS_BADGE, FS_ANNOT,
    K_COLORS,
    apply_style,
)

apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)
OUT_GIF = os.path.join(FIGS, "pacman_gate.gif")

FPS = 30
FIG_W = 11.0
FIG_H = 5.0

# Empirical nearest-ghost-distance per chosen K (from paper Fig 5).
CENTERS = {1: 6.83, 2: 8.40, 4: 11.59}
SIGMA = 1.5
K_LIST = [1, 2, 3, 4]

N_KEYFRAMES = 9                                # 0 = far, 8 = close

# Ghost positions [col, row] in Jumanji's PacMan convention. The reset state
# leaves ghosts at the bottom of the ghost house (min L1 ≈ 8 — already in K=2
# territory), so we hand-pick a FAR set above the ghost house that gives min
# L1 ≈ 11 (matching the paper's empirical K=4 center of 11.59).
FAR_GHOSTS = np.array(
    [[13, 11], [11, 11], [15, 11], [13, 12]],
    dtype=np.int32,
)
CLOSE_GHOSTS = np.array(
    [[11, 23], [15, 23], [11, 20], [15, 20]],
    dtype=np.int32,
)

# Timing
T_INTRO       = 0.6
T_PER_KF      = 0.3
T_CLOSE_HOLD  = 0.6
T_OUTRO       = 0.4
T_APPROACH    = T_PER_KF * N_KEYFRAMES
T_RETREAT     = T_PER_KF * N_KEYFRAMES
T_TOTAL       = T_INTRO + T_APPROACH + T_CLOSE_HOLD + T_RETREAT + T_OUTRO
N_FRAMES      = int(FPS * T_TOTAL)


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def p_k(d):
    logits = []
    for k in K_LIST:
        if k == 3:
            logits.append(-50.0)
        else:
            logits.append(-((d - CENTERS[k]) / SIGMA) ** 2)
    return _softmax(logits)


def _min_l1(ghosts, player_x, player_y):
    """L1 distance from each ghost [col, row] to player (row, col); returns min."""
    return min(
        abs(player_x - g[1]) + abs(player_y - g[0]) for g in ghosts
    )


def build_keyframes():
    import jax
    import jax.numpy as jnp
    from jumanji.environments.routing.pac_man import PacMan
    from jumanji.environments.routing.pac_man.viewer import create_grid_image

    env = PacMan()
    key = jax.random.PRNGKey(0)
    base_state, _ = jax.jit(env.reset)(key)

    player_x = int(base_state.player_locations.x)
    player_y = int(base_state.player_locations.y)
    print(f"Player at row={player_x}, col={player_y}")
    print(f"Far ghosts (chosen):  {FAR_GHOSTS.tolist()}  -> min L1 = {_min_l1(FAR_GHOSTS, player_x, player_y)}")
    print(f"Close ghosts (chosen): {CLOSE_GHOSTS.tolist()} -> min L1 = {_min_l1(CLOSE_GHOSTS, player_x, player_y)}")

    rgb_frames = []
    distances = []
    for kf in range(N_KEYFRAMES):
        alpha = kf / (N_KEYFRAMES - 1)
        # rounded ghost positions for the sprite render
        gh_render = np.round((1 - alpha) * FAR_GHOSTS + alpha * CLOSE_GHOSTS).astype(np.int32)
        state = base_state.replace(ghost_locations=jnp.array(gh_render, dtype=jnp.int32))
        rgb_frames.append(np.array(create_grid_image(state)))
        # unrounded continuous-position distance for P(K) (avoids the round-snap hiccups)
        gh_cont = (1 - alpha) * FAR_GHOSTS + alpha * CLOSE_GHOSTS
        ds = [abs(player_x - g[1]) + abs(player_y - g[0]) for g in gh_cont]
        distances.append(float(min(ds)))
    print(f"Per-keyframe nearest-ghost distance (continuous): {[f'{d:.1f}' for d in distances]}")
    return rgb_frames, distances


def kf_at_time(t):
    if t < T_INTRO:
        return 0
    t -= T_INTRO
    if t < T_APPROACH:
        return min(int(t / T_PER_KF), N_KEYFRAMES - 1)
    t -= T_APPROACH
    if t < T_CLOSE_HOLD:
        return N_KEYFRAMES - 1
    t -= T_CLOSE_HOLD
    if t < T_RETREAT:
        rev = min(int(t / T_PER_KF), N_KEYFRAMES - 1)
        return (N_KEYFRAMES - 1) - rev
    return 0


def d_at_time(t, distances):
    d_far, d_close = distances[0], distances[-1]
    if t < T_INTRO:
        return d_far
    t -= T_INTRO
    if t < T_APPROACH:
        u = t / T_APPROACH
        return d_far + u * (d_close - d_far)
    t -= T_APPROACH
    if t < T_CLOSE_HOLD:
        return d_close
    t -= T_CLOSE_HOLD
    if t < T_RETREAT:
        u = t / T_RETREAT
        return d_close + u * (d_far - d_close)
    return d_far


def build_figure(rgb0, p0, d0):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = gridspec.GridSpec(
        1, 2, width_ratios=[1.0, 0.85],
        wspace=0.18, left=0.04, right=0.97,
        top=0.92, bottom=0.16,
    )
    ax_maze = fig.add_subplot(gs[0])
    ax_bars = fig.add_subplot(gs[1])

    ax_maze.set_axis_off()
    ax_maze.set_title("Pac-Man: ghosts approach", fontsize=FS_TITLE, pad=6)
    im = ax_maze.imshow(rgb0, interpolation="nearest")

    badge = ax_maze.text(
        0.96, 0.96, "K=4",
        transform=ax_maze.transAxes,
        ha="right", va="top",
        fontsize=FS_BADGE + 2, color=C_NAVY,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor=C_WHITE, edgecolor=C_NAVY, lw=1.8,
        ),
        zorder=10,
    )

    bars = ax_bars.bar(
        K_LIST, p0, color=K_COLORS,
        edgecolor=C_BLACK, linewidth=0.8, width=0.72,
    )
    ax_bars.set_ylim(0, 1.05)
    ax_bars.set_xticks(K_LIST)
    ax_bars.set_xticklabels([f"K={k}" for k in K_LIST], fontsize=FS_TICK)
    ax_bars.set_ylabel("gate probability  P(K)", fontsize=FS_LABEL)
    ax_bars.set_title("Adaptive planning budget", fontsize=FS_TITLE, pad=6)
    ax_bars.tick_params(axis="y", labelsize=FS_TICK)
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)

    dist_txt = ax_bars.text(
        0.5, -0.22, f"nearest-ghost distance = {d0:.1f}",
        transform=ax_bars.transAxes,
        ha="center", va="top",
        fontsize=FS_ANNOT, color=C_DARK_GRAY, style="italic",
    )

    return fig, im, badge, bars, dist_txt


def update(frame_idx, rgb_frames, distances, im, badge, bars, dist_txt):
    t = frame_idx / FPS
    kf = kf_at_time(t)
    d = d_at_time(t, distances)
    p = p_k(d)

    im.set_array(rgb_frames[kf])
    for bar, h in zip(bars, p):
        bar.set_height(h)
    badge.set_text(f"K={K_LIST[int(np.argmax(p))]}")
    dist_txt.set_text(f"nearest-ghost distance = {d:.1f}")

    return [im, badge, dist_txt, *bars]


def main():
    print("Building keyframes...")
    rgb_frames, distances = build_keyframes()
    print(f"  {len(rgb_frames)} keyframes built; loop length {T_TOTAL:.1f}s, {N_FRAMES} frames")

    p0 = p_k(distances[0])
    fig, im, badge, bars, dist_txt = build_figure(rgb_frames[0], p0, distances[0])

    anim = FuncAnimation(
        fig, update,
        frames=N_FRAMES,
        fargs=(rgb_frames, distances, im, badge, bars, dist_txt),
        interval=1000.0 / FPS, blit=False,
    )
    print(f"Rendering -> {OUT_GIF}")
    anim.save(OUT_GIF, writer=PillowWriter(fps=FPS), dpi=140)
    plt.close(fig)
    print(f"Saved: {OUT_GIF}")


if __name__ == "__main__":
    main()
