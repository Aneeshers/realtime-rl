#!/usr/bin/env python3
"""Animated two-GPU deployment-timeline GIF for blog release.

Left panel: schematic of the two-GPU async loop (GPU 0 = env+reflex, GPU 1 = MCTS)
            with state/action packets flying between the boards each meta-step.
Right panel: Gantt-style execution timeline with a vertical playhead sweeping
             across the K=4 + K=1 + start-of-K=2 sequence at ~9.4x slowdown.

Both panels are tightly coupled: the left circuit "reacts" to events on the
right timeline.

Saves: figures/deployment_timeline.gif
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

from plot_config import (
    C_BLACK, C_DARK_GRAY, C_LIGHT_GRAY,
    apply_style,
)

# Reuse the static drawing helpers + constants from plot_deployment.py so the
# look exactly matches paper Fig 7.
from plot_deployment import (
    _plot_timeline,
    _plot_timeline_schematic,
    FRAME_MS,
    TIMELINE_LIMIT,
    GPU1_BLOCKS,   # GPU 0 reflex-frame ticks (2.5 ms each at k*FRAME_MS)
    GPU2_BLOCKS,   # GPU 1 MCTS computation intervals (start_ms, duration_ms)
    GPU1_COLOR,    # blue
    GPU2_COLOR,    # purple
)

apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)
OUT_GIF = os.path.join(FIGS, "deployment_timeline.gif")

# ── Timing ────────────────────────────────────────────────────────────────────
FPS_ANIM    = 30
T_HEAD_HOLD = 0.3
T_SWEEP     = 6.4
T_TAIL_HOLD = 0.3
T_TOTAL     = T_HEAD_HOLD + T_SWEEP + T_TAIL_HOLD
N_FRAMES    = int(FPS_ANIM * T_TOTAL)

# ── Event times (ms world time) ───────────────────────────────────────────────
GPU0_TICK_STARTS = [s for (s, _) in GPU1_BLOCKS]                 # reflex frames
MCTS_INTERVALS   = [(s, s + d) for (s, d) in GPU2_BLOCKS]
MCTS_STARTS      = [s for (s, _e) in MCTS_INTERVALS]
MCTS_ENDS        = [e for (_s, e) in MCTS_INTERVALS]

PACKET_DURATION_MS = 90.0
GLOW_SIGMA_MS = 30.0
GPU0_BLINK_SIGMA_MS = 22.0
GLOW_PEAK_ALPHA = 0.55

# ── Left-panel circuit coordinates (axes-fraction; matches
# _plot_timeline_schematic in plot_deployment.py) ─────────────────────────────
TOP_ARROW_Y    = 0.53
BOT_ARROW_Y    = 0.37
ARROW_X_LEFT   = 0.41
ARROW_X_RIGHT  = 0.59
GPU0_CENTER    = (0.22, 0.45)   # GPU 0 board centre (board x=0.04, w=0.36)
GPU1_CENTER    = (0.78, 0.45)   # GPU 1 board centre (board x=0.60, w=0.36)
GLOW_W         = 0.46
GLOW_H         = 0.50

GLOW_COLOR = "#FFD27A"   # warm halo
PACKET_RADIUS = 0.022


def _gpu0_glow_alpha(t_world):
    """Soft peaks around each reflex-frame tick."""
    a = 0.0
    for s in GPU0_TICK_STARTS:
        a = max(a, np.exp(-((t_world - s) / GPU0_BLINK_SIGMA_MS) ** 2))
    return GLOW_PEAK_ALPHA * a


def _gpu1_glow_alpha(t_world):
    """Full alpha while inside an MCTS interval, exponential decay after end."""
    a = 0.0
    for s, e in MCTS_INTERVALS:
        if s <= t_world <= e:
            a = 1.0
            break
        elif t_world > e:
            a = max(a, np.exp(-((t_world - e) / GLOW_SIGMA_MS) ** 2))
        elif t_world < s:
            # tiny ramp-in just before the MCTS block starts
            a = max(a, 0.5 * np.exp(-((t_world - s) / (0.5 * GLOW_SIGMA_MS)) ** 2))
    return GLOW_PEAK_ALPHA * a


def _packet_state(t_world, fire_times, x_from, x_to, y, duration=PACKET_DURATION_MS):
    """Return (x, y, alpha) for a packet that fires at any t in fire_times."""
    for ft in fire_times:
        if ft <= t_world <= ft + duration:
            f = (t_world - ft) / duration
            # ease in then ease out on alpha for a soft flash
            alpha = np.sin(np.pi * f) ** 0.6
            x = x_from + f * (x_to - x_from)
            return x, y, float(alpha)
    return x_from, y, 0.0


def _t_world_at(t_anim):
    if t_anim < T_HEAD_HOLD:
        return 0.0
    if t_anim < T_HEAD_HOLD + T_SWEEP:
        return ((t_anim - T_HEAD_HOLD) / T_SWEEP) * TIMELINE_LIMIT
    return TIMELINE_LIMIT


def build_figure():
    fig, axes = plt.subplots(
        1, 2, figsize=(14.0, 4.0),
        gridspec_kw={"width_ratios": [1.0, 1.7]},
    )
    ax_left, ax_right = axes

    # Static layouts via the existing helpers.
    _plot_timeline_schematic(ax_left)
    _plot_timeline(ax_right, show_inset=False, title="Execution Timeline (9 FPS)")

    # ── Left: GPU "glow" halos (behind the boards, zorder=0) ──
    gpu0_glow = Rectangle(
        (GPU0_CENTER[0] - GLOW_W / 2, GPU0_CENTER[1] - GLOW_H / 2),
        GLOW_W, GLOW_H,
        facecolor=GLOW_COLOR, edgecolor="none",
        alpha=0.0, zorder=0,
        transform=ax_left.transAxes, clip_on=False,
    )
    gpu1_glow = Rectangle(
        (GPU1_CENTER[0] - GLOW_W / 2, GPU1_CENTER[1] - GLOW_H / 2),
        GLOW_W, GLOW_H,
        facecolor=GLOW_COLOR, edgecolor="none",
        alpha=0.0, zorder=0,
        transform=ax_left.transAxes, clip_on=False,
    )
    ax_left.add_patch(gpu0_glow)
    ax_left.add_patch(gpu1_glow)

    # ── Left: packet dots (top of zorder so they fly over everything) ──
    state_packet = Circle(
        (ARROW_X_LEFT, TOP_ARROW_Y), PACKET_RADIUS,
        facecolor=GPU2_COLOR, edgecolor=C_BLACK, linewidth=0.6,
        alpha=0.0, zorder=10,
        transform=ax_left.transAxes, clip_on=False,
    )
    action_packet = Circle(
        (ARROW_X_RIGHT, BOT_ARROW_Y), PACKET_RADIUS,
        facecolor=GPU1_COLOR, edgecolor=C_BLACK, linewidth=0.6,
        alpha=0.0, zorder=10,
        transform=ax_left.transAxes, clip_on=False,
    )
    ax_left.add_patch(state_packet)
    ax_left.add_patch(action_packet)

    # ── Right: vertical playhead ──
    playhead = Line2D(
        [0, 0], [-2, 38],
        color=C_DARK_GRAY, lw=1.8, alpha=0.85, zorder=8,
    )
    ax_right.add_line(playhead)
    # Subtle "now" tick at the top
    playhead_dot = Circle(
        (0, 36), 6.0,
        facecolor=C_DARK_GRAY, edgecolor="white", linewidth=1.0,
        zorder=9,
    )
    ax_right.add_patch(playhead_dot)

    return fig, ax_left, ax_right, gpu0_glow, gpu1_glow, state_packet, action_packet, playhead, playhead_dot


def update(frame_idx, ctx):
    (gpu0_glow, gpu1_glow, state_packet, action_packet,
     playhead, playhead_dot) = ctx

    t_anim  = frame_idx / FPS_ANIM
    t_world = _t_world_at(t_anim)

    # Playhead
    playhead.set_xdata([t_world, t_world])
    playhead_dot.center = (t_world, 36)

    # GPU glows
    gpu0_glow.set_alpha(_gpu0_glow_alpha(t_world))
    gpu1_glow.set_alpha(_gpu1_glow_alpha(t_world))

    # State packet (GPU 0 → GPU 1, top arrow)
    sx, sy, sa = _packet_state(
        t_world, MCTS_STARTS,
        x_from=ARROW_X_LEFT, x_to=ARROW_X_RIGHT, y=TOP_ARROW_Y,
    )
    state_packet.center = (sx, sy)
    state_packet.set_alpha(sa)

    # Action packet (GPU 1 → GPU 0, bottom arrow)
    ax_, ay, aa = _packet_state(
        t_world, MCTS_ENDS,
        x_from=ARROW_X_RIGHT, x_to=ARROW_X_LEFT, y=BOT_ARROW_Y,
    )
    action_packet.center = (ax_, ay)
    action_packet.set_alpha(aa)

    return []


def main():
    (fig, ax_left, ax_right,
     gpu0_glow, gpu1_glow, state_packet, action_packet,
     playhead, playhead_dot) = build_figure()
    fig.tight_layout(pad=0.9, w_pad=2.5)

    ctx = (gpu0_glow, gpu1_glow, state_packet, action_packet,
           playhead, playhead_dot)

    anim = FuncAnimation(
        fig, update, frames=N_FRAMES,
        fargs=(ctx,),
        interval=1000.0 / FPS_ANIM, blit=False,
    )
    print(f"Rendering {N_FRAMES} frames at {FPS_ANIM} fps -> {OUT_GIF}")
    anim.save(OUT_GIF, writer=PillowWriter(fps=FPS_ANIM), dpi=140)
    plt.close(fig)
    print(f"Saved: {OUT_GIF}")


if __name__ == "__main__":
    main()
