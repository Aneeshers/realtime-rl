"""Shared visual style for all paper figures.

Import this module in every plot script:

    from plot_config import <names>
    apply_style()          # call after matplotlib backend is set
"""

import os
from matplotlib import font_manager as _fm

_HERE = os.path.dirname(os.path.abspath(__file__))
_fm.fontManager.addfont(os.path.join(_HERE, "BerkeleyMonoTrial-Regular.otf"))

# ── Font ─────────────────────────────────────────────────────────────────────
FONT_FAMILY = "Berkeley Mono Trial"

# ── Font sizes ────────────────────────────────────────────────────────────────
FS_TITLE  = 22   # subplot / panel titles
FS_LABEL  = 18   # axis labels
FS_TICK   = 16   # tick labels
FS_LEGEND = 16   # legend entries
FS_BADGE  = 20   # K=N badges and prominent inline labels
FS_ANNOT  = 14   # small italic annotations

# ── Colors ────────────────────────────────────────────────────────────────────
C_BLUE   = "#2c6fad"   # primary blue  — performance curves, bars
C_RED    = "#C94040"   # primary red   — latency, gating, MCTS
C_NAVY   = "#1a3a6b"   # dark navy     — K-value badges
C_STATE  = "#d0e4f7"   # light blue    — state boxes in timeline diagrams
C_COMMIT = "#bbbbbb"   # neutral gray  — committed-action boxes

# Neutrals
C_BLACK      = "black"
C_WHITE      = "white"
C_DARK_GRAY  = "#333333"
C_MID_GRAY   = "#888888"
C_LIGHT_GRAY = "#cccccc"

# Ordered palette for K=1 … K=4 strategy lines
K_COLORS = ["#2171b5", "#e6550d", "#31a354", "#756bb1"]

# ── Plot element defaults ─────────────────────────────────────────────────────
FILL_ALPHA  = 0.18   # fill_between variance bands
BAR_ALPHA   = 0.85   # bar face opacity
LINE_LW     = 2.0    # primary line weight
MARKER_SIZE = 7      # scatter / line marker size
CAPSIZE     = 3      # error-bar cap size (pts)
ERR_LW      = 0.9    # error-bar line weight


def apply_style() -> None:
    """Apply shared matplotlib/seaborn rcParams.

    Call this once per script, after matplotlib.use() and pyplot import.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="white", font_scale=1.0)
    plt.rcParams.update({
        "font.family":       FONT_FAMILY,
        "font.weight":       "normal",
        "axes.titleweight":  "normal",
        "axes.labelweight":  "normal",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
    })
