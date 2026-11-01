"""One visual identity for every figure in the repo (README, research notes and LinkedIn slides).

Colors come from a CVD-validated categorical palette and are assigned in fixed order:
the same entity keeps the same color in every chart. Text is always ink, never a series color.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots, fixed order (blue, orange, aqua, yellow, magenta, green, violet, red).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES

# Diverging blue <-> gray <-> red for signed heatmaps (e.g. Sharpe across a parameter grid).
DIVERGING = ["#9e2e2e", "#e34948", "#f0a8a7", "#f0efec", "#9ec5f4", "#2a78d6", "#104281"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.edgecolor": BASELINE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def subtitle(ax: plt.Axes, text: str) -> None:
    """Secondary line under the title: what the reader should take away."""
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4),
                textcoords="offset points", ha="left", va="bottom", fontsize=10.5, color=INK_2)


def source(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, -0.02, text, fontsize=8.5, color=MUTED, ha="left", va="top")


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
