"""Shared matplotlib styling for the benchmark charts.

Colours are the validated reference categorical palette (slots 1-3; the
validator passes CVD/normal-vision separation on the light surface). Aqua sits
below 3:1 contrast, so every bar is direct-labelled with its value and every
chart has a table in the README: colour never carries the information alone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua


def setup(width: float = 8.0, height: float = 4.2):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 26,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "legend.frameon": False,
        }
    )
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    return fig, ax


def label_bars(ax, bars, fmt: str = "{:.3f}", pad: float = 3) -> None:
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, pad),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=TEXT,
        )


def grouped_bars(ax, groups: list[str], series: dict[str, list[float]], fmt: str = "{:.3f}"):
    """Grouped bars: one colour per series in fixed order, a small gap between bars, direct labels."""
    n = len(series)
    width = 0.8 / n
    for i, (name, values) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(xs, values, width * 0.92, label=name, color=SERIES[i % len(SERIES)], zorder=3)
        label_bars(ax, bars, fmt)
    ax.set_xticks(range(len(groups)), groups)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=n, fontsize=9)


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
