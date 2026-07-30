"""Shared visual language for every figure in the paper.

The palette is not chosen by eye. Three separate encoding jobs appear in this
paper and each gets the palette built for that job:

* **Series identity** (static vs. regime vs. baselines) is categorical. The four
  hues below clear the all-pairs colour-vision-deficiency separation gate
  (worst pair dE 9.2 deutan) and the normal-vision floor (worst pair dE 16.3),
  so no two lines collapse for a colourblind reader.
* **Regime level** is *ordinal* — calm < turbulent < crisis — so it uses a
  single-hue ramp with monotone lightness rather than three arbitrary hues. That
  survives greyscale printing, which matters for a conference proceedings, and
  is inherently CVD-safe because the channel carrying meaning is lightness.
* **Transition probability** is continuous magnitude, so it uses the same hue as
  a sequential ramp.

Regime state is drawn as a dedicated ribbon beneath each time-series panel
rather than as a background wash. A wash light enough to keep the return lines
readable cannot separate three levels from each other *and* from the page; on
its own strip the ramp can run at full strength and be directly labelled.
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Categorical series colours (validated all-pairs, light surface) ---------
SERIES_COLORS = {
    "static": "#2a78d6",
    "regime": "#eb6834",
    "baseline": "#1baf7a",
    "control": "#4a3aa7",
}
SERIES_CYCLE = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")

# --- Ordinal regime ramp (single hue, monotone lightness) -------------------
REGIME_COLORS = {
    "CALM": "#86b6ef",
    "TURBULENT": "#2a78d6",
    "CRISIS": "#104281",
}
REGIME_ORDER = ("CALM", "TURBULENT", "CRISIS")

# --- Sequential ramp for magnitude (transition-probability heatmap) ---------
SEQUENTIAL_HUE = "Blues"

# --- Chrome and ink ---------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FIGURE_DPI = 300
"""Print resolution for camera-ready submission."""


def figure_style() -> dict:
    """Matplotlib rcParams implementing the conventions described above."""
    return {
        "figure.facecolor": SURFACE,
        "figure.dpi": 120,
        "savefig.dpi": FIGURE_DPI,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=list(SERIES_CYCLE)),
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,  # thin marks; 2px is the spec
        "lines.markersize": 4.0,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK_SECONDARY,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9.5,
        # Type 42 (TrueType) fonts are required by the ACM camera-ready checker;
        # the default Type 3 subsetting fails submission validation.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


@contextmanager
def paper_style():
    """Apply the paper's figure style for the duration of a plotting block."""
    with plt.rc_context(figure_style()):
        yield


def as_percent(axis, decimals: int = 0) -> None:
    """Format an axis as percentages — returns and drawdowns are always percentages."""
    axis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=decimals))


def add_caption(ax, text: str, offset_points: float = -34.0) -> None:
    """Place an explanatory note below `ax`, anchored to the axes.

    Anchoring to the axes rather than to figure coordinates matters: a caption
    pinned at figure x=0 fixes the left edge of the tight bounding box there, and
    any tick label extending further left is then cropped out of the saved file.
    """
    ax.annotate(
        text,
        xy=(0.0, 0.0),
        xycoords="axes fraction",
        xytext=(0.0, offset_points),
        textcoords="offset points",
        fontsize=7.5,
        color=INK_MUTED,
        ha="left",
        va="top",
        annotation_clip=False,
    )


def pad_right(ax, fraction: float = 0.16) -> None:
    """Widen the x-axis so end-of-line labels have room inside the figure.

    Without this the direct labels run off the right edge and are clipped when
    the figure is placed in a two-column layout.
    """
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + (right - left) * fraction)


def label_line_end(ax, x, y, text: str, color: str) -> None:
    """Direct-label a series at its right end.

    Direct labelling is what lets the reader identify a line without hopping to a
    legend, and it is the required relief for the one series colour that sits
    below 3:1 contrast against the page.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(4, 0),
        textcoords="offset points",
        color=color,
        fontsize=8.5,
        fontweight="bold",
        va="center",
        ha="left",
        annotation_clip=False,
    )
