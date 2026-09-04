"""nlr_lines.py — the slide LINE-chart style, companion to nlr_bars.py.

For a KPI measured across an ordered sweep (here: team size — 2, 3 and 4
jammers), where the interesting thing is the TREND rather than the individual
values. One routine, so every sensitivity figure in the deck shares its
geometry, colours and confidence-interval treatment.

Each series is drawn as a line through its measured points, with the 95%
confidence interval shown TWICE over:
  · a light filled band between the interval's bounds, which carries the trend,
  · a capped error bar at each measured point, which keeps it honest that the
    measurements are at DISCRETE team sizes and the line between them is
    interpolation, not data.

NLR house colours, and the same series order as the bar figures (dark blue =
the complete model, terra = the baseline), so a viewer reads the two chart types
the same way.

y-AXIS NOTE: unlike a bar chart, these axes do NOT force a zero baseline — a
line chart encodes its values by POSITION, not by bar length, and zero-anchoring
a trend that lives in a narrow band flattens it into a straight line. `zero_base`
turns the zero baseline back on where a KPI genuinely wants one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .nlr_bars import SERIES_COLORS
from .nlr_style import NLR_DARKGRAY, NLR_GRAY

# ---------------------------------------------------------------------
#  Per-series encodings (colour + line style + marker, in a FIXED order)
# ---------------------------------------------------------------------
# Colour already separates the series; the line style and marker repeat that
# information, so the figure survives greyscale printing and colour-blind
# viewing without relying on hue alone.
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]

# ---------------------------------------------------------------------
#  Geometry / typography (one panel; two of these sit side by side)
# ---------------------------------------------------------------------
FIG_W = 6.0               # inches — a pair fits a 16:9 slide side by side
FIG_H = 4.6

LINE_WIDTH = 2.4
MARKER_SIZE = 8
MARKER_EDGE = 1.6         # white keyline, so markers stay separate where lines cross
BAND_ALPHA = 0.16         # the filled CI band
CAP_SIZE = 5
CAP_LW = 1.4
CAP_ALPHA = 0.85

TITLE_FS = 13
AXIS_LABEL_FS = 12
TICK_FS = 11
LEGEND_FS = 11

Y_PAD = 0.12              # blank fraction of the data range above and below


@dataclass
class Point:
    """One measured x with its mean and 95% CI (lo/hi may be NaN)."""
    x: float
    mean: float
    lo: float = float("nan")
    hi: float = float("nan")


@dataclass
class Series:
    """One policy's curve across the sweep."""
    label: str
    points: List[Point] = field(default_factory=list)

    def arrays(self):
        """(x, mean, lo, hi) over the FINITE means, sorted by x."""
        pts = sorted((p for p in self.points if np.isfinite(p.mean)),
                     key=lambda p: p.x)
        if not pts:
            return (np.zeros(0),) * 4
        return (np.array([p.x for p in pts], dtype=float),
                np.array([p.mean for p in pts], dtype=float),
                np.array([p.lo for p in pts], dtype=float),
                np.array([p.hi for p in pts], dtype=float))


def _limits(series: Sequence[Series], zero_base: bool):
    """y-limits covering every band, padded so caps never touch the frame."""
    lo_vals, hi_vals = [], []
    for s in series:
        _x, mean, lo, hi = s.arrays()
        if mean.size == 0:
            continue
        lo_vals.append(np.nanmin(np.where(np.isfinite(lo), lo, mean)))
        hi_vals.append(np.nanmax(np.where(np.isfinite(hi), hi, mean)))
    if not lo_vals:
        return 0.0, 1.0
    lo, hi = float(min(lo_vals)), float(max(hi_vals))
    span = (hi - lo) or (abs(hi) or 1.0)
    lo = 0.0 if zero_base else lo - Y_PAD * span
    return lo, hi + Y_PAD * span


def draw(series: Sequence[Series], out_png: Path, *, xlabel: str, ylabel: str,
         title: Optional[str] = None, dpi: int = 600, zero_base: bool = False,
         percent: bool = False, colors: Optional[Sequence[str]] = None,
         band: bool = True, caps: bool = True, legend: bool = True,
         transparent: bool = False, xticks: Optional[Sequence[float]] = None,
         figsize: Optional[tuple] = None) -> Path:
    """Render one sensitivity line chart and save it as a high-resolution PNG.

    Every series gets its own colour, line style and marker, in a fixed order.
    Stable filenames: a re-run replaces the figure in place."""
    colors = list(colors or SERIES_COLORS)
    fig, ax = plt.subplots(figsize=figsize or (FIG_W, FIG_H))

    for i, s in enumerate(series):
        x, mean, lo, hi = s.arrays()
        if x.size == 0:
            continue
        color = colors[i % len(colors)]
        has_ci = np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
        if band and has_ci:
            ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, lw=0,
                            zorder=1)
        if caps and has_ci:
            ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="none",
                        ecolor=color, elinewidth=CAP_LW, capsize=CAP_SIZE,
                        capthick=CAP_LW, alpha=CAP_ALPHA, zorder=2)
        ax.plot(x, mean, color=color, lw=LINE_WIDTH,
                ls=LINE_STYLES[i % len(LINE_STYLES)],
                marker=MARKERS[i % len(MARKERS)], ms=MARKER_SIZE,
                markeredgecolor="white", markeredgewidth=MARKER_EDGE,
                label=s.label, zorder=3)

    ax.set_ylim(*_limits(series, zero_base))
    if percent:
        ax.set_yticklabels([f"{v * 100:.0f}%" for v in ax.get_yticks()])
    if xticks is not None:
        ax.set_xticks(list(xticks))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS)
    if title:
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=10,
                     color=NLR_DARKGRAY)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend:
        # A long handle so the marker does not break a SOLID line into what
        # looks like a dashed one — the line style is part of the identity.
        ax.legend(fontsize=LEGEND_FS, frameon=False, loc="best", handlelength=3.2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    existed = out_png.exists()
    fig.savefig(out_png, dpi=dpi, transparent=transparent,
                facecolor="none" if transparent else fig.get_facecolor())
    plt.close(fig)
    w, h = figsize or (FIG_W, FIG_H)
    print(f"{'Overwrote' if existed else 'Wrote'} {out_png.name}  "
          f"({int(w * dpi)}×{int(h * dpi)} px @ {dpi} dpi)  ->  {out_png}")
    return out_png
