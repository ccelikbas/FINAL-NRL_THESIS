"""nlr_bars.py — the slide bar-chart style, shared by every presentation figure.

One drawing routine, so the mission-performance and ablation slides come out
visually identical: same bar geometry, same 95%-CI error bars, same NLR colours,
same legend. Only the numbers and the series differ, and those come from the
analysis dumps in eval_results/analysis_data/ (see eval_tools/analysis_data.py) —
nothing here computes or re-runs anything.

Layout (matching the slide mockups): the RATE KPIs share ONE panel as grouped
bars on a 0–100% axis, and each differently-scaled KPI (mission duration, in
simulation steps) gets its OWN panel beside it. Panels are width-weighted by
their group count, so bars are the same width everywhere in the figure.

    Panel(percent=True,  groups=[Group("Targets Destroyed", bars), Group("Survival Rate", bars)])
    Panel(percent=False, groups=[Group("Mission Duration",  bars)])

Every Bar carries its own 95% confidence interval, drawn as a cap on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .nlr_style import (
    NLR_DARKBLUE, NLR_TERRA, NLR_LIGHTBLUE_50, NLR_TERRA_20, NLR_LIGHTBLUE,
    NLR_TERRA_50, NLR_DARKGRAY, NLR_GRAY,
)

# ---------------------------------------------------------------------
#  Series colours — NLR house palette, in a FIXED order.
# ---------------------------------------------------------------------
# Series 1/2 are the headline pair (complete vs baseline): NLR dark blue vs NLR
# terra. Further series are the lighter tints of the same two hues, so an
# ablation reads as "blue family = has communication, orange family = does not"
# while staying inside the house style. The order is fixed, never cycled — the
# n-th policy always gets the n-th colour, so colour means the same thing on
# every slide.
SERIES_COLORS = [NLR_DARKBLUE, NLR_TERRA, NLR_LIGHTBLUE_50, NLR_TERRA_20,
                 NLR_LIGHTBLUE, NLR_TERRA_50]

# ---------------------------------------------------------------------
#  Geometry / typography (tuned for a 16:9 slide)
# ---------------------------------------------------------------------
BAR_SPAN = 0.80           # fraction of a group slot filled by its bars
BAR_PAD = 0.10            # gap between bars inside a group, as a fraction of one bar
GROUP_PAD = 0.62          # blank space left and right of the outermost group

ERR_COLOR = NLR_DARKGRAY  # 95% CI cap colour
ERR_LW = 1.5
ERR_CAPSIZE = 5

# The figure is sized in INCHES while the type is in POINTS, so these two blocks
# together set the apparent text size. Keep the canvas generous (a slide figure
# is viewed large) or 12pt labels start to dominate the bars and collide.
PANEL_H = 4.8             # figure height (inches), before the legend strip
GROUP_W = 2.5             # width (inches) contributed by each group
PANEL_GAP_W = 1.0         # extra width per panel, for its own y-axis

GROUP_LABEL_FS = 12       # the bold KPI names under the bars
TICK_FS = 11
YLABEL_FS = 11
LEGEND_FS = 11.5
VALUE_FS = 10.5
LEGEND_MIN_COL_W = 2.1    # inches a legend entry needs; below this the legend wraps

RATE_TOP = 1.0            # rate panels always show a full 0–100% axis
HEADROOM = 1.09           # blank space above the tallest (bar + CI)
VALUE_HEADROOM = 1.17     # …more when the value labels are printed on top


@dataclass
class Bar:
    """One bar: its mean and the 95% CI drawn as a cap on top.

    `lo`/`hi` may be NaN (no interval — then no cap is drawn); a NaN `mean`
    means the series is absent here and the bar is skipped entirely."""
    mean: float
    lo: float = float("nan")
    hi: float = float("nan")

    @property
    def drawable(self) -> bool:
        return bool(np.isfinite(self.mean))

    @property
    def yerr(self) -> Optional[np.ndarray]:
        if not (self.drawable and np.isfinite(self.lo) and np.isfinite(self.hi)):
            return None
        return np.array([[max(self.mean - self.lo, 0.0)],
                         [max(self.hi - self.mean, 0.0)]])

    @property
    def top(self) -> float:
        """Highest drawn y for this bar (the CI cap, or the bar itself)."""
        if not self.drawable:
            return 0.0
        e = self.yerr
        return self.mean + (float(e[1, 0]) if e is not None else 0.0)


@dataclass
class Group:
    """One x-axis category (a KPI): one bar per series, in series order."""
    label: str
    bars: List[Bar]


@dataclass
class Panel:
    """One sub-plot: the groups sharing a y-axis (hence the same unit).

    `percent` renders a 0–100% axis for rate KPIs; otherwise the axis is scaled
    to the data. Never put differently-scaled KPIs in one panel — that is what
    the extra panel is for."""
    groups: List[Group]
    percent: bool = False
    ylabel: str = ""
    value_fmt: str = "{:.1f}"
    ymax: Optional[float] = None
    grid_step: Optional[float] = field(default=None)


def _bar_positions(n_series: int):
    """(x offsets, bar width) within one group, centred on the group's slot.

    The bars of a group fill BAR_SPAN of its slot with a BAR_PAD gap between
    them, so a 2-series and a 4-series figure keep the same group footprint."""
    if n_series <= 0:
        return np.zeros(0), 0.0
    w = BAR_SPAN / (n_series + BAR_PAD * (n_series - 1))
    step = w * (1.0 + BAR_PAD)
    first = -0.5 * (step * (n_series - 1))
    return first + step * np.arange(n_series), w


def _panel_ymax(panel: Panel, value_labels: bool) -> float:
    if panel.ymax is not None:
        return panel.ymax
    head = VALUE_HEADROOM if value_labels else HEADROOM
    if panel.percent:
        return RATE_TOP * head
    top = max((b.top for g in panel.groups for b in g.bars), default=1.0)
    return (top * head) or 1.0


def _draw_panel(ax, panel: Panel, colors: Sequence[str], value_labels: bool) -> None:
    n_series = max((len(g.bars) for g in panel.groups), default=0)
    offsets, width = _bar_positions(n_series)
    ymax = _panel_ymax(panel, value_labels)

    for gi, group in enumerate(panel.groups):
        for si, bar in enumerate(group.bars):
            if not bar.drawable:
                continue                       # series absent here → no bar
            x = gi + offsets[si]
            ax.bar(x, bar.mean, width=width, color=colors[si % len(colors)],
                   zorder=2)
            err = bar.yerr
            if err is not None:
                ax.errorbar(x, bar.mean, yerr=err, fmt="none", ecolor=ERR_COLOR,
                            elinewidth=ERR_LW, capsize=ERR_CAPSIZE,
                            capthick=ERR_LW, zorder=3)
            if value_labels:
                label = (f"{bar.mean * 100:.0f}%" if panel.percent
                         else panel.value_fmt.format(bar.mean))
                ax.text(x, bar.top + 0.02 * ymax, label, ha="center", va="bottom",
                        fontsize=VALUE_FS, color=NLR_DARKGRAY)

    ax.set_ylim(0.0, ymax)
    if panel.percent:
        step = panel.grid_step or 0.2
        ticks = np.arange(0.0, RATE_TOP + 1e-9, step)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int(round(v * 100))}%" for v in ticks])
    elif panel.grid_step:
        ax.set_yticks(np.arange(0.0, ymax, panel.grid_step))

    ax.set_xticks(np.arange(len(panel.groups)))
    ax.set_xticklabels([g.label for g in panel.groups],
                       fontsize=GROUP_LABEL_FS, fontweight="bold")
    ax.set_xlim(-GROUP_PAD, len(panel.groups) - 1 + GROUP_PAD)
    if panel.ylabel:
        ax.set_ylabel(panel.ylabel, fontsize=YLABEL_FS, color=NLR_GRAY)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.grid(True, axis="y", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw(panels: Sequence[Panel], series_labels: Sequence[str], out_png: Path,
         dpi: int = 600, colors: Optional[Sequence[str]] = None,
         value_labels: bool = False, title: Optional[str] = None,
         transparent: bool = False, legend: bool = True,
         height: Optional[float] = None) -> Path:
    """Render one slide figure and save it as a high-resolution PNG.

    `panels` are laid out left to right, width-weighted by their group count so
    every bar in the figure has the same width. `series_labels` name the series
    in bar order and drive the legend (identity must never be colour-alone —
    drop the legend only for a figure that sits beside one carrying it).
    `height` overrides the figure height, for a compact single-KPI tile."""
    colors = list(colors or SERIES_COLORS)
    widths = [max(len(p.groups), 1) * GROUP_W + PANEL_GAP_W for p in panels]
    fig_h = height or PANEL_H

    fig, axes = plt.subplots(
        1, len(panels), figsize=(sum(widths), fig_h), squeeze=False,
        gridspec_kw={"width_ratios": widths})
    for ax, panel in zip(axes[0], panels):
        _draw_panel(ax, panel, colors, value_labels)

    legend_rows = 0
    if legend and series_labels:
        handles = [Patch(facecolor=colors[i % len(colors)], label=lab)
                   for i, lab in enumerate(series_labels)]
        # A narrow figure (a single-KPI panel) cannot fit four labels on one
        # line, so the legend wraps instead of running off the canvas.
        ncol = len(handles)
        while ncol > 1 and sum(widths) / ncol < LEGEND_MIN_COL_W:
            ncol -= 1
        legend_rows = int(np.ceil(len(handles) / ncol))
        fig.legend(handles=handles, loc="lower center", ncol=ncol,
                   fontsize=LEGEND_FS, frameon=False, bbox_to_anchor=(0.5, 0.008),
                   handlelength=1.1, handletextpad=0.5, columnspacing=1.8)

    # Just enough room for the legend strip — a slide figure should not ship
    # with whitespace the user has to crop off. One legend row is about 0.26 in
    # of type, so reserve that as a fraction of THIS figure's height rather than
    # a fixed fraction (which over-reserves on a short figure).
    bottom = ((0.26 * legend_rows + 0.06) / fig_h) if legend_rows else 0.02
    top = 0.90 if title else 1.0
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color=NLR_DARKGRAY)
    fig.tight_layout(rect=(0.0, bottom, 1.0, top))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # Stable filenames: a re-run REPLACES the figure in place, so the copy sitting
    # in the presentation folder is always the current one — no dated variants to
    # pick between when building the deck.
    existed = out_png.exists()
    fig.savefig(out_png, dpi=dpi, transparent=transparent,
                facecolor="none" if transparent else fig.get_facecolor())
    plt.close(fig)
    print(f"{'Overwrote' if existed else 'Wrote'} {out_png.name}  "
          f"({int(sum(widths) * dpi)}×{int(fig_h * dpi)} px @ {dpi} dpi)"
          f"  ->  {out_png}")
    return out_png
