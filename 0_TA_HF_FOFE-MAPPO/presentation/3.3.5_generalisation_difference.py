"""3.3.5_generalisation_difference.py — team-size generalisation, one slide figure.

THREE difference maps side by side in a single high-resolution PNG:

    presentation/3.3.5_generalisation_difference.png

    [ Targets Destroyed ]   [ Survival Rate ]   [ Mission Duration ]

Each map is the team-size grid — strikers (rows) × jammers (columns) — coloured
by the DIFFERENCE between the two policies in that team composition.

Built for a slide, so three things are done in service of reading it fast:

 1. BLUE ALWAYS MEANS Comm-FOFE-MAPPO IS BETTER. The difference is ORIENTED by
    each KPI's own "better" direction, which the analysis records: for targets
    and survival that is (complete − baseline), but duration is better when
    LOWER, so its map shows (baseline − complete) — steps SAVED. Without this
    flip the duration panel would be blue where the model is slower, and the
    slide would say the opposite of the truth.
 2. THE COLOUR SCALE IS CENTRED ON ZERO, so white is "no difference" and the two
    hues are the two directions. Each panel scales to its own largest absolute
    difference (the units differ — percentage points vs steps), so the colours
    are comparable WITHIN a panel, not across panels; the numbers in the cells
    and each panel's own colour bar carry the magnitude.
 3. CELLS WHERE THE DIFFERENCE IS NOT SIGNIFICANT ARE HATCHED (paired one-sided
    Wilcoxon, p >= 0.05, as computed by the analysis). A pale-blue cell that is
    hatched is not evidence of an advantage, and on a slide that distinction is
    otherwise invisible.

The TRAINING REGION is outlined: both policies were trained only on those team
sizes, so every cell outside the outline is generalisation to a team size never
seen in training.

This script runs NO rollouts and loads NO checkpoints. Its only input is the
dump that eval_tools/3.3.5_team_size_generalisation_analysis.py wrote at the end
of its run:

    eval_results/analysis_data/3.3.5_team_size_generalisation.json

The name is STABLE, so a re-run overwrites the figure in place.

NOTE ON "CONTOURS": the grid is 3×6 DISCRETE team sizes, so the default figure
draws discrete cells. --contour smooths them into filled contour bands, which
looks softer on a slide but interpolates across team sizes that do not exist
(there is no 2.5-striker team); the cell numbers stay in place either way.

Run (repo root, project venv):
  .venv\\Scripts\\python.exe "0_TA_HF_FOFE-MAPPO\\presentation\\3.3.5_generalisation_difference.py"
  …--dpi 900 --contour --transparent
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR.parent
_EVAL_TOOLS_DIR = _PKG_DIR / "eval_tools"
_PKG_NAME = "fofe_mappo"
if __package__ in (None, ""):
    sys.path.insert(0, str(_PKG_DIR.parent))
    if _PKG_NAME not in sys.modules:
        _pkg = types.ModuleType(_PKG_NAME)
        _pkg.__path__ = [str(_PKG_DIR), str(_EVAL_TOOLS_DIR), str(_THIS_DIR)]
        _pkg.__package__ = _PKG_NAME
        _pkg.__file__ = str(_EVAL_TOOLS_DIR / "__init__.py")
        sys.modules[_PKG_NAME] = _pkg
    __package__ = _PKG_NAME

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from .analysis_data import load_analysis
from .nlr_style import (
    NLR_DARKBLUE, NLR_LIGHTBLUE, NLR_LIGHTBLUE_20, NLR_LIGHTBLUE_50, NLR_TERRA,
    NLR_TERRA_20, NLR_TERRA_50, NLR_DARKGRAY, NLR_GRAY,
)

# =====================================================================
#  >>>  INPUT  (which analysis dump to draw)  <<<
# =====================================================================

# Written by eval_tools/3.3.5_team_size_generalisation_analysis.py. [CLI: --data_name]
DATA_NAME = "3.3.5_team_size_generalisation"

# The panels, left to right. `key` must be one of the dump's compared KPIs (only
# those carry per-cell p-values). `unit` picks how a difference is written:
#   "pp"    a rate difference, shown in PERCENTAGE POINTS (0.25 -> +25)
#   "steps" a step-count difference, shown as-is
PANELS = [
    dict(key="targets_destroyed", title="Targets destroyed", unit="rate"),
    dict(key="survival", title="Survival rate", unit="rate"),
    dict(key="duration", title="Mission duration", unit="steps"),
]

# Value / confidence-interval formats per unit. Rates are shown in their own
# 0–1 units (the KPI's own scale), step counts as steps.
VALUE_FMT = {"rate": "{:+.2f}", "steps": "{:+.1f}"}
CI_FMT = {"rate": "±{:.3f}", "steps": "±{:.1f}"}

# Every panel's colour bar says the same thing, because the difference is always
# ORIENTED so that positive favours Comm-FOFE-MAPPO.
CBAR_LABEL = "Δ  (higher = better)"

# The subtitle spells out which subtraction the panel shows. Duration is better
# when LOWER, so its Δ is the other way round — saying so is the difference
# between a reader trusting the figure and misreading it.
SUBTITLE = {
    "higher": "Δ = Comm-FOFE-MAPPO − MAPPO Baseline",
    "lower": "Δ = MAPPO Baseline − Comm-FOFE-MAPPO   (steps saved)",
}

# No headline by default — the panels carry their own KPI titles. The one thing
# a viewer cannot infer, which way the colours point, moves into the small key
# under the figure instead.                               [CLI: --suptitle]
SUPTITLE = ("Team-size generalisation — Comm-FOFE-MAPPO vs MAPPO Baseline "
            "(blue = Comm-FOFE-MAPPO better)")

# Significance threshold for the hatching (the analysis' own per-cell p-values).
ALPHA = 0.05

# ---------------------------------------------------------------------
#  Style
# ---------------------------------------------------------------------
# Diverging: NLR terra (baseline better) → white (no difference) → NLR blue
# (Comm-FOFE-MAPPO better). A diverging scale needs a NEUTRAL midpoint, never a
# third hue, or "no difference" stops being readable.
#
# BALANCE: the two arms carry the same number of stops AND comparable visual
# weight at each step out from white. NLR dark blue is far heavier than plain
# terra, so the orange arm ends on a deep burnt terra rather than the house
# mid-tone — otherwise every blue cell shouts and every orange cell whispers,
# and the eye reads an advantage that the numbers do not support.
NLR_TERRA_DEEP = "#a5510e"    # burnt terra: the orange counterweight to dark blue
NLR_DIV = LinearSegmentedColormap.from_list(
    "nlr_div", [NLR_TERRA_DEEP, NLR_TERRA, NLR_TERRA_50, NLR_TERRA_20,
                "#ffffff",
                NLR_LIGHTBLUE_20, NLR_LIGHTBLUE_50, NLR_LIGHTBLUE, NLR_DARKBLUE])
NLR_DIV.set_bad("#eeeeee")

# Each PANEL is a square box (set_box_aspect(1)); the cells inside it are
# therefore taller than they are wide, which suits a value stacked over its
# confidence interval.
# A SQUARE panel over a 6-wide × 3-tall grid gives cells that are narrow and
# tall, so the panel has to be big enough for a value to fit across one cell:
# at PANEL_SIDE inches a cell is PANEL_SIDE/6 wide, and "-0.16" at CELL_FS must
# sit inside that with room to spare, or neighbouring cells collide.
PANEL_SIDE = 3.1          # inches — the square plotting box
PANEL_PAD_W = 0.9         # inches per panel for its y-axis and the gap between
PANEL_CHROME_H = 1.8      # inches for title, subtitle, x-axis, bar and key
TITLE_FS = 12.5
SUBTITLE_FS = 8.5
AXIS_FS = 10
TICK_FS = 10
CELL_FS = 8               # the difference value
CELL_CI_FS = 6            # the ± confidence interval under it
CBAR_FS = 8.5
KEY_FS = 9.5              # the small key under the figure
SUPTITLE_FS = 13.5

# The training region is called out in NLR terra: it is an annotation about the
# EXPERIMENT, not a data value, so it must not be one of the data hues.
TRAIN_EDGE = NLR_TERRA
TRAIN_LW = 2.0
TRAIN_DIVIDER = True      # dashed line at the edge of the trained team sizes

# Clean-look details: tiles are separated by a hairline of the page background
# rather than butted together, and non-significant tiles are FADED instead of
# hatched — hatching six-plus cells turns the panel into noise, while a veil
# lets them recede and keeps the significant cells the thing you see first.
TILE_GAP_LW = 2.0         # white hairline between tiles
NS_VEIL = 0.62            # opacity of the white veil over a non-significant tile


# Outputs: written straight into the presentation folder under a STABLE name, so
# the figure keeps one recognisable filename and a re-run overwrites it.
OUT_DIR = "presentation"
OUT_NAME = "3.3.5_generalisation_difference"
DPI = 600

# =====================================================================


def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_PKG_DIR / p)


# =====================================================================
#  Reading the analysis dump
# =====================================================================

class SweepData:
    """Thin reader over the 3.3.5 dump. The grids, the per-cell p-values and the
    swept world — all computed by the analysis, nothing recomputed here."""

    def __init__(self, doc: dict):
        self.doc = doc
        self.data = doc["data"]
        self.meta = doc.get("meta", {})
        axes = self.data.get("axes", {})
        self.strikers: List[int] = [int(s) for s in axes.get("strikers", [])]
        self.jammers: List[int] = [int(j) for j in axes.get("jammers", [])]
        self.grids = self.data.get("grids", {})
        self.policies = self.data.get("policies", {})
        # key -> "higher"/"lower": which direction of the KPI is BETTER.
        self.direction: Dict[str, str] = {
            c["key"]: c.get("direction", "higher")
            for c in self.data.get("compare_kpis", [])}
        # (strikers, jammers) -> {kpi: {pvalue, n_pairs, ...}}
        self.pvals: Dict[tuple, dict] = {
            (int(r["n_strikers"]), int(r["n_jammers"])): r
            for r in self.data.get("pvalues", [])}

    def has(self, key: str) -> bool:
        return (key in self.grids.get("complete", {})
                and key in self.grids.get("baseline", {}))

    def oriented_delta(self, key: str) -> np.ndarray:
        """The difference grid, ORIENTED so POSITIVE always favours the complete
        policy: (complete − baseline) for "higher is better" KPIs, and the flip
        of that for "lower is better" ones (duration → steps saved)."""
        c = np.asarray(self.grids["complete"][key], dtype=float)
        b = np.asarray(self.grids["baseline"][key], dtype=float)
        d = c - b
        return -d if self.direction.get(key) == "lower" else d

    def ci_grid(self, key: str) -> np.ndarray:
        """Half-width of the 95% CI of the PAIRED difference, per cell.

        The analysis computes it on the per-episode (complete − baseline) pairs,
        so it is the interval that says whether a cell's Δ differs from zero —
        not the two policies' separate intervals combined. Orienting the Δ flips
        its sign, never the width, so the same grid serves both directions."""
        ci = (self.data.get("ci", {}) or {}).get("difference", {})
        g = ci.get(key)
        if g is None:
            return np.full((len(self.strikers), len(self.jammers)), np.nan)
        return np.asarray(g, dtype=float)

    def pvalue_grid(self, key: str) -> np.ndarray:
        """Per-cell paired p-value, aligned to the (strikers × jammers) grid."""
        g = np.full((len(self.strikers), len(self.jammers)), np.nan)
        for si, ns in enumerate(self.strikers):
            for ji, nj in enumerate(self.jammers):
                cell = (self.pvals.get((ns, nj)) or {}).get(key) or {}
                v = cell.get("pvalue")
                g[si, ji] = float(v) if v is not None else np.nan
        return g

    def train_region(self):
        return ([int(s) for s in self.meta.get("train_strikers", [])],
                [int(j) for j in self.meta.get("train_jammers", [])])

    def describe(self) -> str:
        m = self.meta
        lines = [
            f"  Data               : {self.doc.get('name')} "
            f"(written {self.doc.get('saved_at')} by {self.doc.get('source')})",
            f"  Grid               : strikers {self.strikers} × jammers {self.jammers}"
            f"  ({len(self.strikers) * len(self.jammers)} team sizes)",
            f"  Episodes per cell  : {m.get('n_episodes_per_cell')}   "
            f"base seed: {m.get('base_seed')}",
            f"  Complete policy    : {self.policies.get('complete')}",
            f"  Baseline policy    : {self.policies.get('baseline')}",
            f"  Training region    : strikers {m.get('train_strikers')}, "
            f"jammers {m.get('train_jammers')} (outlined on the figure)",
            f"  Significance       : paired one-sided Wilcoxon, hatched where "
            f"p >= {ALPHA}",
        ]
        return "\n".join(lines)


# =====================================================================
#  Drawing
# =====================================================================

def _fmt_delta(v: float, unit: str) -> str:
    """The difference, in the KPI's OWN units."""
    if not np.isfinite(v):
        return ""
    return VALUE_FMT.get(unit, "{:+.2f}").format(v)


def _fmt_ci(h: float, unit: str) -> str:
    """"±<half-width>" in the same units as the value above it."""
    if not np.isfinite(h):
        return ""
    return CI_FMT.get(unit, "±{:.3f}").format(h)


def _text_color(cmap, norm, value: float) -> str:
    """Black or white cell text, whichever stays legible on the mapped colour."""
    r, g, b, _ = cmap(norm(value))
    return "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "#14202b"


def _color_limit(delta: np.ndarray, pv: np.ndarray, full_scale: bool) -> float:
    """The |Δ| the colour ramp saturates at.

    By default this is the largest SIGNIFICANT difference, not the largest one
    outright: a single huge non-significant cell (a team size where one policy
    barely ever completes the mission) would otherwise stretch the ramp and wash
    every real, tested difference to near-white. Cells beyond the limit saturate
    — they stay hatched and still print their own number, so nothing is hidden.
    `full_scale` restores the plain max."""
    finite = np.isfinite(delta)
    if not full_scale:
        sig = finite & np.isfinite(pv) & (pv < ALPHA)
        if sig.any():
            return float(np.nanmax(np.abs(delta[sig])))
    return float(np.nanmax(np.abs(delta[finite]))) if finite.any() else 1.0


def _draw_panel(fig, ax, d: SweepData, spec: dict, contour: bool,
                annotate: bool, full_scale: bool = False) -> None:
    key, unit = spec["key"], spec["unit"]
    delta = d.oriented_delta(key)
    pv = d.pvalue_grid(key)
    ci = d.ci_grid(key)

    # Symmetric limits so zero sits exactly at the neutral midpoint of the ramp.
    vmax = _color_limit(delta, pv, full_scale) or 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)
    shown = delta

    nS, nJ = len(d.strikers), len(d.jammers)
    if contour:
        # Smoothed bands. Interpolates BETWEEN team sizes that do not exist —
        # prettier, less literal; the per-cell numbers stay authoritative.
        xs, ys = np.meshgrid(np.arange(nJ), np.arange(nS))
        im = ax.contourf(xs, ys, np.ma.masked_invalid(shown), levels=21,
                         cmap=NLR_DIV, norm=norm, extend="both")
        ax.contour(xs, ys, np.ma.masked_invalid(shown), levels=[0.0],
                   colors=[NLR_DARKGRAY], linewidths=1.2, linestyles="--")
    else:
        # pcolormesh rather than imshow, so the tiles can be separated by a
        # hairline of the page background instead of butting together.
        xe = np.arange(nJ + 1) - 0.5
        ye = np.arange(nS + 1) - 0.5
        im = ax.pcolormesh(xe, ye, np.ma.masked_invalid(shown), cmap=NLR_DIV,
                           norm=norm, edgecolors="white", linewidth=TILE_GAP_LW)

    # Non-significant cells recede behind a white veil, so the cells that carry
    # evidence are the ones the eye lands on.
    for si in range(nS):
        for ji in range(nJ):
            p = pv[si, ji]
            if np.isfinite(p) and p >= ALPHA:
                ax.add_patch(Rectangle((ji - 0.5, si - 0.5), 1, 1,
                                       facecolor="white", alpha=NS_VEIL,
                                       edgecolor="none", zorder=3))

    if annotate:
        # Value above centre, its 95% CI half-width below it — the same
        # arrangement the analysis' own heatmaps use. Non-significance is left
        # to the hatch, so the cell stays legible at slide size.
        for si in range(nS):
            for ji in range(nJ):
                v = shown[si, ji]
                if not np.isfinite(v):
                    continue
                p = pv[si, ji]
                faded = np.isfinite(p) and p >= ALPHA
                # A veiled tile is pale whatever its colour, so its text is
                # always the dark ink; only full-strength tiles need the
                # light-on-dark test.
                col = NLR_DARKGRAY if faded else _text_color(NLR_DIV, norm, v)
                h = _fmt_ci(ci[si, ji], unit)
                ax.text(ji, si + (0.15 if h else 0.0), _fmt_delta(delta[si, ji], unit),
                        ha="center", va="center", fontsize=CELL_FS, color=col,
                        zorder=4)
                if h:
                    ax.text(ji, si - 0.22, h, ha="center", va="center",
                            fontsize=CELL_CI_FS, color=col, alpha=0.75, zorder=4)

    # Training region: ONE outline around the whole block (per-cell boxes drew
    # internal lines that read as extra structure). Everything outside it is a
    # team size never trained on.
    train_s, train_j = d.train_region()
    rows = [i for i, s in enumerate(d.strikers) if s in train_s]
    cols = [j for j, n in enumerate(d.jammers) if n in train_j]
    if rows and cols:
        ax.add_patch(Rectangle((min(cols) - 0.5, min(rows) - 0.5),
                               max(cols) - min(cols) + 1,
                               max(rows) - min(rows) + 1,
                               fill=False, edgecolor=TRAIN_EDGE, lw=TRAIN_LW,
                               zorder=6))
        if TRAIN_DIVIDER and max(cols) < nJ - 1:
            # Everything right of this line is a LARGER team than anything seen
            # in training — the generalisation claim lives there.
            ax.axvline(max(cols) + 0.5, color=NLR_GRAY, lw=1.2, ls="--",
                       alpha=0.9, zorder=6)

    ax.set_xticks(range(nJ)); ax.set_xticklabels(d.jammers, fontsize=TICK_FS)
    ax.set_yticks(range(nS)); ax.set_yticklabels(d.strikers, fontsize=TICK_FS)
    ax.set_xlabel("Jammers", fontsize=AXIS_FS, color=NLR_GRAY, labelpad=6)
    ax.set_ylabel("Strikers", fontsize=AXIS_FS, color=NLR_GRAY, labelpad=6)
    # Title, and under it the subtraction the panel actually shows.
    ax.set_title(spec["title"], fontsize=TITLE_FS, fontweight="bold",
                 color=NLR_DARKGRAY, pad=24)
    ax.text(0.5, 1.035, SUBTITLE.get(d.direction.get(key, "higher"), ""),
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=SUBTITLE_FS, color=NLR_GRAY)
    ax.set_xlim(-0.5, nJ - 0.5)
    ax.set_ylim(-0.5, nS - 0.5)
    # The SQUARE plotting box the user asked for: the panel is square whatever
    # the grid's own proportions are.
    ax.set_box_aspect(1)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, colors=NLR_GRAY)

    # Colour bar BELOW the panel, slim and lightly ticked: every tile already
    # prints its own number, so the bar only has to establish the direction and
    # the range.
    # pad clears the x tick labels AND the "Jammers" label, which the colorbar
    # placement does not account for on its own.
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.17,
                        fraction=0.05, aspect=30, extend="both",
                        ticks=[-vmax, 0.0, vmax])
    cbar.set_label(CBAR_LABEL, fontsize=CBAR_FS, color=NLR_GRAY)
    cbar.ax.tick_params(labelsize=CBAR_FS, colors=NLR_GRAY, length=0)
    cbar.ax.set_xticklabels([_fmt_delta(-vmax, unit), "0", _fmt_delta(vmax, unit)])
    cbar.outline.set_visible(False)


def draw(d: SweepData, panels: List[dict], out_png: Path, dpi: int,
         contour: bool, annotate: bool, suptitle: Optional[str],
         transparent: bool, full_scale: bool = False) -> None:
    # Square cells fix each panel's proportions, so the figure is sized FROM the
    # grid rather than the other way round: a panel is as wide as its jammer
    # axis and as tall as its striker axis, plus room for the chrome.
    fig_w = (PANEL_SIDE + PANEL_PAD_W) * len(panels)
    fig_h = PANEL_SIDE + PANEL_CHROME_H + (0.35 if suptitle else 0.0)

    fig, axes = plt.subplots(1, len(panels), figsize=(fig_w, fig_h), squeeze=False)
    for ax, spec in zip(axes[0], panels):
        _draw_panel(fig, ax, d, spec, contour, annotate, full_scale)

    if suptitle:
        fig.suptitle(suptitle, fontsize=SUPTITLE_FS, fontweight="bold",
                     color=NLR_DARKGRAY)
    # A one-line key for everything that is not the numbers: which way the
    # colours point (the panels have their own titles, so this is the only place
    # that says it) and the two non-colour marks.
    fig.text(0.5, 0.12 / fig_h,
             "orange outline = team sizes seen in training     ·     "
             f"faded tile = difference not significant (p ≥ {ALPHA})",
             ha="center", va="bottom", fontsize=KEY_FS, color=NLR_GRAY)
    # Margins are reserved in INCHES converted to figure fractions rather than
    # left to tight_layout: the panels have a FIXED aspect, which tight_layout
    # cannot account for, and it clips the panel titles when it tries.
    fig.tight_layout(rect=(0, 0.42 / fig_h, 1,
                           1.0 - ((0.40 if suptitle else 0.28) / fig_h)))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    existed = out_png.exists()
    fig.savefig(out_png, dpi=dpi, transparent=transparent,
                facecolor="none" if transparent else fig.get_facecolor())
    plt.close(fig)
    print(f"{'Overwrote' if existed else 'Wrote'} {out_png.name}  "
          f"({int(fig_w * dpi)}×{int(fig_h * dpi)} px @ {dpi} dpi)  ->  {out_png}")


# =====================================================================
#  Console echo of the plotted numbers
# =====================================================================

def print_values(d: SweepData, panels: List[dict]) -> None:
    """Print every plotted cell, so a slide can be checked against the analysis."""
    print("\n" + "=" * 92)
    print("  VALUES PLOTTED  (oriented difference: POSITIVE = Comm-FOFE-MAPPO better)")
    print("=" * 92)
    for spec in panels:
        key, unit = spec["key"], spec["unit"]
        delta, pv = d.oriented_delta(key), d.pvalue_grid(key)
        unit_txt = "percentage points" if unit == "pp" else "steps saved"
        print(f"\n  {spec['title']}  [{unit_txt}]   "
              f"(* = p < {ALPHA}, blank = not significant)")
        head = "  strikers \\ jammers  " + "".join(f"{j:>10}" for j in d.jammers)
        print(head)
        print("  " + "-" * (len(head) - 2))
        for si, ns_ in enumerate(d.strikers):
            cells = []
            for ji in range(len(d.jammers)):
                v = _fmt_delta(delta[si, ji], unit)
                p = pv[si, ji]
                cells.append(f"{v}{'*' if (np.isfinite(p) and p < ALPHA) else ' '}")
            print(f"  {ns_:>18}  " + "".join(f"{c:>10}" for c in cells))
    n_ns = sum(int(np.isfinite(p) and p >= ALPHA)
               for spec in panels for p in d.pvalue_grid(spec["key"]).ravel())
    total = len(panels) * len(d.strikers) * len(d.jammers)
    print(f"\n  {total - n_ns} of {total} cells are significant at p < {ALPHA}; "
          f"{n_ns} are hatched on the figure.\n")


# =====================================================================
#  CLI
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Team-size generalisation difference maps (targets / "
        "survival / duration, oriented so blue = Comm-FOFE-MAPPO better) as ONE "
        "slide PNG, built from the dump written by eval_tools/3.3.5_team_size_"
        "generalisation_analysis.py. Runs no rollouts.")
    p.add_argument("--data_name", type=str, default=DATA_NAME,
                   help="Analysis dump to read from eval_results/analysis_data/.")
    p.add_argument("--out_dir", type=str, default=OUT_DIR,
                   help="Directory for the PNG (relative to the project dir).")
    p.add_argument("--name", type=str, default=OUT_NAME, help="Output filename stem.")
    p.add_argument("--dpi", type=int, default=DPI, help=f"Output DPI (default {DPI}).")
    p.add_argument("--contour", action="store_true",
                   help="Smooth the cells into filled contour bands (interpolates "
                        "between team sizes that do not exist — see the docstring).")
    p.add_argument("--no_values", action="store_true",
                   help="Drop the per-cell numbers, leaving colour only.")
    p.add_argument("--suptitle", action="store_true",
                   help="Add a headline above the panels (off by default — the "
                        "panels carry their own KPI titles, and the colour "
                        "orientation is stated in the key underneath).")
    p.add_argument("--full_scale", action="store_true",
                   help="Scale each colour ramp to the largest difference "
                        "outright, instead of the largest SIGNIFICANT one (see "
                        "_color_limit — the default keeps one huge untested cell "
                        "from washing out the rest of the panel).")
    p.add_argument("--transparent", action="store_true",
                   help="Transparent background (for coloured slide masters).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    d = SweepData(load_analysis(args.data_name))

    panels = [s for s in PANELS if d.has(s["key"])]
    for s in PANELS:
        if s not in panels:
            print(f"  ! '{s['key']}' is not in the dump — that panel is skipped.")
    if not panels:
        raise SystemExit("None of the configured KPIs are in the dump.")
    if not (d.strikers and d.jammers):
        raise SystemExit("The dump has no team-size axes to plot.")

    print("─" * 78)
    print("  TEAM-SIZE GENERALISATION — difference maps from saved analysis data")
    print("─" * 78)
    print(d.describe())
    print("─" * 78)
    print_values(d, panels)

    draw(d, panels, _resolve_out(args.out_dir) / f"{args.name}.png", args.dpi,
         args.contour, not args.no_values,
         SUPTITLE if args.suptitle else None, args.transparent, args.full_scale)


if __name__ == "__main__":
    main()
