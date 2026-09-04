"""3.3.2_mission_performance_bars.py — presentation bar charts from SAVED data.

Draws, per scenario (S1 and S2), THREE BAR CHARTS SIDE BY SIDE

    Targets destroyed  ·  Survival rate  ·  Mission duration

with the 95% CONFIDENCE INTERVAL as an error bar on top of every bar. NLR house
colours: Comm-FOFE-MAPPO = NLR dark blue, MAPPO Baseline = NLR terra orange.

This script runs NO rollouts and loads NO checkpoints. Its input is the data dump
that eval_tools/3.3.2_mission_performance_analysis.py writes at the end of its
run:

    eval_results/analysis_data/3.3.2_mission_performance.json

so the analysis (episodes, seeds, pairing, success-conditioning, the statistical
tests) stays entirely in eval_tools with its own config, and this file only ever
reads the numbers and styles them. To refresh the numbers, re-run the analysis;
to restyle the figures, re-run this — in seconds, on a laptop, offline.

Outputs (presentation/figures/):
    mission_bars_S1.png    3 panels, S1
    mission_bars_S2.png    3 panels, S2
    mission_bars_all.png   both scenarios (one row each)

What the bars mean (as computed by the analysis — see its docstring):
  · Both policies of a scenario are rolled out on the SAME per-episode initial
    conditions (common random numbers), so the comparison is PAIRED.
  · The bar is the mean over those episodes; the error bar is the two-sided 95%
    t interval of that mean (mean ± t·s/√n).
  · Mission duration is measured only over JOINTLY-SUCCESSFUL episodes (both
    policies destroyed all targets) — a policy whose agents die early must not
    look "faster". Its n is therefore smaller; it is printed in the console.
  · Overlapping error bars do NOT mean "no difference" for paired data: the
    paired p-value and Cohen's d_z from the analysis are printed to the console
    (and can be shown on the figure with --annotate).

Run (repo root, project venv):
  .venv\\Scripts\\python.exe "0_TA_HF_FOFE-MAPPO\\presentation\\3.3.2_mission_performance_bars.py"
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# This file lives in <project>/presentation/, one level DEEPER than eval_tools/,
# so the package path must expose BOTH the project dir (nlr_style) and
# eval_tools/ (analysis_data).
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
from matplotlib.patches import Patch

from .analysis_data import load_analysis
# NLR house palette (auto-applied to matplotlib on import).
from .nlr_style import NLR_PRIMARY, NLR_ACCENT, NLR_DARKGRAY

# =====================================================================
#  >>>  INPUT  (which analysis dump to draw)  <<<
# =====================================================================

# Written by eval_tools/3.3.2_mission_performance_analysis.py.  [CLI: --data_name]
DATA_NAME = "3.3.2_mission_performance"

# The THREE panels, left to right. Keys must exist in the dump's "results".
BAR_KPIS = ["targets", "survival", "duration"]

# Panel titles (presentation wording; the dump's own KPI label is the fallback).
PANEL_TITLES = {
    "targets": "Targets destroyed",
    "survival": "Survival rate",
    "duration": "Mission duration",
}

# =====================================================================
#  >>>  FIGURE STYLE  (NLR house colours)  <<<
# =====================================================================
MAIN_COLOR = NLR_PRIMARY      # #004d7d  NLR dark blue  — Comm-FOFE-MAPPO
BASE_COLOR = NLR_ACCENT       # #ed7914  NLR terra      — MAPPO Baseline

BAR_WIDTH = 0.46          # bar thickness (x units; bars sit at x = 0 and BAR_GAP)
BAR_GAP = 0.62            # centre-to-centre distance between the two bars
ERR_COLOR = NLR_DARKGRAY  # error-bar (95% CI) colour
ERR_LW = 1.4
ERR_CAPSIZE = 5
VALUE_LABELS = True       # print the mean on top of each bar (above its CI cap)
PANEL_W = 2.6             # inches per panel
PANEL_H = 3.4             # inches per panel row

# Headroom above the tallest (bar + CI) so value labels never clip.
RATE_YMAX = 1.16          # rate panels are pinned to a 0–100% axis + headroom
DUR_HEADROOM = 1.28       # step-unit panels: ymax = max(bar + CI) * this

# Outputs (relative paths resolved against the project dir 0_TA_...).
OUT_DIR = "presentation/figures"
OUT_PREFIX = "mission_bars"
DPI = 500

# =====================================================================


def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_PKG_DIR / p)


# =====================================================================
#  Reading the analysis dump
# =====================================================================

class Analysis:
    """Thin reader over one 3.3.2 data dump: the numbers plus the labels, units
    and provenance the figures need. Everything here was computed by the
    analysis; nothing is recomputed."""

    def __init__(self, doc: dict):
        self.doc = doc
        self.data = doc["data"]
        self.meta = doc.get("meta", {})
        self.results: Dict[str, Dict[str, dict]] = self.data["results"]
        self.kpis: Dict[str, dict] = self.data.get("kpis", {})
        self.scenarios: List[str] = list(self.data.get("scenario_order", []))
        labels = self.data.get("labels", {})
        self.main_label = labels.get("main_display") or labels.get("main") or "Complete"
        self.base_label = labels.get("base_display") or labels.get("base") or "Baseline"

    # -- per-KPI metadata -------------------------------------------------
    def title(self, key: str) -> str:
        return PANEL_TITLES.get(key) or self.kpis.get(key, {}).get("label", key)

    def unit(self, key: str) -> str:
        return self.kpis.get(key, {}).get("unit", "")

    def is_rate(self, key: str) -> bool:
        return self.unit(key) == "rate"

    def is_success_conditioned(self, key: str) -> bool:
        return bool(self.kpis.get(key, {}).get("success_conditioned"))

    def test(self, key: str) -> str:
        return self.kpis.get(key, {}).get("test", "")

    # -- per-(KPI, scenario) numbers --------------------------------------
    def result(self, key: str, scen: str) -> dict:
        return self.results.get(key, {}).get(scen, {})

    def bar(self, key: str, scen: str, slot: str) -> Tuple[float, Tuple[float, float]]:
        """(mean, (ci_lo, ci_hi)) for one bar; NaNs when that policy is absent."""
        r = self.result(key, scen)
        mean = float(r.get(f"mean_{slot}", float("nan")))
        ci = r.get(f"ci_{slot}") or (float("nan"), float("nan"))
        return mean, (float(ci[0]), float(ci[1]))

    def missing_kpis(self) -> List[str]:
        return [k for k in BAR_KPIS if k not in self.results]

    def describe(self) -> str:
        m = self.meta
        pol = m.get("policies", {})
        lines = [
            f"  Data               : {self.doc.get('name')} "
            f"(written {self.doc.get('saved_at')} by {self.doc.get('source')})",
            f"  Scenarios          : {', '.join(self.scenarios)}",
            f"  Series             : {self.main_label} (NLR dark blue) vs "
            f"{self.base_label} (NLR terra)",
            f"  Panels             : {', '.join(self.title(k) for k in BAR_KPIS)}",
            f"  Paired episodes    : {m.get('n_episodes')}   base seed: {m.get('base_seed')}",
            f"  Error bars         : {int(round(float(m.get('ci_level', 0.95)) * 100))}% CI "
            f"of the mean (t interval)",
        ]
        for scn in self.scenarios:
            p = pol.get(scn, {})
            for slot, label in (("main", self.main_label), ("base", self.base_label)):
                f = (p.get(slot) or {}).get("policy_file")
                lines.append(f"      [{scn}] {label}: {f or '(none)'}")
        cond = [self.title(k) for k in BAR_KPIS if self.is_success_conditioned(k)]
        if cond:
            n = self.data.get("joint_success_n", {})
            lines.append(f"  Success-conditioned: {', '.join(cond)} — jointly-successful "
                         f"pairs only ({', '.join(f'{s}: {n.get(s)}' for s in self.scenarios)})")
        return "\n".join(lines)


# =====================================================================
#  Formatting helpers
# =====================================================================

def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _bar_label(a: Analysis, key: str, mean: float) -> str:
    """On-bar value label: percentages for rates, one decimal otherwise."""
    if not np.isfinite(mean):
        return ""
    return f"{mean * 100:.0f}%" if a.is_rate(key) else f"{mean:.1f}"


def _yerr(mean: float, ci: Tuple[float, float]) -> Optional[np.ndarray]:
    """Asymmetric yerr column [[down], [up]] for one bar's CI, or None when the
    bar has no interval (single observation / missing policy)."""
    lo, hi = ci
    if not (np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)):
        return None
    return np.array([[max(mean - lo, 0.0)], [max(hi - mean, 0.0)]])


# =====================================================================
#  The bar charts
# =====================================================================

def _panel_top(a: Analysis, key: str, scen: str) -> float:
    """Highest drawn y in a panel (tallest bar + its upper CI cap); 0 if empty."""
    tops = []
    for slot in ("main", "base"):
        mean, ci = a.bar(key, scen, slot)
        if not np.isfinite(mean):
            continue
        err = _yerr(mean, ci)
        tops.append(mean + (float(err[1, 0]) if err is not None else 0.0))
    return max(tops) if tops else 0.0


def _draw_panel(ax, a: Analysis, key: str, scen: str, title_fs: float,
                tick_fs: float, value_fs: float, annotate: bool = False,
                ymax_override: Optional[float] = None) -> None:
    """One KPI panel: two bars (main = NLR dark blue, base = NLR terra orange)
    with 95% CI error bars on top and the mean printed above each cap.

    `ymax_override` pins the y-axis top so the same KPI can share ONE scale across
    scenarios (used by the combined figure — otherwise each row would auto-scale
    and the rows could not be compared by eye)."""
    xs = [0.0, BAR_GAP]
    colors = [MAIN_COLOR, BASE_COLOR]
    bars = [a.bar(key, scen, slot) for slot in ("main", "base")]

    for x, (mean, ci), color in zip(xs, bars, colors):
        if not np.isfinite(mean):
            continue                       # policy absent in the analysis → no bar
        ax.bar(x, mean, width=BAR_WIDTH, color=color, zorder=2)
        err = _yerr(mean, ci)
        if err is not None:
            ax.errorbar(x, mean, yerr=err, fmt="none", ecolor=ERR_COLOR,
                        elinewidth=ERR_LW, capsize=ERR_CAPSIZE, capthick=ERR_LW,
                        zorder=3)

    # y-axis: rates on a fixed 0–100% scale; other units auto-scaled with headroom
    # so the CI cap and its value label never touch the panel top.
    if a.is_rate(key):
        ymax = ymax_override or RATE_YMAX
        ticks = np.arange(0.0, 1.01, 0.2)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int(round(v * 100))}%" for v in ticks])
    else:
        ymax = ymax_override or (_panel_top(a, key, scen) * DUR_HEADROOM) or 1.0
        ax.set_ylabel(a.unit(key), fontsize=tick_fs)
    ax.set_ylim(0.0, ymax)

    if VALUE_LABELS:
        for x, (mean, ci) in zip(xs, bars):
            if not np.isfinite(mean):
                continue
            err = _yerr(mean, ci)
            top = mean + (float(err[1, 0]) if err is not None else 0.0)
            ax.text(x, top + 0.028 * ymax, _bar_label(a, key, mean), ha="center",
                    va="bottom", fontsize=value_fs, color=NLR_DARKGRAY)

    if annotate:
        # The paired test from the analysis — the rigorous comparison, which the
        # marginal error bars do not show.
        p = float(a.result(key, scen).get("pvalue", float("nan")))
        if np.isfinite(p):
            ax.text(0.5, 0.965, _fmt_p(p), transform=ax.transAxes, ha="center",
                    va="top", fontsize=value_fs - 0.5, color=NLR_DARKGRAY,
                    style="italic")

    ax.set_title(a.title(key), fontsize=title_fs, fontweight="bold", pad=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["", ""])            # identity comes from the legend
    ax.set_xlim(xs[0] - BAR_GAP * 0.85, xs[-1] + BAR_GAP * 0.85)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.tick_params(axis="x", length=0)
    ax.grid(True, axis="y", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _legend_handles(a: Analysis):
    return [Patch(facecolor=MAIN_COLOR, label=a.main_label),
            Patch(facecolor=BASE_COLOR, label=a.base_label)]


def plot_scenario_bars(a: Analysis, scen: str, out_png: Path, dpi: int,
                       annotate: bool = False, title: Optional[str] = None) -> None:
    """The presentation figure: three bar charts side by side for ONE scenario."""
    fig, axes = plt.subplots(1, len(BAR_KPIS),
                             figsize=(PANEL_W * len(BAR_KPIS), PANEL_H),
                             squeeze=False)
    for j, key in enumerate(BAR_KPIS):
        _draw_panel(axes[0, j], a, key, scen, title_fs=11, tick_fs=9, value_fs=9,
                    annotate=annotate)

    fig.legend(handles=_legend_handles(a), loc="lower center", ncol=2, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, 0.005), handlelength=1.1,
               handletextpad=0.5, columnspacing=1.6)
    if title:
        fig.suptitle(title, fontsize=12.5, fontweight="bold")
        rect = (0, 0.085, 1, 0.94)
    else:
        rect = (0, 0.085, 1, 1.0)
    fig.tight_layout(rect=rect)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved {scen} bar figure to: {out_png}")


def plot_all_scenarios(a: Analysis, scenarios: List[str], out_png: Path, dpi: int,
                       annotate: bool = False) -> None:
    """Both scenarios in one figure: one ROW per scenario, three panels per row.

    Non-rate panels share ONE y-axis across the rows, so the rows can also be
    compared vertically (a per-row autoscale would silently rescale them)."""
    shared_ymax = {}
    for key in BAR_KPIS:
        if a.is_rate(key):
            continue
        top = max((_panel_top(a, key, s) for s in scenarios), default=0.0)
        shared_ymax[key] = (top * DUR_HEADROOM) or 1.0

    fig, axes = plt.subplots(len(scenarios), len(BAR_KPIS),
                             figsize=(PANEL_W * len(BAR_KPIS), PANEL_H * len(scenarios)),
                             squeeze=False)
    for i, scen in enumerate(scenarios):
        for j, key in enumerate(BAR_KPIS):
            ax = axes[i, j]
            _draw_panel(ax, a, key, scen, title_fs=11, tick_fs=9, value_fs=9,
                        annotate=annotate, ymax_override=shared_ymax.get(key))
            if i > 0:                        # titles only on the top row
                ax.set_title("")
            if j == 0:
                ax.text(-0.30, 0.5, scen, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=12, fontweight="bold",
                        color=NLR_DARKGRAY)

    fig.legend(handles=_legend_handles(a), loc="lower center", ncol=2, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, 0.004), handlelength=1.1,
               handletextpad=0.5, columnspacing=1.6)
    fig.tight_layout(rect=(0.02, 0.05, 1, 1.0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved combined bar figure to: {out_png}")


# =====================================================================
#  Console echo of the plotted numbers
# =====================================================================

def print_values(a: Analysis, scenarios: List[str]) -> None:
    """Print exactly what the bars and error bars show, plus the paired test the
    figure cannot show, so the slide can be checked against the analysis."""
    pct = int(round(float(a.meta.get("ci_level", 0.95)) * 100))
    print("\n" + "=" * 92)
    print(f"  VALUES PLOTTED  (bar = mean, error bar = {pct}% CI; paired test from "
          f"the analysis)")
    print("=" * 92)
    for key in BAR_KPIS:
        cond = "  [jointly-successful pairs only]" if a.is_success_conditioned(key) else ""
        test = f"  [test: {a.test(key)}]" if a.test(key) else ""
        print(f"\n  {a.title(key)}{cond}{test}")
        header = ["Scenario", a.main_label, a.base_label, "p", "d_z", "n"]
        rows = []
        for scen in scenarios:
            r = a.result(key, scen)

            def _cell(slot):
                mean, ci = a.bar(key, scen, slot)
                if not np.isfinite(mean):
                    return ""
                f = "{:.3f}" if a.is_rate(key) else "{:.1f}"
                return (f"{f.format(mean)} [{f.format(ci[0])}, {f.format(ci[1])}]")

            p = float(r.get("pvalue", float("nan")))
            dz = float(r.get("dz", float("nan")))
            rows.append([scen, _cell("main"), _cell("base"),
                         (_fmt_p(p).replace("p=", "").replace("p", "")
                          if np.isfinite(p) else ""),
                         (f"{dz:.2f}" if np.isfinite(dz) else ""),
                         str(r.get("n", "") or "")])
        widths = [max(len(h), *(len(r[c]) for r in rows)) for c, h in enumerate(header)]
        fmt = lambda cells: "  ".join(  # noqa: E731
            c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
            for i, c in enumerate(cells))
        print("  " + fmt(header))
        print("  " + "  ".join("-" * w for w in widths))
        for r in rows:
            print("  " + fmt(r))
    print()


# =====================================================================
#  CLI
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Presentation bar charts (targets destroyed / survival rate / "
        "mission duration, 95% CI error bars, NLR colours) for S1 and S2, built "
        "from the data dump written by eval_tools/3.3.2_mission_performance_"
        "analysis.py. Runs no rollouts.")
    p.add_argument("--data_name", type=str, default=DATA_NAME,
                   help="Analysis dump to read from eval_results/analysis_data/.")
    p.add_argument("--scenarios", type=str, default=None,
                   help="Comma-separated subset of scenarios to draw "
                        "(default: all in the dump, in its order).")
    p.add_argument("--out_dir", type=str, default=OUT_DIR,
                   help="Directory for the PNGs (relative to the project dir).")
    p.add_argument("--prefix", type=str, default=OUT_PREFIX,
                   help="Filename prefix for the PNGs.")
    p.add_argument("--dpi", type=int, default=DPI)
    p.add_argument("--titles", action="store_true",
                   help="Add a 'Scenario Sx' suptitle to each per-scenario figure.")
    p.add_argument("--annotate", action="store_true",
                   help="Print the paired p-value inside each panel.")
    p.add_argument("--no_combined", action="store_true",
                   help="Skip the combined S1+S2 overview figure.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    a = Analysis(load_analysis(args.data_name))
    scenarios = ([s.strip() for s in args.scenarios.split(",") if s.strip()]
                 if args.scenarios else a.scenarios)
    unknown = [s for s in scenarios if s not in a.scenarios]
    if unknown:
        raise SystemExit(f"Scenario(s) {unknown} are not in the dump "
                         f"(it has: {', '.join(a.scenarios)}).")
    missing = a.missing_kpis()
    if missing:
        raise SystemExit(f"KPI(s) {missing} are not in the dump — re-run the "
                         f"analysis with them in its TABLE_KPIS.")

    print("─" * 78)
    print("  MISSION PERFORMANCE BAR CHARTS — drawn from saved analysis data")
    print("─" * 78)
    print(a.describe())
    print("─" * 78)

    print_values(a, scenarios)

    out_dir = _resolve_out(args.out_dir)
    for scen in scenarios:
        plot_scenario_bars(a, scen, out_dir / f"{args.prefix}_{scen}.png", args.dpi,
                           annotate=args.annotate,
                           title=f"Scenario {scen}" if args.titles else None)

    if not args.no_combined and len(scenarios) > 1:
        plot_all_scenarios(a, scenarios, out_dir / f"{args.prefix}_all.png",
                           args.dpi, annotate=args.annotate)


if __name__ == "__main__":
    main()
