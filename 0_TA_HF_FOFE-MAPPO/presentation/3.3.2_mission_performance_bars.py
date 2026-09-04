"""3.3.2_mission_performance_bars.py — mission-performance slide bar charts.

Per scenario (S1, S2): Comm-FOFE-MAPPO vs MAPPO Baseline as grouped bars —

    [ Targets Destroyed | Survival Rate ]   [ Mission Duration ]
              one 0–100% axis                  its own axis (steps)

with the 95% CONFIDENCE INTERVAL as a cap on top of every bar. NLR house
colours: dark blue = Comm-FOFE-MAPPO, terra orange = MAPPO Baseline.

Saved as high-resolution PNGs straight into this folder, one per scenario:

    presentation/3.3.2_mission_performance_S1.png
    presentation/3.3.2_mission_performance_S2.png

The names are STABLE, so re-running this script overwrites those files in place
and the deck always points at the current figure.

This script runs NO rollouts and loads NO checkpoints. Its only input is the
dump that eval_tools/3.3.2_mission_performance_analysis.py wrote at the end of
its run:

    eval_results/analysis_data/3.3.2_mission_performance.json

Re-running THIS script only restyles the figures; refreshing the numbers means
re-running that analysis. The shared slide style lives in nlr_bars.py.

What the bars mean (as computed by the analysis — see its docstring):
  · Both policies are rolled out on the SAME per-episode initial conditions
    (common random numbers), so the comparison is PAIRED.
  · The bar is the mean over those episodes; the cap is the two-sided 95% t
    interval of that mean (mean ± t·s/√n).
  · Mission duration is measured only over JOINTLY-SUCCESSFUL episodes (both
    policies destroyed all targets) — a policy whose agents die early must not
    look "faster". Its n is smaller, and is printed to the console.
  · Overlapping caps do NOT mean "no difference" for paired data: the paired
    p-value and Cohen's d_z are printed to the console, and --annotate prints
    the p-value on the figure.

Run (repo root, project venv):
  .venv\\Scripts\\python.exe "0_TA_HF_FOFE-MAPPO\\presentation\\3.3.2_mission_performance_bars.py"
  …--dpi 900 --values --transparent
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Dict, List

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

from .analysis_data import load_analysis
from . import nlr_bars
from .nlr_bars import Bar, Group, Panel

# =====================================================================
#  >>>  INPUT  (which analysis dump to draw)  <<<
# =====================================================================

# Written by eval_tools/3.3.2_mission_performance_analysis.py.  [CLI: --data_name]
DATA_NAME = "3.3.2_mission_performance"

# The RATE KPIs, grouped in the first panel on a shared 0–100% axis.
RATE_KPIS = ["targets", "survival"]
# The KPIs that need their own axis (different unit), one panel each, in order.
OWN_AXIS_KPIS = ["duration"]

# Slide wording for the bold labels under each group of bars.
GROUP_LABELS = {
    "targets": "Targets Destroyed",
    "survival": "Survival Rate",
    "duration": "Mission Duration",
    "reward": "Episode Reward",
    "fragmentation": "Coalition Fragmentation",
}

# y-axis label of the rates panel (the other panels stay unlabelled — their KPI
# name is already under the bars).
RATE_YLABEL = "Mean"

# Outputs: written straight into the presentation folder under a STABLE name per
# scenario — "3.3.2_mission_performance_<scenario>.png" — so each figure keeps
# one recognisable filename and every re-run overwrites it in place.
# (Relative paths are resolved against the project dir 0_TA_...)
OUT_DIR = "presentation"
OUT_PREFIX = "3.3.2_mission_performance"
DPI = 600

# =====================================================================


def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_PKG_DIR / p)


# =====================================================================
#  Reading the analysis dump
# =====================================================================

class MissionData:
    """Thin reader over the 3.3.2 dump: the numbers plus the labels and
    provenance the slide needs. Everything was computed by the analysis."""

    def __init__(self, doc: dict):
        self.doc = doc
        self.data = doc["data"]
        self.meta = doc.get("meta", {})
        self.results: Dict[str, Dict[str, dict]] = self.data["results"]
        self.kpis: Dict[str, dict] = self.data.get("kpis", {})
        self.scenarios: List[str] = list(self.data.get("scenario_order", []))
        labels = self.data.get("labels", {})
        self.series = [labels.get("main_display") or labels.get("main") or "Complete",
                       labels.get("base_display") or labels.get("base") or "Baseline"]

    def label(self, key: str) -> str:
        return GROUP_LABELS.get(key) or self.kpis.get(key, {}).get("label", key)

    def unit(self, key: str) -> str:
        return self.kpis.get(key, {}).get("unit", "")

    def conditioned(self, key: str) -> bool:
        return bool(self.kpis.get(key, {}).get("success_conditioned"))

    def result(self, key: str, scen: str) -> dict:
        return self.results.get(key, {}).get(scen, {})

    def bar(self, key: str, scen: str, slot: str) -> Bar:
        r = self.result(key, scen)
        ci = r.get(f"ci_{slot}") or (float("nan"), float("nan"))
        return Bar(float(r.get(f"mean_{slot}", float("nan"))),
                   float(ci[0]), float(ci[1]))

    def group(self, key: str, scen: str) -> Group:
        return Group(self.label(key),
                     [self.bar(key, scen, "main"), self.bar(key, scen, "base")])

    def describe(self) -> str:
        m = self.meta
        pct = int(round(float(m.get("ci_level", 0.95)) * 100))
        lines = [
            f"  Data               : {self.doc.get('name')} "
            f"(written {self.doc.get('saved_at')} by {self.doc.get('source')})",
            f"  Scenarios          : {', '.join(self.scenarios)}",
            f"  Series             : {self.series[0]} (NLR dark blue) vs "
            f"{self.series[1]} (NLR terra)",
            f"  Paired episodes    : {m.get('n_episodes')}   base seed: {m.get('base_seed')}",
            f"  Error bars         : {pct}% CI of the mean (t interval)",
        ]
        for scn in self.scenarios:
            p = (m.get("policies") or {}).get(scn, {})
            for slot, name in zip(("main", "base"), self.series):
                lines.append(f"      [{scn}] {name}: "
                             f"{(p.get(slot) or {}).get('policy_file') or '(none)'}")
        n = self.data.get("joint_success_n", {})
        cond = [self.label(k) for k in RATE_KPIS + OWN_AXIS_KPIS if self.conditioned(k)]
        if cond:
            lines.append(f"  Success-conditioned: {', '.join(cond)} — jointly-successful "
                         f"pairs only ({', '.join(f'{s}: {n.get(s)}' for s in self.scenarios)})")
        return "\n".join(lines)


# =====================================================================
#  Figures
# =====================================================================

def build_panels(d: MissionData, scen: str, annotate: bool = False) -> List[Panel]:
    """The slide layout: one grouped 0–100% panel for the rate KPIs, then one
    panel per differently-scaled KPI (mission duration, in steps)."""
    panels = [Panel(groups=[d.group(k, scen) for k in RATE_KPIS],
                    percent=True, ylabel=RATE_YLABEL)]
    for key in OWN_AXIS_KPIS:
        panels.append(Panel(groups=[d.group(key, scen)], percent=False,
                            value_fmt="{:.1f}"))
    if annotate:
        for panel in panels:
            for group in panel.groups:
                key = next(k for k in RATE_KPIS + OWN_AXIS_KPIS
                           if d.label(k) == group.label)
                p = float(d.result(key, scen).get("pvalue", float("nan")))
                if np.isfinite(p):
                    group.label += ("\np<0.001" if p < 0.001 else f"\np={p:.3f}")
    return panels


def plot_scenario(d: MissionData, scen: str, out_png: Path, dpi: int,
                  values: bool, transparent: bool, titled: bool,
                  annotate: bool) -> None:
    nlr_bars.draw(build_panels(d, scen, annotate), d.series, out_png, dpi=dpi,
                  value_labels=values, transparent=transparent,
                  title=f"Scenario {scen}" if titled else None)


# =====================================================================
#  Console echo of the plotted numbers
# =====================================================================

def print_values(d: MissionData, scenarios: List[str]) -> None:
    """Print exactly what the bars and caps show, plus the paired test the
    figure cannot show, so a slide can be checked against the analysis."""
    pct = int(round(float(d.meta.get("ci_level", 0.95)) * 100))
    print("\n" + "=" * 88)
    print(f"  VALUES PLOTTED  (bar = mean, cap = {pct}% CI; paired test from the analysis)")
    print("=" * 88)
    for key in RATE_KPIS + OWN_AXIS_KPIS:
        cond = "  [jointly-successful pairs only]" if d.conditioned(key) else ""
        print(f"\n  {d.label(key)}{cond}")
        header = ["Scenario", d.series[0], d.series[1], "p", "d_z", "n"]
        rows = []
        for scen in scenarios:
            r = d.result(key, scen)
            fmt = "{:.3f}" if d.unit(key) == "rate" else "{:.1f}"

            def _cell(slot):
                b = d.bar(key, scen, slot)
                if not b.drawable:
                    return ""
                return f"{fmt.format(b.mean)} [{fmt.format(b.lo)}, {fmt.format(b.hi)}]"

            p = float(r.get("pvalue", float("nan")))
            dz = float(r.get("dz", float("nan")))
            rows.append([scen, _cell("main"), _cell("base"),
                         ("<0.001" if p < 0.001 else f"{p:.3f}") if np.isfinite(p) else "",
                         f"{dz:.2f}" if np.isfinite(dz) else "",
                         str(r.get("n", "") or "")])
        widths = [max(len(h), *(len(r[c]) for r in rows)) for c, h in enumerate(header)]

        def _row(cells):
            return "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
                             for i, c in enumerate(cells))

        print("  " + _row(header))
        print("  " + "  ".join("-" * w for w in widths))
        for r in rows:
            print("  " + _row(r))
    print()


# =====================================================================
#  CLI
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mission-performance slide bar charts (targets destroyed / "
        "survival rate / mission duration, 95% CI caps, NLR colours) per "
        "scenario, built from the dump written by eval_tools/3.3.2_mission_"
        "performance_analysis.py. Runs no rollouts.")
    p.add_argument("--data_name", type=str, default=DATA_NAME,
                   help="Analysis dump to read from eval_results/analysis_data/.")
    p.add_argument("--scenarios", type=str, default=None,
                   help="Comma-separated subset to draw (default: all in the dump).")
    p.add_argument("--out_dir", type=str, default=OUT_DIR,
                   help="Directory for the PNGs (relative to the project dir).")
    p.add_argument("--prefix", type=str, default=OUT_PREFIX)
    p.add_argument("--dpi", type=int, default=DPI, help=f"Output DPI (default {DPI}).")
    p.add_argument("--values", action="store_true",
                   help="Print each mean on top of its bar.")
    p.add_argument("--annotate", action="store_true",
                   help="Add the paired p-value under each KPI label.")
    p.add_argument("--titles", action="store_true",
                   help="Add a 'Scenario Sx' title above each figure.")
    p.add_argument("--transparent", action="store_true",
                   help="Transparent background (for coloured slide masters).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    d = MissionData(load_analysis(args.data_name))

    scenarios = ([s.strip() for s in args.scenarios.split(",") if s.strip()]
                 if args.scenarios else d.scenarios)
    unknown = [s for s in scenarios if s not in d.scenarios]
    if unknown:
        raise SystemExit(f"Scenario(s) {unknown} are not in the dump "
                         f"(it has: {', '.join(d.scenarios)}).")
    missing = [k for k in RATE_KPIS + OWN_AXIS_KPIS if k not in d.results]
    if missing:
        raise SystemExit(f"KPI(s) {missing} are not in the dump — re-run the "
                         f"analysis with them in its TABLE_KPIS.")

    print("─" * 78)
    print("  MISSION PERFORMANCE — slide bar charts from saved analysis data")
    print("─" * 78)
    print(d.describe())
    print("─" * 78)
    print_values(d, scenarios)

    out_dir = _resolve_out(args.out_dir)
    for scen in scenarios:
        plot_scenario(d, scen, out_dir / f"{args.prefix}_{scen}.png", args.dpi,
                      args.values, args.transparent, args.titles, args.annotate)


if __name__ == "__main__":
    main()
