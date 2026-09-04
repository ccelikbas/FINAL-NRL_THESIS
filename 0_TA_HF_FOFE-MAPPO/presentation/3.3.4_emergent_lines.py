"""3.3.4_emergent_lines.py — emergent-behaviour sensitivity line charts.

TWO figures, sized identically so they sit side by side on one slide:

    presentation/3.3.4_emergent_fragmentation.png    coalition fragmentation
    presentation/3.3.4_emergent_duration.png         mission duration

Each plots the KPI against TEAM SIZE (2, 3 and 4 jammers) for both policies —
Comm-FOFE-MAPPO (NLR dark blue, solid) and MAPPO Baseline (NLR terra, dashed) —
with the 95% confidence interval as a filled band plus a capped error bar at
each measured team size. The names are STABLE, so a re-run overwrites them in
place.

This script runs NO rollouts and loads NO checkpoints. Its only input is the
dump that eval_tools/3.3.4_emergent_team_behaviour_analysis.py wrote at the end
of its run:

    eval_results/analysis_data/3.3.4_emergent_team_behaviour.json

Refreshing the numbers means re-running that analysis; re-running this only
restyles the figures. The shared slide style lives in nlr_lines.py.

Two things the analysis defines and these figures inherit:
  · Both policies are rolled out on the SAME per-episode initial conditions at
    every team size (common random numbers), so each comparison is PAIRED. The
    paired p-value per team size is printed to the console.
  · MISSION DURATION is measured only over JOINTLY-SUCCESSFUL episodes (both
    policies destroyed all targets) — a policy whose agents die early must not
    look "faster" — so its points rest on fewer episodes than the fragmentation
    points. Both counts are printed to the console.

FRAGMENTATION is an index, not a rate: its axis is zero-anchored (0 means a team
that never splits). DURATION is a trend in step counts, so its axis is scaled to
the data — see the y-axis note in nlr_lines.py.

Run (repo root, project venv):
  .venv\\Scripts\\python.exe "0_TA_HF_FOFE-MAPPO\\presentation\\3.3.4_emergent_lines.py"
  …--dpi 900 --transparent --no_titles
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Dict, List

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
from . import nlr_lines
from .nlr_lines import Point, Series

# =====================================================================
#  >>>  INPUT  (which analysis dump to draw)  <<<
# =====================================================================

# Written by eval_tools/3.3.4_emergent_team_behaviour_analysis.py. [CLI: --data_name]
DATA_NAME = "3.3.4_emergent_team_behaviour"

# One figure per KPI, in this order. Each entry is:
#   key        the KPI key in the dump
#   slug       the filename suffix (the figure's recognisable name)
#   title      the bold title on the figure
#   ylabel     the y-axis label (name the UNIT — the numbers are meaningless without it)
#   zero_base  anchor the y-axis at zero (see the y-axis note in nlr_lines.py)
FIGURES = [
    dict(key="fragmentation", slug="fragmentation",
         title="Coalition Fragmentation", ylabel="Fragmentation (index)",
         zero_base=True),
    dict(key="duration", slug="duration",
         title="Mission Duration", ylabel="Duration (steps)",
         zero_base=False),
]

# x-axis of every figure: the swept team size.
XLABEL = "Number of jammers"

# Outputs: written straight into the presentation folder under STABLE names —
# "3.3.4_emergent_<slug>.png" — so each figure keeps one recognisable filename
# and every re-run overwrites it in place.
OUT_DIR = "presentation"
OUT_PREFIX = "3.3.4_emergent"
DPI = 600

# =====================================================================


def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_PKG_DIR / p)


# =====================================================================
#  Reading the analysis dump
# =====================================================================

class EmergentData:
    """Thin reader over the 3.3.4 dump: the per-composition numbers plus the
    labels and provenance the slide needs. Nothing is recomputed here."""

    def __init__(self, doc: dict):
        self.doc = doc
        self.data = doc["data"]
        self.meta = doc.get("meta", {})
        self.results: Dict[str, Dict[str, dict]] = self.data["results"]
        self.kpis: Dict[str, dict] = self.data.get("kpis", {})
        self.scenarios: List[str] = list(self.data.get("scenario_order", []))
        labels = self.data.get("labels", {})
        self.series_labels = [
            labels.get("main_display") or labels.get("main") or "Complete",
            labels.get("base_display") or labels.get("base") or "Baseline",
        ]

    def jammers(self, scen: str) -> float:
        """The x value of one composition — its jammer count, as recorded by the
        analysis (it reads the section, so '2S3J' cannot drift out of sync)."""
        return float((self.data.get("scenarios", {}).get(scen, {})
                      or {}).get("jammer_count", float("nan")))

    def result(self, key: str, scen: str) -> dict:
        return self.results.get(key, {}).get(scen, {})

    def conditioned(self, key: str) -> bool:
        return bool(self.kpis.get(key, {}).get("success_conditioned"))

    def series(self, key: str) -> List[Series]:
        """One Series per policy: its mean and CI at every swept team size."""
        out = []
        for slot, label in zip(("main", "base"), self.series_labels):
            pts = []
            for scen in self.scenarios:
                r = self.result(key, scen)
                ci = r.get(f"ci_{slot}") or (float("nan"), float("nan"))
                pts.append(Point(self.jammers(scen),
                                 float(r.get(f"mean_{slot}", float("nan"))),
                                 float(ci[0]), float(ci[1])))
            out.append(Series(label, pts))
        return out

    def n_at(self, key: str, scen: str) -> str:
        return str(self.result(key, scen).get("n", "") or "")

    def describe(self) -> str:
        m = self.meta
        pct = int(round(float(m.get("ci_level", 0.95)) * 100))
        pol = m.get("policies") or {}
        lines = [
            f"  Data               : {self.doc.get('name')} "
            f"(written {self.doc.get('saved_at')} by {self.doc.get('source')})",
            f"  Compositions       : {', '.join(self.scenarios)}  "
            f"(jammers: {', '.join(f'{self.jammers(s):g}' for s in self.scenarios)})",
            f"  Series             : {self.series_labels[0]} (NLR dark blue, solid) vs "
            f"{self.series_labels[1]} (NLR terra, dashed)",
            f"  Paired episodes    : {m.get('n_episodes')}   base seed: {m.get('base_seed')}",
            f"  Confidence bands   : {pct}% CI of the mean (t interval)",
        ]
        for slot, name in zip(("main", "base"), self.series_labels):
            lines.append(f"      {name}: {(pol.get(slot) or {}).get('policy_file') or '(none)'}")
        n = self.data.get("joint_success_n", {})
        if n:
            lines.append("  Duration n         : jointly-successful pairs only "
                         f"({', '.join(f'{s}: {n.get(s)}' for s in self.scenarios)})")
        return "\n".join(lines)


# =====================================================================
#  Console echo of the plotted numbers
# =====================================================================

def print_values(d: EmergentData, keys: List[str]) -> None:
    """Print exactly what the lines and bands show, plus the paired test at each
    team size, so a slide can be checked against the analysis."""
    pct = int(round(float(d.meta.get("ci_level", 0.95)) * 100))
    print("\n" + "=" * 90)
    print(f"  VALUES PLOTTED  (point = mean, band/cap = {pct}% CI; paired test "
          f"from the analysis)")
    print("=" * 90)
    for key in keys:
        cond = ("  [jointly-successful pairs only]" if d.conditioned(key) else "")
        label = d.kpis.get(key, {}).get("label", key)
        print(f"\n  {label}{cond}")
        header = ["Jammers", d.series_labels[0], d.series_labels[1], "p", "n"]
        rows = []
        for scen in d.scenarios:
            r = d.result(key, scen)

            def _cell(slot):
                mean = float(r.get(f"mean_{slot}", float("nan")))
                ci = r.get(f"ci_{slot}") or (float("nan"), float("nan"))
                if not np.isfinite(mean):
                    return ""
                return f"{mean:.3f} [{float(ci[0]):.3f}, {float(ci[1]):.3f}]"

            p = float(r.get("pvalue", float("nan")))
            rows.append([f"{d.jammers(scen):g}", _cell("main"), _cell("base"),
                         ("<0.001" if p < 0.001 else f"{p:.3f}") if np.isfinite(p) else "",
                         d.n_at(key, scen)])
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
        description="Emergent-behaviour sensitivity line charts (fragmentation "
        "and mission duration vs number of jammers, 95% CI bands, NLR colours), "
        "built from the dump written by eval_tools/3.3.4_emergent_team_"
        "behaviour_analysis.py. Runs no rollouts.")
    p.add_argument("--data_name", type=str, default=DATA_NAME,
                   help="Analysis dump to read from eval_results/analysis_data/.")
    p.add_argument("--out_dir", type=str, default=OUT_DIR,
                   help="Directory for the PNGs (relative to the project dir).")
    p.add_argument("--prefix", type=str, default=OUT_PREFIX)
    p.add_argument("--dpi", type=int, default=DPI, help=f"Output DPI (default {DPI}).")
    p.add_argument("--no_titles", action="store_true",
                   help="Drop the figure titles (the y-axis label still names the KPI).")
    p.add_argument("--no_band", action="store_true",
                   help="Draw only the capped error bars, without the filled CI band.")
    p.add_argument("--no_caps", action="store_true",
                   help="Draw only the filled CI band, without the error-bar caps.")
    p.add_argument("--transparent", action="store_true",
                   help="Transparent background (for coloured slide masters).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    d = EmergentData(load_analysis(args.data_name))

    missing = [f["key"] for f in FIGURES if f["key"] not in d.results]
    if missing:
        raise SystemExit(f"KPI(s) {missing} are not in the dump — re-run the "
                         f"analysis with them in its TABLE_KPIS.")
    if not d.scenarios:
        raise SystemExit("The dump has no compositions to plot.")

    print("─" * 78)
    print("  EMERGENT BEHAVIOUR — sensitivity line charts from saved analysis data")
    print("─" * 78)
    print(d.describe())
    print("─" * 78)
    print_values(d, [f["key"] for f in FIGURES])

    out_dir = _resolve_out(args.out_dir)
    xticks = [d.jammers(s) for s in d.scenarios]
    for fig in FIGURES:
        nlr_lines.draw(
            d.series(fig["key"]),
            out_dir / f"{args.prefix}_{fig['slug']}.png",
            xlabel=XLABEL, ylabel=fig["ylabel"],
            title=None if args.no_titles else fig["title"],
            dpi=args.dpi, zero_base=fig["zero_base"],
            band=not args.no_band, caps=not args.no_caps,
            transparent=args.transparent, xticks=xticks)


if __name__ == "__main__":
    main()
