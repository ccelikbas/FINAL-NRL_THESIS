"""3.3.3_ablation_bars.py — ablation-study slide bar charts.

All four policies side by side, in the same slide style as the mission-
performance figure —

    [ Targets Destroyed | Survival Rate ]   [ Mission Duration ]
              one 0–100% axis                  its own axis (steps)

with the 95% CONFIDENCE INTERVAL as a cap on top of every bar. NLR house
colours, assigned in a fixed order so the families read at a glance:

    Comm-FOFE-MAPPO   NLR dark blue        (FOFE on,  comms on)
    MAPPO Baseline    NLR terra            (FOFE off, comms off)
    No-Com            NLR light blue 50%   (FOFE on,  comms off)
    No-FOFE           NLR terra 20%        (FOFE off, comms on)

Saved as a high-resolution PNG straight into this folder:

    presentation/3.3.3_ablation_study_S1.png

The name is STABLE, so re-running this script overwrites that file in place and
the deck always points at the current figure.

This script runs NO rollouts and loads NO checkpoints. Its only input is the
dump that eval_tools/3.3.3_ablation_study_analysis.py wrote at the end of its
run:

    eval_results/analysis_data/3.3.3_ablation_study.json

Re-running THIS script only restyles the figure; refreshing the numbers means
re-running that analysis. The shared slide style lives in nlr_bars.py.

Two things the analysis defines and this figure inherits:
  · Every comparison is PAIRED against Comm-FOFE-MAPPO on common random numbers,
    so the p-values printed to the console are paired one-sided tests, not
    between-group ones. Overlapping caps do not settle a paired comparison —
    --annotate puts the p-value on the figure.
  · Mission duration is each policy's mean over its OWN successful episodes
    (a policy that dies early must not look "faster"), so the duration bars rest
    on different episode counts — printed to the console as "own successes".

Run (repo root, project venv):
  .venv\\Scripts\\python.exe "0_TA_HF_FOFE-MAPPO\\presentation\\3.3.3_ablation_bars.py"
  …--dpi 900 --values --transparent
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
from . import nlr_bars
from .nlr_bars import Bar, Group, Panel

# =====================================================================
#  >>>  INPUT  (which analysis dump to draw)  <<<
# =====================================================================

# Written by eval_tools/3.3.3_ablation_study_analysis.py.       [CLI: --data_name]
DATA_NAME = "3.3.3_ablation_study"

# The RATE KPIs, grouped in the first panel on a shared 0–100% axis.
RATE_KPIS = ["targets", "survival"]
# The KPIs that need their own axis (different unit), one panel each, in order.
OWN_AXIS_KPIS = ["duration"]

# KPIs that ALSO get a figure of their OWN — a single narrow panel, one bar per
# policy, written as "<prefix>_<kpi>_<scenario>.png". Coalition fragmentation is
# on its own because it is an INDEX on a different scale from the rates, and
# because it makes a point of its own on a slide (the blue family splits the
# team; the orange family does not).
SOLO_KPIS = ["fragmentation"]

# Filename slug per solo KPI (keeps the output name short and recognisable).
SOLO_SLUGS = {"fragmentation": "fragmentation"}

# Height (inches) of a solo tile. Shorter than the main figure, so the single
# panel comes out roughly square instead of a narrow column.
SOLO_HEIGHT = 3.4

# Slide wording for the bold labels under each group of bars.
GROUP_LABELS = {
    "targets": "Targets Destroyed",
    "survival": "Survival Rate",
    "duration": "Mission Duration",
    "fragmentation": "Fragmentation",
}

# Legend wording per policy (dump name -> slide name). Names not listed keep the
# name the analysis gave them.
POLICY_LABELS = {
    "Comm-FOFE-MAPPO": "Comm-FOFE-MAPPO",
    "MAPPO Baseline": "MAPPO Baseline",
    "No-Com": "No-Com (FOFE, no comms)",
    "No-FOFE": "No-FOFE (comms, no FOFE)",
}

# y-axis label of the rates panel.
RATE_YLABEL = "Mean"

# Outputs: written straight into the presentation folder under a STABLE name —
# "3.3.3_ablation_study_<scenario>.png" — so the figure keeps one recognisable
# filename and every re-run overwrites it in place.
# (Relative paths are resolved against the project dir 0_TA_...)
OUT_DIR = "presentation"
OUT_PREFIX = "3.3.3_ablation_study"
DPI = 600

# =====================================================================


def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_PKG_DIR / p)


# =====================================================================
#  Reading the analysis dump
# =====================================================================

class AblationData:
    """Thin reader over the 3.3.3 dump. Bars use the DISPLAYED value the
    analysis computed (own-successes mean for the success-conditioned KPIs,
    marginal mean otherwise) with the matching CI — the same numbers as its
    LaTeX table."""

    def __init__(self, doc: dict):
        self.doc = doc
        self.data = doc["data"]
        self.meta = doc.get("meta", {})
        self.results: Dict[str, dict] = self.data["results"]
        self.kpis: Dict[str, dict] = self.data.get("kpis", {})
        self.main: str = self.data.get("main", "")
        # Fixed order: the reference policy first, then the ablations as
        # configured — so colour always means the same policy.
        self.policies: List[str] = [self.main] + list(self.data.get("comparison_order", []))
        self.scenario: str = (self.data.get("scenario") or {}).get("name", "")

    def series_labels(self) -> List[str]:
        return [POLICY_LABELS.get(p, p) for p in self.policies]

    def label(self, key: str) -> str:
        return GROUP_LABELS.get(key) or self.kpis.get(key, {}).get("label", key)

    def unit(self, key: str) -> str:
        return self.kpis.get(key, {}).get("unit", "")

    def conditioned(self, key: str) -> bool:
        return bool(self.kpis.get(key, {}).get("success_conditioned"))

    def cell(self, policy: str, key: str) -> dict:
        return self.results.get(policy, {}).get(key, {})

    def bar(self, policy: str, key: str) -> Bar:
        c = self.cell(policy, key)
        ci = c.get("ci") or (float("nan"), float("nan"))
        return Bar(float(c.get("value", float("nan"))), float(ci[0]), float(ci[1]))

    def group(self, key: str) -> Group:
        return Group(self.label(key), [self.bar(p, key) for p in self.policies])

    def missing_kpis(self) -> List[str]:
        want = RATE_KPIS + OWN_AXIS_KPIS
        have = set(self.kpis) or set(self.cell(self.main, "").keys())
        return [k for k in want if k not in have]

    def describe(self) -> str:
        m = self.meta
        pct = int(round(float(m.get("ci_level", 0.95)) * 100))
        lines = [
            f"  Data               : {self.doc.get('name')} "
            f"(written {self.doc.get('saved_at')} by {self.doc.get('source')})",
            f"  Scenario           : {self.scenario}",
            f"  Reference policy   : {self.main} (all p-values are vs this)",
            f"  Episodes / policy  : {m.get('n_episodes')}   base seed: {m.get('base_seed')}",
            f"  Error bars         : {pct}% CI of the mean (t interval)",
        ]
        pol = m.get("policies") or {}
        for i, p in enumerate(self.policies):
            colour = nlr_bars.SERIES_COLORS[i % len(nlr_bars.SERIES_COLORS)]
            rec = pol.get(p) or {}
            lines.append(f"      {colour}  {POLICY_LABELS.get(p, p):<26} "
                         f"{rec.get('policy_file') or '(none)'}")
        cond = [self.label(k) for k in RATE_KPIS + OWN_AXIS_KPIS if self.conditioned(k)]
        if cond:
            ns = ", ".join(f"{p}: {self.results[p].get('success_n')}" for p in self.policies)
            lines.append(f"  Success-conditioned: {', '.join(cond)} — each policy's OWN "
                         f"successful episodes ({ns})")
        return "\n".join(lines)


# =====================================================================
#  Figure
# =====================================================================

def build_panels(d: AblationData, annotate: bool = False) -> List[Panel]:
    """The slide layout: one grouped 0–100% panel for the rate KPIs, then one
    panel per differently-scaled KPI (mission duration, in steps)."""
    panels = [Panel(groups=[d.group(k) for k in RATE_KPIS], percent=True,
                    ylabel=RATE_YLABEL)]
    for key in OWN_AXIS_KPIS:
        panels.append(Panel(groups=[d.group(key)], percent=False, value_fmt="{:.1f}"))
    if annotate:
        # With four series a per-bar p-value would be unreadable, so the label
        # carries the range of the paired p-values against the reference policy.
        for panel in panels:
            for group in panel.groups:
                key = next(k for k in RATE_KPIS + OWN_AXIS_KPIS
                           if d.label(k) == group.label)
                ps = [float(d.cell(p, key).get("pvalue", float("nan")))
                      for p in d.policies[1:]]
                ps = [p for p in ps if np.isfinite(p)]
                if ps:
                    worst = max(ps)
                    group.label += ("\nall p<0.001" if worst < 0.001
                                    else f"\nmax p={worst:.3f}")
    return panels


# =====================================================================
#  Console echo of the plotted numbers
# =====================================================================

def print_values(d: AblationData) -> None:
    """Print exactly what the bars and caps show, plus the paired test vs the
    reference policy, so a slide can be checked against the analysis."""
    pct = int(round(float(d.meta.get("ci_level", 0.95)) * 100))
    print("\n" + "=" * 88)
    print(f"  VALUES PLOTTED  (bar = mean, cap = {pct}% CI; p and d_z are PAIRED "
          f"vs {d.main})")
    print("=" * 88)
    for key in RATE_KPIS + OWN_AXIS_KPIS:
        cond = "  [each policy's own successful episodes]" if d.conditioned(key) else ""
        print(f"\n  {d.label(key)}{cond}")
        header = ["Policy", f"mean ({pct}% CI)", "p vs ref", "d_z"]
        rows = []
        fmt = "{:.3f}" if d.unit(key) == "rate" else "{:.1f}"
        for p in d.policies:
            b = d.bar(p, key)
            c = d.cell(p, key)
            pv = float(c.get("pvalue", float("nan")))
            dz = float(c.get("dz", float("nan")))
            rows.append([
                POLICY_LABELS.get(p, p),
                (f"{fmt.format(b.mean)} [{fmt.format(b.lo)}, {fmt.format(b.hi)}]"
                 if b.drawable else ""),
                ("(reference)" if p == d.main else
                 ("<0.001" if pv < 0.001 else f"{pv:.3f}") if np.isfinite(pv) else ""),
                f"{dz:.2f}" if np.isfinite(dz) else "",
            ])
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
        description="Ablation-study slide bar charts (targets destroyed / "
        "survival rate / mission duration, 95% CI caps, NLR colours) for all "
        "policies, built from the dump written by eval_tools/3.3.3_ablation_"
        "study_analysis.py. Runs no rollouts.")
    p.add_argument("--data_name", type=str, default=DATA_NAME,
                   help="Analysis dump to read from eval_results/analysis_data/.")
    p.add_argument("--policies", type=str, default=None,
                   help="Comma-separated subset/order of policies to draw "
                        "(default: reference first, then the dump's order).")
    p.add_argument("--out_dir", type=str, default=OUT_DIR,
                   help="Directory for the PNGs (relative to the project dir).")
    p.add_argument("--prefix", type=str, default=OUT_PREFIX)
    p.add_argument("--dpi", type=int, default=DPI, help=f"Output DPI (default {DPI}).")
    p.add_argument("--values", action="store_true",
                   help="Print each mean on top of its bar.")
    p.add_argument("--annotate", action="store_true",
                   help="Add the paired p-value range under each KPI label.")
    p.add_argument("--title", action="store_true",
                   help="Add an 'Ablation study (Sx)' title above the figure.")
    p.add_argument("--no_legend", action="store_true",
                   help="Drop the legend everywhere (only safe when the figure "
                        "sits beside one that carries it — colour alone must not "
                        "be the only thing identifying a policy).")
    p.add_argument("--solo_legend", action="store_true",
                   help="Give the solo single-KPI tiles their own legend, for "
                        "when one is shown without the main figure beside it.")
    p.add_argument("--transparent", action="store_true",
                   help="Transparent background (for coloured slide masters).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    d = AblationData(load_analysis(args.data_name))

    if args.policies:
        wanted = [p.strip() for p in args.policies.split(",") if p.strip()]
        unknown = [p for p in wanted if p not in d.results]
        if unknown:
            raise SystemExit(f"Policy/policies {unknown} are not in the dump "
                             f"(it has: {', '.join(d.policies)}).")
        d.policies = wanted
    missing = d.missing_kpis()
    if missing:
        raise SystemExit(f"KPI(s) {missing} are not in the dump — re-run the "
                         f"analysis with them in its TABLE_KPIS.")
    solo = [k for k in SOLO_KPIS if k in d.kpis]
    for k in SOLO_KPIS:
        if k not in solo:
            print(f"  ! '{k}' is not in the dump — its solo figure is skipped.")

    print("─" * 78)
    print("  ABLATION STUDY — slide bar charts from saved analysis data")
    print("─" * 78)
    print(d.describe())
    print("─" * 78)
    print_values(d)

    out_dir = _resolve_out(args.out_dir)
    scen = d.scenario or "all"
    nlr_bars.draw(build_panels(d, args.annotate), d.series_labels(),
                  out_dir / f"{args.prefix}_{scen}.png",
                  dpi=args.dpi, value_labels=args.values,
                  transparent=args.transparent, legend=not args.no_legend,
                  title=(f"Ablation study ({d.scenario})" if args.title else None))

    # One extra figure per solo KPI: a compact single-panel tile, one bar per
    # policy. It is meant to sit BESIDE the figure above, which already carries
    # the legend, so it ships without one (four long labels on a narrow tile
    # would dwarf the bars) — pass --solo_legend when it stands alone, or the
    # colours are the only thing naming the policies.
    for key in solo:
        panel = Panel(groups=[d.group(key)], percent=False, value_fmt="{:.3f}")
        nlr_bars.draw([panel], d.series_labels(),
                      out_dir / f"{args.prefix}_{SOLO_SLUGS.get(key, key)}_{scen}.png",
                      dpi=args.dpi, value_labels=args.values,
                      transparent=args.transparent, height=SOLO_HEIGHT,
                      legend=args.solo_legend and not args.no_legend,
                      title=(f"{d.label(key)} ({d.scenario})" if args.title else None))


if __name__ == "__main__":
    main()
