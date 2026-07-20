"""emergent_team_behaviour_analysis.py — team-composition table (paired t-test).

Reuses the evaluation engine in evaluate_policy.py (PAIRED rollouts via common
random numbers) and compares a COMPLETE policy against a BASELINE policy across
TEAM COMPOSITIONS, for four KPIs:

    Targets destroyed · Survival · Duration · Fragmentation

Each KPI has a Complete column and a "Baseline ($p$)" column, the latter showing
the baseline mean with the one-sided PAIRED t-test p-value vs Complete in
parentheses, e.g. "0.774 (0.031)". Rows are the compositions in EVAL_SCENARIOS
(e.g. 2S2J, 2S3J, 2S4J).

STATISTICS (same methodology as 3.3.2): the raw per-episode KPIs are not normal,
but the SAMPLE MEAN is (CLT at n=600, per statistical_analysis.py), which
justifies a parametric PAIRED t-test on the per-episode differences
d_i = KPI_complete_i − KPI_baseline_i (matched by common random numbers),
one-sided in each KPI's better direction (targets/survival: greater; duration:
less; fragmentation: two-sided). Duration is conditioned on jointly-successful
pairs so a method whose agents die early can't look "faster".

Outputs: a LaTeX table + console table (the analysis), plus TWO sensitivity plots
(a team-size sensitivity analysis, overwritten each run):
  1. rates — targets / survival / fragmentation vs number of jammers, and
  2. duration vs number of jammers,
both with Complete and Baseline and 95% CI error bars at each point.

Run (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\emergent_team_behaviour_analysis.py
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\emergent_team_behaviour_analysis.py --n_episodes 100
"""
from __future__ import annotations

import argparse
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR.parent
_PKG_NAME = "fofe_mappo"
if __package__ in (None, ""):
    sys.path.insert(0, str(_PKG_DIR.parent))
    if _PKG_NAME not in sys.modules:
        _pkg = types.ModuleType(_PKG_NAME)
        _pkg.__path__ = [str(_PKG_DIR), str(_THIS_DIR)]
        _pkg.__package__ = _PKG_NAME
        _pkg.__file__ = str(_THIS_DIR / "__init__.py")
        sys.modules[_PKG_NAME] = _pkg
    __package__ = _PKG_NAME

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel
from scipy.stats import t as _student_t

# Reuse the evaluation engine (paired rollouts, KPI defs). The engine also runs
# Wilcoxon internally, but we ignore that and compute the PAIRED t-test locally.
from .evaluate_policy import (
    PolicyInput, evaluate_comparison, _LoadedCheckpoint, _resolve_policy_path,
    _print_policy_diagnostics, _ALT, KPIS,
)
from .run_curriculum import CurriculumSection
# NLR house palette (auto-applied to matplotlib on import).
from .nlr_style import NLR_PRIMARY, NLR_ACCENT

# =====================================================================
#  >>>  TEST CONFIG  (how many runs, seeding, significance)  <<<
# =====================================================================

# Number of RUNS = paired episodes per policy per composition (test N and the
# "n=" printed in the caption).                            [CLI: --n_episodes]
N_EPISODES = 600
# Parallel envs per rollout chunk (lower this if you hit GPU OOM). [CLI: --chunk]
CHUNK_EPISODES = 300
# Base RNG seed; both policies share it so episodes are paired 1:1.  [CLI: --seed]
BASE_SEED = 42
# Significance level for the tests.                                 [CLI: --alpha]
ALPHA = 0.05
# Confidence level for the reported intervals (two-sided).            [CLI: --ci]
CI_LEVEL = 0.95

# =====================================================================
#  >>>  POLICIES  (Complete = reference column, Baseline = compared)  <<<
# =====================================================================

MAIN_POLICY = PolicyInput(
    name="Complete",
    policy_file="runs/FINALV2/complete_stage7of8_DR_j2-4_k0_25.pt",
    communicate=True,
)
COMPARISON_POLICIES: List[PolicyInput] = [
    PolicyInput(
        name="Baseline",
        policy_file="runs/FINALV2/Final_Baseline_Cont_4.pt",
        communicate=False,
    ),
]

# =====================================================================
#  >>>  TEAM COMPOSITIONS  (the ROWS of the table)  <<<
#  Fixed world; only the team size changes per row. The row label is the section
#  `name` (printed in math mode, e.g. $2S2J$).
# =====================================================================

EVAL_SCENARIOS: List[CurriculumSection] = [
    CurriculumSection(
        name="2S2J",
        n_iters=1,
        n_strikers=2, n_jammers=2,
        n_known_targets=(2, 4), n_unknown_targets=0,
        n_known_radars=(4, 6), n_unknown_radars=0,
        radar_kill_probability=0.25,
        scenario="S2",
    ),
    CurriculumSection(
        name="2S3J",
        n_iters=1,
        n_strikers=2, n_jammers=3,
        n_known_targets=(2, 4), n_unknown_targets=0,
        n_known_radars=(4, 6), n_unknown_radars=0,
        radar_kill_probability=0.25,
        scenario="S2",
    ),
    CurriculumSection(
        name="2S4J",
        n_iters=1,
        n_strikers=2, n_jammers=4,
        n_known_targets=(2, 4), n_unknown_targets=0,
        n_known_radars=(4, 6), n_unknown_radars=0,
        radar_kill_probability=0.25,
        scenario="S2",
    ),
]

# KPIs shown (in column order). Keys must exist in evaluate_policy.KPIS.
TABLE_KPIS = ["targets", "survival", "duration", "fragmentation"]

# KPIs restricted to SUCCESSFUL episodes. For these, means, CIs and the (one-sided,
# directional) paired t-test use only episode pairs where BOTH policies completed
# the mission. Duration is only comparable between missions that actually finished.
# (Fragmentation stays over all episodes and is tested two-sided.)
SUCCESS_CONDITIONED_KPIS = {"duration"}

# Per-episode success flag: "completion" == mission_complete == (~target_alive).all()
# == "all targets destroyed" (see environment.py).
SUCCESS_KEY = "completion"

# Outputs (relative paths resolved against the project dir 0_TA_...). The two
# sensitivity plots use STABLE names (overwritten on every run).
OUT_PATH = "eval_results/emergent_team_behaviour.tex"
RATES_PLOT_OUT = "eval_results/emergent_team_rates.png"        # targets/survival/frag vs #jammers
DURATION_PLOT_OUT = "eval_results/emergent_team_duration.png"  # duration vs #jammers
DPI = 200

# =====================================================================

_KPI_BY_KEY = {s.key: s for s in KPIS}


# =====================================================================
#  Formatting helpers
# =====================================================================

def _fmt_val(key: str, mean: float) -> str:
    if not np.isfinite(mean):
        return "--"
    return _KPI_BY_KEY[key].fmt.format(mean)


def _fmt_signed(key: str, val: float) -> str:
    if not np.isfinite(val):
        return "--"
    s = _KPI_BY_KEY[key].fmt.format(val)
    return s if s.startswith("-") else "+" + s


def _fmt_ci(key: str, lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "[--, --]"
    f = _KPI_BY_KEY[key].fmt
    return f"[{f.format(lo)}, {f.format(hi)}]"


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _stars(p: float, alpha: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < alpha:
        return "*"
    return ""


# =====================================================================
#  Paired one-sided t-test (computed from the raw paired `samples`)
# =====================================================================

def _joint_success_mask(scn, main, base, samples) -> np.ndarray:
    """Boolean mask over episodes where BOTH policies completed the mission."""
    sc = np.asarray(samples[scn.name][main.name][SUCCESS_KEY], dtype=float) >= 0.5
    sb = np.asarray(samples[scn.name][base.name][SUCCESS_KEY], dtype=float) >= 0.5
    return sc & sb


def _ci_mean(vals: np.ndarray, tcrit: float) -> Tuple[float, float]:
    """Two-sided t confidence interval for the mean of `vals`."""
    n = vals.size
    if n < 2:
        m = float(vals[0]) if n == 1 else float("nan")
        return (m, m)
    sem = float(vals.std(ddof=1)) / np.sqrt(n)
    mu = float(vals.mean())
    return (mu - tcrit * sem, mu + tcrit * sem)


def _paired_ttest(m: np.ndarray, b: np.ndarray, direction: str,
                  ci_level: float) -> Dict[str, Any]:
    """PAIRED one-sided t-test on d = m − b, plus per-method and difference CIs.
    The difference is ORIENTED so a positive value means the main policy is better
    (for 'lower' KPIs the sign is flipped)."""
    m = np.asarray(m, dtype=float)
    b = np.asarray(b, dtype=float)
    n = m.size
    sign = -1.0 if direction == "lower" else 1.0
    alt = _ALT[direction]
    out: Dict[str, Any] = {
        "n": int(n), "alternative": alt,
        "mean_main": float("nan"), "mean_base": float("nan"),
        "mean_diff_oriented": float("nan"),
        "ci_main": (float("nan"), float("nan")),
        "ci_base": (float("nan"), float("nan")),
        "ci_diff_oriented": (float("nan"), float("nan")),
        "t": float("nan"), "pvalue": float("nan"),
    }
    if n < 2:
        if n == 1:
            out["mean_main"], out["mean_base"] = float(m[0]), float(b[0])
            out["mean_diff_oriented"] = sign * float(m[0] - b[0])
        return out

    d = m - b
    tcrit = float(_student_t.ppf(1.0 - (1.0 - ci_level) / 2.0, n - 1))
    mean_d = float(d.mean())
    out["mean_main"], out["mean_base"] = float(m.mean()), float(b.mean())
    out["mean_diff_oriented"] = sign * mean_d
    out["ci_main"], out["ci_base"] = _ci_mean(m, tcrit), _ci_mean(b, tcrit)
    d_lo, d_hi = _ci_mean(d, tcrit)
    out["ci_diff_oriented"] = tuple(sorted((sign * d_lo, sign * d_hi)))

    sd = float(d.std(ddof=1))
    if sd == 0.0:
        if mean_d == 0.0:
            out["t"], out["pvalue"] = 0.0, 1.0
        else:
            better = (mean_d > 0) if alt == "greater" else (
                mean_d < 0) if alt == "less" else True
            out["t"] = float("inf") if mean_d > 0 else float("-inf")
            out["pvalue"] = 0.0 if better else 1.0
        return out
    res = ttest_rel(m, b, alternative=alt)
    out["t"], out["pvalue"] = float(res.statistic), float(res.pvalue)
    return out


def _paired_result(scn, key, main, base, samples, ci_level) -> Dict[str, Any]:
    """Full paired result for one (composition, KPI), applying success-conditioning
    for the KPIs in SUCCESS_CONDITIONED_KPIS."""
    direction = _KPI_BY_KEY[key].direction
    mv = np.asarray(samples[scn.name][main.name][key], dtype=float)
    if base is None:
        mask = np.isfinite(mv)
        if key in SUCCESS_CONDITIONED_KPIS:
            sm = np.asarray(samples[scn.name][main.name][SUCCESS_KEY], dtype=float) >= 0.5
            mask &= sm[:mv.size]
        m = mv[mask]
        n = m.size
        tcrit = float(_student_t.ppf(1.0 - (1.0 - ci_level) / 2.0, max(n - 1, 1)))
        return {"n": int(n),
                "mean_main": float(m.mean()) if n else float("nan"),
                "mean_base": float("nan"), "mean_diff_oriented": float("nan"),
                "ci_main": _ci_mean(m, tcrit) if n else (float("nan"), float("nan")),
                "ci_base": (float("nan"), float("nan")),
                "ci_diff_oriented": (float("nan"), float("nan")),
                "t": float("nan"), "pvalue": float("nan")}

    bv = np.asarray(samples[scn.name][base.name][key], dtype=float)
    n = min(mv.size, bv.size)
    mv, bv = mv[:n], bv[:n]
    mask = np.isfinite(mv) & np.isfinite(bv)
    if key in SUCCESS_CONDITIONED_KPIS:
        mask &= _joint_success_mask(scn, main, base, samples)[:n]
    return _paired_ttest(mv[mask], bv[mask], direction, ci_level)


def compute_results(scenarios, main, base, samples, ci_level) -> Dict[str, Dict[str, dict]]:
    return {key: {scn.name: _paired_result(scn, key, main, base, samples, ci_level)
                  for scn in scenarios}
            for key in TABLE_KPIS}


# =====================================================================
#  LaTeX table
# =====================================================================

def _complete_baseline_cells(scn, key, results) -> Tuple[str, str]:
    """(complete, 'baseline (p)') strings for one (composition, KPI). The baseline
    cell is '' when there is no comparison policy."""
    r = results[key][scn.name]
    c = _fmt_val(key, r["mean_main"])
    if not np.isfinite(r["mean_base"]):
        return c, ""
    return c, f"{_fmt_val(key, r['mean_base'])} ({_fmt_p(r['pvalue'])})"


def _duration_n_phrase(scenarios, results) -> str:
    """Caption phrase reporting the effective episode count for Duration (the
    jointly-successful pairs), auto-filled for copy-paste. One number if all
    compositions match, else listed per composition."""
    ns = [(scn.name, int(results["duration"][scn.name]["n"])) for scn in scenarios]
    uniq = {n for _, n in ns}
    if len(uniq) == 1:
        return f"$n_\\mathrm{{dur}}={ns[0][1]}$"
    return "$n_\\mathrm{dur}$: " + ", ".join(f"{name}={n}" for name, n in ns)


_HEADER = r"""\begin{table}[htbp]
    \centering
    \caption{Emergent team behaviour across compositions (n=__N__ paired
    episodes; one-sided \emph{paired} $t$-test on common-random-number pairs;
    each ``Baseline ($p$)'' cell is the baseline mean with the paired-$t$
    $p$-value vs Complete; Duration is conditioned on jointly-successful
    episodes, __DURN__).}
    \label{tab:team-composition-comparison}
    \small
    \resizebox{\textwidth}{!}{%
        \begin{tabular}{l*{8}{c}}
            \toprule
            \textbf{Team Size}
            & \multicolumn{2}{c}{\textbf{Targets destroyed}}
            & \multicolumn{2}{c}{\textbf{Survival}}
            & \multicolumn{2}{c}{\textbf{Duration}}
            & \multicolumn{2}{c}{\textbf{Fragmentation}} \\

            \cmidrule(lr){2-3}
            \cmidrule(lr){4-5}
            \cmidrule(lr){6-7}
            \cmidrule(lr){8-9}

            & Complete & Baseline ($p$)
            & Complete & Baseline ($p$)
            & Complete & Baseline ($p$)
            & Complete & Baseline ($p$) \\
            \midrule

"""

_FOOTER = r"""            \bottomrule
        \end{tabular}%
    }
\end{table}
"""


def build_latex(scenarios, results, n_episodes) -> str:
    blocks = []
    for scn in scenarios:
        t = _complete_baseline_cells(scn, "targets", results)
        s = _complete_baseline_cells(scn, "survival", results)
        d = _complete_baseline_cells(scn, "duration", results)
        f = _complete_baseline_cells(scn, "fragmentation", results)
        blocks.append(
            f"            ${scn.name}$\n"
            f"            & {t[0]} & {t[1]}\n"
            f"            & {s[0]} & {s[1]}\n"
            f"            & {d[0]} & {d[1]}\n"
            f"            & {f[0]} & {f[1]} \\\\\n"
        )
    header = (_HEADER.replace("__N__", str(n_episodes))
              .replace("__DURN__", _duration_n_phrase(scenarios, results)))
    return header + "\n".join(blocks) + _FOOTER


# =====================================================================
#  Console table
# =====================================================================

def print_console_tables(scenarios, main, base, results, ci_level, alpha) -> None:
    pct = int(round(ci_level * 100))
    print("\n" + "=" * 96)
    print(f"  EMERGENT TEAM BEHAVIOUR — one-sided PAIRED t-test  "
          f"(CRN pairs; {pct}% CIs; * p<{alpha:g} ** p<0.01 *** p<0.001)")
    print("=" * 96)
    for key in TABLE_KPIS:
        spec = _KPI_BY_KEY[key]
        better = ("higher is better" if spec.direction == "higher"
                  else "lower is better" if spec.direction == "lower"
                  else "two-sided")
        cond = "  [successful pairs only]" if key in SUCCESS_CONDITIONED_KPIS else ""
        headers = ["Composition", f"{main.name} ({pct}% CI)"]
        if base is not None:
            headers += [f"{base.name} ({pct}% CI)",
                        f"Δ {main.name}−{base.name} ({pct}% CI)", "p", "n"]
        rows: List[List[str]] = []
        for scn in scenarios:
            r = results[key][scn.name]
            row = [scn.name,
                   f"{_fmt_val(key, r['mean_main'])} {_fmt_ci(key, *r['ci_main'])}"]
            if base is not None:
                row += [
                    f"{_fmt_val(key, r['mean_base'])} {_fmt_ci(key, *r['ci_base'])}",
                    f"{_fmt_signed(key, r['mean_diff_oriented'])} "
                    f"{_fmt_ci(key, *r['ci_diff_oriented'])}",
                    _fmt_p(r["pvalue"]) + _stars(r["pvalue"], alpha), str(r["n"])]
            rows.append(row)

        widths = [len(h) for h in headers]
        for row in rows:
            for c, cell in enumerate(row):
                widths[c] = max(widths[c], len(cell))

        def _fmt_row(cells):
            return "  ".join(cell.ljust(widths[c]) if c == 0 else cell.rjust(widths[c])
                             for c, cell in enumerate(cells))

        print(f"\n  KPI: {spec.label}  ({better}){cond}")
        print("  " + _fmt_row(headers))
        print("  " + "  ".join("-" * w for w in widths))
        for row in rows:
            print("  " + _fmt_row(row))
    print()


# =====================================================================
#  Sensitivity plots (KPI vs number of jammers) — the visual deliverable
# =====================================================================

def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_PKG_DIR / p)


def _jammer_count(section) -> float:
    """Number of jammers for a composition (the x-axis of the sensitivity plots).
    Reads section.n_jammers (scalar → itself; (lo,hi) → midpoint); falls back to
    the digits before 'J' in the section name (e.g. '2S3J' → 3)."""
    nj = getattr(section, "n_jammers", None)
    try:
        return float(int(nj))
    except (TypeError, ValueError):
        pass
    if isinstance(nj, (tuple, list)) and nj:
        return float(np.mean([float(x) for x in nj]))
    m = re.search(r"(\d+)\s*[Jj]", getattr(section, "name", ""))
    return float(m.group(1)) if m else float("nan")


# Consistent policy families across BOTH figures: Complete = shades of BLUE,
# Baseline = shades of ORANGE, so policy is obvious (blue vs orange) side by side.
# In the rates plot each KPI gets its OWN shade + marker (dark→light within the
# family) so the three lines are easy to tell apart. Line style also encodes
# policy (Complete solid, Baseline dashed).
RATE_KPIS = ["targets", "survival", "fragmentation"]
_POLICY_LS = ["-", "--"]                      # Complete solid, Baseline dashed
KPI_MARKER = {"targets": "o", "survival": "s", "fragmentation": "^", "duration": "D"}
# per-KPI shades (dark → light) within each policy's colour family.
_COMPLETE_KPI_COLOR = {"targets": "#004d7d", "survival": "#1f88bd", "fragmentation": "#5bc0eb"}
_BASELINE_KPI_COLOR = {"targets": "#a5510e", "survival": "#ed7914", "fragmentation": "#f6b26b"}
_DURATION_COLOR = [NLR_PRIMARY, NLR_ACCENT]   # duration plot: Complete blue, Baseline orange
_MARKER_SIZE = 9


def _sorted_scen_x(scenarios):
    scen = sorted(scenarios, key=_jammer_count)
    return scen, np.array([_jammer_count(s) for s in scen], dtype=float)


def _mean_ci_keys(pi: int):
    """(mean_key, ci_key) for policy index pi (0 = Complete/main, else Baseline)."""
    return ("mean_main", "ci_main") if pi == 0 else ("mean_base", "ci_base")


def _rate_color(pi: int, key: str) -> str:
    return (_COMPLETE_KPI_COLOR if pi == 0 else _BASELINE_KPI_COLOR).get(key, NLR_PRIMARY)


def plot_rates_sensitivity(scenarios, main, base, results, ci_level, out_png) -> None:
    """Plot 1: targets / survival / fragmentation RATES vs number of jammers, for
    Complete (blue shades) and Baseline (orange shades), with 95% CI error bars.
    Each KPI = its own shade + marker; line style also encodes the policy."""
    scen, xs = _sorted_scen_x(scenarios)
    pct = int(round(ci_level * 100))
    policies = [main] + ([base] if base is not None else [])

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for key in RATE_KPIS:
        if key not in results:
            continue
        mk = KPI_MARKER.get(key, "o")
        for pi in range(len(policies)):
            mkey, ckey = _mean_ci_keys(pi)
            color = _rate_color(pi, key)
            ys = np.array([results[key][s.name][mkey] for s in scen], dtype=float)
            err = np.array([(results[key][s.name][ckey][1]
                             - results[key][s.name][ckey][0]) / 2.0 for s in scen], dtype=float)
            ax.errorbar(xs, ys, yerr=err, color=color, ls=_POLICY_LS[pi % len(_POLICY_LS)],
                        marker=mk, ms=_MARKER_SIZE, lw=1.9, capsize=4,
                        mec="white", mew=0.7)

    ax.set_xlabel("Number of jammers (strikers = 2)", fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)}" for x in xs])
    ax.grid(True, alpha=0.3)

    # One explicit legend: each (policy · KPI) with its exact colour/marker/style.
    handles, labels = [], []
    for pi, pol in enumerate(policies):
        for key in RATE_KPIS:
            if key not in results:
                continue
            handles.append(Line2D([0], [0], color=_rate_color(pi, key),
                                  ls=_POLICY_LS[pi % len(_POLICY_LS)], lw=2.0,
                                  marker=KPI_MARKER[key], ms=8, mec="white", mew=0.7))
            labels.append(f"{pol.name} · {_KPI_BY_KEY[key].label}")
    leg = ax.legend(handles, labels, title="policy · KPI", loc="center left",
                    bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=True)
    ax.set_title(f"Team-size sensitivity — rates ({pct}% CI)", fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", bbox_extra_artists=[leg])
    plt.close(fig)
    print(f"Saved rates sensitivity plot to: {out_png}")


def plot_duration_sensitivity(scenarios, main, base, results, ci_level, out_png) -> None:
    """Plot 2: Duration (steps) vs number of jammers, for Complete (blue) and
    Baseline (orange), with 95% CI error bars — same policy colour families and
    line styles as the rates plot. Duration is over jointly-successful pairs."""
    scen, xs = _sorted_scen_x(scenarios)
    pct = int(round(ci_level * 100))
    policies = [main] + ([base] if base is not None else [])
    mk = KPI_MARKER["duration"]

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for pi, pol in enumerate(policies):
        mkey, ckey = _mean_ci_keys(pi)
        color = _DURATION_COLOR[pi % len(_DURATION_COLOR)]
        ys = np.array([results["duration"][s.name][mkey] for s in scen], dtype=float)
        err = np.array([(results["duration"][s.name][ckey][1]
                         - results["duration"][s.name][ckey][0]) / 2.0 for s in scen], dtype=float)
        ax.errorbar(xs, ys, yerr=err, color=color, ls=_POLICY_LS[pi % len(_POLICY_LS)],
                    marker=mk, ms=_MARKER_SIZE, lw=1.9, capsize=4, label=pol.name,
                    mec="white", mew=0.7)

    ax.set_xlabel("Number of jammers (strikers = 2)", fontsize=11)
    ax.set_ylabel("Duration (steps, successful pairs)", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)}" for x in xs])
    ax.grid(True, alpha=0.3)
    ax.legend(title="policy", fontsize=10, frameon=True)
    ax.set_title(f"Team-size sensitivity — duration ({pct}% CI)", fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved duration sensitivity plot to: {out_png}")


# =====================================================================
#  Config banner
# =====================================================================

def _print_config(main, comparison, scenarios, n_episodes, base_seed, alpha,
                  ci_level) -> None:
    print("─" * 78)
    print("  EMERGENT TEAM BEHAVIOUR — PAIRED t-TEST (common random numbers)")
    print("─" * 78)
    print(f"  Main policy        : {main.name}  [{main.policy_file}]")
    if comparison:
        print("  Comparison policies: "
              + ", ".join(f"{p.name} [{p.policy_file}]" for p in comparison))
    print(f"  Compositions       : {len(scenarios)}  "
          f"({', '.join(s.name for s in scenarios)})")
    print(f"  Test               : one-sided PAIRED t-test on d=main−baseline "
          f"(paired by common seed)")
    print(f"  Normality basis    : CLT on the sample mean (see statistical_analysis.py)")
    print(f"  N (paired episodes): {n_episodes}   base seed: {base_seed}")
    print(f"  Significance / CI  : alpha={alpha:g}   {int(round(ci_level*100))}% CIs")
    print("  Hypotheses (H1, main vs comparison):")
    for key in TABLE_KPIS:
        spec = _KPI_BY_KEY[key]
        if spec.direction == "two-sided":
            print(f"      {spec.label:18s} main ≠ comparison  (alternative='two-sided')")
        else:
            rel = "higher" if spec.direction == "higher" else "lower"
            print(f"      {spec.label:18s} main {rel:6s}  (alternative='{_ALT[spec.direction]}')")
    if SUCCESS_CONDITIONED_KPIS:
        print(f"  Success-conditioned: {', '.join(sorted(SUCCESS_CONDITIONED_KPIS))} "
              f"(jointly-successful pairs only)")
    print("─" * 78)


# =====================================================================
#  CLI
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Team-composition LaTeX table + console table "
        "(Targets/Survival/Duration/Fragmentation) for a Complete vs Baseline "
        "policy across EVAL_SCENARIOS, using a one-sided PAIRED t-test.")
    p.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                   help="Default .pt for any policy that leaves policy_file=None.")
    p.add_argument("--n_episodes", type=int, default=N_EPISODES)
    p.add_argument("--chunk", type=int, default=CHUNK_EPISODES)
    p.add_argument("--seed", type=int, default=BASE_SEED)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--ci", type=float, default=CI_LEVEL,
                   help=f"Confidence level for intervals (default: {CI_LEVEL}).")
    p.add_argument("--device", type=str, default=None, metavar="DEVICE")
    p.add_argument("--out", type=str, default=OUT_PATH)
    p.add_argument("--rates_out", type=str, default=RATES_PLOT_OUT)
    p.add_argument("--duration_out", type=str, default=DURATION_PLOT_OUT)
    p.add_argument("--no_plots", action="store_true", help="Skip the sensitivity plots.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not EVAL_SCENARIOS:
        raise RuntimeError("EVAL_SCENARIOS is empty — define at least one composition.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be strictly between 0 and 1.")
    if not 0.0 < args.ci < 1.0:
        raise ValueError("--ci must be strictly between 0 and 1.")

    default_ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    _print_config(MAIN_POLICY, COMPARISON_POLICIES, EVAL_SCENARIOS,
                  args.n_episodes, args.seed, args.alpha, args.ci)
    cache: Dict[str, _LoadedCheckpoint] = {}
    _print_policy_diagnostics(MAIN_POLICY, COMPARISON_POLICIES, EVAL_SCENARIOS,
                              default_ckpt_path, device, cache)

    _summary, _tests, samples = evaluate_comparison(
        EVAL_SCENARIOS, MAIN_POLICY, COMPARISON_POLICIES, default_ckpt_path,
        n_episodes=args.n_episodes, chunk=args.chunk, base_seed=args.seed,
        p_adjust="none", device=device, cache=cache,
    )

    base = COMPARISON_POLICIES[0] if COMPARISON_POLICIES else None
    results = compute_results(EVAL_SCENARIOS, MAIN_POLICY, base, samples, args.ci)

    # Effective (jointly-successful) n for the success-conditioned KPIs.
    if SUCCESS_CONDITIONED_KPIS and base is not None:
        print(f"[note] {', '.join(sorted(SUCCESS_CONDITIONED_KPIS))} computed over "
              f"jointly-successful episodes only (both policies completed the mission):")
        for scn in EVAL_SCENARIOS:
            n_ok = int(_joint_success_mask(scn, MAIN_POLICY, base, samples).sum())
            print(f"         {scn.name}: {n_ok}/{args.n_episodes} pairs")

    print_console_tables(EVAL_SCENARIOS, MAIN_POLICY, base, results, args.ci, args.alpha)

    latex = build_latex(EVAL_SCENARIOS, results, args.n_episodes)
    print(latex)
    out = _resolve_out(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table to: {out}")

    if not args.no_plots:
        plot_rates_sensitivity(EVAL_SCENARIOS, MAIN_POLICY, base, results,
                               args.ci, _resolve_out(args.rates_out))
        plot_duration_sensitivity(EVAL_SCENARIOS, MAIN_POLICY, base, results,
                                  args.ci, _resolve_out(args.duration_out))


if __name__ == "__main__":
    main()
