"""mission_performance_analysis.py — mission-performance table + CI dashboard.

Reuses the evaluation engine in evaluate_policy.py (PAIRED rollouts: both methods
are evaluated on the SAME per-episode initial conditions via common random
numbers) and compares a COMPLETE policy against a BASELINE policy on a set of
scenarios, for three KPIs:

    Targets destroyed · Survival · Duration     (each: Complete, Baseline, p-value)

──────────────────────────────────────────────────────────────────────────────
STATISTICAL METHODOLOGY  (paired one-sided t-test)
──────────────────────────────────────────────────────────────────────────────
The RAW per-episode KPIs are not normal (completion is binary, targets/survival
are bounded fractions, duration is right-skewed). But statistical_analysis.py
showed — via the bootstrap sampling distribution of the mean — that the SAMPLE
MEAN of each KPI is approximately normal (central limit theorem at n=600). That
normality justifies a PARAMETRIC test on the means.

Because the environment is fully seeded, episode i faces the SAME randomised
layout for both methods (common random numbers), so (complete_i, baseline_i) is a
MATCHED PAIR. We therefore use a PAIRED t-test on the per-episode differences
    d_i = KPI_complete_i − KPI_baseline_i ,
one-sided in each KPI's "better" direction (targets/survival: Complete greater;
duration: Complete less). Pairing removes the shared between-layout variance, so
the test has more power than an unpaired comparison.

Duration is conditioned on JOINTLY-SUCCESSFUL pairs (both methods destroyed all
targets): duration is only comparable between missions that actually finished, so
a method whose agents die early cannot look "faster". Its effective n < 600.

──────────────────────────────────────────────────────────────────────────────
OUTPUTS
──────────────────────────────────────────────────────────────────────────────
  1. a LaTeX table  (Complete / Baseline / one-sided paired-t p per KPI),
  2. the same table printed to the console (with 95% CIs and the mean diff), and
  3. a CI dashboard: per-method mean ± 95% CI (eyeball whether they OVERLAP) plus
     a companion panel of the paired mean-difference ± 95% CI vs a zero line (the
     rigorous visual analog of the paired test — overlapping marginal CIs do NOT
     imply a null result for paired data, but a difference CI that excludes zero does).

Run (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\mission_performance_analysis.py
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\mission_performance_analysis.py --n_episodes 100
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from scipy.stats import ttest_rel
from scipy.stats import t as _student_t

# Reuse the evaluation engine (paired rollouts, KPI defs, path resolution) from
# evaluate_policy. The engine also runs Wilcoxon internally, but we ignore that
# here and compute the PAIRED t-test locally from the raw paired `samples`.
from .evaluate_policy import (
    PolicyInput, evaluate_comparison, _LoadedCheckpoint, _resolve_policy_path,
    _print_policy_diagnostics, _ALT, KPIS,
)
from .run_curriculum import CurriculumSection
# NLR house palette (auto-applied to matplotlib on import).
from .nlr_style import NLR_PRIMARY, NLR_ACCENT, NLR_SECONDARY, NLR_GRAY, NLR_DARKGRAY

# =====================================================================
#  >>>  TEST CONFIG  (how many runs, seeding, significance)  <<<
# =====================================================================

# Number of RUNS = paired episodes per policy per scenario (this is the test N,
# and the "n=" printed in the table caption).             [CLI: --n_episodes]
N_EPISODES = 600
# Parallel envs per rollout chunk (lower this if you hit GPU OOM). [CLI: --chunk]
CHUNK_EPISODES = 300
# Base RNG seed; both policies share it so episodes are paired 1:1.  [CLI: --seed]
BASE_SEED = 42
# Significance level for the tests.                                 [CLI: --alpha]
ALPHA = 0.05
# Confidence level for the reported / plotted intervals (two-sided). [CLI: --ci]
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
        policy_file="runs/FINALV2/Final_Baseline_Cont_3.pt",
        communicate=False,
    ),
]

# =====================================================================
#  >>>  SCENARIOS  (the ROWS of the table)  <<<
#  Same CurriculumSection format as evaluate_policy.py. The row label is the
#  section `name` (printed in math mode, e.g. $S1$). Communication is per-policy.
# =====================================================================

EVAL_SCENARIOS: List[CurriculumSection] = [
    CurriculumSection(
        name="S1",
        n_iters=1,
        n_strikers=2, n_jammers=4,
        n_known_targets=(2, 4), n_unknown_targets=0,
        n_known_radars=(4, 6), n_unknown_radars=0,
        radar_kill_probability=0.25,
        scenario="S2",
    ),
    # CurriculumSection(
    #     name="S2",
    #     n_iters=1,
    #     n_strikers=2, n_jammers=(2, 4),
    #     n_known_targets=(2, 4), n_unknown_targets=0,
    #     n_known_radars=(4, 6), n_unknown_radars=0,
    #     radar_kill_probability=0.25,
    #     scenario="S2",
    # ),
]

# KPIs shown (in column order). Keys must exist in evaluate_policy.KPIS.
TABLE_KPIS = ["targets", "survival", "duration"]

# KPIs restricted to SUCCESSFUL episodes. For these, means, CIs and the (one-sided,
# directional) paired t-test are computed only over episode pairs where BOTH
# policies completed the mission (all targets destroyed). Duration is only
# comparable between missions that actually finished — a policy whose agents die
# early ends its episode early, which would otherwise make it look "faster".
SUCCESS_CONDITIONED_KPIS = {"duration"}

# The per-episode success flag: the "completion" KPI == mission_complete ==
# (~target_alive).all() == "all targets destroyed" (see environment.py).
SUCCESS_KEY = "completion"

# Outputs (relative paths resolved against the project dir 0_TA_...).
OUT_PATH = "eval_results/mission_performance.tex"
DASHBOARD_OUT = "eval_results/mission_performance_ci.png"
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

    `direction` is the KPI's "better" direction: "higher" → H1 mean(d) > 0,
    "lower" → H1 mean(d) < 0, "two-sided" → H1 mean(d) != 0. The reported
    difference is ORIENTED so a positive value always means the main policy is
    better (for "lower" KPIs the sign is flipped)."""
    m = np.asarray(m, dtype=float)
    b = np.asarray(b, dtype=float)
    n = m.size
    sign = -1.0 if direction == "lower" else 1.0
    alt = _ALT[direction]
    out: Dict[str, Any] = {
        "n": int(n), "alternative": alt,
        "mean_main": float("nan"), "mean_base": float("nan"),
        "mean_diff": float("nan"), "mean_diff_oriented": float("nan"),
        "ci_main": (float("nan"), float("nan")),
        "ci_base": (float("nan"), float("nan")),
        "ci_diff_oriented": (float("nan"), float("nan")),
        "t": float("nan"), "pvalue": float("nan"), "reason": None,
    }
    if n < 2:
        out["reason"] = "fewer than 2 paired observations"
        if n == 1:
            out["mean_main"], out["mean_base"] = float(m[0]), float(b[0])
            out["mean_diff"] = float(m[0] - b[0])
            out["mean_diff_oriented"] = sign * out["mean_diff"]
        return out

    d = m - b
    tcrit = float(_student_t.ppf(1.0 - (1.0 - ci_level) / 2.0, n - 1))
    mean_d = float(d.mean())
    out["mean_main"], out["mean_base"] = float(m.mean()), float(b.mean())
    out["mean_diff"], out["mean_diff_oriented"] = mean_d, sign * mean_d
    out["ci_main"], out["ci_base"] = _ci_mean(m, tcrit), _ci_mean(b, tcrit)
    d_lo, d_hi = _ci_mean(d, tcrit)
    out["ci_diff_oriented"] = tuple(sorted((sign * d_lo, sign * d_hi)))

    sd = float(d.std(ddof=1))
    if sd == 0.0:                                   # constant difference
        if mean_d == 0.0:
            out["t"], out["pvalue"] = 0.0, 1.0
        else:
            better = (mean_d > 0) if alt == "greater" else (
                mean_d < 0) if alt == "less" else True
            out["t"] = float("inf") if mean_d > 0 else float("-inf")
            out["pvalue"] = 0.0 if better else 1.0
        out["reason"] = "all paired differences are equal"
        return out
    res = ttest_rel(m, b, alternative=alt)
    out["t"], out["pvalue"] = float(res.statistic), float(res.pvalue)
    return out


def _main_only(m: np.ndarray, ci_level: float) -> Dict[str, Any]:
    """Main-policy mean + CI when there is no comparison policy."""
    m = np.asarray(m, dtype=float)
    n = m.size
    tcrit = float(_student_t.ppf(1.0 - (1.0 - ci_level) / 2.0, max(n - 1, 1)))
    return {
        "n": int(n),
        "mean_main": float(m.mean()) if n else float("nan"),
        "mean_base": float("nan"), "mean_diff": float("nan"),
        "mean_diff_oriented": float("nan"),
        "ci_main": _ci_mean(m, tcrit) if n else (float("nan"), float("nan")),
        "ci_base": (float("nan"), float("nan")),
        "ci_diff_oriented": (float("nan"), float("nan")),
        "t": float("nan"), "pvalue": float("nan"), "reason": None,
    }


def _paired_result(scn, key, main, base, samples, ci_level) -> Dict[str, Any]:
    """Full paired result for one (scenario, KPI), applying success-conditioning
    for the KPIs in SUCCESS_CONDITIONED_KPIS."""
    direction = _KPI_BY_KEY[key].direction
    mv = np.asarray(samples[scn.name][main.name][key], dtype=float)
    if base is None:
        mask = np.isfinite(mv)
        if key in SUCCESS_CONDITIONED_KPIS:
            sm = np.asarray(samples[scn.name][main.name][SUCCESS_KEY], dtype=float) >= 0.5
            mask &= sm[:mv.size]
        return _main_only(mv[mask], ci_level)

    bv = np.asarray(samples[scn.name][base.name][key], dtype=float)
    n = min(mv.size, bv.size)
    mv, bv = mv[:n], bv[:n]
    mask = np.isfinite(mv) & np.isfinite(bv)
    if key in SUCCESS_CONDITIONED_KPIS:
        mask &= _joint_success_mask(scn, main, base, samples)[:n]
    return _paired_ttest(mv[mask], bv[mask], direction, ci_level)


def compute_results(scenarios, main, base, samples, ci_level) -> Dict[str, Dict[str, dict]]:
    """results[kpi_key][scenario_name] = paired result dict."""
    return {key: {scn.name: _paired_result(scn, key, main, base, samples, ci_level)
                  for scn in scenarios}
            for key in TABLE_KPIS}


# =====================================================================
#  LaTeX table
# =====================================================================

def _cells(scn, key, results) -> Tuple[str, str, str]:
    """(complete, baseline, p) strings for one (scenario, KPI)."""
    r = results[key][scn.name]
    c = _fmt_val(key, r["mean_main"])
    if not np.isfinite(r["mean_base"]):
        return c, "", ""
    return c, _fmt_val(key, r["mean_base"]), _fmt_p(r["pvalue"])


_HEADER = r"""\begin{table}[htbp]
    \centering
    \caption{Mission performance across scenarios (n=__N__ paired episodes;
    one-sided \emph{paired} $t$-test on common-random-number pairs; Duration is
    conditioned on jointly-successful episodes, __DURN__).}
    \label{tab:mission-performence}
    \small
    \resizebox{\textwidth}{!}{%
        \begin{tabular}{l*{9}{c}}
            \toprule
            \textbf{Configuration}
            & \multicolumn{3}{c}{\textbf{Targets destroyed}}
            & \multicolumn{3}{c}{\textbf{Survival}}
            & \multicolumn{3}{c}{\textbf{Duration}} \\

            \cmidrule(lr){2-4}
            \cmidrule(lr){5-7}
            \cmidrule(lr){8-10}

            & Complete & Baseline & $p$-value
            & Complete & Baseline & $p$-value
            & Complete & Baseline & $p$-value \\
            \midrule

"""

_FOOTER = r"""            \bottomrule
        \end{tabular}%
    }
\end{table}
"""


def _duration_n_phrase(scenarios, results) -> str:
    """Caption phrase reporting the effective episode count for Duration (the
    jointly-successful pairs), auto-filled so the caption is copy-paste ready.
    One number if a single scenario / all equal, else listed per scenario."""
    ns = [(scn.name, int(results["duration"][scn.name]["n"])) for scn in scenarios]
    uniq = {n for _, n in ns}
    if len(uniq) == 1:
        return f"$n_\\mathrm{{dur}}={ns[0][1]}$"
    return "$n_\\mathrm{dur}$: " + ", ".join(f"{name}={n}" for name, n in ns)


def build_latex(scenarios, results, n_episodes) -> str:
    blocks = []
    for scn in scenarios:
        t = _cells(scn, "targets", results)
        s = _cells(scn, "survival", results)
        d = _cells(scn, "duration", results)
        blocks.append(
            f"            ${scn.name}$\n"
            f"            & {t[0]} & {t[1]} & {t[2]}\n"
            f"            & {s[0]} & {s[1]} & {s[2]}\n"
            f"            & {d[0]} & {d[1]} & {d[2]} \\\\\n"
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
    print(f"  MISSION PERFORMANCE — one-sided PAIRED t-test  "
          f"(CRN pairs; {pct}% CIs; * p<{alpha:g} ** p<0.01 *** p<0.001)")
    print("=" * 96)
    for key in TABLE_KPIS:
        spec = _KPI_BY_KEY[key]
        better = ("higher is better" if spec.direction == "higher"
                  else "lower is better" if spec.direction == "lower"
                  else "two-sided")
        cond = "  [successful pairs only]" if key in SUCCESS_CONDITIONED_KPIS else ""
        headers = ["Scenario", f"{main.name} ({pct}% CI)"]
        if base is not None:
            headers += [f"{base.name} ({pct}% CI)",
                        f"Δ {main.name}−{base.name} ({pct}% CI)", "p", "n"]
        rows: List[List[str]] = []
        for scn in scenarios:
            r = results[key][scn.name]
            row = [scn.name,
                   f"{_fmt_val(key, r['mean_main'])} {_fmt_ci(key, *r['ci_main'])}"]
            if base is not None:
                p_txt = _fmt_p(r["pvalue"]) + _stars(r["pvalue"], alpha)
                row += [
                    f"{_fmt_val(key, r['mean_base'])} {_fmt_ci(key, *r['ci_base'])}",
                    f"{_fmt_signed(key, r['mean_diff_oriented'])} "
                    f"{_fmt_ci(key, *r['ci_diff_oriented'])}",
                    p_txt, str(r["n"])]
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
#  CI dashboard
# =====================================================================

def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_PKG_DIR / p)


def plot_dashboard(scenarios, main, base, results, ci_level, alpha, out_png) -> None:
    """Top row: per-method mean ± CI per KPI (eyeball overlap). Bottom row: the
    paired mean-difference ± CI vs a zero line (rigorous paired check)."""
    keys = TABLE_KPIS
    K = len(keys)
    scen = list(scenarios)
    xn = np.arange(len(scen))
    pct = int(round(ci_level * 100))

    fig, axes = plt.subplots(2, K, figsize=(4.7 * K, 7.6), squeeze=False)

    for j, key in enumerate(keys):
        spec = _KPI_BY_KEY[key]
        is_rate = spec.unit == "rate"
        ax_top, ax_bot = axes[0, j], axes[1, j]

        m_mean = np.array([results[key][s.name]["mean_main"] for s in scen])
        m_err = np.array([(results[key][s.name]["ci_main"][1]
                           - results[key][s.name]["ci_main"][0]) / 2.0 for s in scen])
        # ── top: per-method means ± CI (dodged) ──
        ax_top.errorbar(xn - 0.11, m_mean, yerr=m_err, fmt="o", ms=6, capsize=5,
                        color=NLR_PRIMARY, label=main.name, lw=1.6)
        if base is not None:
            b_mean = np.array([results[key][s.name]["mean_base"] for s in scen])
            b_err = np.array([(results[key][s.name]["ci_base"][1]
                               - results[key][s.name]["ci_base"][0]) / 2.0 for s in scen])
            ax_top.errorbar(xn + 0.11, b_mean, yerr=b_err, fmt="s", ms=6, capsize=5,
                            color=NLR_ACCENT, label=base.name, lw=1.6)
        cond = "  (successful pairs)" if key in SUCCESS_CONDITIONED_KPIS else ""
        ax_top.set_title(f"{spec.label}{cond}", fontsize=11)
        ax_top.set_ylabel(spec.unit)
        ax_top.set_xticks(xn)
        ax_top.set_xticklabels([s.name for s in scen])
        ax_top.set_xlim(-0.5, len(scen) - 0.5)
        if is_rate:
            ax_top.set_ylim(0.0, 1.02)
        ax_top.grid(True, axis="y", alpha=0.3)
        if j == 0:
            ax_top.legend(fontsize=9, frameon=True)

        # ── bottom: oriented paired difference ± CI vs zero ──
        ax_bot.axhline(0.0, color=NLR_GRAY, ls="--", lw=1.2)
        if base is not None:
            d_mean = np.array([results[key][s.name]["mean_diff_oriented"] for s in scen])
            d_err = np.array([(results[key][s.name]["ci_diff_oriented"][1]
                               - results[key][s.name]["ci_diff_oriented"][0]) / 2.0
                              for s in scen])
            # colour by significance of the one-sided paired test
            sig = np.array([np.isfinite(results[key][s.name]["pvalue"])
                            and results[key][s.name]["pvalue"] < alpha for s in scen])
            ax_bot.errorbar(xn, d_mean, yerr=d_err, fmt="D", ms=6, capsize=5,
                            color=NLR_SECONDARY, lw=1.6, zorder=3)
            for xi, s in enumerate(scen):
                r = results[key][s.name]
                mark = "sig." if sig[xi] else "n.s."
                ax_bot.annotate(f"p={_fmt_p(r['pvalue'])}\n{mark}",
                                (xi, d_mean[xi]), textcoords="offset points",
                                xytext=(8, 0), va="center", fontsize=7.5,
                                color=NLR_DARKGRAY)
        ax_bot.set_title(f"Δ {spec.label}  ({main.name} better →)", fontsize=10)
        ax_bot.set_ylabel(f"mean diff ({spec.unit})")
        ax_bot.set_xticks(xn)
        ax_bot.set_xticklabels([s.name for s in scen])
        ax_bot.set_xlim(-0.5, len(scen) - 0.5)
        ax_bot.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Mission performance — {pct}% confidence intervals "
                 f"(top: per-method overlap check; bottom: paired difference vs 0)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print(f"Saved CI dashboard to: {out_png}")


# =====================================================================
#  Config banner
# =====================================================================

def _print_config(main, comparison, scenarios, n_episodes, base_seed, alpha,
                  ci_level) -> None:
    print("─" * 78)
    print("  MISSION PERFORMANCE — PAIRED t-TEST (common random numbers)")
    print("─" * 78)
    print(f"  Main policy        : {main.name}  [{main.policy_file}]")
    if comparison:
        print("  Comparison policies: "
              + ", ".join(f"{p.name} [{p.policy_file}]" for p in comparison))
    else:
        print("  Comparison policies: (none — values only, no p-values)")
    print(f"  Scenarios          : {len(scenarios)}  "
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
        description="Mission-performance LaTeX table + CI dashboard "
        "(Targets/Survival/Duration) for a Complete vs Baseline policy across "
        "EVAL_SCENARIOS, using a one-sided PAIRED t-test.")
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
    p.add_argument("--dashboard_out", type=str, default=DASHBOARD_OUT)
    p.add_argument("--no_dashboard", action="store_true", help="Skip the CI dashboard.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not EVAL_SCENARIOS:
        raise RuntimeError("EVAL_SCENARIOS is empty — define at least one scenario.")
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

    # The engine collects the PAIRED per-episode KPIs we need (it also runs its
    # own Wilcoxon, which we ignore — `_tests` unused; we t-test from `samples`).
    _summary, _tests, samples = evaluate_comparison(
        EVAL_SCENARIOS, MAIN_POLICY, COMPARISON_POLICIES, default_ckpt_path,
        n_episodes=args.n_episodes, chunk=args.chunk, base_seed=args.seed,
        p_adjust="none", device=device, cache=cache,
    )

    base = COMPARISON_POLICIES[0] if COMPARISON_POLICIES else None
    results = compute_results(EVAL_SCENARIOS, MAIN_POLICY, base, samples, args.ci)

    # Report the effective (jointly-successful) n for success-conditioned KPIs.
    if SUCCESS_CONDITIONED_KPIS and base is not None:
        print(f"[note] {', '.join(sorted(SUCCESS_CONDITIONED_KPIS))} computed over "
              f"jointly-successful episodes only (both policies completed the mission):")
        for scn in EVAL_SCENARIOS:
            n_ok = int(_joint_success_mask(scn, MAIN_POLICY, base, samples).sum())
            print(f"         {scn.name}: {n_ok}/{args.n_episodes} pairs")

    # ── console table ──
    print_console_tables(EVAL_SCENARIOS, MAIN_POLICY, base, results, args.ci, args.alpha)

    # ── LaTeX table ──
    latex = build_latex(EVAL_SCENARIOS, results, args.n_episodes)
    print(latex)
    out = _resolve_out(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table to: {out}")

    # ── CI dashboard ──
    if not args.no_dashboard:
        plot_dashboard(EVAL_SCENARIOS, MAIN_POLICY, base, results, args.ci,
                       args.alpha, _resolve_out(args.dashboard_out))


if __name__ == "__main__":
    main()
