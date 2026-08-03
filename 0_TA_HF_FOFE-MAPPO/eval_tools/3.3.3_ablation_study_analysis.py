"""ablation_study_analysis.py — ablation LaTeX table + console table for ONE scenario.

Reuses the evaluation engine in evaluate_policy.py (PAIRED rollouts via common
random numbers) and emits a table for FOUR policies on ONE scenario, with the
KPIs as ROWS and the policies as COLUMNS:

    KPI  |  Complete (mean)  |  Baseline (mean, p, d_z)  |  No-Com (…)  |  No-FOFE (…)

Complete is the reference (mean only). Each other policy shows its mean, a
one-sided PAIRED p-value versus Complete, and the paired effect size d_z.

STATISTICS (same methodology as 3.3.2): the raw per-episode KPIs are not normal,
but the SAMPLE MEAN of most of them is (CLT at n=600, per statistical_analysis.py),
which justifies a parametric PAIRED t-test on the per-episode differences
d_i = KPI_complete_i − KPI_policy_i (matched by common random numbers),
one-sided in each KPI's better direction (targets/survival: greater; duration:
less; fragmentation: two-sided). --p_adjust holm corrects across the comparison
columns within each KPI.

TWO TESTS: the Shapiro-Wilk check in statistical_analysis.py REJECTED normality
of the mean's sampling distribution for a few (KPI, model) combinations — see
NORMALITY_REJECTED below. Those KPIs drop the parametric assumption and use the
one-sided WILCOXON SIGNED-RANK test on the same paired differences instead; the
rest keep the paired t-test. Which test was used is named per KPI on the console;
in LaTeX the MINORITY test carries a dagger while the caption names the majority
one as the default (see MARKED_TEST — Wilcoxon now covers 3 of the 4 KPIs, so it
is the paired-t p-values that are daggered). The switch applies to EVERY
comparison column: Complete appears in every pair, and the two ablation
checkpoints have no normality result of their own, so the distribution-free test
is the conservative (and defensible) choice for the whole row.

EFFECT SIZE: Cohen's PAIRED-SAMPLE d_z = mean(d)/sd(d) accompanies every p-value,
computed on exactly the same paired subset the test used. The sign is ORIENTED so
a POSITIVE d_z always favours Complete (for "lower is better" KPIs the sign is
flipped); fragmentation is two-sided and keeps the raw Complete−policy sign. d_z
is reported regardless of which test produced the p-value — it is descriptive and
does not itself assume normality. Being a WITHIN-PAIR effect size (common random
numbers remove the between-layout variance) it must not be read against unpaired
Cohen's d benchmarks.

Duration is SUCCESS-CONDITIONED (see SUCCESS_CONDITIONED_KPIS): each policy's value
is its mean duration over its OWN successful missions, while its p and d_z compare
Complete vs that policy over the episodes BOTH completed (removes the early-death
confound).

Run (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\ablation_study_analysis.py
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\ablation_study_analysis.py --n_episodes 100
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Dict, List

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
from scipy.stats import ttest_rel, wilcoxon
from scipy.stats import t as _student_t

# Reuse the evaluation engine (paired rollouts, KPI defs). The engine also runs
# its own Wilcoxon internally, but we ignore that and compute the paired tests
# locally (so test choice, success-conditioning and effect size all match).
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

# Number of RUNS = paired episodes per policy (the test N).   [CLI: --n_episodes]
N_EPISODES = 1000
# Parallel envs per rollout chunk (lower this if you hit GPU OOM). [CLI: --chunk]
CHUNK_EPISODES = 500
# Base RNG seed; all policies share it so episodes are paired 1:1.   [CLI: --seed]
BASE_SEED = 42
# Significance level.                                               [CLI: --alpha]
ALPHA = 0.05
# Multiplicity correction across the comparison columns: "none" or "holm". [--p_adjust]
P_ADJUST = "none"

# =====================================================================
#  >>>  POLICIES  (the COLUMNS)  <<<
#  MAIN_POLICY = "Complete" (reference, value-only column). Each COMPARISON is one
#  (value, p) column pair, tested vs Complete. The column headers are the policy
#  `name`s. communicate=... must match how each ablation was TRAINED.
#
#  NOTE: No-Com / No-FOFE below are PLACEHOLDERS pointing at existing checkpoints
#  so the tool runs out of the box — REPLACE their policy_file with your actual
#  separately-trained ablation checkpoints (No-Com = FOFE on / comms off,
#  No-FOFE = FOFE off / comms on). FOFE on/off is baked into the checkpoint.
# =====================================================================

MAIN_POLICY = PolicyInput(
    name="Comm-FOFE-MAPPO",
    policy_file="runs/FINALV2/complete_stage7of8_DR_j2-4_k0_25.pt",
    communicate=True,
)
COMPARISON_POLICIES: List[PolicyInput] = [
    PolicyInput(
        name="MAPPO Baseline",
        policy_file="runs/FINALV2/Final_Baseline_Cont_4.pt",
        communicate=False,
    ),
    PolicyInput(  # PLACEHOLDER — replace with your FOFE-on / comms-off checkpoint
        name="No-Com",
        policy_file="runs/FINALV6/NO_COMM.pt",
        communicate=False,
    ),
    PolicyInput(  # PLACEHOLDER — replace with your FOFE-off / comms-on checkpoint
        name="No-FOFE",
        policy_file="runs/FINALV6/NO_FOFE.pt",
        communicate=True,
    ),
]

# =====================================================================
#  >>>  SCENARIO  (exactly ONE — the ablation is for a single scenario)  <<<
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
]

# KPI rows (top → bottom). Keys must exist in evaluate_policy.KPIS.
TABLE_KPIS = ["targets", "survival", "duration", "fragmentation"]

# Row labels in the LaTeX table. Falls back to the engine's KPISpec.label.
ROW_LABELS = {
    "targets": "Targets destroyed",
    "survival": "Survival",
    "duration": "Duration",
    "fragmentation": "Fragmentation",
    "reward": "Reward",
}

# =====================================================================
#  >>>  WHICH TEST PER KPI  (t-test vs Wilcoxon signed-rank)  <<<
# =====================================================================
# statistical_analysis.py runs a Shapiro-Wilk test on the BOOTSTRAP SAMPLING
# DISTRIBUTION OF EACH KPI MEAN, per model. Normality was NOT rejected for most
# (KPI, model) combinations, so those KPIs keep the parametric paired t-test. It
# WAS formally rejected for the combinations below; for those we fall back to the
# distribution-free one-sided Wilcoxon signed-rank test on the same differences.
#
# Keys are policy NAMES (must match MAIN_POLICY / COMPARISON_POLICIES above),
# values are KPI keys. A KPI switches to Wilcoxon in EVERY comparison column when
# it appears here for ANY policy: the reference policy is a member of every pair,
# and No-Com / No-FOFE have no normality result of their own, so applying the
# distribution-free test across the whole row is the conservative choice (and
# keeps the KPI tested the same way as in the 3.3.2 mission-performance table).
NORMALITY_REJECTED: Dict[str, set] = {
    "Comm-FOFE-MAPPO": {"targets"},                  # target-destruction rate
    "MAPPO Baseline": {"fragmentation", "duration"},  # coalition frag. + duration
}

# Derived: the KPIs tested with Wilcoxon rather than the paired t-test.
WILCOXON_KPIS = {k for keys in NORMALITY_REJECTED.values() for k in keys}

# Which test carries the dagger in the LaTeX table. Wilcoxon is now the MAJORITY
# test (3 of 4 KPIs), so the caption states Wilcoxon as the default and daggers
# the paired-t minority. Flip to "wilcoxon" if NORMALITY_REJECTED ever shrinks so
# that the t-test becomes the majority again.
MARKED_TEST = "t"
TEST_MARK_TEX = r"\(^{\dagger}\)"

# KPIs restricted to SUCCESSFUL episodes (mission complete == all targets destroyed).
# Each policy's value = mean over its OWN successful episodes; each p = one-sided
# PAIRED t-test vs Complete over the episodes BOTH completed. Removes the
# early-death confound so duration's lower-is-better holds.
SUCCESS_CONDITIONED_KPIS = {"duration"}
SUCCESS_KEY = "completion"

# LaTeX caption / label / output. The caption's episode counts are auto-filled.
# (EVAL_SCENARIOS below defines the S1 world, so the caption/label say S1.)
CAPTION_PREFIX = "Ablation-study results for Scenario~S1"
LABEL = "tab:ablation-study-s1"
OUT_PATH = "eval_results/ablation_study.tex"
DASHBOARD_OUT = "eval_results/ablation_study_ci.png"
CI_LEVEL = 0.95
DPI = 200

# =====================================================================

_KPI_BY_KEY = {s.key: s for s in KPIS}


def _fmt_val(key: str, mean: float) -> str:
    if not np.isfinite(mean):
        return "--"
    return _KPI_BY_KEY[key].fmt.format(mean)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _fmt_p_tex(p: float) -> str:
    """LaTeX p-value ('<' must be in math mode)."""
    if not np.isfinite(p):
        return "--"
    return r"\(<0.001\)" if p < 0.001 else f"{p:.3f}"


def _fmt_dz(dz: float) -> str:
    """Cohen's paired-sample d_z (oriented: positive = main policy better)."""
    if not np.isfinite(dz):
        return "--"
    return f"{dz:.2f}"


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
#  Values (per-policy) + paired one-sided t-test p-values
# =====================================================================

def _own_success_mean(scn, policy, key, samples) -> float:
    """Mean of a KPI over the policy's OWN successful (mission-complete) episodes."""
    arr = np.asarray(samples[scn.name][policy.name][key], dtype=float)
    succ = np.asarray(samples[scn.name][policy.name][SUCCESS_KEY], dtype=float) >= 0.5
    fin = succ & np.isfinite(arr)
    return float(arr[fin].mean()) if fin.any() else float("nan")


def _own_success_n(scn, policy, samples) -> int:
    succ = np.asarray(samples[scn.name][policy.name][SUCCESS_KEY], dtype=float) >= 0.5
    return int(succ.sum())


def _value(scn, policy, key, summary, samples) -> float:
    if key in SUCCESS_CONDITIONED_KPIS:
        return _own_success_mean(scn, policy, key, samples)
    return summary[(scn.name, policy.name, key)][0]


def _test_for(key: str) -> str:
    """Which paired test this KPI uses: 'wilcoxon' where the Shapiro-Wilk check
    in statistical_analysis.py rejected normality of a mean's sampling
    distribution (NORMALITY_REJECTED), else the parametric 't'."""
    return "wilcoxon" if key in WILCOXON_KPIS else "t"


_TEST_LABEL = {"t": "paired t", "wilcoxon": "Wilcoxon"}


def _paired_stats(m: np.ndarray, b: np.ndarray, direction: str,
                  test: str) -> Dict[str, float]:
    """One-sided PAIRED test on d = m − b over the finite pairs, plus d_z.

    `test` is "t" (parametric paired t-test) or "wilcoxon" (distribution-free
    signed-rank test, used where normality was rejected). Both run on the SAME
    differences and the same one-sided alternative; only the p-value differs.

    d_z = mean(d)/sd(d) is Cohen's paired-sample effect size, ORIENTED so that a
    positive value always favours the main policy ("lower is better" KPIs flip
    sign). It is computed identically for both tests."""
    m = np.asarray(m, dtype=float)
    b = np.asarray(b, dtype=float)
    nn = min(m.size, b.size)
    m, b = m[:nn], b[:nn]
    mask = np.isfinite(m) & np.isfinite(b)
    m, b = m[mask], b[mask]
    out = {"pvalue": float("nan"), "dz": float("nan"), "n": float(m.size)}
    if m.size < 2:
        return out
    d = m - b
    alt = _ALT[direction]
    sign = -1.0 if direction == "lower" else 1.0
    sd = float(np.std(d, ddof=1))
    out["dz"] = (sign * float(d.mean()) / sd) if sd > 0.0 else float("nan")
    if sd == 0.0:                                  # constant difference
        mean_d = float(d.mean())
        if mean_d == 0.0:
            out["pvalue"] = 1.0
        elif alt == "greater":
            out["pvalue"] = 0.0 if mean_d > 0 else 1.0
        elif alt == "less":
            out["pvalue"] = 0.0 if mean_d < 0 else 1.0
        else:
            out["pvalue"] = 0.0
        return out
    if test == "wilcoxon":
        # zero_method="wilcox" drops exact ties (episodes where both policies
        # scored identically), matching evaluate_policy._wilcoxon_pair.
        if not np.any(d != 0.0):
            out["pvalue"] = 1.0
            return out
        try:
            out["pvalue"] = float(wilcoxon(d, alternative=alt,
                                           zero_method="wilcox",
                                           correction=False).pvalue)
        except ValueError:                          # degenerate inputs
            pass
        return out
    out["pvalue"] = float(ttest_rel(m, b, alternative=alt).pvalue)
    return out


def _pairwise_stats(scn, key, main, comp, samples) -> Dict[str, float]:
    """Complete-vs-comp one-sided paired test (t or Wilcoxon per _test_for) plus
    d_z. Success-conditioned KPIs use only the episodes BOTH completed, so the
    p-value and the effect size describe the same paired subset."""
    mv = np.asarray(samples[scn.name][main.name][key], dtype=float)
    cv = np.asarray(samples[scn.name][comp.name][key], dtype=float)
    if key in SUCCESS_CONDITIONED_KPIS:
        sm = np.asarray(samples[scn.name][main.name][SUCCESS_KEY], dtype=float) >= 0.5
        sc = np.asarray(samples[scn.name][comp.name][SUCCESS_KEY], dtype=float) >= 0.5
        keep = sm & sc & np.isfinite(mv) & np.isfinite(cv)
        mv, cv = mv[keep], cv[keep]
    return _paired_stats(mv, cv, _KPI_BY_KEY[key].direction, _test_for(key))


def _holm_adjust(pvals: List[float]) -> List[float]:
    """Holm–Bonferroni step-down adjustment; NaNs pass through untouched."""
    idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    out = list(pvals)
    m = len(idx)
    if m == 0:
        return out
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)
        out[i] = running
    return out


def compute_pvalues(scn, main, comparisons, samples, p_adjust):
    """(pmap, dzmap) where
        pmap[kpi_key][comp.name]  = displayed p-value (Holm-adjusted across the
                                    comparison columns within each KPI if asked),
        dzmap[kpi_key][comp.name] = Cohen's paired-sample d_z (never adjusted —
                                    multiplicity affects inference, not the size
                                    of the effect)."""
    pmap: Dict[str, Dict[str, float]] = {}
    dzmap: Dict[str, Dict[str, float]] = {}
    for key in TABLE_KPIS:
        stats = [_pairwise_stats(scn, key, main, c, samples) for c in comparisons]
        raw = [s["pvalue"] for s in stats]
        adj = _holm_adjust(raw) if p_adjust == "holm" else raw
        pmap[key] = {c.name: p for c, p in zip(comparisons, adj)}
        dzmap[key] = {c.name: s["dz"] for c, s in zip(comparisons, stats)}
    return pmap, dzmap


# =====================================================================
#  LaTeX table
# =====================================================================

def _duration_n_phrase(scn, policies, samples) -> str:
    """Caption phrase: each policy's own successful-episode count feeding Duration."""
    parts = [f"{p.name}={_own_success_n(scn, p, samples)}" for p in policies]
    return "successful episodes: " + ", ".join(parts)


def _marked_kpi_phrase() -> str:
    """Caption fragment naming the KPIs whose p-value carries the dagger."""
    names = [ROW_LABELS.get(k, _KPI_BY_KEY[k].label).lower()
             for k in TABLE_KPIS if _test_for(k) == MARKED_TEST]
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def build_latex(scn, main, comparisons, summary, samples, pmap, dzmap,
                n_episodes, alpha, p_adjust="none") -> str:
    K = len(comparisons)
    colspec = "l" + "c" * (1 + 3 * K)     # KPI + reference(1) + 3 per comparison
    dur_phrase = _duration_n_phrase(scn, [main] + comparisons, samples)
    mark = TEST_MARK_TEX
    adj_phrase = (r" \(p\)-values are Holm-corrected across the ablation columns "
                  r"within each KPI." if p_adjust == "holm" else "")
    caption = (
        f"{CAPTION_PREFIX} based on {n_episodes:,} paired episodes (common random "
        f"numbers). Each ablation is compared with {main.name}. "
        rf"\(p\)-values are from one-sided Wilcoxon signed-rank tests, except {mark}, "
        r"which are from one-sided \emph{paired} \(t\)-tests (used where the "
        r"Shapiro--Wilk test did not reject normality of the KPI mean's sampling "
        f"distribution: {_marked_kpi_phrase()}). Coalition fragmentation is "
        r"tested two-sided. Effect size is reported as Cohen's paired-sample "
        rf"\(d_z\), oriented such that a positive value favours {main.name}."
        f"{adj_phrase} Duration uses each policy's own successful missions "
        f"({dur_phrase}); its \\(p\\) and \\(d_z\\) use the jointly-successful pairs."
    )

    lines: List[str] = [
        r"\begin{table}[htbp]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        rf"    \label{{{LABEL}}}",
        r"    \small",
        r"    \resizebox{\textwidth}{!}{%",
        rf"        \begin{{tabular}}{{{colspec}}}",
        r"            \toprule",
    ]

    # Header row 1: KPI | reference | <each ablation> (spanning 3).
    hdr = [r"            \textbf{KPI}",
           rf"            & \multicolumn{{1}}{{c}}{{\textbf{{{main.name}}}}}"]
    for c in comparisons:
        hdr.append(rf"            & \multicolumn{{3}}{{c}}{{\textbf{{{c.name}}}}}")
    lines.append("\n".join(hdr) + r" \\")
    lines.append("")

    lines.append(r"            \cmidrule(lr){2-2}")
    col = 3
    for _c in comparisons:
        lines.append(rf"            \cmidrule(lr){{{col}-{col + 2}}}")
        col += 3
    lines.append("")

    # Header row 2: Mean | (Mean, p, d_z) per ablation.
    h2 = [r"            & Mean"]
    for _c in comparisons:
        h2.append(r"            & Mean & \(\boldsymbol{p}\) & \(\boldsymbol{d_z}\)")
    lines.append("\n".join(h2) + r" \\")
    lines.append(r"            \midrule")

    # Body: one row per KPI (label on its own line, cells below).
    for key in TABLE_KPIS:
        cells = [_fmt_val(key, _value(scn, main, key, summary, samples))]
        for c in comparisons:
            p = pmap[key][c.name]
            p_txt = _fmt_p_tex(p)
            if np.isfinite(p) and _test_for(key) == MARKED_TEST:
                p_txt += mark
            cells += [_fmt_val(key, _value(scn, c, key, summary, samples)),
                      p_txt, _fmt_dz(dzmap[key][c.name])]
        lines.append(f"            {ROW_LABELS.get(key, _KPI_BY_KEY[key].label)}")
        lines.append("            & " + " & ".join(cells) + r" \\")
        lines.append("")

    lines += [r"            \bottomrule", r"        \end{tabular}%", r"    }",
              r"\end{table}", ""]
    return "\n".join(lines)


# =====================================================================
#  Console table
# =====================================================================

def print_console_table(scn, main, comparisons, summary, samples, pmap, dzmap,
                        alpha) -> None:
    print("\n" + "=" * 96)
    print(f"  ABLATION — one-sided PAIRED tests (t / Wilcoxon) vs {main.name}  "
          f"(CRN pairs; * p<{alpha:g} ** p<0.01 *** p<0.001)")
    print("=" * 96)
    headers = ["KPI", main.name] + [f"{c.name} (p, d_z)" for c in comparisons]
    rows: List[List[str]] = []
    for key in TABLE_KPIS:
        cond = "  (own successes)" if key in SUCCESS_CONDITIONED_KPIS else ""
        row = [f"{_KPI_BY_KEY[key].label}{cond}  [{_TEST_LABEL[_test_for(key)]}]",
               _fmt_val(key, _value(scn, main, key, summary, samples))]
        for c in comparisons:
            v = _fmt_val(key, _value(scn, c, key, summary, samples))
            p = pmap[key][c.name]
            row.append(f"{v} ({_fmt_p(p)}{_stars(p, alpha)}, "
                       f"d={_fmt_dz(dzmap[key][c.name])})")
        rows.append(row)

    widths = [len(h) for h in headers]
    for row in rows:
        for c, cell in enumerate(row):
            widths[c] = max(widths[c], len(cell))

    def _fmt_row(cells):
        return "  ".join(cell.ljust(widths[c]) if c == 0 else cell.rjust(widths[c])
                         for c, cell in enumerate(cells))

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


def _ci_mean(vals: np.ndarray, tcrit: float):
    n = vals.size
    if n < 2:
        m = float(vals[0]) if n == 1 else float("nan")
        return (m, m)
    sem = float(vals.std(ddof=1)) / np.sqrt(n)
    mu = float(vals.mean())
    return (mu - tcrit * sem, mu + tcrit * sem)


def _policy_mean_ci(scn, policy, key, samples, ci_level):
    """(mean, lo, hi) using the SAME data as the displayed value: marginal for
    non-conditioned KPIs, the policy's OWN successful episodes for conditioned."""
    arr = np.asarray(samples[scn.name][policy.name][key], dtype=float)
    if key in SUCCESS_CONDITIONED_KPIS:
        succ = np.asarray(samples[scn.name][policy.name][SUCCESS_KEY], dtype=float) >= 0.5
        vals = arr[succ & np.isfinite(arr)]
    else:
        vals = arr[np.isfinite(arr)]
    n = vals.size
    if n < 1:
        return (float("nan"), float("nan"), float("nan"))
    tcrit = float(_student_t.ppf(1.0 - (1.0 - ci_level) / 2.0, max(n - 1, 1)))
    lo, hi = _ci_mean(vals, tcrit)
    return (float(vals.mean()), lo, hi)


def _pairwise_diff_ci(scn, key, main, comp, samples, ci_level):
    """(oriented mean diff, lo, hi) for Complete − comp over the paired subset
    (jointly-successful for conditioned KPIs); positive = Complete better."""
    direction = _KPI_BY_KEY[key].direction
    sign = -1.0 if direction == "lower" else 1.0
    mv = np.asarray(samples[scn.name][main.name][key], dtype=float)
    cv = np.asarray(samples[scn.name][comp.name][key], dtype=float)
    mask = np.isfinite(mv) & np.isfinite(cv)
    if key in SUCCESS_CONDITIONED_KPIS:
        sm = np.asarray(samples[scn.name][main.name][SUCCESS_KEY], dtype=float) >= 0.5
        sc = np.asarray(samples[scn.name][comp.name][SUCCESS_KEY], dtype=float) >= 0.5
        mask &= sm & sc
    m, c = mv[mask], cv[mask]
    if m.size < 2:
        return (float("nan"), float("nan"), float("nan"))
    d = m - c
    tcrit = float(_student_t.ppf(1.0 - (1.0 - ci_level) / 2.0, d.size - 1))
    lo, hi = _ci_mean(d, tcrit)
    olo, ohi = sorted((sign * lo, sign * hi))
    return (sign * float(d.mean()), olo, ohi)


def plot_dashboard(scn, main, comparisons, samples, pmap, dzmap, ci_level, alpha,
                   out_png) -> None:
    """Top row: each KPI's per-policy mean ± CI (eyeball overlap). Bottom row:
    each comparison's paired mean-difference vs Complete ± CI, vs a zero line."""
    from matplotlib.lines import Line2D
    keys = TABLE_KPIS
    K = len(keys)
    policies = [main] + list(comparisons)
    pol_names = [p.name for p in policies]
    comp_names = [c.name for c in comparisons]
    pct = int(round(ci_level * 100))
    pol_colors = [NLR_PRIMARY, NLR_ACCENT, NLR_SECONDARY, NLR_DARKGRAY, NLR_GRAY]

    fig, axes = plt.subplots(2, K, figsize=(4.7 * K, 7.8), squeeze=False)
    xs = np.arange(len(policies))
    xc = np.arange(len(comparisons))

    for j, key in enumerate(keys):
        spec = _KPI_BY_KEY[key]
        two_sided = spec.direction == "two-sided"
        is_rate = spec.unit == "rate"
        ax_top, ax_bot = axes[0, j], axes[1, j]

        # top: per-policy mean ± CI
        for xi, pol in enumerate(policies):
            mu, lo, hi = _policy_mean_ci(scn, pol, key, samples, ci_level)
            if not np.isfinite(mu):
                continue
            ax_top.errorbar([xi], [mu], yerr=[[mu - lo], [hi - mu]],
                            fmt=("o" if xi == 0 else "s"), ms=6, capsize=5,
                            color=pol_colors[xi % len(pol_colors)], lw=1.6)
        cond = "  (own successes)" if key in SUCCESS_CONDITIONED_KPIS else ""
        ax_top.set_title(f"{spec.label}{cond}", fontsize=11)
        ax_top.set_ylabel(spec.unit)
        ax_top.set_xticks(xs)
        ax_top.set_xticklabels(pol_names, rotation=20, ha="right", fontsize=8)
        ax_top.set_xlim(-0.5, len(policies) - 0.5)
        if is_rate:
            ax_top.set_ylim(0.0, 1.02)
        ax_top.grid(True, axis="y", alpha=0.3)

        # bottom: oriented paired difference vs Complete ± CI vs zero
        ax_bot.axhline(0.0, color=NLR_GRAY, ls="--", lw=1.2)
        for xi, comp in enumerate(comparisons):
            md, lo, hi = _pairwise_diff_ci(scn, key, main, comp, samples, ci_level)
            if np.isfinite(md):
                ax_bot.errorbar([xi], [md], yerr=[[md - lo], [hi - md]], fmt="D",
                                ms=6, capsize=5, color=NLR_SECONDARY, lw=1.6, zorder=3)
            p = pmap[key][comp.name]
            sig = np.isfinite(p) and p < alpha
            ax_bot.annotate(f"p={_fmt_p(p)} ({_TEST_LABEL[_test_for(key)]})\n"
                            f"$d_z$={_fmt_dz(dzmap[key][comp.name])}  "
                            f"{'sig.' if sig else 'n.s.'}",
                            (xi, md if np.isfinite(md) else 0.0),
                            textcoords="offset points", xytext=(8, 0), va="center",
                            fontsize=7.5, color=NLR_DARKGRAY)
        subtitle = (f"Δ {spec.label}  ({main.name} − policy)" if two_sided
                    else f"Δ {spec.label}  ({main.name} better →)")
        ax_bot.set_title(subtitle, fontsize=10)
        ax_bot.set_ylabel(f"mean diff ({spec.unit})")
        ax_bot.set_xticks(xc)
        ax_bot.set_xticklabels(comp_names, rotation=20, ha="right", fontsize=8)
        ax_bot.set_xlim(-0.5, len(comparisons) - 0.5)
        ax_bot.grid(True, axis="y", alpha=0.3)

    handles = [Line2D([0], [0], marker=("o" if i == 0 else "s"), color="w",
                      markerfacecolor=pol_colors[i % len(pol_colors)], markersize=8,
                      label=pol_names[i]) for i in range(len(policies))]
    axes[0, 0].legend(handles=handles, fontsize=8, frameon=True)
    fig.suptitle(f"Ablation ({scn.name}) — {pct}% confidence intervals "
                 f"(top: per-policy overlap check; bottom: paired difference vs {main.name})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print(f"Saved CI dashboard to: {out_png}")


# =====================================================================
#  Config banner
# =====================================================================

def _print_config(main, comparison, scn, n_episodes, base_seed, alpha, p_adjust) -> None:
    print("─" * 78)
    print("  ABLATION STUDY — PAIRED TESTS (common random numbers)")
    print("─" * 78)
    print(f"  Main policy        : {main.name}  [{main.policy_file}]")
    print("  Comparison policies: "
          + ", ".join(f"{p.name} [{p.policy_file}]" for p in comparison))
    print(f"  Scenario           : {scn.name}")
    print(f"  Test               : one-sided PAIRED test on d=main−policy "
          f"(paired by common seed)")
    print(f"  Normality basis    : CLT on the sample mean (see statistical_analysis.py)")
    print("  Test per KPI (Wilcoxon where Shapiro-Wilk rejected normality):")
    for key in TABLE_KPIS:
        spec = _KPI_BY_KEY[key]
        who = sorted(n for n, keys in NORMALITY_REJECTED.items() if key in keys)
        why = f"  <- normality rejected for: {', '.join(who)}" if who else ""
        print(f"      {spec.label:18s} {_TEST_LABEL[_test_for(key)]}{why}")
    print(f"  Effect size        : Cohen's paired-sample d_z = mean(d)/sd(d), "
          f"oriented (+ favours {main.name})")
    print(f"  N (paired episodes): {n_episodes}   base seed: {base_seed}")
    print(f"  Significance       : alpha={alpha:g}   correction across columns: {p_adjust}")
    if SUCCESS_CONDITIONED_KPIS:
        print(f"  Success-conditioned: {', '.join(sorted(SUCCESS_CONDITIONED_KPIS))} "
              f"(value=own successes; p=jointly-successful pairs)")
    print("─" * 78)


# =====================================================================
#  CLI
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ablation LaTeX table + console table (KPIs × policies) for "
        "ONE scenario: Complete vs Baseline / No-Com / No-FOFE via one-sided "
        "PAIRED tests (t-test, or Wilcoxon signed-rank for the KPIs in "
        "NORMALITY_REJECTED) plus Cohen's d_z.")
    p.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                   help="Default .pt for any policy that leaves policy_file=None.")
    p.add_argument("--n_episodes", type=int, default=N_EPISODES)
    p.add_argument("--chunk", type=int, default=CHUNK_EPISODES)
    p.add_argument("--seed", type=int, default=BASE_SEED)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--p_adjust", type=str, default=P_ADJUST, choices=["none", "holm"])
    p.add_argument("--device", type=str, default=None, metavar="DEVICE")
    p.add_argument("--out", type=str, default=OUT_PATH)
    p.add_argument("--dashboard_out", type=str, default=DASHBOARD_OUT)
    p.add_argument("--ci", type=float, default=CI_LEVEL,
                   help=f"Confidence level for the dashboard intervals (default: {CI_LEVEL}).")
    p.add_argument("--no_dashboard", action="store_true", help="Skip the CI dashboard.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if len(EVAL_SCENARIOS) != 1:
        raise RuntimeError("ablation table expects exactly ONE scenario in EVAL_SCENARIOS.")
    if not COMPARISON_POLICIES:
        raise RuntimeError("need at least one comparison policy for the ablation columns.")
    scn = EVAL_SCENARIOS[0]

    default_ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    _print_config(MAIN_POLICY, COMPARISON_POLICIES, scn,
                  args.n_episodes, args.seed, args.alpha, args.p_adjust)
    cache: Dict[str, _LoadedCheckpoint] = {}
    _print_policy_diagnostics(MAIN_POLICY, COMPARISON_POLICIES, EVAL_SCENARIOS,
                              default_ckpt_path, device, cache)

    summary, _tests, samples = evaluate_comparison(
        EVAL_SCENARIOS, MAIN_POLICY, COMPARISON_POLICIES, default_ckpt_path,
        n_episodes=args.n_episodes, chunk=args.chunk, base_seed=args.seed,
        p_adjust="none", device=device, cache=cache,
    )

    pmap, dzmap = compute_pvalues(scn, MAIN_POLICY, COMPARISON_POLICIES, samples,
                                  args.p_adjust)

    # Effective n per policy for the success-conditioned KPIs.
    if SUCCESS_CONDITIONED_KPIS:
        print(f"[note] {', '.join(sorted(SUCCESS_CONDITIONED_KPIS))} value = mean over "
              f"each policy's OWN successful episodes; p and d_z = Complete vs policy "
              f"over jointly-successful episodes. Successful episodes / {args.n_episodes}:")
        for pol in [MAIN_POLICY] + COMPARISON_POLICIES:
            print(f"         {pol.name:<10}: {_own_success_n(scn, pol, samples)}")

    print_console_table(scn, MAIN_POLICY, COMPARISON_POLICIES, summary, samples,
                        pmap, dzmap, args.alpha)

    latex = build_latex(scn, MAIN_POLICY, COMPARISON_POLICIES, summary, samples,
                        pmap, dzmap, args.n_episodes, args.alpha, args.p_adjust)
    print(latex)

    out = _resolve_out(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table to: {out}")

    if not args.no_dashboard:
        plot_dashboard(scn, MAIN_POLICY, COMPARISON_POLICIES, samples, pmap, dzmap,
                       args.ci, args.alpha, _resolve_out(args.dashboard_out))


if __name__ == "__main__":
    main()
