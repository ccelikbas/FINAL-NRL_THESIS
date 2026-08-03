"""mission_performance_analysis.py — mission-performance table + CI dashboard.

Reuses the evaluation engine in evaluate_policy.py (PAIRED rollouts: both methods
are evaluated on the SAME per-episode initial conditions via common random
numbers) and compares a COMPLETE policy against a BASELINE policy for three KPIs:

    Targets destroyed · Survival · Duration     (each: Complete, Baseline, p-value)

There are FOUR policy inputs: each SCENARIO ROW carries its OWN (Complete,
Baseline) pair, so the S1 row is Complete-S1 vs Baseline-S1 and the S2 row is
Complete-S2 vs Baseline-S2 (configured in SCENARIO_POLICIES below). Any slot may
be left empty — that cell/section of the table is then simply rendered blank, and
a p-value is produced for a row only when BOTH of its policies are present.

──────────────────────────────────────────────────────────────────────────────
STATISTICAL METHODOLOGY  (paired one-sided tests: t OR Wilcoxon per KPI)
──────────────────────────────────────────────────────────────────────────────
The RAW per-episode KPIs are not normal (completion is binary, targets/survival
are bounded fractions, duration is right-skewed). But statistical_analysis.py
showed — via the bootstrap sampling distribution of the mean — that the SAMPLE
MEAN of MOST KPIs is approximately normal (central limit theorem at n=600). That
normality justifies a PARAMETRIC test on those means.

Normality was NOT universal, though: the Shapiro-Wilk test in statistical_analysis.py
REJECTED it for a few (KPI, model) combinations — see NORMALITY_REJECTED below.
For every KPI where EITHER member of the compared pair failed that check we drop
the parametric assumption and use the one-sided WILCOXON SIGNED-RANK test on the
same paired differences instead; the remaining KPIs keep the paired t-test. Which
test produced which p-value is marked in the outputs: on the console each KPI
block names its test, and in LaTeX the MINORITY test carries a dagger while the
caption names the majority one as the default (see MARKED_TEST — Wilcoxon now
covers 3 of the 5 KPIs, so it is the paired-t p-values that are daggered).

Because the environment is fully seeded, episode i faces the SAME randomised
layout for both methods (common random numbers), so (complete_i, baseline_i) is a
MATCHED PAIR. Both tests therefore run on the per-episode differences
    d_i = KPI_complete_i − KPI_baseline_i ,
one-sided in each KPI's "better" direction (targets/survival: Complete greater;
duration: Complete less; fragmentation is two-sided). Pairing removes the shared
between-layout variance, so the tests have more power than unpaired comparisons.

EFFECT SIZE: alongside the p-value we report Cohen's PAIRED-SAMPLE d_z,
    d_z = mean(d) / sd(d)          (sd with ddof=1)
i.e. the mean paired difference in units of its own standard deviation. It is
reported for every KPI regardless of which test was used (d_z is descriptive; it
does not itself assume normality). The sign is ORIENTED so a POSITIVE d_z always
means the Complete policy is better — for "lower is better" KPIs (duration) the
sign is flipped, exactly like the oriented difference CIs. Note that d_z is a
WITHIN-PAIR effect size: because common random numbers remove the between-layout
variance it is legitimately larger than an unpaired Cohen's d on the same data,
and the two must not be compared.

Duration is conditioned on JOINTLY-SUCCESSFUL pairs (both methods destroyed all
targets): duration is only comparable between missions that actually finished, so
a method whose agents die early cannot look "faster". Its effective n < 600.

──────────────────────────────────────────────────────────────────────────────
OUTPUTS
──────────────────────────────────────────────────────────────────────────────
  1. a LaTeX table  (per scenario: Complete / Baseline / p-value / d_z per KPI),
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
from dataclasses import dataclass
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
from scipy.stats import ttest_rel, wilcoxon
from scipy.stats import t as _student_t

# Reuse the evaluation engine (paired rollouts, KPI defs, path resolution) from
# evaluate_policy. The engine also runs its own Wilcoxon internally, but we ignore
# that here and compute the paired tests locally from the raw paired `samples`
# (so the test choice, the success-conditioning and the effect size all match).
from .evaluate_policy import (
    PolicyInput, evaluate_comparison, _LoadedCheckpoint, _resolve_policy_path,
    _print_policy_diagnostics, _ALT, KPIS, SEED_STRIDE,
)
from .run_curriculum import CurriculumSection
# NLR house palette (auto-applied to matplotlib on import).
from .nlr_style import NLR_PRIMARY, NLR_ACCENT, NLR_SECONDARY, NLR_GRAY, NLR_DARKGRAY

# =====================================================================
#  >>>  TEST CONFIG  (how many runs, seeding, significance)  <<<
# =====================================================================

# Number of RUNS = paired episodes per policy per scenario (this is the test N,
# and the "n=" printed in the table caption).             [CLI: --n_episodes]
N_EPISODES = 1000
# Parallel envs per rollout chunk (lower this if you hit GPU OOM). [CLI: --chunk]
CHUNK_EPISODES = 500
# Base RNG seed; both policies share it so episodes are paired 1:1.  [CLI: --seed]
BASE_SEED = 42
# Significance level for the tests.                                 [CLI: --alpha]
ALPHA = 0.05
# Confidence level for the reported / plotted intervals (two-sided). [CLI: --ci]
CI_LEVEL = 0.95

# =====================================================================
#  >>>  POLICIES  (four inputs: a (Complete, Baseline) pair per scenario)  <<<
# =====================================================================
#  The table has TWO generic columns — Complete (reference) and Baseline
#  (compared) — but each SCENARIO ROW is filled by its OWN trained pair:
#      S1 row  ->  Complete-S1  vs  Baseline-S1   (evaluated on the S1 world)
#      S2 row  ->  Complete-S2  vs  Baseline-S2   (evaluated on the S2 world)
#  So there are FOUR policy inputs in total. Any slot may be left EMPTY
#  (policy_file=None, or the whole entry set to None): the corresponding cells
#  are rendered blank and that section of the table is simply left empty — no
#  p-value is computed for a row unless BOTH of its policies are present.
#
#  Fill in each `policy_file` below (same path rules as evaluate_policy.py:
#  bare name -> runs/, "runs/FINALV2/…/x.pt" -> project-dir-relative, absolute
#  -> as-is). `communicate` must match how each model was trained (Complete
#  usually True, Baseline usually False).

# Generic column headers, shared by every scenario row.
MAIN_LABEL = "Complete"
BASE_LABEL = "Baseline"


@dataclass
class ScenarioPolicies:
    """The (Complete, Baseline) policy pair evaluated for ONE scenario row.

    `main` fills the Complete column, `base` the Baseline column. Either may be
    None (or carry policy_file=None) to leave that cell/section blank."""
    main: Optional[PolicyInput] = None
    base: Optional[PolicyInput] = None


# scenario `name`  ->  its (Complete, Baseline) pair. Keys MUST match the
# EVAL_SCENARIOS names below (S1, S2). Leave a policy_file=None to blank a slot.
SCENARIO_POLICIES: Dict[str, ScenarioPolicies] = {
    "S1": ScenarioPolicies(
        main=PolicyInput(name=MAIN_LABEL, policy_file="runs/FINALV2/complete_stage7of8_DR_j2-4_k0_25.pt", communicate=True),
        base=PolicyInput(name=BASE_LABEL, policy_file="runs/FINALV2/Final_Baseline_Cont_4.pt", communicate=False),
    ),
    "S2": ScenarioPolicies(
        main=PolicyInput(name=MAIN_LABEL, policy_file="runs/FINALV5/complete_S2_BEST.pt", communicate=True),
        base=PolicyInput(name=BASE_LABEL, policy_file="runs/FINALV2/S2_Baseline_stage9of9_S2_DR_j2-4_k0_25_FINAL.pt", communicate=False),
    ),
}

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
    CurriculumSection(
        name="S2",
        n_iters=1,
        n_strikers=2, n_jammers=(2, 4),
        n_known_targets=(1, 2), n_unknown_targets=(1, 2),
           n_known_radars=(3, 4), n_unknown_radars=(1, 2),
        radar_kill_probability=0.25,
        scenario="S2",
    ),
]

# KPIs shown (in row order). Keys must exist in evaluate_policy.KPIS.
TABLE_KPIS = ["reward", "targets", "survival", "duration", "fragmentation"]

# Row labels in the LaTeX table. Falls back to the engine's KPISpec.label.
ROW_LABELS = {
    "reward": "Reward",
    "targets": "Targets destroyed",
    "survival": "Survival",
    "duration": "Duration",
    "fragmentation": "Fragmentation",
}

# =====================================================================
#  >>>  WHICH TEST PER KPI  (t-test vs Wilcoxon signed-rank)  <<<
# =====================================================================
# statistical_analysis.py runs a Shapiro-Wilk test on the BOOTSTRAP SAMPLING
# DISTRIBUTION OF EACH KPI MEAN, per model. Normality was NOT rejected for most
# (KPI, model) combinations, so those KPIs keep the parametric paired t-test.
# It WAS formally rejected for the combinations listed below; for those the CLT
# justification does not hold and we fall back to the DISTRIBUTION-FREE one-sided
# Wilcoxon signed-rank test on the same paired differences.
#
# Keys are the policy ROLE (MAIN_LABEL / BASE_LABEL), values are KPI keys from
# TABLE_KPIS. A KPI switches to Wilcoxon when EITHER member of the compared pair
# appears here (the conservative choice: the parametric assumption must hold for
# both arms of a paired comparison). The switch applies to EVERY scenario row.
NORMALITY_REJECTED: Dict[str, set] = {
    MAIN_LABEL: {"targets"},                      # Complete: target-destruction rate
    BASE_LABEL: {"fragmentation", "duration"},    # Baseline: coalition frag. + duration
}

# Derived: the KPIs tested with Wilcoxon rather than the paired t-test.
WILCOXON_KPIS = {k for keys in NORMALITY_REJECTED.values() for k in keys}

# Which test carries the dagger in the LaTeX table. Wilcoxon is now the MAJORITY
# test (3 of 5 KPIs), so the caption states Wilcoxon as the default and daggers
# the paired-t minority. Flip to "wilcoxon" if NORMALITY_REJECTED ever shrinks so
# that the t-test becomes the majority again.
MARKED_TEST = "t"
TEST_MARK_TEX = r"\(^{\dagger}\)"

# KPIs restricted to SUCCESSFUL episodes. For these, means, CIs and the (one-sided,
# directional) paired t-test are computed only over episode pairs where BOTH
# policies completed the mission (all targets destroyed). Duration is only
# comparable between missions that actually finished — a policy whose agents die
# early ends its episode early, which would otherwise make it look "faster".
SUCCESS_CONDITIONED_KPIS = {"duration"}

# The per-episode success flag: the "completion" KPI == mission_complete, as
# recorded by the env. With RewardConfig.mission_discovered_targets_only (change
# #3, default on) that means "all DISCOVERED targets destroyed" — an unknown
# pop-up the team never sees does not block success; otherwise it is the legacy
# "all present targets destroyed" (~target_alive).all(). See environment.py
# (_mission_complete_mask). Likewise the "targets" KPI (targets_frac) is measured
# over the discovered set when the flag is on.
SUCCESS_KEY = "completion"

# Outputs (relative paths resolved against the project dir 0_TA_...).
OUT_PATH = "eval_results/mission_performance.tex"
DASHBOARD_OUT = "eval_results/mission_performance_ci.png"
COMPACT_TOP_OUT = "eval_results/mission_performance_ci_top.png"
DPI = 500

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
#  Paired one-sided test — t OR Wilcoxon (from the raw paired `samples`)
# =====================================================================

def _test_for(key: str) -> str:
    """Which paired test this KPI uses: 'wilcoxon' where the Shapiro-Wilk check
    in statistical_analysis.py rejected normality of a mean's sampling
    distribution (NORMALITY_REJECTED), else the parametric 't'."""
    return "wilcoxon" if key in WILCOXON_KPIS else "t"


_TEST_LABEL = {"t": "paired t", "wilcoxon": "Wilcoxon"}

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


def _paired_test(m: np.ndarray, b: np.ndarray, direction: str, ci_level: float,
                 test: str) -> Dict[str, Any]:
    """PAIRED one-sided test on d = m − b, plus per-method and difference CIs
    and Cohen's paired-sample d_z.

    `direction` is the KPI's "better" direction: "higher" → H1 mean(d) > 0,
    "lower" → H1 mean(d) < 0, "two-sided" → H1 mean(d) != 0. The reported
    difference is ORIENTED so a positive value always means the main policy is
    better (for "lower" KPIs the sign is flipped); d_z carries the same
    orientation.

    `test` is "t" (parametric paired t-test, the default when the KPI mean's
    sampling distribution passed the normality check) or "wilcoxon" (the
    distribution-free signed-rank test, used where normality was rejected). Both
    run on the SAME paired differences and the same one-sided alternative; only
    the p-value differs. The CIs and d_z are computed identically either way."""
    m = np.asarray(m, dtype=float)
    b = np.asarray(b, dtype=float)
    n = m.size
    sign = -1.0 if direction == "lower" else 1.0
    alt = _ALT[direction]
    out: Dict[str, Any] = {
        "n": int(n), "alternative": alt, "test": test,
        "mean_main": float("nan"), "mean_base": float("nan"),
        "mean_diff": float("nan"), "mean_diff_oriented": float("nan"),
        "ci_main": (float("nan"), float("nan")),
        "ci_base": (float("nan"), float("nan")),
        "ci_diff_oriented": (float("nan"), float("nan")),
        "statistic": float("nan"), "pvalue": float("nan"),
        "dz": float("nan"), "reason": None,
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
    # Cohen's paired-sample effect size d_z = mean(d)/sd(d), oriented so that a
    # POSITIVE value always favours the main policy. Undefined when sd == 0.
    out["dz"] = (sign * mean_d / sd) if sd > 0.0 else float("nan")
    if sd == 0.0:                                   # constant difference
        if mean_d == 0.0:
            out["statistic"], out["pvalue"] = 0.0, 1.0
        else:
            better = (mean_d > 0) if alt == "greater" else (
                mean_d < 0) if alt == "less" else True
            out["statistic"] = float("inf") if mean_d > 0 else float("-inf")
            out["pvalue"] = 0.0 if better else 1.0
        out["reason"] = "all paired differences are equal"
        return out

    if test == "wilcoxon":
        # Signed-rank test on the same differences. zero_method="wilcox" drops
        # exact ties (episodes where both policies scored identically), matching
        # evaluate_policy._wilcoxon_pair.
        if not np.any(d != 0.0):
            out["statistic"], out["pvalue"] = 0.0, 1.0
            out["reason"] = "all paired differences are zero"
            return out
        try:
            res = wilcoxon(d, alternative=alt, zero_method="wilcox",
                           correction=False)
        except ValueError as exc:                   # degenerate inputs
            out["reason"] = f"wilcoxon failed: {exc}"
            return out
    else:
        res = ttest_rel(m, b, alternative=alt)
    out["statistic"], out["pvalue"] = float(res.statistic), float(res.pvalue)
    return out


def _empty_result() -> Dict[str, Any]:
    """A fully-blank result (no policy present for this scenario/KPI)."""
    return {
        "n": 0, "alternative": None, "test": None,
        "mean_main": float("nan"), "mean_base": float("nan"),
        "mean_diff": float("nan"), "mean_diff_oriented": float("nan"),
        "ci_main": (float("nan"), float("nan")),
        "ci_base": (float("nan"), float("nan")),
        "ci_diff_oriented": (float("nan"), float("nan")),
        "statistic": float("nan"), "pvalue": float("nan"),
        "dz": float("nan"), "reason": None,
    }


def _single_result(scn_samples: dict, pol: "PolicyInput", key: str,
                   ci_level: float, slot: str) -> Dict[str, Any]:
    """Mean + CI for a lone policy (only one of Complete/Baseline present).

    `slot` is "main" (Complete column) or "base" (Baseline column). The value is
    stored under mean_<slot>/ci_<slot>; there is no p-value without a pair.
    Applies success-conditioning for the KPIs in SUCCESS_CONDITIONED_KPIS."""
    v = np.asarray(scn_samples[pol.name][key], dtype=float)
    mask = np.isfinite(v)
    if key in SUCCESS_CONDITIONED_KPIS:
        sm = np.asarray(scn_samples[pol.name][SUCCESS_KEY], dtype=float) >= 0.5
        mask &= sm[:v.size]
    v = v[mask]
    n = v.size
    tcrit = float(_student_t.ppf(1.0 - (1.0 - ci_level) / 2.0, max(n - 1, 1)))
    out = _empty_result()
    out["n"] = int(n)
    out[f"mean_{slot}"] = float(v.mean()) if n else float("nan")
    out[f"ci_{slot}"] = _ci_mean(v, tcrit) if n else (float("nan"), float("nan"))
    return out


def _paired_result(scn, key, main, base, samples, ci_level) -> Dict[str, Any]:
    """Full result for one (scenario, KPI) given the scenario's own policy pair.

    `main`/`base` are this scenario's Complete/Baseline policies; either may be
    None (empty slot). With both present a PAIRED test is run; with one present
    only that column's mean/CI is filled; with neither the result is blank.
    Applies success-conditioning for the KPIs in SUCCESS_CONDITIONED_KPIS."""
    direction = _KPI_BY_KEY[key].direction
    scn_samples = samples.get(scn.name, {})

    if main is not None and base is not None:
        mv = np.asarray(scn_samples[main.name][key], dtype=float)
        bv = np.asarray(scn_samples[base.name][key], dtype=float)
        n = min(mv.size, bv.size)
        mv, bv = mv[:n], bv[:n]
        mask = np.isfinite(mv) & np.isfinite(bv)
        if key in SUCCESS_CONDITIONED_KPIS:
            mask &= _joint_success_mask(scn, main, base, samples)[:n]
        return _paired_test(mv[mask], bv[mask], direction, ci_level,
                            _test_for(key))

    if main is not None:
        return _single_result(scn_samples, main, key, ci_level, "main")
    if base is not None:
        return _single_result(scn_samples, base, key, ci_level, "base")
    return _empty_result()


def compute_results(scenarios, scen_main, scen_base, samples, ci_level) -> Dict[str, Dict[str, dict]]:
    """results[kpi_key][scenario_name] = result dict, using each scenario's own
    (Complete, Baseline) pair from scen_main/scen_base (scenario name -> policy
    or None)."""
    return {key: {scn.name: _paired_result(scn, key, scen_main.get(scn.name),
                                           scen_base.get(scn.name), samples, ci_level)
                  for scn in scenarios}
            for key in TABLE_KPIS}


# =====================================================================
#  LaTeX table
# =====================================================================

def _cells(scn, key, results) -> Tuple[str, str, str, str]:
    """(complete, baseline, p, d_z) LaTeX strings for one (scenario, KPI).

    The p-value carries TEST_MARK_TEX when it came from the MARKED_TEST (the
    minority test, named in the caption). Any absent policy (empty slot) yields
    blank cells so that section of the table is simply empty."""
    r = results[key][scn.name]
    c = _fmt_val(key, r["mean_main"]) if np.isfinite(r["mean_main"]) else ""
    b = _fmt_val(key, r["mean_base"]) if np.isfinite(r["mean_base"]) else ""
    if np.isfinite(r["pvalue"]):
        p = _fmt_p_tex(r["pvalue"])
        if r.get("test") == MARKED_TEST:
            p += TEST_MARK_TEX
    else:
        p = ""
    dz = _fmt_dz(r["dz"]) if np.isfinite(r["dz"]) else ""
    return c, b, p, dz


_CAPTION = (
    r"Mission performance across scenarios based on __N__ paired episodes "
    r"(common random numbers). \(p\)-values are from one-sided Wilcoxon "
    r"signed-rank tests, except __MARK__, which are from one-sided \emph{paired} "
    r"\(t\)-tests (used where the Shapiro--Wilk test did not reject normality of "
    r"the KPI mean's sampling distribution: __MKPIS__). Coalition fragmentation "
    r"is tested two-sided. Effect size is Cohen's paired-sample \(d_z\), oriented "
    r"so that a positive value favours __MAIN__. Duration is evaluated only for "
    r"jointly successful episodes (__DURN__)."
)

_FOOTER = r"""        \bottomrule
    \end{tabular}
\end{table}
"""


def _duration_n_phrase(scenarios, results) -> str:
    """Caption phrase reporting the effective episode count for Duration (the
    jointly-successful pairs), auto-filled so the caption is copy-paste ready.
    One number if a single scenario / all equal, else listed per scenario."""
    ns = [(scn.name, int(results["duration"][scn.name]["n"])) for scn in scenarios]
    uniq = {n for _, n in ns}
    if len(uniq) == 1:
        return f"\\(n={ns[0][1]}\\)"
    return "; ".join(f"{name}: \\(n={n}\\)" for name, n in ns)


def _marked_kpi_phrase() -> str:
    """Caption fragment naming the KPIs whose p-value carries the dagger."""
    names = [ROW_LABELS.get(k, _KPI_BY_KEY[k].label).lower()
             for k in TABLE_KPIS if _test_for(k) == MARKED_TEST]
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def build_latex(scenarios, results, n_episodes) -> str:
    """LaTeX table: one ROW per KPI, one 4-column GROUP per scenario
    (Complete, Baseline, p-value, d_z)."""
    scen = list(scenarios)
    caption = (_CAPTION.replace("__N__", f"{n_episodes:,}")
               .replace("__MARK__", TEST_MARK_TEX)
               .replace("__MKPIS__", _marked_kpi_phrase())
               .replace("__DURN__", _duration_n_phrase(scen, results))
               .replace("__MAIN__", MAIN_LABEL))

    # column groups: 1 label column + 4 columns per scenario
    group_heads, cmidrules, sub_heads = [], [], []
    for i, s in enumerate(scen):
        lo = 2 + 4 * i
        group_heads.append(f"        & \\multicolumn{{4}}{{c}}{{\\textbf{{Scenario {s.name}}}}}")
        cmidrules.append(f"        \\cmidrule(lr){{{lo}-{lo + 3}}}")
        sub_heads.append(f"        & \\textbf{{{MAIN_LABEL}}}\n"
                         f"        & \\textbf{{{BASE_LABEL}}}\n"
                         f"        & \\(\\boldsymbol{{p}}\\)\\textbf{{-value}}\n"
                         f"        & \\(\\boldsymbol{{d_z}}\\)")

    header = (
        "\\begin{table}[htbp]\n"
        "    \\centering\n"
        f"    \\caption{{{caption}}}\n"
        "    \\label{tab:mission-performance}\n"
        "    \\small\n"
        f"    \\begin{{tabular}}{{l{'cccc' * len(scen)}}}\n"
        "        \\toprule\n"
        + "\n".join(group_heads) + " \\\\\n"
        + "\n".join(cmidrules) + "\n\n"
        "        \\textbf{KPI}\n"
        + "\n".join(sub_heads) + " \\\\\n"
        "        \\midrule\n\n"
    )

    blocks = []
    for key in TABLE_KPIS:
        label = ROW_LABELS.get(key, _KPI_BY_KEY[key].label)
        lines = [f"        {label}"]
        for s in scen:
            c, b, p, dz = _cells(s, key, results)
            lines.append(f"        & {c} & {b} & {p} & {dz}")
        blocks.append("\n".join(lines) + " \\\\\n")
    return header + "\n".join(blocks) + _FOOTER


# =====================================================================
#  Console table
# =====================================================================

def print_console_tables(scenarios, scen_main, scen_base, results, ci_level,
                         alpha, main_label, base_label) -> None:
    pct = int(round(ci_level * 100))
    # Show the comparison columns if ANY scenario has a Baseline policy.
    any_base = any(scen_base.get(s.name) is not None for s in scenarios)
    print("\n" + "=" * 96)
    print(f"  MISSION PERFORMANCE — one-sided PAIRED tests (t / Wilcoxon)  "
          f"(CRN pairs; {pct}% CIs; * p<{alpha:g} ** p<0.01 *** p<0.001)")
    print("=" * 96)
    for key in TABLE_KPIS:
        spec = _KPI_BY_KEY[key]
        better = ("higher is better" if spec.direction == "higher"
                  else "lower is better" if spec.direction == "lower"
                  else "two-sided")
        cond = "  [successful pairs only]" if key in SUCCESS_CONDITIONED_KPIS else ""
        test_txt = f"  [test: {_TEST_LABEL[_test_for(key)]}]"
        headers = ["Scenario", f"{main_label} ({pct}% CI)"]
        if any_base:
            headers += [f"{base_label} ({pct}% CI)",
                        f"Δ {main_label}−{base_label} ({pct}% CI)", "p", "d_z", "n"]
        rows: List[List[str]] = []
        for scn in scenarios:
            r = results[key][scn.name]
            main_cell = (f"{_fmt_val(key, r['mean_main'])} {_fmt_ci(key, *r['ci_main'])}"
                         if np.isfinite(r["mean_main"]) else "")
            row = [scn.name, main_cell]
            if any_base:
                base_cell = (f"{_fmt_val(key, r['mean_base'])} {_fmt_ci(key, *r['ci_base'])}"
                             if np.isfinite(r["mean_base"]) else "")
                if np.isfinite(r["pvalue"]):
                    diff_cell = (f"{_fmt_signed(key, r['mean_diff_oriented'])} "
                                 f"{_fmt_ci(key, *r['ci_diff_oriented'])}")
                    p_txt = _fmt_p(r["pvalue"]) + _stars(r["pvalue"], alpha)
                    n_txt = str(r["n"])
                else:
                    diff_cell, p_txt = "", ""
                    n_txt = str(r["n"]) if r["n"] else ""
                dz_txt = _fmt_dz(r["dz"]) if np.isfinite(r["dz"]) else ""
                row += [base_cell, diff_cell, p_txt, dz_txt, n_txt]
            rows.append(row)

        widths = [len(h) for h in headers]
        for row in rows:
            for c, cell in enumerate(row):
                widths[c] = max(widths[c], len(cell))

        def _fmt_row(cells):
            return "  ".join(cell.ljust(widths[c]) if c == 0 else cell.rjust(widths[c])
                             for c, cell in enumerate(cells))

        print(f"\n  KPI: {spec.label}  ({better}){cond}{test_txt}")
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


def plot_dashboard(scenarios, scen_main, scen_base, results, ci_level, alpha,
                   out_png, main_label, base_label) -> None:
    """Top row: per-method mean ± CI per KPI (eyeball overlap). Bottom row: the
    paired mean-difference ± CI vs a zero line (rigorous paired check). Scenarios
    with an empty slot simply contribute no marker for the missing policy."""
    keys = TABLE_KPIS
    K = len(keys)
    scen = list(scenarios)
    xn = np.arange(len(scen))
    pct = int(round(ci_level * 100))
    any_base = any(scen_base.get(s.name) is not None for s in scen)

    fig, axes = plt.subplots(2, K, figsize=(4.7 * K, 7.6), squeeze=False)

    for j, key in enumerate(keys):
        spec = _KPI_BY_KEY[key]
        is_rate = spec.unit == "rate"
        ax_top, ax_bot = axes[0, j], axes[1, j]

        m_mean = np.array([results[key][s.name]["mean_main"] for s in scen])
        m_err = np.array([(results[key][s.name]["ci_main"][1]
                           - results[key][s.name]["ci_main"][0]) / 2.0 for s in scen])
        # ── top: per-method means ± CI (dodged); NaNs (empty slots) drop out ──
        ax_top.errorbar(xn - 0.11, m_mean, yerr=m_err, fmt="o", ms=6, capsize=5,
                        color=NLR_PRIMARY, label=main_label, lw=1.6)
        if any_base:
            b_mean = np.array([results[key][s.name]["mean_base"] for s in scen])
            b_err = np.array([(results[key][s.name]["ci_base"][1]
                               - results[key][s.name]["ci_base"][0]) / 2.0 for s in scen])
            ax_top.errorbar(xn + 0.11, b_mean, yerr=b_err, fmt="s", ms=6, capsize=5,
                            color=NLR_ACCENT, label=base_label, lw=1.6)
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
        if any_base:
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
                if not np.isfinite(r["pvalue"]):
                    continue                     # no pair here → nothing to annotate
                mark = "sig." if sig[xi] else "n.s."
                test_tag = _TEST_LABEL.get(r.get("test"), "")
                ax_bot.annotate(f"p={_fmt_p(r['pvalue'])} ({test_tag})\n"
                                f"$d_z$={_fmt_dz(r['dz'])}  {mark}",
                                (xi, d_mean[xi]), textcoords="offset points",
                                xytext=(8, 0), va="center", fontsize=7.5,
                                color=NLR_DARKGRAY)
        ax_bot.set_title(f"Δ {spec.label}  ({main_label} better →)", fontsize=10)
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


def plot_top_compact(scenarios, scen_main, scen_base, results, ci_level, alpha,
                     out_png, main_label, base_label,
                     spacing=0.5, dodge=0.09, panel_w=1.95, panel_h=2.9,
                     title_fs=9, label_fs=8.5, tick_fs=8, legend_fs=8) -> None:
    """A COMPACT, report-ready version of ONLY the top row of plot_dashboard:
    per-method mean ± CI per KPI, no suptitle, no paired-difference row. The
    scenarios are packed together (small `spacing`) and each panel is narrow
    (`panel_w`) so there is no wasted horizontal whitespace between the S1/S2
    datapoints — same data as the full dashboard, just squeezed in x. The shared
    legend sits at the bottom, tucked close under the panels."""
    keys = TABLE_KPIS
    K = len(keys)
    scen = list(scenarios)
    xn = np.arange(len(scen)) * spacing
    any_base = any(scen_base.get(s.name) is not None for s in scen)

    fig, axes = plt.subplots(1, K, figsize=(panel_w * K, panel_h), squeeze=False)
    handles: list = []
    for j, key in enumerate(keys):
        spec = _KPI_BY_KEY[key]
        is_rate = spec.unit == "rate"
        ax = axes[0, j]

        m_mean = np.array([results[key][s.name]["mean_main"] for s in scen])
        m_err = np.array([(results[key][s.name]["ci_main"][1]
                           - results[key][s.name]["ci_main"][0]) / 2.0 for s in scen])
        h1 = ax.errorbar(xn - dodge, m_mean, yerr=m_err, fmt="o", ms=6, capsize=5,
                         color=NLR_PRIMARY, label=main_label, lw=1.6)
        if any_base:
            b_mean = np.array([results[key][s.name]["mean_base"] for s in scen])
            b_err = np.array([(results[key][s.name]["ci_base"][1]
                               - results[key][s.name]["ci_base"][0]) / 2.0 for s in scen])
            h2 = ax.errorbar(xn + dodge, b_mean, yerr=b_err, fmt="s", ms=6, capsize=5,
                             color=NLR_ACCENT, label=base_label, lw=1.6)
            if not handles:
                handles = [h1, h2]
        elif not handles:
            handles = [h1]

        ax.set_title(spec.label, fontsize=title_fs)
        ax.set_ylabel(spec.unit, fontsize=label_fs)
        ax.set_xticks(xn)
        ax.set_xticklabels([s.name for s in scen])
        ax.tick_params(labelsize=tick_fs)
        ax.set_xlim(xn.min() - spacing * 0.56, xn.max() + spacing * 0.56)
        if is_rate:
            ax.set_ylim(0.0, 1.02)
        ax.grid(True, axis="y", alpha=0.3)

    # Shared legend at the BOTTOM, tucked close under the panels.
    if handles:
        fig.legend(handles=handles,
                   labels=[main_label, base_label] if any_base else [main_label],
                   loc="lower center", ncol=2, fontsize=legend_fs, frameon=True,
                   bbox_to_anchor=(0.5, 0.005), handletextpad=0.4,
                   columnspacing=1.2, borderpad=0.4)
    fig.tight_layout(rect=(0, 0.075, 1, 1.0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print(f"Saved compact top figure to: {out_png}")


# =====================================================================
#  Config banner
# =====================================================================

def _slot_txt(pol: Optional[PolicyInput]) -> str:
    """Human description of one policy slot for the config banner."""
    if pol is None or not pol.policy_file:
        return "(empty — section left blank)"
    return str(pol.policy_file)


def _print_config(scenario_policies, scenarios, n_episodes, base_seed, alpha,
                  ci_level, main_label, base_label) -> None:
    print("─" * 78)
    print("  MISSION PERFORMANCE — PAIRED TESTS (common random numbers)")
    print("─" * 78)
    print(f"  Columns            : {main_label} (reference) vs {base_label} (compared)")
    print("  Per-scenario policy pairs (each row uses its OWN trained pair):")
    for scn in scenarios:
        pair = scenario_policies.get(scn.name)
        m = pair.main if pair else None
        b = pair.base if pair else None
        print(f"      [{scn.name}]  {main_label}: {_slot_txt(m)}")
        print(f"      {' ' * len(scn.name)}    {base_label}: {_slot_txt(b)}")
    print(f"  Scenarios          : {len(scenarios)}  "
          f"({', '.join(s.name for s in scenarios)})")
    print(f"  Test               : one-sided PAIRED test on d=main−baseline "
          f"(paired by common seed)")
    print(f"  Normality basis    : CLT on the sample mean (see statistical_analysis.py)")
    print("  Test per KPI (Wilcoxon where Shapiro-Wilk rejected normality):")
    for key in TABLE_KPIS:
        spec = _KPI_BY_KEY[key]
        who = sorted(role for role, keys in NORMALITY_REJECTED.items() if key in keys)
        why = f"  <- normality rejected for: {', '.join(who)}" if who else ""
        print(f"      {spec.label:18s} {_TEST_LABEL[_test_for(key)]}{why}")
    print(f"  Effect size        : Cohen's paired-sample d_z = mean(d)/sd(d), "
          f"oriented (+ favours {main_label})")
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
        "EVAL_SCENARIOS, using one-sided PAIRED tests (t-test, or Wilcoxon "
        "signed-rank for the KPIs in NORMALITY_REJECTED) plus Cohen's d_z.")
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
    p.add_argument("--compact_top_out", type=str, default=COMPACT_TOP_OUT,
                   help="Compact top-row-only CI figure (no suptitle, x-compressed).")
    p.add_argument("--no_dashboard", action="store_true", help="Skip the CI dashboard.")
    p.add_argument("--no_compact_top", action="store_true",
                   help="Skip the compact top-row-only CI figure.")
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

    _print_config(SCENARIO_POLICIES, EVAL_SCENARIOS, args.n_episodes, args.seed,
                  args.alpha, args.ci, MAIN_LABEL, BASE_LABEL)

    def _present(pol: Optional[PolicyInput]) -> Optional[PolicyInput]:
        """A slot counts as present if it names a checkpoint (or a --checkpoint
        default is available to fall back on); otherwise it is treated as empty."""
        if pol is not None and (pol.policy_file or default_ckpt_path is not None):
            return pol
        return None

    cache: Dict[str, _LoadedCheckpoint] = {}
    scen_main: Dict[str, Optional[PolicyInput]] = {}
    scen_base: Dict[str, Optional[PolicyInput]] = {}
    samples: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    # Each scenario is evaluated against its OWN (Complete, Baseline) pair, so we
    # drive the engine one scenario at a time. The per-scenario seed offset
    # reproduces the engine's cross-scenario seeding (base_seed + si*SEED_STRIDE);
    # within a scenario the pair still shares one seed schedule → matched pairs.
    for si, scn in enumerate(EVAL_SCENARIOS):
        pair = SCENARIO_POLICIES.get(scn.name)
        main = _present(pair.main) if pair else None
        base = _present(pair.base) if pair else None
        scen_main[scn.name] = main
        scen_base[scn.name] = base

        present = [p for p in (main, base) if p is not None]
        if not present:
            print(f"\n[{si + 1}/{len(EVAL_SCENARIOS)}] Scenario '{scn.name}': "
                  f"no policies configured — leaving its row blank.", flush=True)
            continue

        _print_policy_diagnostics(present[0], present[1:], [scn],
                                  default_ckpt_path, device, cache)
        seed = args.seed + si * SEED_STRIDE
        # The engine collects the PAIRED per-episode KPIs we need (it also runs
        # its own Wilcoxon, which we ignore — we t-test from `samples`). present[0]
        # is the reference column (Complete when set, else the lone Baseline).
        _summary, _tests, sub = evaluate_comparison(
            [scn], present[0], present[1:], default_ckpt_path,
            n_episodes=args.n_episodes, chunk=args.chunk, base_seed=seed,
            p_adjust="none", device=device, cache=cache,
        )
        samples.update(sub)

    results = compute_results(EVAL_SCENARIOS, scen_main, scen_base, samples, args.ci)

    # Report the effective (jointly-successful) n for success-conditioned KPIs,
    # for every scenario that has BOTH policies present (i.e. an actual pair).
    paired = [scn for scn in EVAL_SCENARIOS
              if scen_main.get(scn.name) is not None
              and scen_base.get(scn.name) is not None]
    if SUCCESS_CONDITIONED_KPIS and paired:
        print(f"[note] {', '.join(sorted(SUCCESS_CONDITIONED_KPIS))} computed over "
              f"jointly-successful episodes only (both policies completed the mission):")
        for scn in paired:
            n_ok = int(_joint_success_mask(scn, scen_main[scn.name],
                                           scen_base[scn.name], samples).sum())
            print(f"         {scn.name}: {n_ok}/{args.n_episodes} pairs")

    # ── console table ──
    print_console_tables(EVAL_SCENARIOS, scen_main, scen_base, results,
                         args.ci, args.alpha, MAIN_LABEL, BASE_LABEL)

    # ── LaTeX table ──
    latex = build_latex(EVAL_SCENARIOS, results, args.n_episodes)
    print(latex)
    out = _resolve_out(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table to: {out}")

    # ── CI dashboard ──
    if not args.no_dashboard:
        plot_dashboard(EVAL_SCENARIOS, scen_main, scen_base, results, args.ci,
                       args.alpha, _resolve_out(args.dashboard_out),
                       MAIN_LABEL, BASE_LABEL)

    # ── compact top-row-only CI figure (report-ready, x-compressed) ──
    if not args.no_compact_top:
        plot_top_compact(EVAL_SCENARIOS, scen_main, scen_base, results, args.ci,
                         args.alpha, _resolve_out(args.compact_top_out),
                         "Comm-FOFE-MAPPO", "MAPPO Baseline")


if __name__ == "__main__":
    main()
