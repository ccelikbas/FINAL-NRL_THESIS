r"""
statistical_analysis.py
═══════════════════════
Two clearly separated workflows for a FROZEN FOFE-MAPPO policy's KPIs:

  ── PILOT  (--analysis_mode pilot) ── sample-size PLANNING ──────────────────
     Collect a large pool (e.g. 1000 episodes) and decide HOW MANY evaluation
     episodes the final study needs. Everything here is a DIAGNOSTIC:
       • coefficient-of-variation (CoV) stabilisation curves,
       • precision curves: bootstrap CI half-width vs candidate sample size,
       • one recommended common evaluation count  N_eval,
       • EXPECTED sampling distributions of the KPI means at that proposed
         N_eval (labelled as *expected* — NOT a final result).
     Pilot means are the "pilot pooled mean" over the whole pool; pilot CIs are
     planning estimates, never reported as final performance.

  ── FINAL  (--analysis_mode final --eval_episodes N) ── STATISTICAL INFERENCE ─
     Collect a NEW independent set of EXACTLY N evaluation episodes and report
     the final KPI means and 95 % confidence intervals. All KPIs share ONE
     common N (they come from the same episodes). The final mean is computed
     from those N episodes; N is never chosen from the same final sample.

──────────────────────────────────────────────────────────────────────────────
RAW DISTRIBUTIONS  vs  SAMPLING DISTRIBUTIONS OF THE MEAN
──────────────────────────────────────────────────────────────────────────────
Each evaluation episode is the experimental unit; one episode yields one value
for every KPI. The raw episode-level distribution describes mission-to-mission
variation (descriptive; it need not be normal). The SAMPLING distribution of the
MEAN describes uncertainty in the estimated average performance; its standard
deviation is the standard error of the mean (SEM = raw SD / √n). The bootstrap
approximates it by resampling n episodes WITH replacement and recomputing the
mean. The bootstrap SE is the SD of the bootstrap means (NOT divided by √B); the
95 % CI is a percentile interval and does NOT require normality.

──────────────────────────────────────────────────────────────────────────────
DURATION  (two metrics, failed missions never silently dropped)
──────────────────────────────────────────────────────────────────────────────
  • PRIMARY  "Failure-aware duration"  (duration_capped): observed duration if
    the mission completed, else max_steps. Defined for every episode ⇒ same
    sample size as the other KPIs.
  • SECONDARY "Duration among completed missions" (duration_completed):
    conditional on completion; reported WITH the completion rate so a policy
    cannot look fast merely by completing only the easy missions. Its sample
    size is the number of completed episodes, NOT the total.

──────────────────────────────────────────────────────────────────────────────
SEEDS
──────────────────────────────────────────────────────────────────────────────
--n_seeds are EVALUATION RNG seeds for one frozen checkpoint (extra independent
episode pools) — NOT independently trained models. For a frozen policy the pools
may be combined; episodes remain the sampling unit. The seed id is preserved in
the raw CSV. Several INDEPENDENTLY TRAINED checkpoints need a separate
HIERARCHICAL analysis:  episodes → mean per training seed → variation across
training-seed means. Do NOT pool episodes across training seeds as if one policy.

──────────────────────────────────────────────────────────────────────────────
NORMALITY OF BOOTSTRAP MEANS  (diagnostic only, OFF by default)
──────────────────────────────────────────────────────────────────────────────
A D'Agostino-Pearson test on B=10 000 bootstrap means is over-sensitive (B is
not a count of independent experiments) and is NOT a valid rule for deciding
whether the bootstrap CI holds. It is disabled unless
--run_bootstrap_normality_diagnostic is passed. Q-Q plots and skewness remain as
visual shape diagnostics; the percentile CI is reported regardless.

──────────────────────────────────────────────────────────────────────────────
HOW TO USE
──────────────────────────────────────────────────────────────────────────────
    # 1) plan the sample size (pilot):
    .\.venv\Scripts\python.exe 0_TA_HF_FOFE-MAPPO\eval_tools\statistical_analysis.py ^
        --analysis_mode pilot --max_episodes 1000

    # 2) run the final study at the chosen count:
    .\.venv\Scripts\python.exe 0_TA_HF_FOFE-MAPPO\eval_tools\statistical_analysis.py ^
        --analysis_mode final --eval_episodes 100

Edit SCENARIOS + PRECISION_TARGETS below. Outputs land in <script_dir>/stat_results.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Match the training entry points: force Inductor to compile in-process (avoids
# a Triton 'cubin' KeyError on some Linux+CUDA setups). Harmless elsewhere.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# Windows consoles default to cp1252, which can't encode the glyphs (± × σ μ).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── package bootstrap (so the file runs as a script OR as -m fofe_mappo.*) ──
_THIS_DIR = Path(__file__).resolve().parent          # .../0_TA_HF_FOFE-MAPPO/eval_tools
_PKG_DIR = _THIS_DIR.parent                           # .../0_TA_HF_FOFE-MAPPO (sim modules + runs/ live here)
_RUNS_DIR = _PKG_DIR / "runs"      # bare policy_file names resolve here
_PKG_NAME = "fofe_mappo"
if __package__ in (None, ""):
    sys.path.insert(0, str(_PKG_DIR.parent))
    if _PKG_NAME not in sys.modules:
        _pkg = types.ModuleType(_PKG_NAME)
        _pkg.__path__ = [str(_PKG_DIR), str(_THIS_DIR)]  # search parent (sim) + eval_tools (analysis)
        _pkg.__package__ = _PKG_NAME
        _pkg.__file__ = str(_THIS_DIR / "__init__.py")
        sys.modules[_PKG_NAME] = _pkg
    __package__ = _PKG_NAME

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# NOTE: the "pynvml package is deprecated" FutureWarning at import time comes
# from the PyTorch/CUDA dependency stack, not from this analysis. It is left
# untouched on purpose (we do not suppress FutureWarnings globally).
from scipy.stats import normaltest, probplot, skew

from .config import PPOConfig
from .trainer import build_env
# Reuse the EXACT scenario format + config-resolution + checkpoint loading so a
# scenario here behaves identically to one in evaluate_policy.py.
from .run_curriculum import CurriculumSection, _section_to_env_cfg, _section_label
from .evaluate_policy import (
    _LoadedCheckpoint, _build_policy_for_scenario, _resolve_policy_path,
)
# Importing nlr_style auto-applies the NLR house palette to Matplotlib.
from .nlr_style import NLR_CYCLE, NLR_REFERENCE, NLR_ACCENT, NLR_SECONDARY


# =====================================================================
#  >>>  EDIT YOUR SCENARIOS HERE  <<<  (same format as evaluate_policy.py)
# =====================================================================

SCENARIOS: List[CurriculumSection] = [
    CurriculumSection(
        name="FOFE-MAPPO S2",
        policy_file="runs/FINALV2/complete_stage7of8_DR_j2-4_k0_25.pt",
        n_iters=1,  # not used
        n_strikers=2, n_jammers=(2, 4),
        n_known_targets=(2, 4), n_unknown_targets=0,
        n_known_radars=(4, 6), n_unknown_radars=0,
        radar_kill_probability=0.25,
        scenario="S2",
        communicate=True,
    ),
]


# =====================================================================
#  >>>  ANALYSIS CONFIG  <<<  (all overridable on the CLI)
# =====================================================================

# ── pilot: how big the planning pool is ──────────────────────────────
MAX_EPISODES   = 1000     # pilot episode pool per evaluation RNG seed
N_SEEDS        = 1        # evaluation RNG seeds (extra pilot pools; NOT training seeds)
BASE_SEED      = 42       # evaluation RNG base seed; seed s uses BASE_SEED + s*SEED_STRIDE
SEED_STRIDE    = 1_000_000
# Final-mode episodes are drawn with an independent seed so a default final run
# does NOT reuse the default pilot draws.
FINAL_SEED_OFFSET = 7_000_000
CHUNK_EPISODES = 250      # parallel envs per rollout (chunked to avoid OOM)

# ── CoV stabilisation diagnostic (pilot) ─────────────────────────────
N_ORDERINGS    = 64       # random shuffles → order-independent CoV curve
STABILITY_TOL  = 0.05     # "stable" = CoV within ±5 % of its final value …
STABILITY_MIN_N = 10      # … but ignore the very noisy n < this region

# ── precision-based sample-size planning (pilot) ─────────────────────
# User-defined PRACTICAL precision requirements: the target 95 % CI half-width
# for each KPI's mean (same units as the KPI). Easy to edit.
PRECISION_TARGETS: Dict[str, float] = {
    "completion":      0.03,   # ±0.03 on the completion proportion
    "targets":         0.03,   # ±0.03 on the target-destruction fraction
    "survival":        0.03,   # ±0.03 on the survival fraction
    "fragmentation":   0.02,   # ±0.02 on coalition fragmentation
    "duration_capped": 2.0,    # ±2 steps on failure-aware duration
    "reward":          1.0,    # ±1.0 on episode reward
}
PRECISION_STEP      = 10      # candidate sample sizes 10, 20, 30, …, max_episodes
PRECISION_BOOTSTRAP = 2000    # bootstrap resamples for the (many) precision points
ROUND_TO            = 25      # round the recommended N_eval up to a tidy multiple

# ── bootstrap of the sampling distribution of the mean ───────────────
N_BOOTSTRAP    = 10_000   # bootstrap resamples B for the reported distributions
BOOTSTRAP_SEED = 12345    # RNG seed for ALL bootstrap resampling (reproducible)
BOOTSTRAP_MAX_ELEMS = 20_000_000   # cap the [B, n] index matrix before chunking

# ── normality diagnostic (OFF by default; not a decision rule) ───────
NORMALITY_ALPHA = 0.05

SAVE_RAW_DIST = True      # also save the descriptive raw episode-level histograms
WILSON_Z      = 1.959963985   # z for a 95 % Wilson interval (completion, final mode)


# =====================================================================
#  KPI DEFINITIONS
# =====================================================================
#  Raw per-episode stats collected from the env (collection logic UNCHANGED).
RAW_STATS: List[Tuple[str, str]] = [
    ("completion",    "mission_complete"),
    ("targets",       "targets_frac"),
    ("survival",      "survival_frac"),
    ("fragmentation", "coalition_fragmentation"),
    ("duration",      "duration"),
    ("reward",        "episode_total_reward"),
]


@dataclass(frozen=True)
class KpiSpec:
    key: str            # analysis key
    label: str          # display label
    cov: bool           # part of the CoV stabilisation diagnostic (non-negative)
    precision: bool     # part of the precision (CI half-width) planning
    conditional: bool = False   # conditional-on-completion (own sample size)


#  Analysis KPIs used for means / bootstrap / plots. Duration is split into the
#  failure-aware primary metric and the conditional secondary metric.
ANALYSIS_KPIS: List[KpiSpec] = [
    KpiSpec("completion",         "Completion rate",                   cov=True,  precision=True),
    KpiSpec("targets",            "Target-destruction rate",           cov=True,  precision=True),
    KpiSpec("survival",           "Survival rate",                     cov=True,  precision=True),
    KpiSpec("fragmentation",      "Coalition fragmentation",           cov=True,  precision=True),
    KpiSpec("duration_capped",    "Failure-aware duration",            cov=True,  precision=True),
    KpiSpec("reward",             "Episode reward",                    cov=False, precision=True),
    KpiSpec("duration_completed", "Duration among completed missions", cov=False, precision=False,
            conditional=True),
]
PRIMARY_KPIS   = [k for k in ANALYSIS_KPIS if not k.conditional]
COV_KPIS       = [k for k in ANALYSIS_KPIS if k.cov]
PRECISION_KPIS = [k for k in ANALYSIS_KPIS if k.precision]

SCENARIO_TOKEN_RE = re.compile(r"\bS([123])\b", re.IGNORECASE)


@dataclass
class RunOpts:
    """Everything a per-scenario run needs, gathered from CLI + config."""
    analysis_mode: str
    max_episodes: int
    eval_episodes: Optional[int]
    n_seeds: int
    base_seed: int
    chunk: int
    n_orderings: int
    tol: float
    min_n: int
    precision_step: int
    precision_bootstrap: int
    round_to: int
    n_bootstrap: int
    bootstrap_seed: int
    normality_alpha: float
    run_normality: bool
    scenario_name_check: str
    out_dir: Path
    stamp: str
    write_csv: bool
    write_bootstrap_csv: bool
    save_raw_dist: bool


# =====================================================================
#  STEP 1 — collect RAW per-episode KPI values  [COLLECTION LOGIC UNCHANGED]
# =====================================================================

def _chunk_seed(master_seed: int, chunk_idx: int) -> int:
    """Distinct RNG seed per (master seed, chunk) so chunks are independent."""
    return int(master_seed + chunk_idx * 1000)


@torch.no_grad()
def _rollout_collect(policy, env, max_steps: int, n_envs: int,
                     device: torch.device) -> List[Optional[dict]]:
    """Run one parallel batch of `n_envs` episodes; return per-env stats dicts.

    Mirrors the rollout in trainer.evaluate_current_policy, but keeps the RAW
    per-episode stats instead of averaging. Stats for a newly-finished env are
    popped immediately: re-stepping a done env would re-fire its stat with
    reward=0 and an inflated step count, overwriting the real values.
    """
    was_training = policy.training
    policy.eval()
    policy.deterministic = True          # frozen, greedy policy during eval

    done_mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
    td = env.reset()
    stats_by_env: Dict[int, dict] = {}

    for _ in range(max_steps):
        td = policy(td)
        td_next = env.step(td)

        step_done = td_next.get(("next", "done")).reshape(n_envs).bool()
        newly_done = step_done & ~done_mask
        done_mask = done_mask | step_done

        if newly_done.any():
            for s in env.pop_episode_stats():
                ei = int(s.get("env_idx", -1))
                if ei >= 0 and bool(newly_done[ei]) and ei not in stats_by_env:
                    stats_by_env[ei] = s

        if done_mask.all():
            break
        td = td_next.get("next")

    policy.deterministic = False
    if was_training:
        policy.train()

    return [stats_by_env.get(b) for b in range(n_envs)]


def collect_episode_kpis(policy, env_cfg, hf_radar_cfg, n_episodes: int,
                         master_seed: int, device: torch.device,
                         chunk: int) -> Dict[str, np.ndarray]:
    """Return {raw_key: array[n_episodes]} of per-episode values for one seed.

    Episodes are generated in independent parallel chunks (different seed each)
    so the whole pool is `n_episodes` i.i.d. randomised episodes.
    """
    per_kpi: Dict[str, List[float]] = {key: [] for key, _ in RAW_STATS}
    collected = 0
    chunk_idx = 0
    while collected < n_episodes:
        b = min(chunk, n_episodes - collected)
        ppo = PPOConfig(num_envs=b, device=device,
                        seed=_chunk_seed(master_seed, chunk_idx))
        env = build_env(env_cfg, ppo, hf_radar_cfg=hf_radar_cfg)
        stats = _rollout_collect(policy, env, env_cfg.max_steps, b, device)

        for s in stats:
            for key, stat_key in RAW_STATS:
                if s is None:
                    per_kpi[key].append(float("nan"))
                elif stat_key == "mission_complete":      # bool → 0/1 KPI
                    per_kpi[key].append(1.0 if bool(s.get(stat_key, False)) else 0.0)
                else:
                    per_kpi[key].append(float(s.get(stat_key, float("nan"))))

        collected += b
        chunk_idx += 1
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {k: np.asarray(v, dtype=float) for k, v in per_kpi.items()}


def collect_pool(policy, env_cfg, ckpt, n_episodes: int, seeds: List[int],
                 device: torch.device, chunk: int) -> List[Dict[str, np.ndarray]]:
    """Collect one raw-KPI dict per evaluation RNG seed."""
    per_seed: List[Dict[str, np.ndarray]] = []
    for s_idx, master in enumerate(seeds):
        kpis = collect_episode_kpis(policy, env_cfg, ckpt.hf_radar_cfg,
                                    n_episodes, master, device, chunk)
        per_seed.append(kpis)
        print(f"        eval RNG seed {s_idx + 1}/{len(seeds)} (base={master}) "
              f"collected {n_episodes} episodes", flush=True)
    return per_seed


# =====================================================================
#  Derive analysis KPIs (failure-aware + conditional duration) from raw stats
# =====================================================================

def derive_analysis_kpis(raw: Dict[str, np.ndarray],
                         max_steps: int) -> Dict[str, np.ndarray]:
    """Map raw episode stats → the analysis KPI arrays.

    Failure-aware duration uses max_steps for any non-completed episode (failed
    OR never-terminated); conditional duration keeps completed episodes only.
    """
    comp = raw["completion"]
    dur = raw["duration"]
    completed = np.isfinite(comp) & (comp == 1.0)

    dur_capped = np.where(completed, dur, float(max_steps)).astype(float)
    bad = completed & ~np.isfinite(dur)          # completed but missing duration
    dur_capped[bad] = float(max_steps)

    dur_completed = np.where(completed, dur, np.nan)

    return {
        "completion":         comp,
        "targets":            raw["targets"],
        "survival":           raw["survival"],
        "fragmentation":      raw["fragmentation"],
        "duration_capped":    dur_capped,
        "reward":             raw["reward"],
        "duration_completed": dur_completed,
    }


def episode_counts(derived: Dict[str, np.ndarray]) -> Dict[str, int]:
    comp = derived["completion"]
    completed = np.isfinite(comp) & (comp == 1.0)
    total = int(comp.size)
    n_completed = int(completed.sum())
    return {"total": total, "completed": n_completed, "failed": total - n_completed}


# =====================================================================
#  STEP 2 — running coefficient of variation  c_v(n) = σ(n)/μ(n)  [UNCHANGED]
# =====================================================================

def _running_cov_shuffled(x: np.ndarray, n_orderings: int,
                          rng: np.random.Generator) -> np.ndarray:
    """E[c_v(n)] over `n_orderings` random orderings of the samples `x`."""
    N = x.size
    n = np.arange(1, N + 1, dtype=float)
    denom = np.maximum(n - 1.0, 1.0)          # ddof=1; guard the n=1 prefix
    acc = np.zeros(N)
    cnt = np.zeros(N)
    for _ in range(n_orderings):
        xp = x[rng.permutation(N)]
        csum = np.cumsum(xp)
        csum2 = np.cumsum(xp * xp)
        mean = csum / n
        var = np.clip((csum2 - n * mean * mean) / denom, 0.0, None)  # FP guard
        std = np.sqrt(var)
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = std / mean
        finite = np.isfinite(cov)
        acc[finite] += cov[finite]
        cnt[finite] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = acc / cnt
    out[cnt == 0] = np.nan
    return out


def cov_curves(per_seed_derived: List[Dict[str, np.ndarray]], n_orderings: int,
               rng: np.random.Generator) -> Dict[str, Dict[str, np.ndarray]]:
    """Per CoV KPI: across-seed mean / min / max of the order-averaged CoV curve."""
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for spec in COV_KPIS:
        seed_curves = []
        for seed_arr in per_seed_derived:
            x = seed_arr[spec.key]
            x = x[np.isfinite(x)]          # drop episodes that never finished
            if x.size >= 2:
                seed_curves.append(_running_cov_shuffled(x, n_orderings, rng))
        if not seed_curves:
            continue
        L = min(c.size for c in seed_curves)            # align lengths across seeds
        stack = np.vstack([c[:L] for c in seed_curves])
        curves[spec.key] = {
            "n":    np.arange(1, L + 1),
            "mean": np.nanmean(stack, axis=0),
            "min":  np.nanmin(stack, axis=0),
            "max":  np.nanmax(stack, axis=0),
        }
    return curves


def _required_runs(n: np.ndarray, cov_mean: np.ndarray, tol: float,
                   min_n: int) -> Optional[int]:
    """Smallest n* (≥ min_n) past which |c_v(m) − c_v(∞)| ≤ tol·|c_v(∞)| for all m ≥ n*."""
    N = cov_mean.size
    if N == 0:
        return None
    finite = np.isfinite(cov_mean)
    if not finite.any():
        return None
    final = cov_mean[np.where(finite)[0][-1]]
    if final == 0:                          # constant KPI (σ=0): stable at once
        return int(min(max(min_n, 1), N))
    band = tol * abs(final)
    dev = np.abs(cov_mean - final)
    start = N
    for i in range(N - 1, -1, -1):
        if finite[i] and dev[i] <= band:
            start = i
        else:
            break
    if start >= N:
        return None
    idx = max(start, min_n - 1)
    return int(n[idx]) if idx < N else None


# =====================================================================
#  STEP 3 — BOOTSTRAP of the sampling distribution of the mean
# =====================================================================

def bootstrap_sample_means(x: np.ndarray, sample_size: int, n_bootstrap: int,
                           rng: np.random.Generator,
                           max_elems: int = BOOTSTRAP_MAX_ELEMS) -> np.ndarray:
    """Bootstrap approximation of the sampling distribution of the MEAN of `x`.

    Draws `n_bootstrap` resamples, each `sample_size` observations WITH
    replacement, and returns one sample mean per resample → shape [n_bootstrap].
    The [n_bootstrap, sample_size] index matrix is built in row-chunks whenever
    it would exceed `max_elems` entries, to bound peak memory.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]                       # per-KPI, never mixed with others
    if x.size == 0:
        return np.empty(0, dtype=float)
    if sample_size < 2:
        raise ValueError(f"sample_size must be >= 2 (got {sample_size}).")

    N = x.size
    out = np.empty(n_bootstrap, dtype=float)
    per_chunk = max(1, int(max_elems // max(sample_size, 1)))
    done = 0
    while done < n_bootstrap:
        b = min(per_chunk, n_bootstrap - done)
        idx = rng.integers(0, N, size=(b, sample_size))   # WITH replacement
        out[done:done + b] = x[idx].mean(axis=1)          # one mean per resample
        done += b
    return out


def _bootstrap_ci(means: np.ndarray, lo: float = 2.5, hi: float = 97.5):
    """Percentile 95 % CI from the bootstrap means (no normality assumption)."""
    if means.size == 0:
        return (float("nan"), float("nan"))
    a, b = np.percentile(means, [lo, hi])
    return float(a), float(b)


def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> Tuple[float, float]:
    """Wilson score 95 % CI for a binomial proportion k/n (completion rate)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return (centre - half, centre + half)


# ── precision (CI half-width) planning ───────────────────────────────

def precision_curve(pool: np.ndarray, candidate_ns: np.ndarray, n_boot: int,
                    rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """For each candidate n: bootstrap SE and 95 % CI half-width of the mean."""
    hw = np.full(candidate_ns.size, np.nan)
    se = np.full(candidate_ns.size, np.nan)
    pool = pool[np.isfinite(pool)]
    if pool.size < 2:
        return hw, se
    for i, n in enumerate(candidate_ns):
        if n < 2:
            continue
        means = bootstrap_sample_means(pool, int(n), n_boot, rng)
        lo, hiv = _bootstrap_ci(means)
        hw[i] = (hiv - lo) / 2.0
        se[i] = float(means.std(ddof=1)) if means.size > 1 else 0.0
    return hw, se


def precision_required_n(candidate_ns: np.ndarray, hw: np.ndarray,
                         threshold: float) -> Optional[int]:
    """Smallest candidate n whose CI half-width ≤ threshold AND stays below it."""
    idxs = [i for i in range(candidate_ns.size) if np.isfinite(hw[i])]
    for i in idxs:
        if hw[i] <= threshold and all(hw[j] <= threshold for j in idxs if j >= i):
            return int(candidate_ns[i])
    return None


def _round_up(n: Optional[int], base: int) -> Optional[int]:
    if n is None:
        return None
    return int(math.ceil(n / base) * base)


# =====================================================================
#  Summaries of the sampling distribution of the mean
# =====================================================================

def summarise_mean_distribution(spec: KpiSpec, pool: np.ndarray,
                                sample_size: int, n_bootstrap: int,
                                rng: np.random.Generator) -> dict:
    """Bootstrap `pool`'s mean at `sample_size` and return the full summary."""
    pool = np.asarray(pool, dtype=float)
    pool = pool[np.isfinite(pool)]
    n_available = int(pool.size)
    s = {
        "key": spec.key, "label": spec.label, "conditional": spec.conditional,
        "n_available": n_available, "sample_size": int(sample_size),
        "mean": float("nan"), "raw_sd": float("nan"),
        "theoretical_sem": float("nan"), "bootstrap_se": float("nan"),
        "ci_lo": float("nan"), "ci_hi": float("nan"), "ci_half_width": float("nan"),
        "skew": float("nan"), "means": np.empty(0, dtype=float),
        "n_gt_available": bool(sample_size > n_available),
    }
    if n_available == 0 or sample_size < 2:
        return s

    mean = float(pool.mean())
    raw_sd = float(pool.std(ddof=1)) if n_available > 1 else 0.0
    means = bootstrap_sample_means(pool, sample_size, n_bootstrap, rng)
    ci_lo, ci_hi = _bootstrap_ci(means)

    s.update(
        mean=mean, raw_sd=raw_sd,
        theoretical_sem=raw_sd / math.sqrt(sample_size),
        bootstrap_se=float(means.std(ddof=1)) if means.size > 1 else 0.0,
        ci_lo=ci_lo, ci_hi=ci_hi, ci_half_width=(ci_hi - ci_lo) / 2.0,
        skew=float(skew(means)) if means.size > 2 and np.ptp(means) > 0 else 0.0,
        means=means,
    )
    return s


# ── paired Complete-vs-Baseline machinery (prepared for later use) ────

def paired_difference(complete: np.ndarray, baseline: np.ndarray,
                      key: str) -> np.ndarray:
    """Per-episode difference, sign so that positive ⇒ Complete better.

    Duration (lower better): d = duration_baseline − duration_complete.
    Everything else (higher better): d = value_complete − value_baseline.
    Requires MATCHED (same seeds/layouts) equal-length arrays.
    """
    a = np.asarray(complete, dtype=float)
    b = np.asarray(baseline, dtype=float)
    return (b - a) if key.startswith("duration") else (a - b)


def bootstrap_paired_diff_means(complete: np.ndarray, baseline: np.ndarray,
                                key: str, sample_size: int, n_bootstrap: int,
                                rng: np.random.Generator) -> np.ndarray:
    """Bootstrap the mean PAIRED difference (matched pairs kept intact)."""
    a = np.asarray(complete, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        raise ValueError("complete/baseline must be matched, equal-length arrays.")
    mask = np.isfinite(a) & np.isfinite(b)
    d = paired_difference(a[mask], b[mask], key)
    return bootstrap_sample_means(d, sample_size, n_bootstrap, rng)


# =====================================================================
#  Normality diagnostic (OFF by default; NOT a decision rule)
# =====================================================================

def normality_tests_of_means(means_by_kpi: Dict[str, np.ndarray],
                             alpha: float) -> Dict[str, dict]:
    """D'Agostino-Pearson on each KPI's bootstrap means (shape diagnostic only)."""
    results: Dict[str, dict] = {}
    for spec in ANALYSIS_KPIS:
        x = np.asarray(means_by_kpi.get(spec.key, np.empty(0)), dtype=float)
        x = x[np.isfinite(x)]
        result = {"n": int(x.size), "statistic": None, "pvalue": None,
                  "is_normal": None, "reason": None}
        if x.size < 8:
            result["reason"] = "requires at least 8 bootstrap means"
        elif np.ptp(x) == 0.0:
            result["reason"] = "all bootstrap means are identical"
        else:
            test = normaltest(x)
            statistic, pvalue = float(test.statistic), float(test.pvalue)
            if np.isfinite(statistic) and np.isfinite(pvalue):
                result["statistic"] = statistic
                result["pvalue"] = pvalue
                result["is_normal"] = pvalue >= alpha
            else:
                result["reason"] = "test returned a non-finite result"
        results[spec.key] = result
    return results


def _print_normality_of_means(results: Dict[str, dict]) -> None:
    print("      -- OPTIONAL bootstrap-mean shape diagnostic "
          "(D'Agostino-Pearson) --", flush=True)
    print("         This is an optional shape diagnostic only. With B bootstrap "
          "replicates, very small", flush=True)
    print("         deviations from normality may produce small p-values. The "
          "bootstrap confidence", flush=True)
    print("         interval does not require normality.", flush=True)
    for spec in ANALYSIS_KPIS:
        r = results.get(spec.key, {})
        if r.get("is_normal") is None:
            print(f"           {spec.label:34s} not testable ({r.get('reason')})",
                  flush=True)
        else:
            print(f"           {spec.label:34s} K^2={r['statistic']:.4g}  "
                  f"p={r['pvalue']:.4g}", flush=True)


# =====================================================================
#  scenario-name validation
# =====================================================================

def resolve_display_scenario(section: CurriculumSection, mode: str) -> str:
    """Detect a mismatch between the display name's S# token and section.scenario.

    mode='error' raises; 'warn' prints a prominent warning and rewrites the
    token to the resolved scenario for all titles/filenames; 'ignore' keeps the
    original name unchanged.
    """
    name = section.name
    resolved = getattr(section, "scenario", None)
    m = SCENARIO_TOKEN_RE.search(name)
    if not (m and resolved):
        return name
    token = m.group(0).upper()
    if token == str(resolved).upper():
        return name
    msg = (f"scenario display name contains {token}, but section.scenario is "
           f"{resolved}. Rename the scenario or correct its configuration.")
    if mode == "error":
        raise ValueError("Scenario mismatch: " + msg)
    if mode == "ignore":
        print(f"  NOTE (ignored): {msg}", flush=True)
        return name
    corrected = SCENARIO_TOKEN_RE.sub(str(resolved), name)
    print("  " + "!" * 66, flush=True)
    print(f"  WARNING: {msg}", flush=True)
    print(f"           Using resolved scenario in titles/filenames: '{corrected}'.",
          flush=True)
    print("  " + "!" * 66, flush=True)
    return corrected


# =====================================================================
#  PLOTS
# =====================================================================

def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def plot_cov(curves: Dict[str, Dict[str, np.ndarray]], max_eps: int,
             display: str, out_png: Path) -> None:
    """Pilot diagnostic: c_v vs number of episodes, one line per CoV KPI."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    cap_vals = []
    for spec, color in zip(COV_KPIS, NLR_CYCLE):
        c = curves.get(spec.key)
        if c is None:
            continue
        pm = c["n"] >= 2
        nn = c["n"][pm]
        ax.fill_between(nn, c["min"][pm], c["max"][pm], color=color, alpha=0.15, lw=0)
        ax.plot(nn, c["mean"][pm], color=color, lw=1.9, label=spec.label)
        stable = c["n"] >= max(STABILITY_MIN_N, 2)
        if stable.any():
            cap_vals.append(float(np.nanmax(c["max"][stable])))
    ax.set_xlim(2, max_eps)
    if cap_vals:
        ax.set_ylim(0, max(cap_vals) * 1.15)
    ax.set_xlabel("Number of evaluation episodes")
    ax.set_ylabel(r"Coefficient of variation  $c_v = \sigma / \mu$")
    ax.grid(True, alpha=0.4)
    ax.legend(title="KPI (pilot diagnostic)", fontsize=8.5, framealpha=0.9)
    ax.set_title(f"CoV stabilisation diagnostic — {display}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_precision_curves(candidate_ns: np.ndarray,
                          hw_by_kpi: Dict[str, np.ndarray],
                          reqs: Dict[str, Optional[int]],
                          recommended: Optional[int],
                          display: str, out_png: Path) -> None:
    """Pilot diagnostic: CI half-width vs candidate sample size, one panel/KPI."""
    specs = PRECISION_KPIS
    ncol = 3
    nrow = int(np.ceil(len(specs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for idx, spec in enumerate(specs):
        ax = axes[idx]
        color = NLR_CYCLE[idx % len(NLR_CYCLE)]
        hw = hw_by_kpi.get(spec.key)
        thr = PRECISION_TARGETS.get(spec.key)
        if hw is None or not np.isfinite(hw).any():
            ax.set_visible(False)
            continue
        ax.plot(candidate_ns, hw, color=color, lw=1.9, marker="o", ms=2.5)
        if thr is not None:
            ax.axhline(thr, color=NLR_ACCENT, ls="--", lw=1.3,
                       label=f"target ±{thr:g}")
        req = reqs.get(spec.key)
        if req is not None:
            ax.axvline(req, color=NLR_REFERENCE, ls=":", lw=1.3,
                       label=f"n*={req}")
        if recommended is not None:
            ax.axvline(recommended, color=NLR_SECONDARY, ls="-", lw=1.1, alpha=0.8)
        ax.set_title(spec.label, fontsize=9.5)
        ax.set_xlabel("evaluation episodes n")
        ax.set_ylabel("95% CI half-width")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, framealpha=0.9)
    for ax in axes[len(specs):]:
        ax.set_visible(False)
    fig.suptitle("Precision planning: CI half-width vs sample size  "
                 f"(blue line = recommended N_eval={recommended}) — {display}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_mean_sampling_distributions(display: str, summaries: Dict[str, dict],
                                     n_bootstrap: int, mode: str, neval: int,
                                     out_png: Path) -> None:
    """Histograms of the bootstrap sample means (per KPI)."""
    pilot = (mode == "pilot")
    mean_label = "pilot pooled mean" if pilot else "final sample mean"
    specs = ANALYSIS_KPIS
    ncol = 3
    nrow = int(np.ceil(len(specs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for idx, spec in enumerate(specs):
        ax = axes[idx]
        s = summaries.get(spec.key)
        if s is None or s["means"].size == 0:
            ax.set_visible(False)
            continue
        color = NLR_CYCLE[idx % len(NLR_CYCLE)]
        means = s["means"]
        ax.hist(means, bins=45, density=True, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.axvline(s["mean"], color=NLR_REFERENCE, ls="--", lw=1.5, label=mean_label)
        ax.axvline(s["ci_lo"], color=NLR_ACCENT, ls=":", lw=1.4, label="95% CI")
        ax.axvline(s["ci_hi"], color=NLR_ACCENT, ls=":", lw=1.4)
        nlabel = (f"n_completed = {s['sample_size']}" if spec.conditional
                  else f"N_eval = {s['sample_size']}")
        title = (f"{spec.label}\n({nlabel})\n"
                 f"{mean_label} = {s['mean']:.3g}\n"
                 f"SEM(theory) = {s['theoretical_sem']:.2g}   "
                 f"SE(boot) = {s['bootstrap_se']:.2g}\n"
                 f"95% CI = [{s['ci_lo']:.3g}, {s['ci_hi']:.3g}]")
        ax.set_title(title, fontsize=8.2)
        ax.set_xlabel("sample mean")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7.0, framealpha=0.9)
    for ax in axes[len(specs):]:
        ax.set_visible(False)
    if pilot:
        sup = ("Expected sampling distributions of KPI means\n"
               f"Pilot planning analysis, proposed N_eval = {neval}, "
               f"B = {n_bootstrap:,} — {display}")
    else:
        sup = ("Sampling distributions of KPI means\n"
               f"Final statistical inference, N_eval = {neval}, "
               f"B = {n_bootstrap:,} bootstrap resamples — {display}")
    fig.suptitle(sup, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_qq_means(display: str, summaries: Dict[str, dict], out_png: Path) -> None:
    """Q-Q dashboard of each KPI's bootstrap means vs a normal reference."""
    specs = ANALYSIS_KPIS
    ncol = 3
    nrow = int(np.ceil(len(specs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.7 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for idx, spec in enumerate(specs):
        ax = axes[idx]
        s = summaries.get(spec.key)
        if s is None or s["means"].size < 3 or np.ptp(s["means"]) == 0:
            ax.set_visible(False)
            continue
        color = NLR_CYCLE[idx % len(NLR_CYCLE)]
        (osm, osr), (slope, intercept, r) = probplot(s["means"], dist="norm")
        ax.scatter(osm, osr, s=5, color=color, alpha=0.4, edgecolors="none")
        ax.plot(osm, slope * osm + intercept, color=NLR_REFERENCE, lw=1.3)
        ax.set_title(f"{spec.label}   (R²={r * r:.4f}, skew={s['skew']:.2g})",
                     fontsize=8.5)
        ax.set_xlabel("theoretical quantiles")
        ax.set_ylabel("ordered sample means")
        ax.grid(True, alpha=0.3)
    for ax in axes[len(specs):]:
        ax.set_visible(False)
    fig.suptitle("Q-Q plots of the bootstrap sampling distributions of the mean\n"
                 f"(visual shape diagnostic) — {display}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_raw_distributions(display: str, derived: Dict[str, np.ndarray],
                           n_runs: int, out_png: Path) -> None:
    """DESCRIPTIVE dashboard of raw episode-level KPI histograms."""
    specs = ANALYSIS_KPIS
    ncol = 3
    nrow = int(np.ceil(len(specs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for idx, spec in enumerate(specs):
        ax = axes[idx]
        color = NLR_CYCLE[idx % len(NLR_CYCLE)]
        x = derived[spec.key]
        x = x[np.isfinite(x)]
        if x.size == 0:
            ax.set_visible(False)
            continue
        mu = float(x.mean())
        sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
        if spec.key == "completion":                  # binary KPI → two bars
            ax.hist(x, bins=[-0.5, 0.5, 1.5], density=True, color=color,
                    alpha=0.85, edgecolor="white", linewidth=0.5, rwidth=0.9)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["fail", "complete"])
        else:
            ax.hist(x, bins=30, density=True, color=color, alpha=0.85,
                    edgecolor="white", linewidth=0.3)
        ax.axvline(mu, color=NLR_REFERENCE, ls="--", lw=1.2)
        if spec.key != "reward" and mu != 0.0:
            stat = rf"$\mu$={mu:.3g}   $\sigma$={sd:.3g}   $c_v$={sd / abs(mu):.3g}"
        else:
            stat = rf"$\mu$={mu:.3g}   $\sigma$={sd:.3g}"
        ax.set_title(f"{spec.label}\n{stat}", fontsize=9)
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
    for ax in axes[len(specs):]:
        ax.set_visible(False)
    fig.suptitle(f"Raw episode-level KPI distributions — {display}   "
                 f"(descriptive; {n_runs} episodes)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# =====================================================================
#  CSV writers
# =====================================================================

def _write_samples_csv(per_seed: List[Dict[str, np.ndarray]],
                       out_path: Path) -> None:
    """Raw per-episode stats, one row per (evaluation RNG seed, episode)."""
    import csv
    keys = [k for k, _ in RAW_STATS]
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["eval_rng_seed", "episode"] + keys)
        for s_idx, seed_arr in enumerate(per_seed):
            n = max((seed_arr[k].size for k in keys), default=0)
            for i in range(n):
                w.writerow([s_idx, i] + [
                    (seed_arr[k][i] if i < seed_arr[k].size else "") for k in keys])


def _write_bootstrap_means_csv(scenario: str, summaries: Dict[str, dict],
                               out_path: Path) -> None:
    import csv
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "kpi", "sample_size",
                    "bootstrap_iteration", "bootstrap_mean"])
        for spec in ANALYSIS_KPIS:
            s = summaries.get(spec.key)
            if s is None:
                continue
            n = s["sample_size"]
            for b, m in enumerate(s["means"]):
                w.writerow([scenario, spec.key, n, b, float(m)])


def _write_summary_csv(scenario: str, policy: str, mode: str, total_eval: int,
                       summaries: Dict[str, dict], counts: Dict[str, int],
                       out_path: Path) -> None:
    """Compact per-KPI summary of the sampling distribution of the mean."""
    import csv
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "policy", "analysis_mode", "total_eval_episodes",
                    "kpi", "n_available", "bootstrap_sample_size", "mean",
                    "raw_sd", "theoretical_sem", "bootstrap_se",
                    "ci_2_5", "ci_97_5", "ci_half_width",
                    "completion_count", "failure_count",
                    "n_completed", "conditional_on_completion"])
        for spec in ANALYSIS_KPIS:
            s = summaries.get(spec.key)
            if s is None:
                continue
            w.writerow([
                scenario, policy, mode, total_eval, spec.key,
                s["n_available"], s["sample_size"], s["mean"], s["raw_sd"],
                s["theoretical_sem"], s["bootstrap_se"], s["ci_lo"], s["ci_hi"],
                s["ci_half_width"], counts["completed"], counts["failed"],
                (counts["completed"] if spec.conditional else ""),
                (True if spec.conditional else False),
            ])


def _write_recommendation_csv(scenario: str, cov_reqs: Dict[str, Optional[int]],
                              prec_reqs: Dict[str, Optional[int]],
                              cov_common: Optional[int],
                              prec_common: Optional[int],
                              recommended: Optional[int], out_path: Path) -> None:
    import csv
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "kpi", "cov_required_n", "precision_threshold",
                    "precision_required_n", "cov_common_n", "precision_common_n",
                    "recommended_n_eval"])
        for spec in ANALYSIS_KPIS:
            if not (spec.cov or spec.precision):
                continue
            w.writerow([
                scenario, spec.key, cov_reqs.get(spec.key, ""),
                PRECISION_TARGETS.get(spec.key, ""), prec_reqs.get(spec.key, ""),
                cov_common if cov_common is not None else "",
                prec_common if prec_common is not None else "",
                recommended if recommended is not None else "",
            ])


# =====================================================================
#  PILOT-mode driver
# =====================================================================

def _run_pilot_scenario(section, display, ckpt, ckpt_path, env_cfg, policy,
                        device, opts: RunOpts) -> None:
    print("\n  ANALYSIS MODE: PILOT SAMPLE-SIZE PLANNING", flush=True)
    seed_word = "seed" if opts.n_seeds == 1 else "seeds"
    print(f"  Pilot episode pool: {opts.max_episodes} episodes", flush=True)
    print(f"  Evaluation RNG replicates: {opts.n_seeds} ({seed_word})", flush=True)
    print(f"  {_section_label(section, env_cfg)}", flush=True)

    seeds = [opts.base_seed + s * SEED_STRIDE for s in range(opts.n_seeds)]
    per_seed_raw = collect_pool(policy, env_cfg, ckpt, opts.max_episodes, seeds,
                                device, opts.chunk)
    per_seed_der = [derive_analysis_kpis(r, env_cfg.max_steps) for r in per_seed_raw]
    pooled = {k: np.concatenate([d[k] for d in per_seed_der]) for k in per_seed_der[0]}
    counts = episode_counts(pooled)

    shuffle_rng = np.random.default_rng(opts.base_seed)
    boot_rng = np.random.default_rng(opts.bootstrap_seed + opts.scenario_index_hash())

    # ── CoV diagnostic ──
    curves = cov_curves(per_seed_der, opts.n_orderings, shuffle_rng)
    cov_reqs = {spec.key: (_required_runs(curves[spec.key]["n"],
                                          curves[spec.key]["mean"],
                                          opts.tol, opts.min_n)
                           if spec.key in curves else None)
                for spec in COV_KPIS}
    cov_common = max([r for r in cov_reqs.values() if r is not None], default=None)

    print("\n  CoV diagnostic:", flush=True)
    for spec in COV_KPIS:
        r = cov_reqs.get(spec.key)
        print(f"      {spec.label:34s} n*={str(r) if r else 'not reached'}", flush=True)

    # ── precision (CI half-width) planning ──
    step = max(2, opts.precision_step)
    candidate_ns = np.arange(step, opts.max_episodes + 1, step, dtype=int)
    if candidate_ns.size == 0:
        candidate_ns = np.array([opts.max_episodes], dtype=int)
    hw_by_kpi: Dict[str, np.ndarray] = {}
    prec_reqs: Dict[str, Optional[int]] = {}
    print("\n  Precision requirement (target 95% CI half-width):", flush=True)
    for spec in PRECISION_KPIS:
        pool = pooled[spec.key]
        hw, _se = precision_curve(pool, candidate_ns, opts.precision_bootstrap, boot_rng)
        thr = PRECISION_TARGETS.get(spec.key, float("inf"))
        req = precision_required_n(candidate_ns, hw, thr)
        hw_by_kpi[spec.key] = hw
        prec_reqs[spec.key] = req
        print(f"      {spec.label:34s} n*={str(req) if req else 'not reached'}"
              f"   (target ±{thr:g})", flush=True)
    prec_common = max([r for r in prec_reqs.values() if r is not None], default=None)

    raw_reco = max([n for n in (cov_common, prec_common) if n is not None],
                   default=opts.max_episodes)
    recommended = _round_up(raw_reco, opts.round_to)
    proposed = int(opts.eval_episodes) if opts.eval_episodes else int(recommended)

    print(f"\n  CoV-based common count: "
          f"{cov_common if cov_common is not None else 'not reached'}", flush=True)
    print(f"  Precision-based common count: "
          f"{prec_common if prec_common is not None else 'not reached'}", flush=True)
    print(f"  CoV diagnostic suggests at least "
          f"{cov_common if cov_common else '?'} episodes.", flush=True)
    print(f"  Precision analysis suggests at least "
          f"{prec_common if prec_common else '?'} episodes.", flush=True)
    print(f"  Recommended common final evaluation count: {recommended} episodes"
          f"  (round-up of max to nearest {opts.round_to})", flush=True)
    if opts.eval_episodes:
        print(f"  (user override via --eval_episodes: using proposed N_eval="
              f"{proposed} for the expected distributions)", flush=True)

    # ── EXPECTED sampling distributions at the proposed N_eval ──
    p_complete = float(np.nanmean(pooled["completion"])) if pooled["completion"].size else 0.0
    expected_completed = max(2, int(round(p_complete * proposed)))
    print(f"\n  Expected sampling distributions at N_eval={proposed}"
          f"  (B={opts.n_bootstrap:,} bootstrap resamples):", flush=True)
    print(f"      Total pilot episodes: {counts['total']}", flush=True)
    print(f"      Completed episodes available for conditional duration: "
          f"{counts['completed']}", flush=True)
    print(f"      Conditional-duration bootstrap resample size (expected at "
          f"N_eval): {expected_completed}", flush=True)

    summaries: Dict[str, dict] = {}
    for spec in ANALYSIS_KPIS:
        size = expected_completed if spec.conditional else proposed
        s = summarise_mean_distribution(spec, pooled[spec.key], size,
                                        opts.n_bootstrap, boot_rng)
        summaries[spec.key] = s
        if s["means"].size == 0:
            print(f"      {spec.label:34s} no finite episodes — skipped", flush=True)
            continue
        warn = "  [WARN resample size > available pool]" if s["n_gt_available"] else ""
        print(f"      {spec.label}:", flush=True)
        print(f"          pilot pooled mean = {s['mean']:.4g}", flush=True)
        print(f"          available pilot observations = {s['n_available']}", flush=True)
        print(f"          bootstrap resample size = {s['sample_size']}{warn}", flush=True)
        print(f"          theoretical SEM = {s['theoretical_sem']:.3g}", flush=True)
        print(f"          bootstrap SE = {s['bootstrap_se']:.3g}", flush=True)
        print(f"          expected 95% CI = [{s['ci_lo']:.4g}, {s['ci_hi']:.4g}]  "
              f"(half-width {s['ci_half_width']:.3g})", flush=True)

    # ── figures ──
    tag = _safe_name(display)
    cov_png = opts.out_dir / f"pilot_cov_{tag}_{opts.stamp}.png"
    plot_cov(curves, opts.max_episodes, display, cov_png)
    print(f"\n      saved CoV diagnostic     : {cov_png}", flush=True)

    prec_png = opts.out_dir / f"pilot_precision_{tag}_{opts.stamp}.png"
    plot_precision_curves(candidate_ns, hw_by_kpi, prec_reqs, recommended, display, prec_png)
    print(f"      saved precision curves   : {prec_png}", flush=True)

    mean_png = opts.out_dir / f"pilot_expected_means_{tag}_{opts.stamp}.png"
    plot_mean_sampling_distributions(display, summaries, opts.n_bootstrap,
                                     "pilot", proposed, mean_png)
    print(f"      saved expected means     : {mean_png}", flush=True)

    qq_png = opts.out_dir / f"pilot_qq_{tag}_{opts.stamp}.png"
    plot_qq_means(display, summaries, qq_png)
    print(f"      saved Q-Q of means       : {qq_png}", flush=True)

    if opts.save_raw_dist:
        raw_png = opts.out_dir / f"pilot_raw_dist_{tag}_{opts.stamp}.png"
        plot_raw_distributions(display, pooled, counts["total"], raw_png)
        print(f"      saved raw distributions  : {raw_png}", flush=True)

    if opts.run_normality:
        norm = normality_tests_of_means({k: summaries[k]["means"] for k in summaries},
                                        opts.normality_alpha)
        _print_normality_of_means(norm)

    # ── CSVs ──
    if opts.write_csv:
        samp_csv = opts.out_dir / f"pilot_samples_{tag}_{opts.stamp}.csv"
        _write_samples_csv(per_seed_raw, samp_csv)
        print(f"      saved raw samples CSV    : {samp_csv}", flush=True)

        reco_csv = opts.out_dir / f"pilot_recommendation_{tag}_{opts.stamp}.csv"
        _write_recommendation_csv(display, cov_reqs, prec_reqs, cov_common,
                                  prec_common, recommended, reco_csv)
        print(f"      saved recommendation CSV : {reco_csv}", flush=True)

        if opts.write_bootstrap_csv:
            boot_csv = opts.out_dir / f"pilot_bootstrap_means_{tag}_{opts.stamp}.csv"
            _write_bootstrap_means_csv(display, summaries, boot_csv)
            print(f"      saved bootstrap CSV      : {boot_csv}", flush=True)


# =====================================================================
#  FINAL-mode driver
# =====================================================================

def _run_final_scenario(section, display, ckpt, ckpt_path, env_cfg, policy,
                        device, opts: RunOpts) -> None:
    N = int(opts.eval_episodes)
    print("\n  ANALYSIS MODE: FINAL STATISTICAL INFERENCE", flush=True)
    print(f"  Final evaluation episodes: {N}", flush=True)
    print(f"  Bootstrap resamples: {opts.n_bootstrap:,}", flush=True)
    print(f"  Bootstrap resample size: {N}", flush=True)
    print(f"  {_section_label(section, env_cfg)}", flush=True)
    if opts.n_seeds != 1:
        print("  NOTE: final mode uses ONE evaluation RNG seed; --n_seeds is "
              "ignored here.", flush=True)

    # Independent draw from the pilot pool (distinct evaluation RNG seed).
    final_seed = opts.base_seed + FINAL_SEED_OFFSET
    per_seed_raw = collect_pool(policy, env_cfg, ckpt, N, [final_seed],
                                device, opts.chunk)
    derived = derive_analysis_kpis(per_seed_raw[0], env_cfg.max_steps)
    counts = episode_counts(derived)

    boot_rng = np.random.default_rng(opts.bootstrap_seed + opts.scenario_index_hash())

    # One common N_eval for every PRIMARY KPI; conditional duration uses n_completed.
    n_completed = max(2, counts["completed"])
    summaries: Dict[str, dict] = {}
    for spec in ANALYSIS_KPIS:
        size = n_completed if spec.conditional else N
        summaries[spec.key] = summarise_mean_distribution(
            spec, derived[spec.key], size, opts.n_bootstrap, boot_rng)

    # ── report ──
    print(f"\n  Completed missions: {counts['completed']}/{counts['total']}  "
          f"({100.0 * counts['completed'] / max(counts['total'], 1):.1f}%)  |  "
          f"failed: {counts['failed']}", flush=True)

    for spec in PRIMARY_KPIS:
        s = summaries[spec.key]
        if s["means"].size == 0:
            print(f"\n  {spec.label}: no finite episodes — skipped", flush=True)
            continue
        warn = "  [WARN N_eval > available finite pool]" if s["n_gt_available"] else ""
        print(f"\n  {spec.label}:", flush=True)
        print(f"      final sample mean = {s['mean']:.4g}"
              f"   (from {s['n_available']} finite obs of N={N}){warn}", flush=True)
        print(f"      95% bootstrap CI = [{s['ci_lo']:.4g}, {s['ci_hi']:.4g}]  "
              f"(half-width {s['ci_half_width']:.3g})", flush=True)
        print(f"      theoretical SEM = {s['theoretical_sem']:.3g}   "
              f"bootstrap SE = {s['bootstrap_se']:.3g}", flush=True)
        if spec.key == "completion":
            k = counts["completed"]
            wlo, whi = wilson_interval(k, N)
            p_hat = s["mean"]
            npq_ok = (N * p_hat >= 10.0) and (N * (1.0 - p_hat) >= 10.0)
            print(f"      95% Wilson CI = [{wlo:.4g}, {whi:.4g}]"
                  f"   (k={k}, n={N})", flush=True)
            print(f"      normal-approx check (secondary): "
                  f"N·p̂={N * p_hat:.1f}, N·(1−p̂)={N * (1.0 - p_hat):.1f} → "
                  f"{'ok' if npq_ok else 'not met'} "
                  f"(does NOT invalidate the bootstrap/Wilson CI)", flush=True)

    # conditional duration (secondary) — reported with the completion rate.
    cs = summaries["duration_completed"]
    print("\n  Duration among completed missions:", flush=True)
    if cs["means"].size == 0:
        print("      no completed episodes — not available", flush=True)
    else:
        print(f"      completed episodes = {counts['completed']}/{counts['total']}",
              flush=True)
        print(f"      n_completed = {cs['n_available']}", flush=True)
        print(f"      conditional mean = {cs['mean']:.4g}", flush=True)
        print(f"      conditional 95% bootstrap CI = "
              f"[{cs['ci_lo']:.4g}, {cs['ci_hi']:.4g}]", flush=True)
    print("      (report the completion rate alongside this metric: a policy can "
          "look fast by\n       completing only the easy missions.)", flush=True)

    # ── figures ──
    tag = _safe_name(display)
    mean_png = opts.out_dir / f"final_means_{tag}_{opts.stamp}.png"
    plot_mean_sampling_distributions(display, summaries, opts.n_bootstrap,
                                     "final", N, mean_png)
    print(f"\n      saved final means        : {mean_png}", flush=True)

    qq_png = opts.out_dir / f"final_qq_{tag}_{opts.stamp}.png"
    plot_qq_means(display, summaries, qq_png)
    print(f"      saved Q-Q of means       : {qq_png}", flush=True)

    if opts.save_raw_dist:
        raw_png = opts.out_dir / f"final_raw_dist_{tag}_{opts.stamp}.png"
        plot_raw_distributions(display, derived, counts["total"], raw_png)
        print(f"      saved raw distributions  : {raw_png}", flush=True)

    if opts.run_normality:
        norm = normality_tests_of_means({k: summaries[k]["means"] for k in summaries},
                                        opts.normality_alpha)
        _print_normality_of_means(norm)

    # ── CSVs ──
    if opts.write_csv:
        samp_csv = opts.out_dir / f"final_samples_{tag}_{opts.stamp}.csv"
        _write_samples_csv(per_seed_raw, samp_csv)
        print(f"      saved raw samples CSV    : {samp_csv}", flush=True)

        summ_csv = opts.out_dir / f"final_summary_{tag}_{opts.stamp}.csv"
        _write_summary_csv(display, ckpt_path.name, "final", N, summaries, counts,
                           summ_csv)
        print(f"      saved final summary CSV  : {summ_csv}", flush=True)

        if opts.write_bootstrap_csv:
            boot_csv = opts.out_dir / f"final_bootstrap_means_{tag}_{opts.stamp}.csv"
            _write_bootstrap_means_csv(display, summaries, boot_csv)
            print(f"      saved bootstrap CSV      : {boot_csv}", flush=True)


# =====================================================================
#  DRIVER
# =====================================================================

def analyse_scenarios(scenarios: List[CurriculumSection],
                      default_ckpt_path: Optional[Path],
                      opts: RunOpts, device: torch.device) -> None:
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, _LoadedCheckpoint] = {}

    for i, section in enumerate(scenarios):
        opts._scenario_index = i          # for reproducible per-scenario RNG offset
        display = resolve_display_scenario(section, opts.scenario_name_check)

        ckpt_path = _resolve_policy_path(section.policy_file, default_ckpt_path)
        if ckpt_path is None:
            raise RuntimeError(
                f"Scenario '{section.name}' sets no policy_file and no "
                f"--checkpoint default was given — cannot pick a policy.")
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Scenario '{section.name}': policy not found: {ckpt_path}")

        key = str(ckpt_path.resolve())
        if key not in cache:
            print(f"  Loading policy: {ckpt_path.name}", flush=True)
            cache[key] = _LoadedCheckpoint(ckpt_path, device)
        ckpt = cache[key]

        env_cfg = _section_to_env_cfg(section, ckpt.base_env_cfg,
                                      ckpt.reward_cfg, ckpt.fofe_cfg)
        print(f"\n[{i + 1}/{len(scenarios)}] {display}  [{ckpt_path.name}]", flush=True)

        policy = _build_policy_for_scenario(ckpt, env_cfg, device)

        if opts.analysis_mode == "final":
            _run_final_scenario(section, display, ckpt, ckpt_path, env_cfg,
                                policy, device, opts)
        else:
            _run_pilot_scenario(section, display, ckpt, ckpt_path, env_cfg,
                                policy, device, opts)

        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# small helper attached to RunOpts for a reproducible per-scenario RNG offset
def _scenario_index_hash(self: RunOpts) -> int:
    return int(getattr(self, "_scenario_index", 0))


RunOpts.scenario_index_hash = _scenario_index_hash   # type: ignore[attr-defined]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pilot sample-size PLANNING and FINAL statistical INFERENCE "
        "for the SCENARIOS defined at the top of this file.")
    p.add_argument("--analysis_mode", choices=["pilot", "final"], default="pilot",
                   help="'pilot' plans the sample size; 'final' runs the study "
                        "at a fixed --eval_episodes (default: pilot).")
    p.add_argument("--eval_episodes", type=int, default=None, metavar="N",
                   help="FINAL mode: number of final evaluation episodes "
                        "(required). PILOT mode: optional override of the "
                        "proposed N_eval.")
    p.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                   help="Default .pt for scenarios that leave policy_file=None.")
    p.add_argument("--max_episodes", type=int, default=MAX_EPISODES,
                   help=f"PILOT pool size per eval RNG seed (default: {MAX_EPISODES}).")
    p.add_argument("--n_seeds", type=int, default=N_SEEDS,
                   help=f"Evaluation RNG seeds = extra pilot pools (default: {N_SEEDS}).")
    p.add_argument("--seed", type=int, default=BASE_SEED,
                   help=f"Evaluation RNG base seed (default: {BASE_SEED}).")
    p.add_argument("--chunk", type=int, default=CHUNK_EPISODES,
                   help=f"Parallel envs per rollout (default: {CHUNK_EPISODES}).")
    p.add_argument("--n_orderings", type=int, default=N_ORDERINGS,
                   help=f"Shuffles for order-independent CoV (default: {N_ORDERINGS}).")
    p.add_argument("--tol", type=float, default=STABILITY_TOL,
                   help=f"Relative stability band around final CoV (default: {STABILITY_TOL}).")
    p.add_argument("--min_n", type=int, default=STABILITY_MIN_N,
                   help=f"Ignore n below this when detecting CoV n* (default: {STABILITY_MIN_N}).")
    p.add_argument("--precision_step", type=int, default=PRECISION_STEP,
                   help=f"Candidate sample-size step for precision curves (default: {PRECISION_STEP}).")
    p.add_argument("--precision_bootstrap", type=int, default=PRECISION_BOOTSTRAP,
                   help=f"Bootstrap resamples per precision point (default: {PRECISION_BOOTSTRAP}).")
    p.add_argument("--round_to", type=int, default=ROUND_TO,
                   help=f"Round the recommended N_eval up to this multiple (default: {ROUND_TO}).")
    p.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP,
                   help=f"Bootstrap resamples B for reported distributions (default: {N_BOOTSTRAP}).")
    p.add_argument("--bootstrap_seed", type=int, default=BOOTSTRAP_SEED,
                   help=f"RNG seed for all bootstrap resampling (default: {BOOTSTRAP_SEED}).")
    p.add_argument("--normality_alpha", type=float, default=NORMALITY_ALPHA,
                   help=f"Optional normality-diagnostic alpha (default: {NORMALITY_ALPHA}).")
    p.add_argument("--run_bootstrap_normality_diagnostic", action="store_true",
                   help="Run the OPTIONAL D'Agostino-Pearson shape diagnostic on "
                        "the bootstrap means (off by default; NOT a decision rule).")
    p.add_argument("--scenario_name_check", choices=["warn", "error", "ignore"],
                   default="warn",
                   help="How to handle a display-name vs section.scenario mismatch "
                        "(default: warn + use resolved scenario).")
    p.add_argument("--device", type=str, default=None, metavar="DEVICE",
                   help="Torch device, e.g. 'cpu' or 'cuda:0'. Default: auto.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output dir (default: <script_dir>/stat_results).")
    p.add_argument("--no_csv", action="store_true", help="Do not write any CSV.")
    p.add_argument("--no_bootstrap_csv", action="store_true",
                   help="Skip the (large) per-iteration bootstrap-means CSV.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not SCENARIOS:
        raise RuntimeError("SCENARIOS is empty — define at least one scenario.")
    if not 0.0 < args.normality_alpha < 1.0:
        raise ValueError("--normality_alpha must be strictly between 0 and 1.")
    if args.n_bootstrap < 100:
        raise ValueError("--n_bootstrap should be at least 100.")
    if args.analysis_mode == "final":
        if args.eval_episodes is None:
            raise ValueError("--analysis_mode final requires --eval_episodes N "
                             "(the final evaluation count chosen from a pilot run).")
        if args.eval_episodes < 2:
            raise ValueError("--eval_episodes must be >= 2.")
    if args.eval_episodes is not None and args.eval_episodes < 2:
        raise ValueError("--eval_episodes must be >= 2.")

    default_ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if default_ckpt_path is not None and not default_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {default_ckpt_path}")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else (_PKG_DIR / "stat_results")

    opts = RunOpts(
        analysis_mode=args.analysis_mode, max_episodes=args.max_episodes,
        eval_episodes=args.eval_episodes, n_seeds=args.n_seeds, base_seed=args.seed,
        chunk=args.chunk, n_orderings=args.n_orderings, tol=args.tol,
        min_n=args.min_n, precision_step=args.precision_step,
        precision_bootstrap=args.precision_bootstrap, round_to=args.round_to,
        n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed,
        normality_alpha=args.normality_alpha,
        run_normality=args.run_bootstrap_normality_diagnostic,
        scenario_name_check=args.scenario_name_check,
        out_dir=out_dir, stamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        write_csv=not args.no_csv, write_bootstrap_csv=not args.no_bootstrap_csv,
        save_raw_dist=SAVE_RAW_DIST,
    )

    print("─" * 70)
    print(f"  Device        : {device}")
    print(f"  Analysis mode : {args.analysis_mode.upper()}")
    print(f"  Scenarios     : {len(SCENARIOS)}")
    if args.analysis_mode == "pilot":
        print(f"  Pilot pool    : {args.max_episodes} episodes × {args.n_seeds} "
              f"evaluation RNG seed(s)")
    else:
        print(f"  Final N_eval  : {args.eval_episodes} episodes (1 evaluation RNG seed)")
    print(f"  Primary KPIs  : " + ", ".join(k.label for k in PRIMARY_KPIS))
    print(f"  Bootstrap     : B={args.n_bootstrap}, seed={args.bootstrap_seed}")
    print(f"  Normality     : "
          + ("ON (optional diagnostic)" if args.run_bootstrap_normality_diagnostic
             else "OFF (percentile CI needs no normality)"))
    print("─" * 70)

    analyse_scenarios(SCENARIOS, default_ckpt_path, opts, device)


if __name__ == "__main__":
    main()
