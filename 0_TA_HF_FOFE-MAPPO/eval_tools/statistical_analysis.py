r"""
statistical_analysis.py
═══════════════════════
Two linked analyses for a frozen FOFE-MAPPO policy's KPIs:

  (A) HOW MANY episodes are needed for the KPIs to STABILISE
      (coefficient-of-variation, CoV, method — unchanged), and

  (B) INFERENCE about the policy's AVERAGE performance, based on the *sampling
      distribution of the KPI mean* (new — replaces the old inferential use of
      the raw episode-level histograms).

──────────────────────────────────────────────────────────────────────────────
RAW DISTRIBUTIONS  vs  SAMPLING DISTRIBUTIONS OF THE MEAN
──────────────────────────────────────────────────────────────────────────────
Each evaluation episode is the experimental unit. One episode yields one value
for every KPI (completion, targets destroyed, survival, coalition
fragmentation, duration, episode reward). Collecting all KPIs from the same
episode is fine — but the different KPIs of one episode are NOT independent
simulation runs, so we never pool KPIs together.

There are two distinct distributions, and they answer different questions:

  • RAW episode-level distribution of a KPI describes mission-to-mission
    variation: "what is the spread of outcomes across individual episodes?"
    This is purely descriptive. A raw KPI (completion is binary; survival and
    targets are bounded discrete fractions; duration is right-skewed) does NOT
    have to be normal, and its non-normality does NOT invalidate mean-based
    inference.

  • SAMPLING distribution of the MEAN describes uncertainty in the *estimate*
    of average policy performance: "if we repeated the evaluation with n
    episodes, how much would the estimated mean KPI move?" Its standard
    deviation is the standard error of the mean (SEM). The bootstrap
    approximates this distribution by repeatedly drawing n episodes WITH
    replacement from the observed pool and recomputing the mean.

For a KPI K with finite episode values K_1..K_N and evaluation sample size n:
  1. draw B bootstrap resamples, each n values with replacement;
  2. each resample b gives one sample mean  mean_K_b = (1/n) Σ K_i;
  3. {mean_K_1 .. mean_K_B} is the bootstrap sampling distribution of the mean;
  4. we report  observed mean, raw SD, theoretical SEM = raw SD / √n,
     bootstrap SE = SD of the bootstrap means, 95% percentile CI, skewness,
     and a D'Agostino-Pearson diagnostic ON THE BOOTSTRAP MEANS.

The theoretical SEM and the bootstrap SE agree when the bootstrap is correct.
Note: the bootstrap SE is ALREADY the standard error of the original mean — it
is the SD of the bootstrap means; it is NOT divided by √B (B is only the number
of bootstrap repetitions). The 95% CI is a percentile interval of the bootstrap
means and does NOT depend on the sampling distribution being normal.

──────────────────────────────────────────────────────────────────────────────
CHOOSING n (episodes per sample)
──────────────────────────────────────────────────────────────────────────────
The sampling distribution of the mean depends on n, so n is made explicit:
  • --mean_sample_size N  forces a single n for every KPI, or
  • if omitted, each KPI uses ITS OWN required-run count n* from the CoV
    stabilisation analysis; a KPI that never stabilised falls back to the
    overall required count (max over KPIs), and finally to the available pool
    size. The n actually used is printed per KPI.
If n exceeds the number of collected episodes N, resampling with replacement is
still valid, but a warning is issued because the empirical pool is smaller than
the intended evaluation size — prefer collecting at least n raw episodes.

──────────────────────────────────────────────────────────────────────────────
PER-KPI NOTES
──────────────────────────────────────────────────────────────────────────────
  • Completion is binary (0/1); its raw values are never tested for normality.
    Its bootstrap mean is a completion PROPORTION; we report a bootstrap CI and
    whether the normal-approximation conditions n·p̂ ≥ 10 and n·(1−p̂) ≥ 10 hold.
  • Targets destroyed / survival are bounded discrete fractions; only their
    bootstrap means are analysed inferentially.
  • Duration: failed missions are handled explicitly (see
    DURATION_CONDITIONAL_ON_COMPLETION). When conditional on completion, the
    number of completed episodes is reported and n does NOT equal the total
    episode count.
  • Episode reward may be negative; that does not prevent bootstrapping its
    mean. It stays out of the CoV (σ/μ) analysis (μ can sit near zero) but is
    included in the sampling-distribution-of-means analysis.

──────────────────────────────────────────────────────────────────────────────
SEEDS
──────────────────────────────────────────────────────────────────────────────
--n_seeds generates several independent pools of evaluation episodes for the
SAME frozen checkpoint (evaluation RNG seeds, not different trained policies).
For one frozen policy these pools may be combined to estimate expected episode
performance. The seed id is preserved in the raw CSV for seed-level
diagnostics. If several INDEPENDENTLY TRAINED checkpoints are compared later,
do NOT pool their episodes: reduce episodes→mean-per-training-seed→distribution
across training-seed means (a separate hierarchical analysis).

──────────────────────────────────────────────────────────────────────────────
PAIRED MODEL COMPARISONS (prepared for later use)
──────────────────────────────────────────────────────────────────────────────
Helper functions (`paired_difference`, `bootstrap_paired_diff_means`,
`plot_paired_diff_distributions`) support a future Complete-vs-Baseline
comparison on MATCHED scenario seeds/layouts: keep matched episode pairs, form
per-episode differences d_i, resample the PAIR indices with replacement, and
bootstrap the mean paired difference (null reference = 0). Duration uses the
sign d_i = duration_baseline − duration_complete so that a positive difference
always means Complete is better. The two models are never bootstrapped
independently, so the pairing is preserved.

──────────────────────────────────────────────────────────────────────────────
OUTPUTS (per scenario, in <script_dir>/stat_results)
──────────────────────────────────────────────────────────────────────────────
  cov_<scn>_<stamp>.png            CoV stabilisation curves (unchanged)
  mean_sampling_dist_<scn>_..png   histograms of the bootstrap sample means
  mean_qq_<scn>_<stamp>.png        Q-Q plots of the bootstrap means (visual)
  raw_kpi_dist_<scn>_<stamp>.png   raw episode-level histograms (descriptive)
CSVs: raw episodes (+seed), bootstrap means, and a compact per-KPI summary.

──────────────────────────────────────────────────────────────────────────────
HOW TO USE
──────────────────────────────────────────────────────────────────────────────
Edit the SCENARIOS list below (same CurriculumSection format as
evaluate_policy.py / run_curriculum.py), then run with the project venv:

    .\.venv\Scripts\python.exe 0_TA_HF_FOFE-MAPPO\eval_tools\statistical_analysis.py

    # explicit evaluation sample size + fewer episodes for a smoke-test:
    .\.venv\Scripts\python.exe 0_TA_HF_FOFE-MAPPO\eval_tools\statistical_analysis.py ^
        --max_episodes 300 --n_seeds 1 --mean_sample_size 100 --n_bootstrap 2000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
        name="FOFE-MAPPO S1",
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

MAX_EPISODES   = 1000     # episodes collected per seed (the raw pool per seed)
N_SEEDS        = 1        # independent evaluation seeds (extra episode pools)
BASE_SEED      = 42       # seed s uses BASE_SEED + s * SEED_STRIDE
SEED_STRIDE    = 1_000_000
NORMALITY_ALPHA = 0.05    # D'Agostino-Pearson test significance level
CHUNK_EPISODES = 250      # parallel envs per rollout (chunked to avoid OOM)
N_ORDERINGS    = 64       # random shuffles → order-independent CoV curve
STABILITY_TOL  = 0.05     # "stable" = CoV within ±5% of its final value …
STABILITY_MIN_N = 10      # … but ignore the very noisy n < this region

# ── sampling-distribution-of-the-mean (bootstrap) config ─────────────
N_BOOTSTRAP    = 10_000   # bootstrap resamples B (number of sample means)
BOOTSTRAP_SEED = 12345    # RNG seed for ALL bootstrap resampling (reproducible)
MEAN_SAMPLE_SIZE: Optional[int] = None   # n per resample; None → CoV n* per KPI
# Cap on the [B, n] index matrix before the bootstrap chunks its resamples.
BOOTSTRAP_MAX_ELEMS = 20_000_000

# Duration treatment for FAILED missions. When True, duration is defined as
# "duration CONDITIONAL ON COMPLETION": only completed episodes enter the
# duration pool, the completed-episode count is reported, and n for duration is
# NOT the total episode count. When False, every finished episode is kept (an
# episode that never terminates contributes NaN and is dropped). This choice is
# always printed — failed missions are never silently removed.
DURATION_CONDITIONAL_ON_COMPLETION = True

SAVE_RAW_DIST = True      # also save the descriptive raw episode-level histograms


# =====================================================================
#  KPI columns:  (curve key, per-episode stat key, plot label)
#  KPI_COLUMNS = the CoV KPIs (all non-negative). Reward lives in HIST_KPIS only.
# =====================================================================

KPI_COLUMNS = [
    ("completion",    "mission_complete",        "Completion"),
    ("targets",       "targets_frac",            "Targets"),
    ("survival",      "survival_frac",           "Survival"),
    ("fragmentation", "coalition_fragmentation", "Coalition Frag"),
    ("duration",      "duration",                "Duration"),
]

# Reward is excluded from the CoV plot (signed → CoV undefined) but IS analysed
# in the sampling-distribution-of-means dashboard (its sign is irrelevant there).
REWARD_KPI = ("reward", "episode_total_reward", "Episode reward")

# Everything collected per episode + analysed (the CoV KPIs + reward).
HIST_KPIS = KPI_COLUMNS + [REWARD_KPI]


# =====================================================================
#  STEP 1 — collect RAW per-episode KPI values (the Xᵢ samples)  [UNCHANGED]
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
    """Return {kpi_key: array[n_episodes]} of per-episode values for one seed.

    Episodes are generated in independent parallel chunks (different seed each)
    so the whole pool is `n_episodes` i.i.d. randomised episodes.
    """
    per_kpi: Dict[str, List[float]] = {key: [] for key, *_ in HIST_KPIS}
    collected = 0
    chunk_idx = 0
    while collected < n_episodes:
        b = min(chunk, n_episodes - collected)
        ppo = PPOConfig(num_envs=b, device=device,
                        seed=_chunk_seed(master_seed, chunk_idx))
        env = build_env(env_cfg, ppo, hf_radar_cfg=hf_radar_cfg)
        stats = _rollout_collect(policy, env, env_cfg.max_steps, b, device)

        for s in stats:
            for key, stat_key, _ in HIST_KPIS:
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


# =====================================================================
#  STEP 2 — running coefficient of variation  c_v(n) = σ(n)/μ(n)  [UNCHANGED]
# =====================================================================

def _running_cov_shuffled(x: np.ndarray, n_orderings: int,
                          rng: np.random.Generator) -> np.ndarray:
    """E[c_v(n)] over `n_orderings` random orderings of the samples `x`.

    For each shuffle we compute the *cumulative* CoV at every prefix length n
    (= a random subset of size n), then average across shuffles. Uses cumulative
    sums so each ordering is O(N):
        μ(n)  = Σx / n                                          (eq. mean)
        σ(n)  = sqrt[(Σx² − n·μ²) / (n−1)]   (sample std)       (eq. deviation)
        c_v(n)= σ(n) / μ(n)                                     (eq. Cv)
    """
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


def cov_curves(per_seed: List[Dict[str, np.ndarray]], n_orderings: int,
               rng: np.random.Generator) -> Dict[str, Dict[str, np.ndarray]]:
    """Per KPI: across-seed mean / min / max of the order-averaged CoV curve."""
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for key, *_ in KPI_COLUMNS:
        seed_curves = []
        for seed_arr in per_seed:
            x = seed_arr[key]
            x = x[np.isfinite(x)]          # drop episodes that never finished
            if x.size >= 2:
                seed_curves.append(_running_cov_shuffled(x, n_orderings, rng))
        if not seed_curves:
            continue
        L = min(c.size for c in seed_curves)            # align lengths across seeds
        stack = np.vstack([c[:L] for c in seed_curves])
        curves[key] = {
            "n":    np.arange(1, L + 1),
            "mean": np.nanmean(stack, axis=0),
            "min":  np.nanmin(stack, axis=0),
            "max":  np.nanmax(stack, axis=0),
        }
    return curves


# =====================================================================
#  STEP 3 — required run-count: smallest n past which c_v stays "stable"  [UNCHANGED]
# =====================================================================

def _required_runs(n: np.ndarray, cov_mean: np.ndarray, tol: float,
                   min_n: int) -> Optional[int]:
    """Smallest n* (≥ min_n) such that for all m ≥ n*, |c_v(m) − c_v(∞)| ≤ tol·|c_v(∞)|.

    c_v(∞) is approximated by the final (largest-n) value. Returns None if the
    curve never settles inside the band by the end of the sweep.
    """
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
    # Walk back from the end while inside the band → start of the final in-band run.
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
#  STEP 4 — BOOTSTRAP the sampling distribution of the KPI mean
# =====================================================================

def bootstrap_sample_means(x: np.ndarray, sample_size: int, n_bootstrap: int,
                           rng: np.random.Generator,
                           max_elems: int = BOOTSTRAP_MAX_ELEMS) -> np.ndarray:
    """Bootstrap approximation of the sampling distribution of the MEAN of `x`.

    Draws `n_bootstrap` resamples, each `sample_size` episode observations WITH
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
    """Percentile 95% CI from the bootstrap means (no normality assumption)."""
    if means.size == 0:
        return (float("nan"), float("nan"))
    a, b = np.percentile(means, [lo, hi])
    return float(a), float(b)


def _analysis_pool(key: str, pooled: Dict[str, np.ndarray]) -> np.ndarray:
    """Finite per-episode values used for KPI `key`'s bootstrap.

    Duration honours DURATION_CONDITIONAL_ON_COMPLETION: when set, only episodes
    with completion == 1 contribute (duration | completion). Everything else
    simply drops non-finite (never-terminated) episodes.
    """
    x = pooled[key]
    if key == "duration" and DURATION_CONDITIONAL_ON_COMPLETION:
        comp = pooled["completion"]
        mask = np.isfinite(x) & np.isfinite(comp) & (comp == 1.0)
        return x[mask]
    return x[np.isfinite(x)]


def summarise_mean_distribution(key: str, label: str, pool: np.ndarray,
                                sample_size: int, n_bootstrap: int,
                                rng: np.random.Generator) -> dict:
    """Bootstrap `pool`'s mean and return the full sampling-distribution summary."""
    n_available = int(pool.size)
    summary = {
        "key": key, "label": label,
        "n_available": n_available, "sample_size": int(sample_size),
        "observed_mean": float("nan"), "raw_sd": float("nan"),
        "theoretical_sem": float("nan"), "bootstrap_se": float("nan"),
        "ci_lo": float("nan"), "ci_hi": float("nan"), "skew": float("nan"),
        "means": np.empty(0, dtype=float),
        "n_gt_available": bool(sample_size > n_available),
    }
    if n_available == 0 or sample_size < 2:
        return summary

    observed_mean = float(pool.mean())
    raw_sd = float(pool.std(ddof=1)) if n_available > 1 else 0.0
    means = bootstrap_sample_means(pool, sample_size, n_bootstrap, rng)

    summary["observed_mean"] = observed_mean
    summary["raw_sd"] = raw_sd
    summary["theoretical_sem"] = raw_sd / math.sqrt(sample_size)
    summary["bootstrap_se"] = float(means.std(ddof=1)) if means.size > 1 else 0.0
    ci_lo, ci_hi = _bootstrap_ci(means)
    summary["ci_lo"], summary["ci_hi"] = ci_lo, ci_hi
    summary["skew"] = float(skew(means)) if means.size > 2 and np.ptp(means) > 0 else 0.0
    summary["means"] = means
    return summary


# ── paired Complete-vs-Baseline machinery (prepared for later use) ────

def paired_difference(complete: np.ndarray, baseline: np.ndarray,
                      key: str) -> np.ndarray:
    """Per-episode difference with a sign such that positive ⇒ Complete better.

    For duration (lower is better) the sign is flipped:
        d_i = duration_baseline − duration_complete.
    For every other KPI (higher is better):
        d_i = value_complete − value_baseline.
    Requires the two arrays to be MATCHED episode-for-episode (same seeds/layouts).
    """
    a = np.asarray(complete, dtype=float)
    b = np.asarray(baseline, dtype=float)
    return (b - a) if key == "duration" else (a - b)


def bootstrap_paired_diff_means(complete: np.ndarray, baseline: np.ndarray,
                                key: str, sample_size: int, n_bootstrap: int,
                                rng: np.random.Generator) -> np.ndarray:
    """Bootstrap the sampling distribution of the MEAN PAIRED difference.

    Matched pairs are kept intact: finite pairs are formed first, then the PAIR
    indices are resampled with replacement (the two models are never bootstrapped
    independently). Null reference for the returned means is 0.
    """
    a = np.asarray(complete, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        raise ValueError("complete/baseline must be matched, equal-length arrays.")
    mask = np.isfinite(a) & np.isfinite(b)
    d = paired_difference(a[mask], b[mask], key)
    # Resampling the difference array = resampling matched pairs (pairing intact).
    return bootstrap_sample_means(d, sample_size, n_bootstrap, rng)


# =====================================================================
#  STEP 5 — D'Agostino-Pearson normality of the BOOTSTRAP MEANS
# =====================================================================

def normality_tests_of_means(means_by_kpi: Dict[str, np.ndarray],
                             alpha: float) -> Dict[str, dict]:
    """D'Agostino-Pearson omnibus test on each KPI's BOOTSTRAP SAMPLING
    DISTRIBUTION OF THE MEAN (not on the raw episode-level values).

    This is a *diagnostic* on the distribution of means: it does NOT claim the
    raw KPI is normal, and the bootstrap CI never depends on its verdict. With a
    large B the test can flag tiny, practically irrelevant departures, so pair it
    with the Q-Q plots.
    """
    results: Dict[str, dict] = {}
    for key, _, _ in HIST_KPIS:
        x = np.asarray(means_by_kpi.get(key, np.empty(0)), dtype=float)
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
        results[key] = result
    return results


def _print_normality_of_means(results: Dict[str, dict], alpha: float,
                              completion_cond: Optional[dict]) -> None:
    """Print the D'Agostino-Pearson verdict per KPI's distribution of means."""
    print(f"      -- D'Agostino-Pearson normality assessment of the bootstrap "
          f"sampling distribution of the mean (alpha={alpha:g}) --")
    for key, _, label in HIST_KPIS:
        result = results[key]
        if result["is_normal"] is None:
            print(f"        {label:16s}  not testable"
                  f"  ({result['reason']})", flush=True)
            continue
        verdict = ("consistent with normal" if result["is_normal"]
                   else "departs from normal")
        print(f"        {label:16s}  {verdict:24s}"
              f"  (K^2={result['statistic']:.4g}, p={result['pvalue']:.4g})",
              flush=True)
    print("        Note: this tests the DISTRIBUTION OF MEANS, not the raw KPI. "
          "With large B, tiny", flush=True)
    print("              departures can be flagged as significant — judge "
          "normality visually via the Q-Q plots.", flush=True)
    print("        Note: the 95% CI is a bootstrap percentile interval and does "
          "NOT require normality.", flush=True)
    if completion_cond is not None:
        ok = completion_cond["ok"]
        print(f"        Completion normal-approximation conditions "
              f"(n·p̂={completion_cond['n_phat']:.1f}, "
              f"n·(1−p̂)={completion_cond['n_qhat']:.1f}): "
              f"{'satisfied' if ok else 'NOT satisfied'} "
              f"(both must be >= 10). Its raw values are binary and are never "
              f"tested for normality.", flush=True)


# =====================================================================
#  PLOTS + CSV
# =====================================================================

def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def _common_n(sizes: Dict[str, int]) -> Optional[int]:
    vals = set(int(v) for v in sizes.values())
    return vals.pop() if len(vals) == 1 else None


def plot_scenario(curves: Dict[str, Dict[str, np.ndarray]],
                  max_eps: int, out_png: Path) -> None:
    """One figure: c_v vs number of simulations, one line per KPI + min–max band."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    cap_vals = []
    for (key, _, label), color in zip(KPI_COLUMNS, NLR_CYCLE):
        c = curves.get(key)
        if c is None:
            continue
        pm = c["n"] >= 2                       # skip the n=1 artefact (σ≡0)
        nn = c["n"][pm]
        ax.fill_between(nn, c["min"][pm], c["max"][pm], color=color, alpha=0.15, lw=0)
        ax.plot(nn, c["mean"][pm], color=color, lw=1.9, label=label)
        # robust y-cap: ignore the small-n explosion so the plateau is legible
        stable = c["n"] >= max(STABILITY_MIN_N, 2)
        if stable.any():
            cap_vals.append(float(np.nanmax(c["max"][stable])))

    ax.set_xlim(2, max_eps)
    if cap_vals:
        ax.set_ylim(0, max(cap_vals) * 1.15)
    ax.set_xlabel("Number of simulations (episodes)")
    ax.set_ylabel(r"Coefficient of variation  $c_v = \sigma / \mu$")
    ax.grid(True, alpha=0.4)
    ax.legend(title="KPI", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_mean_sampling_distributions(name: str, summaries: Dict[str, dict],
                                     n_bootstrap: int, out_png: Path) -> None:
    """Dashboard of the bootstrap SAMPLING DISTRIBUTIONS OF THE MEAN.

    Each subplot: density histogram of the bootstrap sample means, a dashed line
    at the observed sample mean, dotted lines at the 95% bootstrap CI limits, and
    a title carrying n, observed mean, theoretical SEM, bootstrap SE and CI.
    """
    kpis = HIST_KPIS
    ncol = 3
    nrow = int(np.ceil(len(kpis) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for idx, (key, _, label) in enumerate(kpis):
        ax = axes[idx]
        s = summaries.get(key)
        if s is None or s["means"].size == 0:
            ax.set_visible(False)
            continue
        color = NLR_CYCLE[idx % len(NLR_CYCLE)]
        means = s["means"]
        ax.hist(means, bins=45, density=True, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.axvline(s["observed_mean"], color=NLR_REFERENCE, ls="--", lw=1.5,
                   label="observed mean")
        ax.axvline(s["ci_lo"], color=NLR_ACCENT, ls=":", lw=1.4, label="95% CI")
        ax.axvline(s["ci_hi"], color=NLR_ACCENT, ls=":", lw=1.4)
        title = (f"{label}   (n = {s['sample_size']})\n"
                 f"mean = {s['observed_mean']:.3g}\n"
                 f"SEM(theory) = {s['theoretical_sem']:.2g}   "
                 f"SE(boot) = {s['bootstrap_se']:.2g}\n"
                 f"95% CI = [{s['ci_lo']:.3g}, {s['ci_hi']:.3g}]")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("sample mean")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7.5, framealpha=0.9)

    for ax in axes[len(kpis):]:
        ax.set_visible(False)

    sizes = {k: summaries[k]["sample_size"] for k in summaries}
    common = _common_n(sizes) if sizes else None
    n_txt = f"{common}" if common is not None else "per-KPI (see subplots)"
    fig.suptitle("Sampling distributions of KPI means\n"
                 f"B = {n_bootstrap:,} bootstrap resamples, n = {n_txt} "
                 f"episodes per sample — {name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_qq_means(name: str, summaries: Dict[str, dict], out_png: Path) -> None:
    """Q-Q dashboard of each KPI's bootstrap means vs a normal reference.

    Visual companion to the D'Agostino test: a large B makes the formal test
    over-sensitive, so straightness of the Q-Q line is the practical check.
    """
    kpis = HIST_KPIS
    ncol = 3
    nrow = int(np.ceil(len(kpis) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.7 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for idx, (key, _, label) in enumerate(kpis):
        ax = axes[idx]
        s = summaries.get(key)
        if s is None or s["means"].size < 3 or np.ptp(s["means"]) == 0:
            ax.set_visible(False)
            continue
        color = NLR_CYCLE[idx % len(NLR_CYCLE)]
        (osm, osr), (slope, intercept, r) = probplot(s["means"], dist="norm")
        ax.scatter(osm, osr, s=5, color=color, alpha=0.4, edgecolors="none")
        ax.plot(osm, slope * osm + intercept, color=NLR_REFERENCE, lw=1.3)
        ax.set_title(f"{label}   (R² = {r * r:.4f})", fontsize=9)
        ax.set_xlabel("theoretical quantiles")
        ax.set_ylabel("ordered sample means")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(kpis):]:
        ax.set_visible(False)

    fig.suptitle("Q-Q plots of the bootstrap sampling distributions of the mean\n"
                 f"{name}   (straight line ⇒ approximately normal)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_raw_distributions(name: str, pooled: Dict[str, np.ndarray], n_runs: int,
                           out_png: Path) -> None:
    """DESCRIPTIVE dashboard of raw episode-level KPI histograms (mission-to-
    mission variation). This is NOT the inferential normality assessment — that
    lives in the sampling-distribution-of-means / Q-Q figures.
    """
    kpis = HIST_KPIS
    ncol = 3
    nrow = int(np.ceil(len(kpis) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for idx, (key, _, label) in enumerate(kpis):
        ax = axes[idx]
        color = NLR_CYCLE[idx % len(NLR_CYCLE)]
        x = pooled[key]
        x = x[np.isfinite(x)]
        if x.size == 0:
            ax.set_visible(False)
            continue
        mu = float(x.mean())
        sd = float(x.std(ddof=1)) if x.size > 1 else 0.0

        if key == "completion":                       # binary KPI → two bars
            ax.hist(x, bins=[-0.5, 0.5, 1.5], density=True, color=color,
                    alpha=0.85, edgecolor="white", linewidth=0.5, rwidth=0.9)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["fail", "complete"])
        else:
            ax.hist(x, bins=30, density=True, color=color, alpha=0.85,
                    edgecolor="white", linewidth=0.3)
        ax.axvline(mu, color=NLR_REFERENCE, ls="--", lw=1.2)

        if key != "reward" and mu != 0.0:             # cᵥ only for non-negative KPIs
            stat = rf"$\mu$={mu:.3g}   $\sigma$={sd:.3g}   $c_v$={sd / abs(mu):.3g}"
        else:
            stat = rf"$\mu$={mu:.3g}   $\sigma$={sd:.3g}"
        ax.set_title(f"{label}\n{stat}", fontsize=10)
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(kpis):]:                        # hide unused grid cells
        ax.set_visible(False)

    fig.suptitle(f"Raw episode-level KPI distributions — {name}   "
                 f"(descriptive; n = {n_runs} episodes pooled across seeds)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_paired_diff_distributions(name: str, diffs_by_kpi: Dict[str, dict],
                                   out_png: Path) -> None:
    """Sampling distribution of the MEAN PAIRED difference (Complete − Baseline,
    sign-corrected so positive ⇒ Complete better). Null reference = 0.

    Prepared for later paired Complete-vs-Baseline runs; `diffs_by_kpi[key]` is a
    dict with a "means" bootstrap array and its 95% CI limits.
    """
    kpis = [(k, lbl) for k, _, lbl in HIST_KPIS if k in diffs_by_kpi]
    if not kpis:
        return
    ncol = 3
    nrow = int(np.ceil(len(kpis) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.7 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for idx, (key, label) in enumerate(kpis):
        ax = axes[idx]
        d = diffs_by_kpi[key]
        means = d["means"]
        if means.size == 0:
            ax.set_visible(False)
            continue
        color = NLR_SECONDARY
        ax.hist(means, bins=45, density=True, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.axvline(0.0, color=NLR_REFERENCE, ls="-", lw=1.4, label="null (0)")
        ax.axvline(d["ci_lo"], color=NLR_ACCENT, ls=":", lw=1.4, label="95% CI")
        ax.axvline(d["ci_hi"], color=NLR_ACCENT, ls=":", lw=1.4)
        excludes0 = not (d["ci_lo"] <= 0.0 <= d["ci_hi"])
        ax.set_title(f"{label}\nΔmean = {float(np.mean(means)):.3g}   "
                     f"95% CI=[{d['ci_lo']:.3g}, {d['ci_hi']:.3g}]"
                     f"{'  *' if excludes0 else ''}", fontsize=8.5)
        ax.set_xlabel("mean paired difference (Complete better →)")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7.5, framealpha=0.9)

    for ax in axes[len(kpis):]:
        ax.set_visible(False)

    fig.suptitle("Sampling distribution of the mean paired difference "
                 f"(Complete − Baseline) — {name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _write_csv(rows: List[dict], out_path: Path) -> None:
    import csv
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["scenario", "kpi", "n", "cov_mean", "cov_min", "cov_max"])
        writer.writeheader()
        writer.writerows(rows)


def _write_samples_csv(per_seed: List[Dict[str, np.ndarray]],
                       out_path: Path) -> None:
    """Raw per-episode KPI values, one row per (seed, episode).

    The `seed` column preserves the evaluation-seed identity so seed-level
    diagnostics remain possible even though the pools are combined for the
    single-policy mean analysis.
    """
    import csv
    keys = [k for k, _, _ in HIST_KPIS]
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "episode"] + keys)
        for s_idx, seed_arr in enumerate(per_seed):
            n = max((seed_arr[k].size for k in keys), default=0)
            for i in range(n):
                w.writerow([s_idx, i] + [
                    (seed_arr[k][i] if i < seed_arr[k].size else "") for k in keys])


def _write_bootstrap_means_csv(scenario: str, summaries: Dict[str, dict],
                               out_path: Path) -> None:
    """Every bootstrap sample mean: scenario, kpi, sample_size, iteration, mean."""
    import csv
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "kpi", "sample_size",
                    "bootstrap_iteration", "bootstrap_mean"])
        for key, _, _ in HIST_KPIS:
            s = summaries.get(key)
            if s is None:
                continue
            n = s["sample_size"]
            for b, m in enumerate(s["means"]):
                w.writerow([scenario, key, n, b, float(m)])


def _write_mean_summary_csv(scenario: str, summaries: Dict[str, dict],
                            normality: Dict[str, dict], out_path: Path) -> None:
    """Compact per-KPI summary of the sampling distribution of the mean."""
    import csv
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "kpi", "n_available", "sample_size",
                    "observed_mean", "raw_sd", "theoretical_sem", "bootstrap_se",
                    "ci_2_5", "ci_97_5", "normality_statistic", "normality_pvalue"])
        for key, _, _ in HIST_KPIS:
            s = summaries.get(key)
            if s is None:
                continue
            nrm = normality.get(key, {})
            w.writerow([
                scenario, key, s["n_available"], s["sample_size"],
                s["observed_mean"], s["raw_sd"], s["theoretical_sem"],
                s["bootstrap_se"], s["ci_lo"], s["ci_hi"],
                nrm.get("statistic"), nrm.get("pvalue"),
            ])


# =====================================================================
#  sample-size resolution
# =====================================================================

def _resolve_sample_sizes(per_kpi_req: Dict[str, Optional[int]],
                          overall_req: Optional[int],
                          pool_sizes: Dict[str, int],
                          cli_n: Optional[int]) -> Dict[str, int]:
    """Per-KPI evaluation sample size n for the bootstrap.

    Priority:  explicit --mean_sample_size  →  KPI-specific CoV n*  →
               overall CoV n* (max over KPIs)  →  available pool size.
    """
    sizes: Dict[str, int] = {}
    for key, _, _ in HIST_KPIS:
        if cli_n is not None:
            n = int(cli_n)
        else:
            n = per_kpi_req.get(key)          # reward / non-stabilised → None
            if n is None:
                n = overall_req
            if n is None:
                n = pool_sizes.get(key, 0)
        sizes[key] = max(int(n), 2)           # bootstrap needs n >= 2
    return sizes


# =====================================================================
#  DRIVER
# =====================================================================

def analyse_scenarios(scenarios: List[CurriculumSection],
                      default_ckpt_path: Optional[Path],
                      max_episodes: int, n_seeds: int, base_seed: int,
                      chunk: int, n_orderings: int, tol: float, min_n: int,
                      normality_alpha: float, n_bootstrap: int,
                      bootstrap_seed: int, mean_sample_size: Optional[int],
                      device: torch.device, out_dir: Path,
                      write_csv: bool, write_bootstrap_csv: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache: Dict[str, _LoadedCheckpoint] = {}
    csv_rows: List[dict] = []
    rng = np.random.default_rng(base_seed)        # drives the ordering shuffles

    for i, section in enumerate(scenarios):
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
        print(f"\n[{i + 1}/{len(scenarios)}] {section.name}  "
              f"[{ckpt_path.name}]  {_section_label(section, env_cfg)}", flush=True)
        print(f"      {max_episodes} episodes × {n_seeds} seeds  "
              f"(chunk={chunk}, orderings={n_orderings})", flush=True)

        policy = _build_policy_for_scenario(ckpt, env_cfg, device)

        # ── STEP 1: per seed, gather MAX_EPISODES raw per-episode KPI samples ──
        per_seed: List[Dict[str, np.ndarray]] = []
        for s in range(n_seeds):
            master = base_seed + s * SEED_STRIDE
            kpis = collect_episode_kpis(policy, env_cfg, ckpt.hf_radar_cfg,
                                        max_episodes, master, device, chunk)
            per_seed.append(kpis)
            print(f"        seed {s + 1}/{n_seeds} (base={master}) "
                  f"collected {max_episodes} episodes", flush=True)

        # ── STEP 2: CoV-vs-n curves (mean ± min/max band across seeds) ──
        curves = cov_curves(per_seed, n_orderings, rng)

        # ── STEP 3: required run-count per KPI + overall ──
        per_kpi_req: Dict[str, Optional[int]] = {}
        for kkey, *_ in KPI_COLUMNS:
            c = curves.get(kkey)
            per_kpi_req[kkey] = (_required_runs(c["n"], c["mean"], tol, min_n)
                                 if c is not None else None)
        finite_reqs = [r for r in per_kpi_req.values() if r is not None]
        overall_req = max(finite_reqs) if finite_reqs else None

        # ── report CoV stabilisation ──
        print("      ── required runs for stable CoV (±{:.0%}) ──".format(tol))
        for kkey, _, label in KPI_COLUMNS:
            r = per_kpi_req.get(kkey)
            c = curves.get(kkey)
            cv_final = f"{c['mean'][-1]:.4f}" if c is not None else "n/a"
            print(f"        {label:16s}  n* = {str(r) if r else 'not reached':12s}"
                  f"  (final c_v = {cv_final})", flush=True)
        if overall_req:
            print(f"      → ALL KPIs stable from ~{overall_req} episodes.", flush=True)
        else:
            print("      → some KPIs did not stabilise within "
                  f"{max_episodes} episodes (raise --max_episodes).", flush=True)

        # ── STEP 4a: plot CoV + accumulate CoV CSV rows ──
        out_png = out_dir / f"cov_{_safe_name(section.name)}_{stamp}.png"
        plot_scenario(curves, max_episodes, out_png)
        print(f"      saved CoV plot          : {out_png}", flush=True)

        # Pool the raw episodes across seeds (same frozen policy, same scenario).
        pooled = {k: np.concatenate([s[k] for s in per_seed]) for k, _, _ in HIST_KPIS}

        # ── STEP 4b: bootstrap the SAMPLING DISTRIBUTION OF THE MEAN per KPI ──
        analysis_pools = {k: _analysis_pool(k, pooled) for k, _, _ in HIST_KPIS}
        pool_sizes = {k: int(v.size) for k, v in analysis_pools.items()}
        sizes = _resolve_sample_sizes(per_kpi_req, overall_req, pool_sizes,
                                      mean_sample_size)

        boot_rng = np.random.default_rng(bootstrap_seed + i)  # reproducible per scn
        summaries: Dict[str, dict] = {}
        print(f"      ── sampling distribution of the mean "
              f"(B={n_bootstrap:,} bootstrap resamples) ──", flush=True)
        src = ("--mean_sample_size" if mean_sample_size is not None
               else "CoV required-run counts")
        print(f"        n per KPI from: {src}", flush=True)
        for kkey, _, label in HIST_KPIS:
            pool = analysis_pools[kkey]
            n_use = sizes[kkey]
            summary = summarise_mean_distribution(kkey, label, pool, n_use,
                                                  n_bootstrap, boot_rng)
            summaries[kkey] = summary
            if pool.size == 0:
                print(f"        {label:16s}  no finite episodes — skipped",
                      flush=True)
                continue
            warn = "  [WARN n > available pool]" if summary["n_gt_available"] else ""
            print(f"        {label:16s}  n={n_use:<5d} "
                  f"mean={summary['observed_mean']:.4g}  "
                  f"SEM(theory)={summary['theoretical_sem']:.3g}  "
                  f"SE(boot)={summary['bootstrap_se']:.3g}  "
                  f"95% CI=[{summary['ci_lo']:.4g}, {summary['ci_hi']:.4g}]  "
                  f"skew={summary['skew']:.2g}"
                  f"  (avail={pool.size}){warn}", flush=True)

        # Duration treatment note (never silent).
        if DURATION_CONDITIONAL_ON_COMPLETION:
            n_completed = pool_sizes.get("duration", 0)
            n_total = int(np.isfinite(pooled["duration"]).sum())
            print(f"        Duration is CONDITIONAL ON COMPLETION: "
                  f"{n_completed}/{n_total} finished episodes were completed; "
                  f"n for Duration is the completed count, not the total.",
                  flush=True)
        else:
            print("        Duration includes every finished episode "
                  "(unfinished episodes contribute NaN and are dropped); "
                  "not conditioned on completion.", flush=True)

        # Completion normal-approximation conditions (n·p̂ ≥ 10 and n·(1−p̂) ≥ 10).
        completion_cond = None
        cs = summaries.get("completion")
        if cs is not None and cs["means"].size:
            p_hat = cs["observed_mean"]
            n_c = cs["sample_size"]
            completion_cond = {
                "n_phat": n_c * p_hat, "n_qhat": n_c * (1.0 - p_hat),
                "ok": (n_c * p_hat >= 10.0) and (n_c * (1.0 - p_hat) >= 10.0),
            }

        # ── STEP 4c: mean-sampling-distribution + Q-Q + raw dashboards ──
        mean_png = out_dir / f"mean_sampling_dist_{_safe_name(section.name)}_{stamp}.png"
        plot_mean_sampling_distributions(section.name, summaries, n_bootstrap, mean_png)
        print(f"      saved mean sampling dist : {mean_png}", flush=True)

        qq_png = out_dir / f"mean_qq_{_safe_name(section.name)}_{stamp}.png"
        plot_qq_means(section.name, summaries, qq_png)
        print(f"      saved Q-Q of means       : {qq_png}", flush=True)

        if SAVE_RAW_DIST:
            n_runs = next(iter(pooled.values())).size
            raw_png = out_dir / f"raw_kpi_dist_{_safe_name(section.name)}_{stamp}.png"
            plot_raw_distributions(section.name, pooled, n_runs, raw_png)
            print(f"      saved raw distributions  : {raw_png}", flush=True)

        # ── STEP 5: normality diagnostic on the BOOTSTRAP MEANS ──
        boot_means = {k: summaries[k]["means"] for k in summaries}
        normality = normality_tests_of_means(boot_means, normality_alpha)
        _print_normality_of_means(normality, normality_alpha, completion_cond)

        # ── CSVs ──
        if write_csv:
            samp_csv = out_dir / f"kpi_samples_{_safe_name(section.name)}_{stamp}.csv"
            _write_samples_csv(per_seed, samp_csv)
            print(f"      saved raw samples CSV    : {samp_csv}", flush=True)

            summ_csv = out_dir / f"mean_summary_{_safe_name(section.name)}_{stamp}.csv"
            _write_mean_summary_csv(section.name, summaries, normality, summ_csv)
            print(f"      saved mean summary CSV   : {summ_csv}", flush=True)

            if write_bootstrap_csv:
                boot_csv = out_dir / f"bootstrap_means_{_safe_name(section.name)}_{stamp}.csv"
                _write_bootstrap_means_csv(section.name, summaries, boot_csv)
                print(f"      saved bootstrap CSV      : {boot_csv}", flush=True)

        for kkey, _, _ in KPI_COLUMNS:
            c = curves.get(kkey)
            if c is None:
                continue
            for j in range(c["n"].size):
                csv_rows.append({
                    "scenario": section.name, "kpi": kkey, "n": int(c["n"][j]),
                    "cov_mean": c["mean"][j], "cov_min": c["min"][j],
                    "cov_max": c["max"][j],
                })

        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if write_csv and csv_rows:
        csv_path = out_dir / f"cov_analysis_{stamp}.csv"
        _write_csv(csv_rows, csv_path)
        print(f"\nSaved CoV curves CSV to: {csv_path}\n", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CoV run-count stabilisation + bootstrap sampling-"
        "distribution-of-the-mean analysis for the SCENARIOS defined at the top "
        "of this file.")
    p.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                   help="Default .pt for scenarios that leave policy_file=None.")
    p.add_argument("--max_episodes", type=int, default=MAX_EPISODES,
                   help=f"Episodes collected per seed (default: {MAX_EPISODES}).")
    p.add_argument("--n_seeds", type=int, default=N_SEEDS,
                   help=f"Independent evaluation seeds = extra pools (default: {N_SEEDS}).")
    p.add_argument("--seed", type=int, default=BASE_SEED,
                   help=f"Base RNG seed for collection/shuffles (default: {BASE_SEED}).")
    p.add_argument("--chunk", type=int, default=CHUNK_EPISODES,
                   help=f"Parallel envs per rollout (default: {CHUNK_EPISODES}).")
    p.add_argument("--n_orderings", type=int, default=N_ORDERINGS,
                   help=f"Shuffles for order-independent CoV (default: {N_ORDERINGS}).")
    p.add_argument("--tol", type=float, default=STABILITY_TOL,
                   help=f"Relative stability band around final CoV (default: {STABILITY_TOL}).")
    p.add_argument("--min_n", type=int, default=STABILITY_MIN_N,
                   help=f"Ignore n below this when detecting n* (default: {STABILITY_MIN_N}).")
    p.add_argument("--normality_alpha", type=float, default=NORMALITY_ALPHA,
                   help="D'Agostino-Pearson significance level "
                        f"(default: {NORMALITY_ALPHA}).")
    p.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP,
                   help=f"Bootstrap resamples B (default: {N_BOOTSTRAP}).")
    p.add_argument("--bootstrap_seed", type=int, default=BOOTSTRAP_SEED,
                   help=f"RNG seed for all bootstrap resampling (default: {BOOTSTRAP_SEED}).")
    p.add_argument("--mean_sample_size", type=int, default=MEAN_SAMPLE_SIZE,
                   metavar="N",
                   help="Episodes per bootstrap resample n. Default: per-KPI CoV "
                        "required-run counts.")
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
    if args.mean_sample_size is not None and args.mean_sample_size < 2:
        raise ValueError("--mean_sample_size must be >= 2.")

    default_ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if default_ckpt_path is not None and not default_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {default_ckpt_path}")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else (_PKG_DIR / "stat_results")

    print("─" * 70)
    print(f"  Device        : {device}")
    print(f"  Scenarios     : {len(SCENARIOS)}")
    print(f"  Collection    : up to {args.max_episodes} episodes × {args.n_seeds} seeds")
    print(f"  KPIs (CoV)    : " + ", ".join(lbl for _, _, lbl in KPI_COLUMNS))
    print(f"  KPIs (means)  : " + ", ".join(lbl for _, _, lbl in HIST_KPIS))
    print(f"  Bootstrap     : B={args.n_bootstrap}, seed={args.bootstrap_seed}, "
          + ("n=" + str(args.mean_sample_size) if args.mean_sample_size is not None
             else "n=CoV required-run counts"))
    print(f"  Duration      : "
          + ("conditional on completion" if DURATION_CONDITIONAL_ON_COMPLETION
             else "all finished episodes"))
    print(f"  Normality     : D'Agostino-Pearson on bootstrap means, "
          f"alpha={args.normality_alpha:g}")
    print("─" * 70)

    analyse_scenarios(
        SCENARIOS, default_ckpt_path,
        max_episodes=args.max_episodes, n_seeds=args.n_seeds, base_seed=args.seed,
        chunk=args.chunk, n_orderings=args.n_orderings, tol=args.tol,
        min_n=args.min_n, normality_alpha=args.normality_alpha,
        n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed,
        mean_sample_size=args.mean_sample_size,
        device=device, out_dir=out_dir,
        write_csv=not args.no_csv, write_bootstrap_csv=not args.no_bootstrap_csv,
    )


if __name__ == "__main__":
    main()
