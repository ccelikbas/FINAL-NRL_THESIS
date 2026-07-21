r"""
statistical_analysis.py
═══════════════════════
Two clearly separated workflows for a FROZEN FOFE-MAPPO policy's KPIs:

  ── PILOT  (--analysis_mode pilot) ── sample-size PLANNING ──────────────────
     Collect a large pool (e.g. 1000 episodes) and decide HOW MANY evaluation
     episodes the final study needs, using a PRECISION criterion: bootstrap the
     mean at candidate sample sizes and pick the smallest n whose 95 % CI
     half-width drops below a user-defined target (PRECISION_TARGETS). Outputs:
       • the precision plot (CI half-width vs sample size) + console recommendation,
       • the EXPECTED sampling distributions of the KPI means at the proposed
         N_eval (histograms) — labelled *expected*, not a final result.

  ── FINAL  (--analysis_mode final --eval_episodes N) ── STATISTICAL INFERENCE ─
     Collect a NEW independent set of EXACTLY N evaluation episodes and report
     the final KPI means with 95 % bootstrap confidence intervals. All KPIs
     share ONE common N (they come from the same episodes). Output: the
     sampling distributions of the KPI means (histograms).

──────────────────────────────────────────────────────────────────────────────
WHAT THE HISTOGRAMS SHOW
──────────────────────────────────────────────────────────────────────────────
The histogram dashboard shows the BOOTSTRAP SAMPLING DISTRIBUTION OF EACH KPI's
MEAN. Its spread is the standard error of the mean (SEM = raw SD / √n); the
bootstrap SE is the SD of those bootstrap means (NOT divided by √B). The 95 % CI
is a percentile interval and does NOT require normality. Six KPIs are shown
(completion is excluded):
    Target-destruction rate, Survival rate, Failure-aware duration,
    Duration among completed missions, Coalition fragmentation, Episode reward.

──────────────────────────────────────────────────────────────────────────────
DURATION  (two metrics, failed missions never silently dropped)
──────────────────────────────────────────────────────────────────────────────
  • PRIMARY  "Failure-aware duration"  (duration_capped): observed duration if
    the mission completed, else max_steps. Defined for every episode ⇒ same
    sample size as the other KPIs.
  • SECONDARY "Duration among completed missions" (duration_completed):
    conditional on completion; its sample size is the number of completed
    episodes. Read it together with the completion rate — a policy can look fast
    by completing only the easy missions.

──────────────────────────────────────────────────────────────────────────────
SEEDS
──────────────────────────────────────────────────────────────────────────────
--n_seeds are EVALUATION RNG seeds for one frozen checkpoint (extra independent
episode pools) — NOT independently trained models. For a frozen policy the pools
may be combined; episodes remain the sampling unit. Several INDEPENDENTLY
TRAINED checkpoints need a separate HIERARCHICAL analysis:
    episodes → mean per training seed → variation across training-seed means.

──────────────────────────────────────────────────────────────────────────────
NORMALITY OF THE SAMPLING DISTRIBUTIONS OF THE KPI MEANS  (ON by default)
──────────────────────────────────────────────────────────────────────────────
The CLT predicts that each KPI's mean is approximately normally distributed at
large n. We assess this with a SHAPIRO-WILK test on the bootstrap sampling
distribution of the mean: draw NORMALITY_N_SAMPLES (default 600) bootstrap
sample means — each a resample of the study's episode count — and test THOSE for
normality. The test is fed a modest 600 means (NOT the full B used for the CI)
so its power is calibrated to a realistic sample size; a genuinely near-normal
sampling distribution then passes rather than being rejected on trivial
deviations (as it would if all B≈10 000 means were tested). Output: per-KPI
p-values at α=0.05 and a printed + saved LaTeX table. The percentile CI itself
requires no normality; the test is reported as supporting evidence.

──────────────────────────────────────────────────────────────────────────────
HOW TO USE
──────────────────────────────────────────────────────────────────────────────
    # 1) plan the sample size (pilot):
    .\.venv\Scripts\python.exe 0_TA_HF_FOFE-MAPPO\eval_tools\statistical_analysis.py ^
        --analysis_mode pilot --max_episodes 1000

    # 2) run the final study at the chosen count:
    .\.venv\Scripts\python.exe 0_TA_HF_FOFE-MAPPO\eval_tools\statistical_analysis.py ^
        --analysis_mode final --eval_episodes 100

Edit SCENARIOS + PRECISION_TARGETS below. Figures land in <script_dir>/stat_results.
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
from scipy.stats import shapiro, skew

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
        name="Baseline",
        policy_file="runs/FINALV2/Final_Baseline_Cont_4.pt",
        n_iters=1,  # not used
        n_strikers=2, n_jammers=4,
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

# ── precision-based sample-size planning (pilot) ─────────────────────
# User-defined PRACTICAL precision requirements: the target 95 % CI half-width
# for each KPI's mean (same units as the KPI). Easy to edit.
PRECISION_TARGETS: Dict[str, float] = {
    "completion":      0.03,   # ±0.03 on the completion proportion
    "targets":         0.03,   # ±0.03 on the target-destruction fraction
    "survival":        0.03,   # ±0.03 on the survival fraction
    "fragmentation":   0.02,   # ±0.02 on coalition fragmentation
    "duration_capped": 3.0,    # ±3 steps on failure-aware duration
    "duration_completed": 3.0, # ±3 steps on duration among completed missions
    "reward":          1.0,    # ±1.0 on episode reward
}
PRECISION_STEP      = 10      # candidate sample sizes 10, 20, 30, …, max_episodes
PRECISION_BOOTSTRAP = 2000    # bootstrap resamples for the (many) precision points
ROUND_TO            = 25      # round the recommended N_eval up to a tidy multiple

# ── bootstrap of the sampling distribution of the mean ───────────────
N_BOOTSTRAP    = 10_000   # bootstrap resamples B for the reported distributions
BOOTSTRAP_SEED = 12345    # RNG seed for ALL bootstrap resampling (reproducible)
BOOTSTRAP_MAX_ELEMS = 20_000_000   # cap the [B, n] index matrix before chunking

# ── normality of the sampling distributions of the KPI means (ON) ────
# Shapiro-Wilk is applied to NORMALITY_N_SAMPLES bootstrap sample means per KPI.
# Using a modest count (matching the study's episode budget) keeps the test's
# power calibrated so near-normal sampling distributions are not over-rejected.
NORMALITY_ALPHA     = 0.05
NORMALITY_N_SAMPLES = 600     # bootstrap sample means fed to the Shapiro-Wilk test

WILSON_Z = 1.959963985   # z for a 95 % Wilson interval (completion, final mode)


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
    conditional: bool = False   # conditional-on-completion (own sample size)


#  Analysis KPIs used for means / bootstrap / plots. Duration is split into the
#  failure-aware primary metric and the conditional secondary metric.
ANALYSIS_KPIS: List[KpiSpec] = [
    KpiSpec("completion",         "Completion rate"),
    KpiSpec("targets",            "Target-destruction rate"),
    KpiSpec("survival",           "Survival rate"),
    KpiSpec("fragmentation",      "Coalition fragmentation"),
    KpiSpec("duration_capped",    "Failure-aware duration"),
    KpiSpec("reward",             "Episode reward"),
    KpiSpec("duration_completed", "Duration among completed missions", conditional=True),
]
PRIMARY_KPIS = [k for k in ANALYSIS_KPIS if not k.conditional]
KEY_TO_SPEC  = {k.key: k for k in ANALYSIS_KPIS}

# The six KPIs shown in BOTH the histogram dashboards and the precision plot
# (completion excluded; both duration metrics included) — same order/layout.
HISTOGRAM_KEYS = ["targets", "survival", "duration_capped", "duration_completed",
                  "fragmentation", "reward"]
HISTOGRAM_SPECS = [KEY_TO_SPEC[k] for k in HISTOGRAM_KEYS]
# Precision drives the recommended N_eval only via KPIs measured over ALL
# episodes; the conditional duration is shown for reference (completed-mission
# units) and does not size the total episode count.
PRECISION_RECO_SPECS = [s for s in HISTOGRAM_SPECS if not s.conditional]

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
    precision_step: int
    precision_bootstrap: int
    round_to: int
    n_bootstrap: int
    bootstrap_seed: int
    normality_alpha: float
    normality_n_samples: int
    run_normality: bool
    scenario_name_check: str
    out_dir: Path
    stamp: str

    def scenario_index_hash(self) -> int:
        """Reproducible per-scenario RNG offset (set on the opts during the loop)."""
        return int(getattr(self, "_scenario_index", 0))


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
#  STEP 2 — BOOTSTRAP of the sampling distribution of the mean
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
#  Normality of the sampling distribution of each KPI mean
#  (Shapiro-Wilk on NORMALITY_N_SAMPLES bootstrap sample means)
# =====================================================================

def normality_of_mean_distribution(spec: KpiSpec, pool: np.ndarray,
                                   sample_size: int, n_samples: int,
                                   rng: np.random.Generator,
                                   alpha: float) -> dict:
    """Shapiro-Wilk normality test of a KPI mean's sampling distribution.

    The sampling distribution of the mean at `sample_size` is estimated by
    drawing `n_samples` bootstrap sample means (each a resample of `sample_size`
    observations WITH replacement); Shapiro-Wilk then tests those `n_samples`
    means for normality. Feeding a MODEST count (default 600) — not the full B
    replicates used for the CI — keeps the test's power calibrated to a realistic
    sample size, so a genuinely near-normal sampling distribution is not rejected
    on trivial deviations. The CLT predicts approximate normality at large n.
    """
    pool = np.asarray(pool, dtype=float)
    pool = pool[np.isfinite(pool)]
    result = {
        "key": spec.key, "label": spec.label, "conditional": spec.conditional,
        "sample_size": int(sample_size), "n_samples": 0,
        "statistic": None, "pvalue": None, "is_normal": None, "reason": None,
    }
    if pool.size < 2 or sample_size < 2:
        result["reason"] = "insufficient data"
        return result

    means = bootstrap_sample_means(pool, int(sample_size), int(n_samples), rng)
    means = means[np.isfinite(means)]
    result["n_samples"] = int(means.size)
    if means.size < 3:
        result["reason"] = "need at least 3 bootstrap means"
    elif np.ptp(means) == 0.0:
        result["reason"] = "all bootstrap means identical"
    else:
        test = shapiro(means)
        statistic, pvalue = float(test.statistic), float(test.pvalue)
        if np.isfinite(statistic) and np.isfinite(pvalue):
            result["statistic"] = statistic
            result["pvalue"] = pvalue
            result["is_normal"] = pvalue >= alpha
        else:
            result["reason"] = "test returned a non-finite result"
    return result


def normality_of_means(pools: Dict[str, np.ndarray], summaries: Dict[str, dict],
                       n_samples: int, rng: np.random.Generator,
                       alpha: float) -> Dict[str, dict]:
    """Run the Shapiro-Wilk normality test for every histogram KPI.

    Each KPI's resample size is taken from its summary (the study N_eval, or the
    completed-mission count for the conditional duration) so the tested sampling
    distribution matches the one reported in the histogram dashboard.
    """
    results: Dict[str, dict] = {}
    for spec in HISTOGRAM_SPECS:
        size = int(summaries.get(spec.key, {}).get("sample_size", 0))
        results[spec.key] = normality_of_mean_distribution(
            spec, pools[spec.key], size, n_samples, rng, alpha)
    return results


def _fmt_pvalue(p: Optional[float]) -> str:
    if p is None:
        return "--"
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


def _latex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}")):
        text = text.replace(a, b)
    return text


def build_normality_latex(results: Dict[str, dict], mode: str, display: str,
                          n_samples: int, alpha: float) -> str:
    """Assemble a booktabs LaTeX table of the Shapiro-Wilk normality results."""
    mode_word = "pilot (expected)" if mode == "pilot" else "final"
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        (f"  \\caption{{Shapiro--Wilk normality test of the bootstrap sampling "
         f"distribution of each KPI mean ({mode_word} mode, scenario "
         f"{_latex_escape(display)}). The test is applied to {n_samples} bootstrap "
         f"sample means; a KPI is flagged Normal when $p \\geq \\alpha$ "
         f"($\\alpha={alpha:g}$).}}"),
        f"  \\label{{tab:normality_{_safe_name(display)}_{mode}}}",
        r"  \begin{tabular}{lrrrc}",
        r"    \toprule",
        r"    KPI & Resample $n$ & $W$ & $p$-value & Normal ($\alpha="
        + f"{alpha:g}" + r"$) \\",
        r"    \midrule",
    ]
    for spec in HISTOGRAM_SPECS:
        r = results.get(spec.key, {})
        label = _latex_escape(spec.label)
        size = r.get("sample_size", 0)
        if r.get("is_normal") is None:
            reason = _latex_escape(r.get("reason") or "not testable")
            lines.append(f"    {label} & {size} & "
                         f"\\multicolumn{{3}}{{c}}{{{reason}}} \\\\")
        else:
            w = f"{r['statistic']:.4f}"
            p = _fmt_pvalue(r["pvalue"])
            verdict = "Yes" if r["is_normal"] else "No"
            lines.append(f"    {label} & {size} & {w} & {p} & {verdict} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def emit_normality_table(pools: Dict[str, np.ndarray],
                         summaries: Dict[str, dict], mode: str, display: str,
                         opts: "RunOpts", rng: np.random.Generator) -> None:
    """Compute, print (console + LaTeX) and save the KPI-mean normality table."""
    results = normality_of_means(pools, summaries, opts.normality_n_samples,
                                 rng, opts.normality_alpha)
    print("\n  -- Normality of the sampling distributions of the KPI means "
          "(Shapiro-Wilk) --", flush=True)
    print(f"     Test applied to {opts.normality_n_samples} bootstrap sample "
          f"means per KPI; alpha={opts.normality_alpha:g}.", flush=True)
    for spec in HISTOGRAM_SPECS:
        r = results[spec.key]
        if r.get("is_normal") is None:
            print(f"       {spec.label:34s} not testable ({r.get('reason')})",
                  flush=True)
        else:
            verdict = "normal" if r["is_normal"] else "NOT normal"
            print(f"       {spec.label:34s} W={r['statistic']:.4f}  "
                  f"p={_fmt_pvalue(r['pvalue'])}  [{verdict}]", flush=True)

    latex = build_normality_latex(results, mode, display,
                                  opts.normality_n_samples, opts.normality_alpha)
    print("\n  LaTeX table (also saved to file):\n", flush=True)
    print(latex, flush=True)

    tex_path = opts.out_dir / f"normality_{_safe_name(display)}_{mode}_{opts.stamp}.tex"
    tex_path.write_text(latex + "\n", encoding="utf-8")
    print(f"\n      saved normality table    : {tex_path}", flush=True)


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
#  PLOTS  (only two figures: precision curves + means histograms)
# =====================================================================

def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def plot_precision_curves(candidate_ns: np.ndarray,
                          hw_by_kpi: Dict[str, np.ndarray],
                          reqs: Dict[str, Optional[int]],
                          recommended: Optional[int],
                          recommended_completed: Optional[int],
                          out_png: Path) -> None:
    """Pilot planning: CI half-width vs candidate sample size, one panel per KPI.

    Panels match the histogram dashboard exactly (same six KPIs, same order:
    completion excluded, both duration metrics included). The conditional
    duration panel is sized in COMPLETED-mission units, so its recommended line
    is the EXPECTED completed count at the recommended N_eval.
    """
    specs = HISTOGRAM_SPECS
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
            ax.axhline(thr, color=NLR_ACCENT, ls="--", lw=1.3, label=f"target ±{thr:g}")
        req = reqs.get(spec.key)
        if req is not None:
            ax.axvline(req, color=NLR_REFERENCE, ls=":", lw=1.3, label=f"n*={req}")
        rec_line = recommended_completed if spec.conditional else recommended
        if rec_line is not None:
            ax.axvline(rec_line, color=NLR_SECONDARY, ls="-", lw=1.1, alpha=0.8,
                       label=f"N_eval={rec_line}")
        ax.set_title(spec.label, fontsize=9.5)
        ax.set_xlabel("completed missions n" if spec.conditional
                      else "evaluation episodes n")
        ax.set_ylabel("95% CI half-width")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, framealpha=0.9)
    for ax in axes[len(specs):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_mean_sampling_distributions(summaries: Dict[str, dict], mode: str,
                                     out_png: Path) -> None:
    """Histograms of the bootstrap sample means for the six histogram KPIs."""
    pilot = (mode == "pilot")
    mean_label = "pilot pooled mean" if pilot else "final sample mean"
    specs = HISTOGRAM_SPECS
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
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


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

    boot_rng = np.random.default_rng(opts.bootstrap_seed + opts.scenario_index_hash())

    # ── precision (CI half-width) planning ──
    # Same six KPIs as the histogram dashboard (completion excluded; both
    # duration metrics). The conditional duration is measured in COMPLETED-
    # mission units, so it is reported for reference and does not size N_eval.
    p_complete = float(np.nanmean(pooled["completion"])) if pooled["completion"].size else 0.0
    step = max(2, opts.precision_step)
    candidate_ns = np.arange(step, opts.max_episodes + 1, step, dtype=int)
    if candidate_ns.size == 0:
        candidate_ns = np.array([opts.max_episodes], dtype=int)
    hw_by_kpi: Dict[str, np.ndarray] = {}
    prec_reqs: Dict[str, Optional[int]] = {}
    print("\n  Precision requirement (target 95% CI half-width):", flush=True)
    for spec in HISTOGRAM_SPECS:
        hw, _se = precision_curve(pooled[spec.key], candidate_ns,
                                  opts.precision_bootstrap, boot_rng)
        thr = PRECISION_TARGETS.get(spec.key, float("inf"))
        req = precision_required_n(candidate_ns, hw, thr)
        hw_by_kpi[spec.key] = hw
        prec_reqs[spec.key] = req
        unit = "completed missions" if spec.conditional else "episodes"
        print(f"      {spec.label:34s} n*={str(req) if req else 'not reached':12s}"
              f"  (target ±{thr:g}, {unit})", flush=True)

    # Only KPIs measured over ALL episodes size the total episode count.
    prec_common = max([prec_reqs[s.key] for s in PRECISION_RECO_SPECS
                       if prec_reqs.get(s.key) is not None], default=None)

    raw_reco = prec_common if prec_common is not None else opts.max_episodes
    recommended = _round_up(raw_reco, opts.round_to)
    proposed = int(opts.eval_episodes) if opts.eval_episodes else int(recommended)
    recommended_completed = max(2, int(round(p_complete * recommended)))

    print(f"\n  Precision-based common count: "
          f"{prec_common if prec_common is not None else 'not reached'}", flush=True)
    print(f"  Precision analysis suggests at least "
          f"{prec_common if prec_common else '?'} episodes.", flush=True)
    print(f"  Recommended common final evaluation count: {recommended} episodes"
          f"  (round-up to nearest {opts.round_to})", flush=True)
    print(f"  (Duration among completed missions is shown for reference; its n is "
          f"in completed-mission units and does not size N_eval.)", flush=True)
    if opts.eval_episodes:
        print(f"  (user override via --eval_episodes: using proposed N_eval="
              f"{proposed} for the expected distributions)", flush=True)

    # ── EXPECTED sampling distributions of the KPI means at the proposed N_eval ──
    expected_completed = max(2, int(round(p_complete * proposed)))
    print(f"\n  Expected sampling distributions at N_eval={proposed}"
          f"  (B={opts.n_bootstrap:,} bootstrap resamples):", flush=True)
    print(f"      Total pilot episodes: {counts['total']}", flush=True)
    print(f"      Completed episodes available for conditional duration: "
          f"{counts['completed']}", flush=True)
    print(f"      Conditional-duration bootstrap resample size (expected at "
          f"N_eval): {expected_completed}", flush=True)

    summaries: Dict[str, dict] = {}
    for spec in HISTOGRAM_SPECS:
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

    # ── figures (precision plot + means histograms) ──
    tag = _safe_name(display)
    prec_png = opts.out_dir / f"pilot_precision_{tag}_{opts.stamp}.png"
    plot_precision_curves(candidate_ns, hw_by_kpi, prec_reqs, recommended,
                          recommended_completed, prec_png)
    print(f"\n      saved precision curves   : {prec_png}", flush=True)

    mean_png = opts.out_dir / f"pilot_expected_means_{tag}_{opts.stamp}.png"
    plot_mean_sampling_distributions(summaries, "pilot", mean_png)
    print(f"      saved expected means     : {mean_png}", flush=True)

    if opts.run_normality:
        emit_normality_table(pooled, summaries, "pilot", display, opts, boot_rng)


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

    # ── figure (means histograms, six KPIs) ──
    tag = _safe_name(display)
    mean_png = opts.out_dir / f"final_means_{tag}_{opts.stamp}.png"
    plot_mean_sampling_distributions(summaries, "final", mean_png)
    print(f"\n      saved final means        : {mean_png}", flush=True)

    if opts.run_normality:
        emit_normality_table(derived, summaries, "final", display, opts, boot_rng)


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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pilot sample-size PLANNING (precision-based) and FINAL "
        "statistical INFERENCE for the SCENARIOS defined at the top of this file.")
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
                   help=f"Alpha for the KPI-mean normality test (default: {NORMALITY_ALPHA}).")
    p.add_argument("--normality_n_samples", type=int, default=NORMALITY_N_SAMPLES,
                   help="Bootstrap sample means fed to the Shapiro-Wilk normality "
                        f"test (default: {NORMALITY_N_SAMPLES}).")
    p.add_argument("--skip_normality", action="store_true",
                   help="Skip the Shapiro-Wilk normality test of the KPI-mean "
                        "sampling distributions (ON by default; prints + saves a "
                        "LaTeX table of the p-values).")
    p.add_argument("--scenario_name_check", choices=["warn", "error", "ignore"],
                   default="warn",
                   help="How to handle a display-name vs section.scenario mismatch "
                        "(default: warn + use resolved scenario).")
    p.add_argument("--device", type=str, default=None, metavar="DEVICE",
                   help="Torch device, e.g. 'cpu' or 'cuda:0'. Default: auto.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output dir (default: <script_dir>/stat_results).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not SCENARIOS:
        raise RuntimeError("SCENARIOS is empty — define at least one scenario.")
    if not 0.0 < args.normality_alpha < 1.0:
        raise ValueError("--normality_alpha must be strictly between 0 and 1.")
    if not 3 <= args.normality_n_samples <= 5000:
        raise ValueError("--normality_n_samples must be in [3, 5000] "
                         "(Shapiro-Wilk's supported range).")
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
        chunk=args.chunk, precision_step=args.precision_step,
        precision_bootstrap=args.precision_bootstrap, round_to=args.round_to,
        n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed,
        normality_alpha=args.normality_alpha,
        normality_n_samples=args.normality_n_samples,
        run_normality=not args.skip_normality,
        scenario_name_check=args.scenario_name_check,
        out_dir=out_dir, stamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
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
    print(f"  Histogram KPIs: " + ", ".join(k.label for k in HISTOGRAM_SPECS))
    print(f"  Bootstrap     : B={args.n_bootstrap}, seed={args.bootstrap_seed}")
    print(f"  Normality     : "
          + (f"OFF (--skip_normality)" if args.skip_normality
             else f"ON — Shapiro-Wilk on {args.normality_n_samples} bootstrap "
                  f"means/KPI, alpha={args.normality_alpha:g} (LaTeX table)"))
    print("─" * 70)

    analyse_scenarios(SCENARIOS, default_ckpt_path, opts, device)


if __name__ == "__main__":
    main()
