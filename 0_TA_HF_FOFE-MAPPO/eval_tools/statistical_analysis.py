r"""
statistical_analysis.py
═══════════════════════
Determine HOW MANY simulations (episodes) are needed for a frozen FOFE-MAPPO
policy's KPIs to STABILISE, using the *coefficient of variation* (CoV) method.

──────────────────────────────────────────────────────────────────────────────
WHAT THIS ANSWERS
──────────────────────────────────────────────────────────────────────────────
If the model output can follow any distribution, the coefficient of variation

        c_v = σ(o) / μ(o)                                              (eq. Cv)

is a distribution-free indicator of relative variability. As more simulations
are run, c_v converges to a stable value; once it no longer changes
significantly from one run-count to the next, the output has "stabilised" and
the required number of runs has been reached. The mean and the (sample)
standard deviation are

        μ(o) = (1/n) Σ Xᵢ                                            (eq. mean)
        σ(o) = sqrt( Σ (Xᵢ − X̄)² / (n − 1) )                        (eq. deviation)

The OUTPUT is TWO figures per scenario:
  1. CoV stabilisation: y = c_v, x = number of simulations, one line per KPI.
     When every line has flattened we know how many runs give stable results; a
     vertical marker shows the smallest run-count past which all KPIs stay
     inside a relative tolerance band of their final CoV.
  2. Distribution dashboard: a histogram (empirical probability density) of
     every KPI over ALL pooled runs, so you can see what kind of distribution
     each output follows (the very motivation for using the distribution-free
     CoV). The raw pooled samples are also dumped to CSV for your own fits.

The console also reports a D'Agostino-Pearson omnibus normality test for every
histogrammed KPI. At alpha=0.05, p < alpha rejects the null hypothesis that the
samples come from a normal distribution; p >= alpha means normality is not
rejected (it does not prove that the underlying distribution is normal).

──────────────────────────────────────────────────────────────────────────────
SEEDS vs RUNS-PER-SEED  (why the x-axis is "episodes")
──────────────────────────────────────────────────────────────────────────────
The whole simulation — layout sampling at reset AND the stochastic radar kills
during steps — is driven by ONE torch.Generator seeded once (see
environment.py: _set_seed / radar_kill draws). So re-running the *same* seed is
bit-identical: "many runs per fixed seed" is degenerate. The genuine unit of
randomness is therefore ONE EPISODE = one randomised layout + its stochastic
rollout. A vectorised env with num_envs=B from one seed already yields B
independent episodes; a different seed simply yields more independent episodes.

Hence:
  • x-axis  = number of episodes (the "number of simulations").
  • seeds   = independent REPLICATES of the whole CoV-vs-n curve. We run
              N_SEEDS master seeds, build one curve each, and plot the across-
              seed mean with a min–max band so the stabilisation point is shown
              to be seed-robust (not a fluke of one ordering).
  • Within each seed we also AVERAGE the cumulative CoV over many random
    episode orderings, so the curve estimates E[c_v(n)] over random subsets of
    size n and is not an artefact of the order episodes happened to arrive.

REWARD is excluded FROM THE CoV PLOT: episode reward is negative-shaped, so its
mean can sit near/below zero and c_v = σ/μ explodes or flips sign there (CoV is
only meaningful for non-negative, ratio-scale outputs). The five CoV KPIs are
all non-negative; reward still appears in the distribution dashboard, where its
sign is irrelevant.

──────────────────────────────────────────────────────────────────────────────
HOW TO USE
──────────────────────────────────────────────────────────────────────────────
Edit the SCENARIOS list below (same CurriculumSection format as
evaluate_policy.py / run_curriculum.py), then run with the project venv:

    # from the repo root:
    .\.venv\Scripts\python.exe 0_TA_HF_FOFE-MAPPO\statistical_analysis.py

    # fewer runs for a quick smoke-test:
    .\.venv\Scripts\python.exe 0_TA_HF_FOFE-MAPPO\statistical_analysis.py ^
        --max_episodes 300 --n_seeds 3

Each scenario may set `policy_file` to pick its checkpoint (resolved under
runs/); scenarios that leave it None fall back to --checkpoint. Outputs (one PNG
per scenario + one CSV of the curves) land in <script_dir>/stat_results.
"""

from __future__ import annotations

import argparse
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
from scipy.stats import normaltest

from .config import PPOConfig
from .trainer import build_env
# Reuse the EXACT scenario format + config-resolution + checkpoint loading so a
# scenario here behaves identically to one in evaluate_policy.py.
from .run_curriculum import CurriculumSection, _section_to_env_cfg, _section_label
from .evaluate_policy import (
    _LoadedCheckpoint, _build_policy_for_scenario, _resolve_policy_path,
)
# Importing nlr_style auto-applies the NLR house palette to Matplotlib.
from .nlr_style import NLR_CYCLE, NLR_REFERENCE


# =====================================================================
#  >>>  EDIT YOUR SCENARIOS HERE  <<<  (same format as evaluate_policy.py)
# =====================================================================

SCENARIOS: List[CurriculumSection] = [
    CurriculumSection(
        name="FOFE-MAPPO S2",
        policy_file="2s4j_V1.pt",
        n_iters=1,  # not used
        n_strikers=2, n_jammers=4,
        n_known_targets=(2, 4), n_unknown_targets=0,
        n_known_radars=6, n_unknown_radars=0,
        radar_kill_probability=0.5,
        scenario="S2",
        communicate=True,
    ),
]


# =====================================================================
#  >>>  ANALYSIS CONFIG  <<<  (all overridable on the CLI)
# =====================================================================

MAX_EPISODES   = 1000     # episodes per seed  → x-axis runs from 1..MAX_EPISODES
N_SEEDS        = 1        # independent master seeds (replicates of the curve)
BASE_SEED      = 42       # seed s uses BASE_SEED + s * SEED_STRIDE
SEED_STRIDE    = 1_000_000
NORMALITY_ALPHA = 0.05    # D'Agostino-Pearson test significance level
CHUNK_EPISODES = 250      # parallel envs per rollout (chunked to avoid OOM)
N_ORDERINGS    = 64       # random shuffles → order-independent CoV curve
STABILITY_TOL  = 0.05     # "stable" = CoV within ±5% of its final value …
STABILITY_MIN_N = 10      # … but ignore the very noisy n < this region


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

# Reward is excluded from the CoV plot (signed → CoV undefined) but IS shown in
# the distribution dashboard, where the sign of the data is irrelevant.
REWARD_KPI = ("reward", "episode_total_reward", "Episode reward")

# Everything collected per episode + histogrammed (the CoV KPIs + reward).
HIST_KPIS = KPI_COLUMNS + [REWARD_KPI]


# =====================================================================
#  STEP 1 — collect RAW per-episode KPI values (the Xᵢ samples)
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
#  STEP 2 — running coefficient of variation  c_v(n) = σ(n)/μ(n)
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
#  STEP 3 — required run-count: smallest n past which c_v stays "stable"
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
#  STEP 4 — plot + CSV
# =====================================================================

def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


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


def _write_csv(rows: List[dict], out_path: Path) -> None:
    import csv
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["scenario", "kpi", "n", "cov_mean", "cov_min", "cov_max"])
        writer.writeheader()
        writer.writerows(rows)


def plot_distributions(name: str, pooled: Dict[str, np.ndarray], n_runs: int,
                       out_png: Path) -> None:
    """Dashboard of per-KPI histograms (empirical probability density) over ALL
    pooled runs, so you can see what kind of distribution each output follows.

    Reward is included here (the histogram is sign-agnostic); the dashed line
    marks each KPI's mean and the title carries μ, σ and — for the non-negative
    KPIs — the coefficient of variation cᵥ.
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

    fig.suptitle(f"KPI distributions — {name}   "
                 f"(n = {n_runs} runs pooled across seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _write_samples_csv(pooled: Dict[str, np.ndarray], out_path: Path) -> None:
    """Raw per-episode KPI values (rows = runs, cols = KPIs) for your own fits."""
    import csv
    keys = [k for k, _, _ in HIST_KPIS]
    n = max((pooled[k].size for k in keys), default=0)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run"] + keys)
        for i in range(n):
            w.writerow([i] + [(pooled[k][i] if i < pooled[k].size else "")
                              for k in keys])


# =====================================================================
#  STEP 5 - D'Agostino-Pearson omnibus normality test
# =====================================================================

def normality_tests(pooled: Dict[str, np.ndarray],
                    alpha: float) -> Dict[str, dict]:
    """Test each pooled KPI against H0: the samples are normally distributed.

    SciPy's D'Agostino-Pearson test combines skewness and kurtosis and requires
    at least eight finite observations. ``is_normal`` is True when H0 is not
    rejected at ``alpha``, False when it is rejected, and None when the test
    cannot produce a valid result.
    """
    results: Dict[str, dict] = {}
    for key, _, _ in HIST_KPIS:
        x = pooled[key]
        x = x[np.isfinite(x)]
        result = {
            "n": int(x.size),
            "statistic": None,
            "pvalue": None,
            "is_normal": None,
            "reason": None,
        }
        if x.size < 8:
            result["reason"] = "requires at least 8 finite samples"
        elif np.ptp(x) == 0.0:
            result["reason"] = "all finite samples are identical"
        else:
            test = normaltest(x)
            statistic = float(test.statistic)
            pvalue = float(test.pvalue)
            if np.isfinite(statistic) and np.isfinite(pvalue):
                result["statistic"] = statistic
                result["pvalue"] = pvalue
                result["is_normal"] = pvalue >= alpha
            else:
                result["reason"] = "test returned a non-finite result"
        results[key] = result
    return results


def _print_normality_results(results: Dict[str, dict], alpha: float) -> None:
    """Print one D'Agostino-Pearson normality verdict per histogrammed KPI."""
    print(f"      -- D'Agostino-Pearson normality (alpha={alpha:g}) --")
    for key, _, label in HIST_KPIS:
        result = results[key]
        if result["is_normal"] is None:
            print(f"        {label:16s}  not testable"
                  f"  (n={result['n']}: {result['reason']})", flush=True)
            continue
        verdict = ("normally distributed" if result["is_normal"]
                   else "NOT normally distributed")
        print(f"        {label:16s}  {verdict:24s}"
              f"  (n={result['n']}, K^2={result['statistic']:.4g}, "
              f"p={result['pvalue']:.4g})", flush=True)
    print("        Note: 'normally distributed' means H0 was not rejected; "
          "it does not prove normality.", flush=True)
    print("        Note: Completion is binary, so a continuous normal model "
          "is not appropriate for that KPI.", flush=True)


# =====================================================================
#  DRIVER
# =====================================================================

def analyse_scenarios(scenarios: List[CurriculumSection],
                      default_ckpt_path: Optional[Path],
                      max_episodes: int, n_seeds: int, base_seed: int,
                      chunk: int, n_orderings: int, tol: float, min_n: int,
                      normality_alpha: float,
                      device: torch.device, out_dir: Path,
                      write_csv: bool) -> None:
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

        # ── report ──
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

        # ── STEP 4: plot + CSV rows ──
        out_png = out_dir / f"cov_{_safe_name(section.name)}_{stamp}.png"
        plot_scenario(curves, max_episodes, out_png)
        print(f"      saved CoV plot      : {out_png}", flush=True)

        # ── KPI probability-distribution dashboard (all runs pooled across seeds) ──
        pooled = {k: np.concatenate([s[k] for s in per_seed]) for k, _, _ in HIST_KPIS}
        n_runs = next(iter(pooled.values())).size
        dist_png = out_dir / f"kpi_dist_{_safe_name(section.name)}_{stamp}.png"
        plot_distributions(section.name, pooled, n_runs, dist_png)
        print(f"      saved distributions : {dist_png}", flush=True)

        normality = normality_tests(pooled, normality_alpha)
        _print_normality_results(normality, normality_alpha)

        if write_csv:
            samp_csv = out_dir / f"kpi_samples_{_safe_name(section.name)}_{stamp}.csv"
            _write_samples_csv(pooled, samp_csv)
            print(f"      saved samples CSV   : {samp_csv}", flush=True)

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
        print(f"\nSaved CSV to: {csv_path}\n", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CoV-based run-count stabilisation analysis for the "
        "SCENARIOS defined at the top of this file.")
    p.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                   help="Default .pt for scenarios that leave policy_file=None.")
    p.add_argument("--max_episodes", type=int, default=MAX_EPISODES,
                   help=f"Episodes per seed / x-axis max (default: {MAX_EPISODES}).")
    p.add_argument("--n_seeds", type=int, default=N_SEEDS,
                   help=f"Independent seeds = curve replicates (default: {N_SEEDS}).")
    p.add_argument("--seed", type=int, default=BASE_SEED,
                   help=f"Base RNG seed (default: {BASE_SEED}).")
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
    p.add_argument("--device", type=str, default=None, metavar="DEVICE",
                   help="Torch device, e.g. 'cpu' or 'cuda:0'. Default: auto.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output dir (default: <script_dir>/stat_results).")
    p.add_argument("--no_csv", action="store_true", help="Do not write the CSV.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not SCENARIOS:
        raise RuntimeError("SCENARIOS is empty — define at least one scenario.")
    if not 0.0 < args.normality_alpha < 1.0:
        raise ValueError("--normality_alpha must be strictly between 0 and 1.")

    default_ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if default_ckpt_path is not None and not default_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {default_ckpt_path}")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else (_PKG_DIR / "stat_results")

    print("─" * 70)
    print(f"  Device       : {device}")
    print(f"  Scenarios    : {len(SCENARIOS)}")
    print(f"  Sweep        : up to {args.max_episodes} episodes × {args.n_seeds} seeds")
    print(f"  KPIs (CoV)   : " + ", ".join(lbl for _, _, lbl in KPI_COLUMNS))
    print(f"  KPIs (dist)  : " + ", ".join(lbl for _, _, lbl in HIST_KPIS)
          + "   (reward: histogram only)")
    print(f"  Normality    : D'Agostino-Pearson, alpha={args.normality_alpha:g}")
    print("─" * 70)

    analyse_scenarios(
        SCENARIOS, default_ckpt_path,
        max_episodes=args.max_episodes, n_seeds=args.n_seeds, base_seed=args.seed,
        chunk=args.chunk, n_orderings=args.n_orderings, tol=args.tol,
        min_n=args.min_n, normality_alpha=args.normality_alpha,
        device=device, out_dir=out_dir,
        write_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()
