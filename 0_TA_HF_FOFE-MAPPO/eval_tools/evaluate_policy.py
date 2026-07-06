"""
evaluate_policy.py
══════════════════
Compare a MAIN FOFE-MAPPO policy against zero or more COMPARISON policies on a
set of test scenarios, and decide — per KPI — whether the main policy is
significantly better, using the **Wilcoxon signed-rank test**.

──────────────────────────────────────────────────────────────────────────────
WHAT THIS PRODUCES
──────────────────────────────────────────────────────────────────────────────
For every KPI it builds ONE table (console + LaTeX): rows = scenarios, columns
= policies. The MAIN policy column shows the raw KPI value with NO p-value (it
is the reference). Every COMPARISON policy column shows its KPI value, the
deviation from the main policy in parentheses, and a one-sided Wilcoxon
signed-rank p-value testing the hypothesis below. With a single policy (no
comparisons) you simply get the value columns and no p-values.

Hypotheses tested (H1 = what we hope holds for the MAIN policy):
    Completion, Targets, Survival, Reward   →  main is HIGHER than comparison
    Duration                                →  main is LOWER  than comparison
    Coalition fragmentation                 →  main ≠ comparison (two-sided)

──────────────────────────────────────────────────────────────────────────────
WHY A *SIGNED-RANK* (PAIRED) TEST IS VALID HERE
──────────────────────────────────────────────────────────────────────────────
The environment is fully seeded: a given seed reproduces the same randomised
layout at reset (see statistical_analysis.py / environment.py). Every policy in
a scenario is therefore evaluated on the SAME seed schedule, so episode i faces
the SAME starting layout for the main and each comparison policy. That makes the
per-episode pair (main_i, baseline_i) a matched pair, and the test runs on the
paired differences d_i = main_i − baseline_i (common random numbers → higher
power). Episodes where either policy never finished are dropped pairwise.

The layout draw at reset depends only on the entity counts / scenario / ranges
(which the scenario forces identical across policies) and the seed — not on the
reward shaping, communication flag, FOFE, or the policy itself — so the pairing
holds even when the compared policies are trained variants (e.g. comm on/off).

──────────────────────────────────────────────────────────────────────────────
HOW TO CONFIGURE  (edit the three blocks below)
──────────────────────────────────────────────────────────────────────────────
1. MAIN_POLICY            – the reference checkpoint (the "no-p-value" column).
2. COMPARISON_POLICIES    – the checkpoints to test against it (each a column
                            WITH a p-value). Leave empty for a single-policy run.
3. EVAL_SCENARIOS         – the environments (rows). Same `CurriculumSection`
                            format as run_curriculum.py. Its `policy_file` field
                            is IGNORED here — policies come from blocks 1 & 2.

`policy_file` values resolve like:
    "BaselineV2.pt"                  → under runs/ (bare name, the common case)
    "runs/FINALV1/…/stage4.pt"       → relative to the project dir 0_TA_.../
    r"C:\\path\\to\\x.pt"              → absolute path, used as-is
    None                             → falls back to the --checkpoint default

Run it as a direct script with the project's venv python:

    # from the repo root:
    .\\.venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\evaluate_policy.py

    # quicker smoke-test:
    .\\.venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\evaluate_policy.py --n_episodes 100
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Match the training entry points: force Inductor to compile in-process (avoids
# a Triton 'cubin' KeyError on some Linux+CUDA setups). Harmless elsewhere.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# Windows consoles default to cp1252, which can't encode the table glyphs
# (± × ─ Δ α). Switch the streams to UTF-8 so printing never crashes.
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
from scipy.stats import wilcoxon

from .config import (
    EnvConfig, EnvExtensionsConfig, FOFEConfig, NetworkConfig, PPOConfig,
)
from .rewards import RewardConfig
from .models import make_combined_policy
from .trainer import build_env
# Reuse the EXACT curriculum scenario format + its config-resolution logic so a
# scenario defined here behaves identically to a curriculum section.
from .run_curriculum import CurriculumSection, _section_to_env_cfg, _section_label


# =====================================================================
#  >>>  POLICIES TO COMPARE  (the COLUMNS of every KPI table)  <<<
# =====================================================================

@dataclass
class PolicyInput:
    """One model variant = one column in every KPI table.

    `name`        : column header (e.g. "Complete", "No comm.").
    `policy_file` : checkpoint — bare name → runs/, path with sub-dirs (e.g.
                    "runs/FINALV1/…/x.pt") → relative to the project dir,
                    absolute path → as-is, None → the run's --checkpoint default.
    `communicate` : per-policy communication flag. Set it to match how the
                    model was trained (e.g. True for the complete model, False
                    for a no-comms baseline) — communication is defined HERE,
                    not on the scenarios. None → fall back to the checkpoint's
                    own trained comms setting.
    """
    name: str
    policy_file: Optional[str] = None
    communicate: Optional[bool] = None


# The MAIN policy — shown WITHOUT a p-value (the reference column).
# Complete model: trained WITH communication → evaluate with comms on.
MAIN_POLICY = PolicyInput(
    name="Complete",
    policy_file="runs/FINALV1/complete_S1_20260704/stage4of5_DR_j2-4_k0_1.pt",
    communicate=True,
)

# The COMPARISON policies — each shown WITH a one-sided Wilcoxon p-value vs MAIN.
# Leave this list EMPTY to just tabulate the main policy (no tests, no p-values).
COMPARISON_POLICIES: List[PolicyInput] = [
    # Baseline model: trained WITHOUT communication → evaluate with comms off.
    PolicyInput(
        name="Baseline",
        policy_file="runs/FINALV1/baseline_S1_20260704/stage4of5_DR_j2-4_k0_1.pt",
        communicate=False,
    ),
]

# =====================================================================
#  >>>  EVAL SCENARIOS  (the ROWS of every KPI table)  <<<
#  Same format as the CURRICULUM list in run_curriculum.py. The `policy_file`
#  field is IGNORED here — policies come from the two blocks above.
#  Communication is NOT set here on purpose: it is a per-policy property
#  (see PolicyInput.communicate above), so each model is evaluated with the
#  comms setting it was trained with, on the same scenario world.
# =====================================================================

EVAL_SCENARIOS: List[CurriculumSection] = [
    CurriculumSection(
        name="S1 - Overall",
        n_iters=1,  # not used
        n_strikers=2, n_jammers=(2, 4),
        n_known_targets=(2, 4), n_unknown_targets=0,
        n_known_radars=(4, 6), n_unknown_radars=0,
        radar_kill_probability=0.1,
        scenario="S2",
    ), 
    CurriculumSection(
        name="2s2j",
        n_iters=1,  # not used
        n_strikers=2, n_jammers=2,
        n_known_targets=(2, 4), n_unknown_targets=0,
        n_known_radars=(4, 6), n_unknown_radars=0,
        radar_kill_probability=0.1,
        scenario="S2",
    ), 
    CurriculumSection(
        name="2s3j",
        n_iters=1,  # not used
        n_strikers=2, n_jammers=3,
        n_known_targets=(2, 4), n_unknown_targets=0,
        n_known_radars=(4, 6), n_unknown_radars=0,
        radar_kill_probability=0.1,
        scenario="S2",
    ), 
    CurriculumSection(
        name="2s4j",
        n_iters=1,  # not used
        n_strikers=2, n_jammers=4,
        n_known_targets=(2, 4), n_unknown_targets=0,
        n_known_radars=(4, 6), n_unknown_radars=0,
        radar_kill_probability=0.1,
        scenario="S2",
    )
]


# =====================================================================
#  >>>  TEST / EVALUATION CONFIG  (all the "inputs" for the test)  <<<
# =====================================================================

N_EPISODES   = 500        # paired episodes per policy per scenario (= the test N)
CHUNK_EPISODES = 300      # parallel envs per rollout (chunked to avoid OOM)
BASE_SEED    = 42         # scenario s uses BASE_SEED + s*SEED_STRIDE for every policy
SEED_STRIDE  = 1_000_000  # distinct layouts per scenario, identical across policies
ALPHA        = 0.05       # significance level for declaring "significantly better"
P_ADJUST     = "none"     # multiplicity correction: "none" (default) or "holm".
                          # "holm" corrects across the comparison policies within
                          # each (scenario, KPI).


# =====================================================================
#  KPI definitions
#  direction = the main policy is BETTER when its value is {higher|lower};
#  it maps to the one-sided Wilcoxon alternative on d = main − baseline:
#       higher → "greater"      lower → "less"
# =====================================================================

@dataclass(frozen=True)
class KPISpec:
    key: str          # short id (column key)
    stat_key: str     # per-episode key in env.pop_episode_stats()
    label: str        # human label / caption
    fmt: str          # value format, e.g. "{:.3f}"
    unit: str         # short sub-header, e.g. "rate" / "steps" / "reward"
    direction: str    # "higher" / "lower" (main better when …) or "two-sided"


KPIS: List[KPISpec] = [
    KPISpec("completion",    "mission_complete",        "Task completion",  "{:.3f}", "rate",   "higher"),
    KPISpec("targets",       "targets_frac",            "Targets destroyed","{:.3f}", "rate",   "higher"),
    KPISpec("survival",      "survival_frac",           "Survival",         "{:.3f}", "rate",   "higher"),
    KPISpec("duration",      "duration",                "Duration",         "{:.1f}", "steps",  "lower"),
    KPISpec("reward",        "episode_total_reward",    "Episode reward",   "{:.2f}", "reward", "higher"),
    # Neither more nor less fragmentation is universally "better", so it is
    # tested TWO-SIDED (H1: main != comparison).
    KPISpec("fragmentation", "coalition_fragmentation", "Coalition frag.",  "{:.3f}", "index",  "two-sided"),
]

# No untested extra stats: fragmentation is now a first-class KPI above.
_EXTRA_STAT_KEYS: List[Tuple[str, str]] = []

_ALT = {"higher": "greater", "lower": "less", "two-sided": "two-sided"}  # direction → scipy alternative


# =====================================================================
#  IMPLEMENTATION  —  you usually do not need to edit below this line
# =====================================================================

def _ensure_checkpoint_import_aliases() -> None:
    """Expose the `fofe_mappo.*` module names that pickled configs reference.

    Checkpoints store dataclass instances (EnvConfig, RewardConfig, ...) pickled
    under the `fofe_mappo` package alias the training scripts run under. The
    top-level imports above already register fofe_mappo.config / .rewards, but
    we setdefault them explicitly so torch.load(weights_only=False) is robust.
    """
    cfg_mod = sys.modules.get(EnvConfig.__module__)
    rew_mod = sys.modules.get(RewardConfig.__module__)
    if cfg_mod is not None:
        sys.modules.setdefault(f"{_PKG_NAME}.config", cfg_mod)
    if rew_mod is not None:
        sys.modules.setdefault(f"{_PKG_NAME}.rewards", rew_mod)


def _strip_compile_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Drop torch.compile's `_orig_mod.` key prefix so strict loads work.

    On Linux+GPU the trainer compiles the actor/critic, inserting `_orig_mod.`
    into every parameter key. Curriculum checkpoints already strip this, but
    run.py checkpoints may not — stripping here keeps both portable."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


class _LoadedCheckpoint:
    """Everything needed to rebuild the trained policy, from either producer."""

    def __init__(self, path: Path, device: torch.device):
        _ensure_checkpoint_import_aliases()
        ckpt = torch.load(path, map_location=device, weights_only=False)

        if "policy_state_dict" not in ckpt:
            raise KeyError(
                f"{path.name} has no 'policy_state_dict' — is this a "
                f"checkpoint produced by run.py / run_curriculum.py?"
            )

        self.policy_state_dict = _strip_compile_prefix(ckpt["policy_state_dict"])
        self.net_cfg: NetworkConfig = ckpt.get("net_cfg", NetworkConfig())
        self.fofe_cfg: FOFEConfig = ckpt.get("fofe_cfg", FOFEConfig(use_fofe=True))
        self.ext_cfg: EnvExtensionsConfig = ckpt.get("ext_cfg", EnvExtensionsConfig())

        # Producer detection: run.py stores a full env_cfg; curriculum does not.
        env_cfg: Optional[EnvConfig] = ckpt.get("env_cfg")
        if env_cfg is not None:
            self.producer = "run.py"
            self.base_env_cfg = env_cfg
            self.reward_cfg = copy.deepcopy(env_cfg.reward_config)
        else:
            self.producer = "run_curriculum.py"
            # Curriculum trained on config.py's leading EnvConfig() defaults for
            # every field a section didn't override — mirror that here.
            self.base_env_cfg = EnvConfig()
            self.reward_cfg = ckpt.get("reward_cfg", RewardConfig())

        self.hf_radar_cfg = (
            self.ext_cfg.hf_radar if getattr(self.ext_cfg, "use_hf_radar", False) else None
        )


def _resolve_policy_path(policy_file: Optional[str],
                         default_path: Optional[Path]) -> Optional[Path]:
    """Resolve a `policy_file` to a concrete checkpoint path.

    None                    → the run's --checkpoint default (may be None).
    absolute path           → used as-is.
    path with sub-dirs      → relative to the project dir 0_TA_.../ , so a value
                              like "runs/FINALV1/complete/stage4.pt" works (same
                              convention as compare_compositions.py).
    bare name (no dirs)     → under runs/ (e.g. "BaselineV2.pt").
    """
    if policy_file is None:
        return default_path
    p = Path(policy_file)
    if p.is_absolute():
        return p
    # A path that already carries directory components (e.g. "runs/…/x.pt") is
    # taken relative to the project dir — otherwise prepending _RUNS_DIR would
    # double the "runs/" segment. A bare filename still resolves under runs/.
    if p.parent != Path("."):
        return _PKG_DIR / p
    return _RUNS_DIR / p


def _env_cfg_for_policy(pol: "PolicyInput", section: "CurriculumSection",
                        ckpt: "_LoadedCheckpoint"):
    """EnvConfig for (policy, scenario), applying the policy's `communicate`
    setting so each model is evaluated with the comms it was trained with.
    Scenarios do not set comms; `pol.communicate is None` falls back to the
    checkpoint's own trained value (via `_section_to_env_cfg`)."""
    env_cfg = _section_to_env_cfg(
        section, ckpt.base_env_cfg, ckpt.reward_cfg, ckpt.fofe_cfg
    )
    if pol.communicate is not None:
        env_cfg.communicate = bool(pol.communicate)
    return env_cfg


def _build_policy_for_scenario(ckpt: _LoadedCheckpoint, env_cfg: EnvConfig,
                               device: torch.device):
    """Rebuild a policy sized to env_cfg's role split and strict-load weights.

    FOFE parameter shapes are count-invariant, so the load is exact; only the
    striker/jammer role split (which the wrapper uses to slice actions) depends
    on env_cfg, hence the per-scenario rebuild."""
    probe_ppo = PPOConfig(num_envs=1, device=device)
    probe_env = build_env(env_cfg, probe_ppo, hf_radar_cfg=ckpt.hf_radar_cfg)
    policy = make_combined_policy(
        probe_env,
        hidden=ckpt.net_cfg.actor_hidden,
        depth=ckpt.net_cfg.depth,
        fofe_cfg=ckpt.fofe_cfg if ckpt.fofe_cfg.use_fofe else None,
    )
    try:
        policy.load_state_dict(ckpt.policy_state_dict, strict=True)
    except RuntimeError as exc:
        # Turn torch's cryptic size-mismatch into an actionable message: the
        # checkpoint was trained against a different observation layout than the
        # current env/config produces — almost always an OUTDATED checkpoint
        # (e.g. a flat-MLP policy saved before the obs-slot defaults changed).
        want = next((int(v.shape[1]) for k, v in ckpt.policy_state_dict.items()
                     if k.endswith("net.net.0.weight") and getattr(v, "ndim", 0) == 2),
                    None)
        try:
            have = int(probe_env.observation_spec["agents", "observation"].shape[-1])
        except Exception:
            have = None
        dims = (f" Its flat-MLP actor expects obs_dim={want}, but this scenario "
                f"builds obs_dim={have}." if (want and have and want != have) else "")
        raise RuntimeError(
            f"Cannot load this checkpoint into the policy rebuilt for the "
            f"scenario (FOFE={'on' if ckpt.fofe_cfg.use_fofe else 'off'}).{dims} "
            f"The checkpoint's observation layout does not match what the current "
            f"environment/config produces — almost always an OUTDATED checkpoint. "
            f"Evaluate a policy trained against the current config.\n"
            f"  underlying error: {exc}"
        ) from exc
    return policy.to(device)


# =====================================================================
#  RAW per-episode KPI collection (the Xᵢ samples the test needs)
#  Mirrors statistical_analysis.py: a parallel batch of episodes, keeping the
#  RAW per-episode stats instead of averaging. Stats for a newly-finished env
#  are popped immediately so a re-stepped done env can't overwrite them.
# =====================================================================

_ALL_STAT_KEYS: List[Tuple[str, str]] = (
    [(s.key, s.stat_key) for s in KPIS] + _EXTRA_STAT_KEYS
)


@torch.no_grad()
def _rollout_collect(policy, env, max_steps: int, n_envs: int,
                     device: torch.device) -> List[Optional[dict]]:
    """Run one parallel batch of `n_envs` episodes; return per-env stats dicts."""
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


def _collect_episode_kpis(policy, env_cfg, hf_radar_cfg, n_episodes: int,
                          master_seed: int, device: torch.device,
                          chunk: int) -> Dict[str, np.ndarray]:
    """Return {kpi_key: array[n_episodes]} of per-episode values for one policy.

    Episodes are generated in independent parallel chunks. The chunk seeds are a
    deterministic function of `master_seed` only — so two policies called with
    the SAME master_seed, n_episodes and chunk see the SAME layouts in the SAME
    order, i.e. their returned arrays are aligned 1:1 (paired by episode index).
    """
    per_kpi: Dict[str, List[float]] = {key: [] for key, _ in _ALL_STAT_KEYS}
    collected = 0
    chunk_idx = 0
    while collected < n_episodes:
        b = min(chunk, n_episodes - collected)
        ppo = PPOConfig(num_envs=b, device=device,
                        seed=int(master_seed + chunk_idx * 1000))
        env = build_env(env_cfg, ppo, hf_radar_cfg=hf_radar_cfg)
        stats = _rollout_collect(policy, env, env_cfg.max_steps, b, device)

        for s in stats:
            for key, stat_key in _ALL_STAT_KEYS:
                if s is None:
                    per_kpi[key].append(float("nan"))
                elif stat_key == "mission_complete":           # bool → 0/1 KPI
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
#  Statistics
# =====================================================================

def _finite_stats(arr: np.ndarray) -> Tuple[float, float, int]:
    """(mean, sample-std, n) over the finite entries; (nan, nan, 0) if none."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), 0
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return float(a.mean()), std, int(a.size)


def _wilcoxon_pair(main_vals: np.ndarray, base_vals: np.ndarray,
                   alternative: str) -> Dict[str, Any]:
    """One-sided Wilcoxon signed-rank test on the paired differences.

    d_i = main_i − baseline_i over episodes where BOTH policies finished.
    alternative="greater" → H1 median(d) > 0 (main higher);
    alternative="less"    → H1 median(d) < 0 (main lower).
    """
    m = np.asarray(main_vals, dtype=float)
    b = np.asarray(base_vals, dtype=float)
    n = min(m.size, b.size)
    m, b = m[:n], b[:n]                       # guard against ragged collection
    mask = np.isfinite(m) & np.isfinite(b)
    d = m[mask] - b[mask]

    out: Dict[str, Any] = {
        "alternative": alternative,
        "n_pairs": int(d.size),
        "statistic": float("nan"),
        "pvalue": float("nan"),
        "median_diff": float(np.median(d)) if d.size else float("nan"),
        "mean_diff": float(np.mean(d)) if d.size else float("nan"),
        "reason": None,
    }
    if d.size < 1:
        out["reason"] = "no finite paired episodes"
        return out
    if np.all(d == 0.0):
        # Identical on every pair → no evidence of a difference in either tail.
        out["statistic"], out["pvalue"] = 0.0, 1.0
        out["reason"] = "all paired differences are zero"
        return out
    try:
        res = wilcoxon(d, alternative=alternative, zero_method="wilcox",
                       correction=False)
        out["statistic"] = float(res.statistic)
        out["pvalue"] = float(res.pvalue)
    except ValueError as exc:                 # e.g. degenerate inputs
        out["reason"] = str(exc)
    return out


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
        running = max(running, adj)           # enforce monotonic non-decreasing
        out[i] = running
    return out


# =====================================================================
#  Evaluation driver
# =====================================================================

def _load_checkpoint(path: Path, device: torch.device,
                     cache: Dict[str, _LoadedCheckpoint]) -> _LoadedCheckpoint:
    """Load (and cache) a checkpoint, announcing producer/FOFE/HF-radar once."""
    key = str(path.resolve())
    if key not in cache:
        print(f"  Loading policy: {path.name}", flush=True)
        ck = _LoadedCheckpoint(path, device)
        print(f"      producer={ck.producer}  "
              f"FOFE={'on' if ck.fofe_cfg.use_fofe else 'off'}  "
              f"HF_radar={'on' if ck.hf_radar_cfg is not None else 'off'}",
              flush=True)
        cache[key] = ck
    return cache[key]


def evaluate_comparison(
    scenarios: List[CurriculumSection],
    main_policy: PolicyInput,
    comparison_policies: List[PolicyInput],
    default_ckpt_path: Optional[Path],
    n_episodes: int,
    chunk: int,
    base_seed: int,
    p_adjust: str,
    device: torch.device,
    cache: Optional[Dict[str, _LoadedCheckpoint]] = None,
) -> Tuple[Dict[Tuple[str, str, str], Tuple[float, float, int]],
           Dict[Tuple[str, str, str], Dict[str, Any]],
           Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """Collect paired per-episode KPIs and run the per-KPI Wilcoxon tests.

    Returns:
      summary[(scenario, policy, kpi)]  = (mean, std, n_finite)
      tests[(scenario, baseline, kpi)]  = wilcoxon result dict (raw + adjusted p)
      samples[scenario][policy]         = {kpi: array[n_episodes]}

    `cache` (resolved-path → loaded checkpoint) may be shared with the
    pre-evaluation policy check so each checkpoint is loaded only once.
    """
    policies = [main_policy] + list(comparison_policies)
    cache = {} if cache is None else cache

    summary: Dict[Tuple[str, str, str], Tuple[float, float, int]] = {}
    tests: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    samples: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for si, section in enumerate(scenarios):
        master_seed = base_seed + si * SEED_STRIDE
        print(f"\n[{si + 1}/{len(scenarios)}] Scenario '{section.name}'  "
              f"(seed={master_seed}, N={n_episodes} paired episodes)", flush=True)

        per_policy: Dict[str, Dict[str, np.ndarray]] = {}
        max_steps_seen: Dict[str, int] = {}

        for pol in policies:
            ckpt_path = _resolve_policy_path(pol.policy_file, default_ckpt_path)
            if ckpt_path is None:
                raise RuntimeError(
                    f"Policy '{pol.name}' sets no policy_file and no "
                    f"--checkpoint default was given — cannot pick a checkpoint."
                )
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Policy '{pol.name}': checkpoint not found: {ckpt_path}"
                )
            ckpt = _load_checkpoint(ckpt_path, device, cache)
            env_cfg = _env_cfg_for_policy(pol, section, ckpt)
            max_steps_seen[pol.name] = int(env_cfg.max_steps)
            print(f"    [{pol.name:14s}] {ckpt_path.name:22s} "
                  f"{_section_label(section, env_cfg)}", flush=True)

            policy = _build_policy_for_scenario(ckpt, env_cfg, device)
            arrs = _collect_episode_kpis(
                policy, env_cfg, ckpt.hf_radar_cfg, n_episodes, master_seed,
                device, chunk,
            )
            per_policy[pol.name] = arrs
            for spec in KPIS:
                summary[(section.name, pol.name, spec.key)] = _finite_stats(arrs[spec.key])
            print("        means: " + "  ".join(
                f"{spec.key}={summary[(section.name, pol.name, spec.key)][0]:.3f}"
                for spec in KPIS), flush=True)

            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Horizon must match for a fair paired comparison (duration/completion
        # are horizon-sensitive). Same seed still pairs the START layouts, but a
        # differing horizon means the rollouts are not comparable.
        if len(set(max_steps_seen.values())) > 1:
            print("    ! WARNING: policies resolved to different max_steps "
                  f"{max_steps_seen} — set `max_steps` in the scenario to force "
                  "a common horizon for a fair comparison.", flush=True)

        # ── per-KPI Wilcoxon tests vs the main policy ──
        main_arrs = per_policy[main_policy.name]
        for spec in KPIS:
            raw: List[float] = []
            base_keys: List[Tuple[str, str, str]] = []
            for base in comparison_policies:
                res = _wilcoxon_pair(main_arrs[spec.key],
                                     per_policy[base.name][spec.key],
                                     _ALT[spec.direction])
                key = (section.name, base.name, spec.key)
                tests[key] = res
                base_keys.append(key)
                raw.append(res["pvalue"])
            # Multiplicity correction within (scenario, KPI) across baselines.
            adj = _holm_adjust(raw) if p_adjust == "holm" else list(raw)
            for key, a in zip(base_keys, adj):
                tests[key]["pvalue_adj"] = a

        samples[section.name] = per_policy

    return summary, tests, samples


# =====================================================================
#  Console + LaTeX + CSV output
# =====================================================================

def _fmt_or_na(spec: KPISpec, x: float) -> str:
    return spec.fmt.format(x) if np.isfinite(x) else "n/a"


def _pct_delta(main_mean: float, base_mean: float, tex: bool = False) -> str:
    """Comparison's deviation from the main policy as a signed percentage,
    relative to the main: 100·(base − main)/|main|, e.g. '+12.3%' / '-8.1%'.

    Returns 'n/a' when undefined (non-finite, or main == 0)."""
    pct_sign = r"\%" if tex else "%"
    if not (np.isfinite(main_mean) and np.isfinite(base_mean)) or main_mean == 0.0:
        return "n/a"
    pct = (base_mean - main_mean) / abs(main_mean) * 100.0
    return f"{pct:+.1f}{pct_sign}"


def _stars(p: float, alpha: float) -> int:
    if not np.isfinite(p):
        return 0
    if p < 0.001:
        return 3
    if p < 0.01:
        return 2
    if p < alpha:
        return 1
    return 0


def _p_console(p: float, alpha: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    txt = "<0.001" if p < 0.001 else f"{p:.3f}"
    return txt + "*" * _stars(p, alpha)


def _p_used(res: Dict[str, Any], p_adjust: str) -> float:
    """The p-value to display (adjusted if a correction was requested)."""
    return res.get("pvalue_adj", res["pvalue"]) if p_adjust == "holm" else res["pvalue"]


def _print_config(main_policy: PolicyInput, comparison_policies: List[PolicyInput],
                  scenarios: List[CurriculumSection], n_episodes: int,
                  base_seed: int, alpha: float, p_adjust: str,
                  device: torch.device) -> None:
    print("─" * 78)
    print("  WILCOXON SIGNED-RANK POLICY COMPARISON")
    print("─" * 78)
    print(f"  Main policy        : {main_policy.name}  [{main_policy.policy_file}]")
    if comparison_policies:
        print(f"  Comparison policies: "
              + ", ".join(f"{p.name} [{p.policy_file}]" for p in comparison_policies))
    else:
        print("  Comparison policies: (none — values only, no p-values)")
    print(f"  Scenarios          : {len(scenarios)}  "
          f"({', '.join(s.name for s in scenarios)})")
    print(f"  Test               : Wilcoxon signed-rank, paired by common seed")
    print(f"  N (paired episodes): {n_episodes}   base seed: {base_seed}")
    print(f"  Significance       : alpha={alpha:g}   correction: {p_adjust}")
    print("  Hypotheses (H1, main vs comparison):")
    for spec in KPIS:
        if spec.direction == "two-sided":
            print(f"      {spec.label:18s} main ≠ comparison  (alternative='two-sided')")
        else:
            rel = "higher" if spec.direction == "higher" else "lower"
            print(f"      {spec.label:18s} main {rel:6s}  (alternative='{_ALT[spec.direction]}')")
    print("  Stars: * p<{a:g}   ** p<0.01   *** p<0.001".format(a=alpha))
    print("─" * 78)


def _print_policy_diagnostics(main_policy: PolicyInput,
                              comparison_policies: List[PolicyInput],
                              scenarios: List[CurriculumSection],
                              default_ckpt_path: Optional[Path],
                              device: torch.device,
                              cache: Dict[str, _LoadedCheckpoint]) -> None:
    """Per-policy sanity check printed at the top: whether each policy uses the
    FOFE encoder (off → flat MLP) and whether communication is on/off in the
    environment it is evaluated in. Loads checkpoints into the shared `cache`."""
    entries = ([("Main", main_policy)]
               + [("Comparison", p) for p in comparison_policies])

    loaded = []  # (role, policy, path|None, ckpt|None)
    for role, pol in entries:
        path = _resolve_policy_path(pol.policy_file, default_ckpt_path)
        ck = _load_checkpoint(path, device, cache) if (path and path.exists()) else None
        loaded.append((role, pol, path, ck))

    print("\n  Policy check  (FOFE encoder / communication used in evaluation):")
    for role, pol, path, ck in loaded:
        if ck is None:
            print(f"      [{role:10s}] {pol.name:14s} : checkpoint NOT FOUND "
                  f"({pol.policy_file})", flush=True)
            continue
        fofe_txt = "FOFE on" if ck.fofe_cfg.use_fofe else "FOFE off (flat MLP)"
        comms = [bool(getattr(_env_cfg_for_policy(pol, s, ck),
                    "communicate", True)) for s in scenarios]
        if all(comms):
            comm_txt = "communication on"
        elif not any(comms):
            comm_txt = "communication off (no communication)"
        else:
            comm_txt = "communication " + " ".join(
                f"{s.name}={'on' if c else 'off'}" for s, c in zip(scenarios, comms))
        print(f"      [{role:10s}] {pol.name:14s} : {fofe_txt:19s} | "
              f"{comm_txt:36s} [{path.name}]", flush=True)
    print("─" * 78)


def _print_kpi_console(spec: KPISpec, scenarios: List[CurriculumSection],
                       main_policy: PolicyInput,
                       comparison_policies: List[PolicyInput],
                       summary: Dict[Tuple[str, str, str], Tuple[float, float, int]],
                       tests: Dict[Tuple[str, str, str], Dict[str, Any]],
                       alpha: float, p_adjust: str) -> None:
    better = ("two-sided, no better direction" if spec.direction == "two-sided"
              else f"{spec.direction} is better")
    headers = ["Scenario", f"{main_policy.name} ({spec.unit})"]
    for base in comparison_policies:
        headers += [f"{base.name} ({spec.unit}, Δ%)", "p"]

    table: List[List[str]] = []
    for scn in scenarios:
        m_mean, _, _ = summary[(scn.name, main_policy.name, spec.key)]
        cells = [scn.name, _fmt_or_na(spec, m_mean)]
        for base in comparison_policies:
            b_mean, _, _ = summary[(scn.name, base.name, spec.key)]
            res = tests[(scn.name, base.name, spec.key)]
            b_str = _fmt_or_na(spec, b_mean)
            if np.isfinite(b_mean):                       # value (+-xx%)
                b_str += f" ({_pct_delta(m_mean, b_mean)})"
            cells += [b_str, _p_console(_p_used(res, p_adjust), alpha)]
        table.append(cells)

    widths = [len(h) for h in headers]
    for row in table:
        for c, cell in enumerate(row):
            widths[c] = max(widths[c], len(cell))

    def _fmt_row(cells: List[str]) -> str:
        return "  ".join(cell.ljust(widths[c]) if c == 0 else cell.rjust(widths[c])
                         for c, cell in enumerate(cells))

    sep = "  ".join("-" * w for w in widths)
    print(f"\n  KPI: {spec.label}  ({better})")
    print("  " + _fmt_row(headers))
    print("  " + sep)
    for row in table:
        print("  " + _fmt_row(row))


def _tex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#")):
        text = text.replace(a, b)
    return text


def _p_tex(p: float, alpha: float) -> str:
    if not np.isfinite(p):
        return "--"
    txt = r"$<$0.001" if p < 0.001 else f"{p:.3f}"
    k = _stars(p, alpha)
    return txt + (r"$^{" + "*" * k + "}$" if k else "")


def _kpi_latex_table(spec: KPISpec, scenarios: List[CurriculumSection],
                     main_policy: PolicyInput,
                     comparison_policies: List[PolicyInput],
                     summary: Dict[Tuple[str, str, str], Tuple[float, float, int]],
                     tests: Dict[Tuple[str, str, str], Dict[str, Any]],
                     n_episodes: int, alpha: float, p_adjust: str) -> str:
    ncomp = len(comparison_policies)
    main_name = _tex_escape(main_policy.name)
    if spec.direction == "two-sided":
        better = "two-sided; no better direction"
        h1 = r"main $\neq$ comparison"
        test_txt = "two-sided Wilcoxon signed-rank"
    else:
        better = ("higher is better" if spec.direction == "higher"
                  else "lower is better")
        h1 = (r"main $>$ comparison" if spec.direction == "higher"
              else r"main $<$ comparison")
        test_txt = "one-sided Wilcoxon signed-rank"
    corr = ("Holm-corrected within each scenario" if p_adjust == "holm"
            else "uncorrected")

    caption = (
        f"{spec.label} ({better}). "
        f"Each cell is the mean over $N={n_episodes}$ paired episodes (common "
        f"random layouts). Comparison columns add, in parentheses, the relative "
        f"deviation from the {main_name} column as a percentage "
        f"($\\Delta\\%=100\\,(c-m)/|m|$; $m$={main_name}, $c$=comparison), and "
        f"the {test_txt} $p$-value (H$_1$: {h1}). "
        f"Significance ({corr}): $^{{*}}p<{alpha:g}$, "
        f"$^{{**}}p<0.01$, $^{{***}}p<0.001$; $\\alpha={alpha:g}$."
    )

    colspec = "l c" + " cc" * ncomp
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        "  \\caption{" + caption + "}",
        "  \\label{tab:wilcoxon-" + spec.key + "}",
        r"  \small",
        "  \\begin{tabular}{" + colspec + "}",
        r"    \toprule",
    ]

    # Header row 1: policy group headers.
    h1cells = " & " + main_name
    for base in comparison_policies:
        h1cells += r" & \multicolumn{2}{c}{" + _tex_escape(base.name) + "}"
    lines.append("    " + h1cells + r" \\")

    # cmidrules under each policy group.
    cmid = r"    \cmidrule(lr){2-2}"
    col = 3
    for _ in comparison_policies:
        cmid += r"\cmidrule(lr){%d-%d}" % (col, col + 1)
        col += 2
    lines.append(cmid)

    # Header row 2: units / p sub-headers.
    h2cells = "Scenario & {" + spec.unit + "}"
    for _ in comparison_policies:
        h2cells += " & {" + spec.unit + r" ($\Delta$\%)} & {$p$}"
    lines.append("    " + h2cells + r" \\")
    lines.append(r"    \midrule")

    # Body: one row per scenario.
    for scn in scenarios:
        m_mean, _, _ = summary[(scn.name, main_policy.name, spec.key)]
        row = _tex_escape(scn.name) + " & " + (
            spec.fmt.format(m_mean) if np.isfinite(m_mean) else "--")
        for base in comparison_policies:
            b_mean, _, _ = summary[(scn.name, base.name, spec.key)]
            res = tests[(scn.name, base.name, spec.key)]
            if np.isfinite(b_mean) and np.isfinite(m_mean):
                val = spec.fmt.format(b_mean) + " (" + _pct_delta(m_mean, b_mean, tex=True) + ")"
            else:
                val = "--"
            row += " & " + val + " & " + _p_tex(_p_used(res, p_adjust), alpha)
        lines.append("    " + row + r" \\")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _write_latex(scenarios, main_policy, comparison_policies, summary, tests,
                 n_episodes, alpha, p_adjust, out_path: Path) -> None:
    blocks = [
        "% Wilcoxon signed-rank policy comparison — generated by evaluate_policy.py",
        f"% {datetime.now().isoformat(timespec='seconds')}",
        "% Requires \\usepackage{booktabs} in the preamble.",
        "",
    ]
    for spec in KPIS:
        blocks.append(_kpi_latex_table(spec, scenarios, main_policy,
                                       comparison_policies, summary, tests,
                                       n_episodes, alpha, p_adjust))
        blocks.append("")
    out_path.write_text("\n".join(blocks), encoding="utf-8")


def _write_csv(scenarios, main_policy, comparison_policies, summary, tests,
               n_episodes, base_seed, alpha, p_adjust, out_path: Path) -> None:
    import csv
    main_name = main_policy.name
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "scenario", "kpi", "kpi_label", "direction", "policy", "role",
            "mean", "std", "n_finite", "n_episodes", "base_seed",
            "alternative", "n_pairs", "wilcoxon_stat", "p_raw", "p_adj",
            "p_adjust", "alpha", "significant", "mean_diff_vs_main",
            "median_diff_vs_main", "pct_diff_vs_main", "note",
        ])
        for scn in scenarios:
            for spec in KPIS:
                # main row (reference, no test)
                m_mean, m_std, m_n = summary[(scn.name, main_policy.name, spec.key)]
                w.writerow([scn.name, spec.key, spec.label, spec.direction,
                            main_name, "main", m_mean, m_std, m_n, n_episodes,
                            base_seed, "", "", "", "", "", p_adjust, alpha,
                            "", "", "", "", "reference policy"])
                for base in comparison_policies:
                    b_mean, b_std, b_n = summary[(scn.name, base.name, spec.key)]
                    res = tests[(scn.name, base.name, spec.key)]
                    p_show = _p_used(res, p_adjust)
                    sig = bool(np.isfinite(p_show) and p_show < alpha)
                    pct = ((b_mean - m_mean) / abs(m_mean) * 100.0
                           if (np.isfinite(b_mean) and np.isfinite(m_mean)
                               and m_mean != 0.0) else "")
                    w.writerow([
                        scn.name, spec.key, spec.label, spec.direction,
                        base.name, "comparison", b_mean, b_std, b_n, n_episodes,
                        base_seed, res["alternative"], res["n_pairs"],
                        res["statistic"], res["pvalue"], res.get("pvalue_adj", ""),
                        p_adjust, alpha, sig, res["mean_diff"], res["median_diff"],
                        pct, res["reason"] or "",
                    ])


# =====================================================================
#  CLI
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare a MAIN FOFE-MAPPO policy against COMPARISON "
        "policies on the EVAL_SCENARIOS via the Wilcoxon signed-rank test. "
        "Edit MAIN_POLICY / COMPARISON_POLICIES / EVAL_SCENARIOS at the top.",
    )
    p.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                   help="Default .pt for any policy that leaves policy_file=None.")
    p.add_argument("--n_episodes", type=int, default=N_EPISODES,
                   help=f"Paired episodes per policy per scenario (default: {N_EPISODES}).")
    p.add_argument("--chunk", type=int, default=CHUNK_EPISODES,
                   help=f"Parallel envs per rollout (default: {CHUNK_EPISODES}).")
    p.add_argument("--seed", type=int, default=BASE_SEED,
                   help=f"Base RNG seed (default: {BASE_SEED}).")
    p.add_argument("--alpha", type=float, default=ALPHA,
                   help=f"Significance level (default: {ALPHA}).")
    p.add_argument("--p_adjust", type=str, default=P_ADJUST,
                   choices=["none", "holm"],
                   help=f"Multiplicity correction across baselines (default: {P_ADJUST}).")
    p.add_argument("--device", type=str, default=None, metavar="DEVICE",
                   help="Torch device, e.g. 'cpu' or 'cuda:0'. Default: auto.")
    p.add_argument("--no_csv", action="store_true", help="Do not write the CSV.")
    p.add_argument("--no_tex", action="store_true", help="Do not write the LaTeX.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output dir (default: <script_dir>/eval_results).")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if not EVAL_SCENARIOS:
        raise RuntimeError("EVAL_SCENARIOS is empty — define at least one scenario.")
    if MAIN_POLICY is None:
        raise RuntimeError("MAIN_POLICY is not set.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be strictly between 0 and 1.")

    default_ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if default_ckpt_path is not None and not default_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {default_ckpt_path}")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _print_config(MAIN_POLICY, COMPARISON_POLICIES, EVAL_SCENARIOS,
                  args.n_episodes, args.seed, args.alpha, args.p_adjust, device)

    # Load each checkpoint once (shared cache) and print the per-policy
    # FOFE / communication check before the (slow) rollouts begin.
    cache: Dict[str, _LoadedCheckpoint] = {}
    _print_policy_diagnostics(MAIN_POLICY, COMPARISON_POLICIES, EVAL_SCENARIOS,
                              default_ckpt_path, device, cache)

    summary, tests, _samples = evaluate_comparison(
        EVAL_SCENARIOS, MAIN_POLICY, COMPARISON_POLICIES, default_ckpt_path,
        n_episodes=args.n_episodes, chunk=args.chunk, base_seed=args.seed,
        p_adjust=args.p_adjust, device=device, cache=cache,
    )

    # ── console tables (one per KPI) ──
    print("\n" + "=" * 78)
    print("  RESULTS  (rows = scenarios, columns = policies)")
    print("=" * 78)
    for spec in KPIS:
        _print_kpi_console(spec, EVAL_SCENARIOS, MAIN_POLICY, COMPARISON_POLICIES,
                           summary, tests, args.alpha, args.p_adjust)
    if not COMPARISON_POLICIES:
        print("\n  (single policy — no comparison p-values)")
    print()

    # ── files ──
    out_dir = Path(args.out_dir) if args.out_dir else (_PKG_DIR / "eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.no_tex:
        tex_path = out_dir / f"policy_wilcoxon_{stamp}.tex"
        _write_latex(EVAL_SCENARIOS, MAIN_POLICY, COMPARISON_POLICIES, summary,
                     tests, args.n_episodes, args.alpha, args.p_adjust, tex_path)
        print(f"Saved LaTeX tables to: {tex_path}")

    if not args.no_csv:
        csv_path = out_dir / f"policy_wilcoxon_{stamp}.csv"
        _write_csv(EVAL_SCENARIOS, MAIN_POLICY, COMPARISON_POLICIES, summary,
                   tests, args.n_episodes, args.seed, args.alpha, args.p_adjust,
                   csv_path)
        print(f"Saved CSV to: {csv_path}")
    print()


if __name__ == "__main__":
    main()
