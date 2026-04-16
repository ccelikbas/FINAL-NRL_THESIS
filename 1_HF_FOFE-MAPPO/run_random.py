from __future__ import annotations

"""
run_random.py
─────────────
Domain-randomised training: every iteration a new S/J/T/R configuration is
sampled uniformly, and the policy is trained on it for one iteration before
moving to the next sample.  The actor weights are transferred exactly (strict
load) between iterations; the critic is adapted shape-by-shape.

Training schedule
-----------------
  1. Warm-up   : WARMUP_ITERS iterations on WARMUP_CONFIG (easy, fixed)
  2. Random    : N_RANDOM_ITERS iterations, config sampled each iteration

After training, the frozen policy is evaluated on EVAL_CONFIGS (fixed set).

Usage
-----
    python -m FOFE-MAPPO.run_random
    python -m FOFE-MAPPO.run_random --n_random_iters 1000 --num_envs 256
    python -m FOFE-MAPPO.run_random --load_checkpoint runs/fofe_mappo.pt

Editing randomisation bounds
-----------------------------
Change the BOUNDS block near the top of this file.

Editing evaluation configurations
----------------------------------
Change the EVAL_CONFIGS list near the top of this file.
"""

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_CHECKPOINT_PKG_ALIAS = "fofe_mappo"
DEFAULT_RANDOM_START_CHECKPOINT = (
    r"C:\Users\celikbas\Documents\REPO GIT NLR\runs\saved_runs\3x3_FOFE_14_04.pt"
)

if __package__ in (None, ""):
    import types
    sys.path.insert(0, str(_THIS_DIR.parent))
    if _CHECKPOINT_PKG_ALIAS not in sys.modules:
        _pkg = types.ModuleType(_CHECKPOINT_PKG_ALIAS)
        _pkg.__path__ = [str(_THIS_DIR)]
        _pkg.__package__ = _CHECKPOINT_PKG_ALIAS
        _pkg.__file__ = str(_THIS_DIR / "__init__.py")
        sys.modules[_CHECKPOINT_PKG_ALIAS] = _pkg
    __package__ = _CHECKPOINT_PKG_ALIAS

import torch
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None

from .config import EnvConfig, ExperimentConfig, FOFEConfig, NetworkConfig, PPOConfig
from .models import make_combined_critic, make_combined_policy
from .rewards import RewardConfig
from .trainer import build_env, evaluate_current_policy, train_mappo
from . import trainer as trainer_module
from .visualization import TestRunner, animate_rollout, plot_training


# ======================================================================
#  >>>  EDIT RANDOMISATION BOUNDS HERE  <<<
#
#  Each iteration a config (n_strikers, n_jammers, n_targets, n_radars)
#  is sampled uniformly from [min, max] (inclusive).
# ======================================================================

BOUNDS = {
    "n_strikers": (1, 2),   # (min, max)
    "n_jammers":  (1, 2),
    "n_targets":  (1, 4),
    "n_radars":   (1, 4),
}

# ======================================================================
#  >>>  EDIT WARM-UP HERE  <<<
# ======================================================================

WARMUP_ITERS  = 50
WARMUP_CONFIG = (1, 1, 1, 1)   # (n_strikers, n_jammers, n_targets, n_radars)

# ======================================================================
#  >>>  EDIT EVALUATION CONFIGS HERE  <<<
#
#  Each entry is evaluated with N_EVAL_EPISODES parallel episodes.
#  Use the same key names as EnvConfig (n_known_targets / n_known_radars).
# ======================================================================

EVAL_CONFIGS: List[Dict[str, Any]] = [
    {
        "label":           "1s_1j_1t_1r",
        "n_strikers":      1, "n_jammers": 1,
        "n_known_targets": 1, "n_known_radars": 1,
    },
    {
        "label":           "1s_1j_2t_2r",
        "n_strikers":      1, "n_jammers": 1,
        "n_known_targets": 2, "n_known_radars": 2,
    },
    {
        "label":           "2s_2j_2t_2r",
        "n_strikers":      2, "n_jammers": 2,
        "n_known_targets": 2, "n_known_radars": 2,
    },
        {
        "label":           "2s_2j_1t_1r",
        "n_strikers":      2, "n_jammers": 2,
        "n_known_targets": 1, "n_known_radars": 1,
    },
    {
        "label":           "2s_2j_3t_3r",
        "n_strikers":      2, "n_jammers": 2,
        "n_known_targets": 3, "n_known_radars": 3,
    },
    {
        "label":           "3s_3j_3t_3r",
        "n_strikers":      3, "n_jammers": 3,
        "n_known_targets": 3, "n_known_radars": 3,
    },
]

N_EVAL_EPISODES = 100   # episodes per eval config (runs fully in parallel)

# ======================================================================


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _sample_config(rng: random.Random, bounds: Dict[str, Tuple[int, int]]) -> Tuple[int, int, int, int]:
    ns = rng.randint(*bounds["n_strikers"])
    nj = rng.randint(*bounds["n_jammers"])
    nt = rng.randint(*bounds["n_targets"])
    nr = rng.randint(*bounds["n_radars"])
    return ns, nj, nt, nr


def _build_env_cfg(
    ns: int, nj: int, nt: int, nr: int,
    base_cfg: EnvConfig,
    reward_cfg: RewardConfig,
) -> EnvConfig:
    """Clone base_cfg, replacing only team composition."""
    return EnvConfig(
        n_strikers=ns,
        n_jammers=nj,
        n_known_targets=nt,
        n_unknown_targets=0,
        n_known_radars=nr,
        n_unknown_radars=0,
        world_bounds=base_cfg.world_bounds,
        dt=base_cfg.dt,
        max_steps=base_cfg.max_steps,
        n_env_layouts=base_cfg.n_env_layouts,
        target_spawn_angle_range=base_cfg.target_spawn_angle_range,
        v_max=base_cfg.v_max,
        accel_magnitude=base_cfg.accel_magnitude,
        dpsi_max=base_cfg.dpsi_max,
        h_accel_magnitude_fraction=base_cfg.h_accel_magnitude_fraction,
        min_turn_radius=base_cfg.min_turn_radius,
        R_obs=base_cfg.R_obs,
        R_comm=base_cfg.R_comm,
        striker_engage_range=base_cfg.striker_engage_range,
        striker_engage_fov=base_cfg.striker_engage_fov,
        striker_v_min=base_cfg.striker_v_min,
        jammer_jam_radius=base_cfg.jammer_jam_radius,
        jammer_jam_effect=base_cfg.jammer_jam_effect,
        jammer_v_min=base_cfg.jammer_v_min,
        radar_range=base_cfg.radar_range,
        radar_kill_probability=base_cfg.radar_kill_probability,
        border_thresh=base_cfg.border_thresh,
        reward_config=copy.deepcopy(reward_cfg),
    )


def _adapt_checkpoint(
    checkpoint: Dict[str, Any],
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    net_cfg: NetworkConfig,
    fofe_cfg: FOFEConfig,
) -> Dict[str, Any]:
    """Transfer policy (strict) and critic (shape-matched) to a new team size."""
    temp_env = build_env(env_cfg, ppo_cfg)
    temp_policy = make_combined_policy(
        temp_env, hidden=net_cfg.actor_hidden, depth=net_cfg.depth,
        fofe_cfg=fofe_cfg if fofe_cfg.use_fofe else None,
    )
    temp_critic = make_combined_critic(
        temp_env, hidden=net_cfg.critic_hidden, depth=net_cfg.depth,
        fofe_cfg=fofe_cfg if fofe_cfg.use_fofe else None,
    )

    src_policy = checkpoint.get("policy_state_dict")
    src_critic = checkpoint.get("critic_state_dict")
    dst_critic = temp_critic.state_dict()

    if isinstance(src_policy, dict):
        temp_policy.load_state_dict(src_policy, strict=True)

    matched_critic = 0
    if isinstance(src_critic, dict):
        for key, tensor in src_critic.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if (key in dst_critic
                    and isinstance(dst_critic[key], torch.Tensor)
                    and tuple(dst_critic[key].shape) == tuple(tensor.shape)):
                dst_critic[key] = tensor.detach().to(
                    dst_critic[key].device, dtype=dst_critic[key].dtype
                )
                matched_critic += 1

    return {
        "policy_state_dict": temp_policy.state_dict(),
        "critic_state_dict": dst_critic,
        "reward_normalizer_state_dict": checkpoint.get("reward_normalizer_state_dict"),
    }


def _merge_logs(dst: Dict[str, List[float]], src: Dict[str, List[float]]) -> None:
    for key, values in src.items():
        dst.setdefault(key, [])
        dst[key].extend(values)


def _ensure_checkpoint_import_aliases() -> None:
    """Expose module aliases expected by checkpoints saved as fofe_mappo.*."""
    import types

    if _CHECKPOINT_PKG_ALIAS not in sys.modules:
        alias_pkg = types.ModuleType(_CHECKPOINT_PKG_ALIAS)
        alias_pkg.__path__ = [str(_THIS_DIR)]
        alias_pkg.__package__ = _CHECKPOINT_PKG_ALIAS
        alias_pkg.__file__ = str(_THIS_DIR / "__init__.py")
        sys.modules[_CHECKPOINT_PKG_ALIAS] = alias_pkg

    config_module = sys.modules.get(EnvConfig.__module__)
    rewards_module = sys.modules.get(RewardConfig.__module__)
    if config_module is not None:
        sys.modules.setdefault(f"{_CHECKPOINT_PKG_ALIAS}.config", config_module)
    if rewards_module is not None:
        sys.modules.setdefault(f"{_CHECKPOINT_PKG_ALIAS}.rewards", rewards_module)


def _run_eval(
    checkpoint: Dict[str, Any],
    eval_configs: List[Dict[str, Any]],
    base_env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
    ppo_template: PPOConfig,
    net_cfg: NetworkConfig,
    fofe_cfg: FOFEConfig,
    n_eval_episodes: int,
    title: str = "Post-training evaluation",
    print_table: bool = True,
) -> List[Dict[str, Any]]:
    """Evaluate the frozen policy on each fixed eval config, return result rows."""
    print(f"\n{'─' * 60}")
    print(f"  {title}  ({n_eval_episodes} episodes per config)")
    print(f"{'─' * 60}")

    results: List[Dict[str, Any]] = []

    for cfg_entry in eval_configs:
        label = cfg_entry["label"]
        ns    = cfg_entry["n_strikers"]
        nj    = cfg_entry["n_jammers"]
        nt    = cfg_entry.get("n_known_targets", cfg_entry.get("n_targets", 1))
        nr    = cfg_entry.get("n_known_radars",  cfg_entry.get("n_radars",  1))

        print(f"  {label} ...", end="", flush=True)

        eval_env_cfg = _build_env_cfg(ns, nj, nt, nr, base_env_cfg, reward_cfg)
        eval_ppo_cfg = PPOConfig(
            num_envs=ppo_template.num_envs,
            n_iters=1,
            num_epochs=ppo_template.num_epochs,
            minibatch_size=ppo_template.minibatch_size,
            actor_lr=ppo_template.actor_lr,
            critic_lr=ppo_template.critic_lr,
            clip_eps=ppo_template.clip_eps,
            entropy_coef=ppo_template.entropy_coef,
            normalize_rewards=ppo_template.normalize_rewards,
            seed=ppo_template.seed,
            device=ppo_template.device,
        )
        eval_exp_cfg = ExperimentConfig(
            env=eval_env_cfg, ppo=eval_ppo_cfg,
            net=net_cfg, fofe=fofe_cfg,
        ).finalize()

        adapted = _adapt_checkpoint(checkpoint, eval_exp_cfg.env, eval_exp_cfg.ppo, net_cfg, fofe_cfg)

        eval_env = build_env(eval_exp_cfg.env, eval_exp_cfg.ppo)
        eval_policy = make_combined_policy(
            eval_env, hidden=net_cfg.actor_hidden, depth=net_cfg.depth,
            fofe_cfg=fofe_cfg if fofe_cfg.use_fofe else None,
        )
        eval_policy.load_state_dict(adapted["policy_state_dict"], strict=True)
        eval_policy = eval_policy.to(eval_exp_cfg.ppo.device)

        metrics = evaluate_current_policy(
            eval_policy, eval_exp_cfg.env, eval_exp_cfg.ppo,
            n_eval_episodes=n_eval_episodes,
        )

        results.append({
            "label":           label,
            "n_strikers":      ns,
            "n_jammers":       nj,
            "n_targets":       eval_env_cfg.n_targets,
            "n_radars":        eval_env_cfg.n_radars,
            "completion_rate": metrics["eval_task_completion_rate"],
            "survival_rate":   metrics["eval_survival_rate"],
            "mean_duration":   metrics["eval_mean_duration"],
            "mean_reward":     metrics["eval_mean_episode_total_reward"],
        })

        print(
            f"  completion={results[-1]['completion_rate']:.3f}  "
            f"survival={results[-1]['survival_rate']:.3f}  "
            f"time={results[-1]['mean_duration']:.1f}"
        )

    if print_table:
        _print_eval_table(results, n_eval_episodes)
    return results


def _mean_metric(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    vals = [float(r.get(key, float("nan"))) for r in rows]
    finite = [v for v in vals if v == v]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _plot_eval_history_per_config(
    history: Dict[str, Dict[str, List[float]]],
    n_eval_episodes: int,
    log_every: int,
) -> None:
    """Plot completion and survival lines over training iteration, per config label."""
    if plt is None:
        print("Per-config eval plot skipped: matplotlib is not available.")
        return
    if not history:
        print("Per-config eval plot skipped: no eval history collected.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    ax_c = axes[0]
    for label, series in history.items():
        if series["iters"]:
            ax_c.plot(series["iters"], series["completion"], marker="o", label=label)
    ax_c.set_title(f"Per-Config Completion Rate (eval every {log_every} iters, n={n_eval_episodes})")
    ax_c.set_ylabel("Completion rate")
    ax_c.set_ylim(0.0, 1.0)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=8, ncol=2)

    ax_s = axes[1]
    for label, series in history.items():
        if series["iters"]:
            ax_s.plot(series["iters"], series["survival"], marker="o", label=label)
    ax_s.set_title(f"Per-Config Survival Rate (eval every {log_every} iters, n={n_eval_episodes})")
    ax_s.set_xlabel("Training iteration")
    ax_s.set_ylabel("Survival rate")
    ax_s.set_ylim(0.0, 1.0)
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    plt.show()


def _print_eval_table(rows: List[Dict[str, Any]], n_eval_episodes: int) -> None:
    W_LBL  = max(len("Config"), max(len(r["label"]) for r in rows))
    max_cnt = max(max(r["n_strikers"], r["n_jammers"], r["n_targets"], r["n_radars"]) for r in rows)
    W_CNT  = max(1, len(str(max_cnt)))
    W_C    = max(len("Completion"), 6)
    W_S    = max(len("Survival"), 6)
    W_T    = max(len("Mean Time"), max(len(f"{r['mean_duration']:.1f}") for r in rows))
    W_R    = max(len("Mean Reward"), max(len(f"{r['mean_reward']:.4f}") for r in rows))

    sep = "  ".join([
        "-" * W_LBL,
        "  ".join(["-" * W_CNT] * 4),
        "-" * W_C, "-" * W_S, "-" * W_T, "-" * W_R,
    ])
    title = f"  Domain-Randomised Policy — Evaluation Results  (n={n_eval_episodes} per config)"
    bar   = "=" * max(len(sep), len(title) + 2)

    print(f"\n{bar}")
    print(title)
    print(bar)
    hdr = (
        f"{'Config':<{W_LBL}}  "
        f"{'S':>{W_CNT}}  {'J':>{W_CNT}}  {'T':>{W_CNT}}  {'R':>{W_CNT}}  "
        f"{'Completion':>{W_C}}  {'Survival':>{W_S}}  "
        f"{'Mean Time':>{W_T}}  {'Mean Reward':>{W_R}}"
    )
    print(hdr)
    print(sep)
    for r in rows:
        print(
            f"{r['label']:<{W_LBL}}  "
            f"{r['n_strikers']:>{W_CNT}}  {r['n_jammers']:>{W_CNT}}  "
            f"{r['n_targets']:>{W_CNT}}  {r['n_radars']:>{W_CNT}}  "
            f"{r['completion_rate']:>{W_C}.4f}  {r['survival_rate']:>{W_S}.4f}  "
            f"{r['mean_duration']:>{W_T}.1f}  {r['mean_reward']:>{W_R}.4f}"
        )
    print(sep)
    print()


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    env_d   = EnvConfig()
    ppo_d   = PPOConfig()
    net_d   = NetworkConfig()
    fofe_d  = FOFEConfig()
    rew_d   = RewardConfig()

    p = argparse.ArgumentParser(
        description=(
            "Domain-randomised MAPPO training: team composition is re-sampled "
            "every iteration within configurable bounds."
        )
    )

    # ── Randomisation schedule ────────────────────────────────────────
    p.add_argument("--use_warmup", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable warm-up phase before random training (default: disabled)")
    p.add_argument("--warmup_iters",    type=int, default=WARMUP_ITERS,
                   help=f"Fixed warm-up iterations on {WARMUP_CONFIG} (default: {WARMUP_ITERS})")
    p.add_argument("--n_random_iters",  type=int, default=100,
                   help="Number of random-config iterations after warm-up (default: 100)")

    # ── Randomisation bounds ──────────────────────────────────────────
    p.add_argument("--min_strikers", type=int, default=BOUNDS["n_strikers"][0])
    p.add_argument("--max_strikers", type=int, default=BOUNDS["n_strikers"][1])
    p.add_argument("--min_jammers",  type=int, default=BOUNDS["n_jammers"][0])
    p.add_argument("--max_jammers",  type=int, default=BOUNDS["n_jammers"][1])
    p.add_argument("--min_targets",  type=int, default=BOUNDS["n_targets"][0])
    p.add_argument("--max_targets",  type=int, default=BOUNDS["n_targets"][1])
    p.add_argument("--min_radars",   type=int, default=BOUNDS["n_radars"][0])
    p.add_argument("--max_radars",   type=int, default=BOUNDS["n_radars"][1])

    # ── PPO / training ────────────────────────────────────────────────
    p.add_argument("--num_envs",       type=int,   default=ppo_d.num_envs)
    p.add_argument("--max_steps",      type=int,   default=env_d.max_steps)
    p.add_argument("--num_epochs",     type=int,   default=ppo_d.num_epochs)
    p.add_argument("--minibatch_size", type=int,   default=ppo_d.minibatch_size)
    p.add_argument("--actor_lr",       type=float, default=ppo_d.actor_lr)
    p.add_argument("--critic_lr",      type=float, default=ppo_d.critic_lr)
    p.add_argument("--clip_eps",       type=float, default=ppo_d.clip_eps)
    p.add_argument("--entropy_coef",   type=float, default=ppo_d.entropy_coef)
    p.add_argument("--normalize_rewards", action=argparse.BooleanOptionalAction,
                   default=ppo_d.normalize_rewards)
    p.add_argument("--seed",           type=int,   default=ppo_d.seed)
    p.add_argument("--log_every",      type=int,   default=ppo_d.log_every)

    # ── Network ───────────────────────────────────────────────────────
    p.add_argument("--actor_hidden",   type=int, default=net_d.actor_hidden)
    p.add_argument("--critic_hidden",  type=int, default=net_d.critic_hidden)
    p.add_argument("--depth",          type=int, default=net_d.depth)

    # ── FOFE ──────────────────────────────────────────────────────────
    p.add_argument("--use_fofe", action=argparse.BooleanOptionalAction,
                   default=fofe_d.use_fofe)

    # ── Reward weights ────────────────────────────────────────────────
    p.add_argument("--target_destroyed",       type=float, default=rew_d.target_destroyed)
    p.add_argument("--agent_destroyed",        type=float, default=rew_d.agent_destroyed)
    p.add_argument("--timestep_penalty",       type=float, default=rew_d.timestep_penalty)
    p.add_argument("--team_spirit",            type=float, default=rew_d.team_spirit)
    p.add_argument("--striker_progress_scale", type=float, default=rew_d.striker_progress_scale)
    p.add_argument("--jammer_progress_scale",  type=float, default=rew_d.jammer_progress_scale)
    p.add_argument("--jammer_jam_bonus",       type=float, default=rew_d.jammer_jam_bonus)

    # ── Eval ──────────────────────────────────────────────────────────
    p.add_argument("--n_eval_episodes", type=int, default=N_EVAL_EPISODES)

    # ── Save / load ───────────────────────────────────────────────────
    p.add_argument("--save_dir",        type=str, default="runs")
    p.add_argument("--save_name",       type=str, default="fofe_mappo_random.pt")
    p.add_argument("--load_checkpoint", type=str, default=DEFAULT_RANDOM_START_CHECKPOINT,
                   help="Start random training from this checkpoint (default: saved 3x3 FOFE policy)")
    p.add_argument("--no_plot",    action="store_true")
    p.add_argument("--no_animate", action="store_true")

    return p


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()

    # run_random-specific speedup: skip expensive env spec checks each outer iteration
    trainer_module._safe_check = lambda _env: None

    # ── Validate bounds ───────────────────────────────────────────────
    for lo, hi, name in [
        (args.min_strikers, args.max_strikers, "strikers"),
        (args.min_jammers,  args.max_jammers,  "jammers"),
        (args.min_targets,  args.max_targets,  "targets"),
        (args.min_radars,   args.max_radars,   "radars"),
    ]:
        if lo < 1:
            raise ValueError(f"min_{name} must be >= 1")
        if hi < lo:
            raise ValueError(f"max_{name} must be >= min_{name}")

    bounds = {
        "n_strikers": (args.min_strikers, args.max_strikers),
        "n_jammers":  (args.min_jammers,  args.max_jammers),
        "n_targets":  (args.min_targets,  args.max_targets),
        "n_radars":   (args.min_radars,   args.max_radars),
    }

    reward_cfg = RewardConfig(
        target_destroyed=args.target_destroyed,
        agent_destroyed=args.agent_destroyed,
        timestep_penalty=args.timestep_penalty,
        team_spirit=args.team_spirit,
        striker_progress_scale=args.striker_progress_scale,
        jammer_progress_scale=args.jammer_progress_scale,
        jammer_jam_bonus=args.jammer_jam_bonus,
    )
    fofe_cfg = FOFEConfig(use_fofe=args.use_fofe)
    net_cfg  = NetworkConfig(
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        depth=args.depth,
    )

    # Base env config: holds all physics/sensor/reward params.
    # Team composition fields are overridden per iteration.
    base_env_cfg = EnvConfig(
        n_strikers=1, n_jammers=1,
        n_known_targets=1, n_known_radars=1,
        max_steps=args.max_steps,
        reward_config=reward_cfg,
    )

    ppo_template = PPOConfig(
        num_envs=args.num_envs,
        n_iters=1,               # always 1 — we loop externally
        num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        normalize_rewards=args.normalize_rewards,
        seed=args.seed,
        log_every=args.log_every,
    )

    effective_warmup_iters = args.warmup_iters if args.use_warmup else 0
    rng = random.Random(args.seed)
    total_iters = effective_warmup_iters + args.n_random_iters

    # ── Print plan ────────────────────────────────────────────────────
    print("=" * 70)
    print("  Domain-Randomised MAPPO Training")
    print("=" * 70)
    warmup_desc = (
        f"{effective_warmup_iters} iters on {WARMUP_CONFIG}"
        if args.use_warmup else "disabled"
    )
    print(f"  Warm-up  : {warmup_desc}")
    print(f"  Random   : {args.n_random_iters} iters, config sampled each iteration")
    print(f"  Bounds   : S{bounds['n_strikers']}  J{bounds['n_jammers']}  "
          f"T{bounds['n_targets']}  R{bounds['n_radars']}")
    print(f"  FOFE     : {'enabled' if fofe_cfg.use_fofe else 'disabled (legacy)'}")
    print(f"  Total    : {total_iters} iterations")
    print(f"  Eval     : {len(EVAL_CONFIGS)} configs × {args.n_eval_episodes} episodes")
    print("=" * 70)

    # ── Load initial checkpoint (optional) ───────────────────────────
    incoming_checkpoint: Optional[Dict[str, Any]] = None
    if args.load_checkpoint:
        ckpt_path = Path(args.load_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"load_checkpoint not found: {ckpt_path}")
        _ensure_checkpoint_import_aliases()
        incoming_checkpoint = torch.load(ckpt_path, map_location=ppo_template.device, weights_only=False)
        print(f"  Loaded initial checkpoint from: {ckpt_path}")

    all_logs: Dict[str, List[float]] = {}
    sampled_configs: List[Tuple[int, int, int, int]] = []
    eval_history_per_config: Dict[str, Dict[str, List[float]]] = {
        cfg["label"]: {"iters": [], "completion": [], "survival": []}
        for cfg in EVAL_CONFIGS
    }
    last_env_cfg  = None
    last_policy   = None
    last_critic   = None
    last_rn       = None

    # ── Training loop ─────────────────────────────────────────────────
    for global_it in range(total_iters):
        is_warmup = global_it < effective_warmup_iters

        if is_warmup:
            ns, nj, nt, nr = WARMUP_CONFIG
            phase_label = f"warmup {global_it + 1}/{effective_warmup_iters}"
        else:
            ns, nj, nt, nr = _sample_config(rng, bounds)
            rand_it = global_it - effective_warmup_iters + 1
            phase_label = f"random {rand_it}/{args.n_random_iters}"

        sampled_configs.append((ns, nj, nt, nr))

        iter_env_cfg = _build_env_cfg(ns, nj, nt, nr, base_env_cfg, reward_cfg)
        iter_ppo_cfg = PPOConfig(
            num_envs=ppo_template.num_envs,
            n_iters=1,
            num_epochs=ppo_template.num_epochs,
            minibatch_size=ppo_template.minibatch_size,
            actor_lr=ppo_template.actor_lr,
            critic_lr=ppo_template.critic_lr,
            clip_eps=ppo_template.clip_eps,
            entropy_coef=ppo_template.entropy_coef,
            normalize_rewards=ppo_template.normalize_rewards,
            seed=ppo_template.seed + global_it,
            log_every=0,
        )
        iter_ppo_cfg.iteration_offset = global_it
        iter_ppo_cfg.profile_iters = 0
        iter_exp_cfg = ExperimentConfig(
            env=iter_env_cfg, ppo=iter_ppo_cfg,
            net=net_cfg, fofe=fofe_cfg,
        ).finalize()

        adapted = None
        if incoming_checkpoint is not None:
            adapted = _adapt_checkpoint(
                incoming_checkpoint,
                iter_exp_cfg.env, iter_exp_cfg.ppo,
                net_cfg, fofe_cfg,
            )

        do_eval_this_iter = args.log_every > 0 and ((global_it + 1) % args.log_every == 0)

        # Keep iteration headers aligned with the evaluation cadence only.
        if do_eval_this_iter:
            print(
                f"\n[{global_it + 1:4d}/{total_iters}] {phase_label} | "
                f"S={ns} J={nj} T={nt} R={nr}"
            )

        _, policy, critic, iter_logs, reward_normalizer = train_mappo(
            iter_exp_cfg.env,
            iter_exp_cfg.ppo,
            iter_exp_cfg.net,
            fofe_cfg=fofe_cfg,
            checkpoint=adapted,
        )

        _merge_logs(all_logs, iter_logs)

        incoming_checkpoint = {
            "policy_state_dict": policy.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "reward_normalizer_state_dict": (
                reward_normalizer.state_dict() if reward_normalizer is not None else None
            ),
        }

        # Multi-config periodic evaluation during training
        if do_eval_this_iter and incoming_checkpoint is not None:
            eval_rows = _run_eval(
                checkpoint=incoming_checkpoint,
                eval_configs=EVAL_CONFIGS,
                base_env_cfg=base_env_cfg,
                reward_cfg=reward_cfg,
                ppo_template=ppo_template,
                net_cfg=net_cfg,
                fofe_cfg=fofe_cfg,
                n_eval_episodes=args.n_eval_episodes,
                title=f"Training evaluation @ iter {global_it + 1}",
                print_table=False,
            )

            for row in eval_rows:
                lbl = row["label"]
                if lbl not in eval_history_per_config:
                    eval_history_per_config[lbl] = {"iters": [], "completion": [], "survival": []}
                eval_history_per_config[lbl]["iters"].append(global_it + 1)
                eval_history_per_config[lbl]["completion"].append(float(row["completion_rate"]))
                eval_history_per_config[lbl]["survival"].append(float(row["survival_rate"]))

            # Keep aggregate eval logs as mean across prescribed eval configs
            all_logs["eval_task_completion_rate"][-1] = _mean_metric(eval_rows, "completion_rate")
            all_logs["eval_survival_rate"][-1] = _mean_metric(eval_rows, "survival_rate")
            all_logs["eval_mean_duration"][-1] = _mean_metric(eval_rows, "mean_duration")
            all_logs["eval_mean_episode_total_reward"][-1] = _mean_metric(eval_rows, "mean_reward")

        last_env_cfg = iter_exp_cfg.env
        last_policy  = policy
        last_critic  = critic
        last_rn      = reward_normalizer

    # ── Post-training evaluation ──────────────────────────────────────
    eval_results: List[Dict[str, Any]] = []
    if last_policy is not None and incoming_checkpoint is not None:
        eval_results = _run_eval(
            checkpoint=incoming_checkpoint,
            eval_configs=EVAL_CONFIGS,
            base_env_cfg=base_env_cfg,
            reward_cfg=reward_cfg,
            ppo_template=ppo_template,
            net_cfg=net_cfg,
            fofe_cfg=fofe_cfg,
            n_eval_episodes=args.n_eval_episodes,
        )

    # ── Save checkpoint ───────────────────────────────────────────────
    save_dir  = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / args.save_name
    torch.save(
        {
            "policy_state_dict": last_policy.state_dict() if last_policy else None,
            "critic_state_dict": last_critic.state_dict() if last_critic else None,
            "env_cfg":           last_env_cfg,
            "ppo_cfg":           ppo_template,
            "net_cfg":           net_cfg,
            "fofe_cfg":          fofe_cfg,
            "logs":              all_logs,
            "reward_normalizer_state_dict": (
                last_rn.state_dict() if last_rn is not None else None
            ),
            "sampled_configs":  sampled_configs,
            "eval_results":     eval_results,
            "eval_history_per_config": eval_history_per_config,
            "bounds":           bounds,
        },
        save_path,
    )
    print(f"\nSaved checkpoint to: {save_path}")

    # ── Training curve plot ───────────────────────────────────────────
    if not args.no_plot and all_logs:
        try:
            plot_training(all_logs)
        except Exception as exc:
            print(f"plot_training warning (continuing): {type(exc).__name__}: {exc}")
        try:
            _plot_eval_history_per_config(
                eval_history_per_config,
                n_eval_episodes=args.n_eval_episodes,
                log_every=args.log_every,
            )
        except Exception as exc:
            print(f"per-config eval plot warning (continuing): {type(exc).__name__}: {exc}")

    # ── Rollout animations ────────────────────────────────────────────
    if not args.no_animate and last_policy is not None and last_env_cfg is not None:
        try:
            from .visualization import animate_rollout
            for r in range(5):
                tester = TestRunner(
                    last_policy, env_cfg=last_env_cfg,
                    device=ppo_template.device, seed=999 + r,
                )
                frames = tester.rollout()
                animate_rollout(frames, tester.env)
                print(f"Visualised rollout {r + 1}/5 with {len(frames)} frames")
        except Exception as exc:
            print(f"animate_rollout warning (continuing): {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
