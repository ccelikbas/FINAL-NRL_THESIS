"""
Run comparison: MAPPO (Legacy) vs FOFE-MAPPO.

Trains both methods sequentially with identical environment / PPO
configuration, then produces:
  1. A comparison dashboard (2x3 grid) with both methods overlaid.
  2. Side-by-side rollout animations using the two trained policies.

Usage:
    python run_comparison.py --n_strikers 2 --n_jammers 2 --n_iters 200
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

if __package__ in (None, ""):
    import types
    _this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_this_dir.parent))
    _pkg_name = "fofe_mappo"
    _pkg = types.ModuleType(_pkg_name)
    _pkg.__path__ = [str(_this_dir)]
    _pkg.__package__ = _pkg_name
    _pkg.__file__ = str(_this_dir / "__init__.py")
    sys.modules[_pkg_name] = _pkg
    __package__ = _pkg_name

from .config import (
    EnvConfig,
    EnvExtensionsConfig,
    ExperimentConfig,
    FOFEConfig,
    NetworkConfig,
    PPOConfig,
)
from .trainer import train_mappo
from .rewards import RewardConfig
from .visualization import (
    TestRunner,
    animate_comparison_rollout,
    plot_comparison,
)
from .HF_visualization import HFTestRunner

import torch


def build_parser() -> argparse.ArgumentParser:
    env_defaults = EnvConfig()
    ext_defaults = EnvExtensionsConfig()
    ppo_defaults = PPOConfig()
    net_defaults = NetworkConfig()
    reward_defaults = RewardConfig()

    p = argparse.ArgumentParser(
        description="Compare MAPPO (Legacy) vs FOFE-MAPPO on the same environment",
    )
    # Counts
    p.add_argument("--n_strikers", type=int, default=env_defaults.n_strikers)
    p.add_argument("--n_jammers", type=int, default=env_defaults.n_jammers)
    p.add_argument("--n_targets", type=int, default=env_defaults.n_targets)
    p.add_argument("--n_radars", type=int, default=env_defaults.n_radars)
    p.add_argument("--n_known_targets", type=int, default=env_defaults.n_known_targets)
    p.add_argument("--n_unknown_targets", type=int, default=env_defaults.n_unknown_targets)
    p.add_argument("--n_known_radars", type=int, default=env_defaults.n_known_radars)
    p.add_argument("--n_unknown_radars", type=int, default=env_defaults.n_unknown_radars)
    # PPO
    p.add_argument("--num_envs", type=int, default=ppo_defaults.num_envs)
    p.add_argument("--max_steps", type=int, default=env_defaults.max_steps)
    p.add_argument("--n_iters", type=int, default=ppo_defaults.n_iters)
    p.add_argument("--num_epochs", type=int, default=ppo_defaults.num_epochs)
    p.add_argument("--minibatch_size", type=int, default=ppo_defaults.minibatch_size)
    p.add_argument("--actor_lr", type=float, default=ppo_defaults.actor_lr)
    p.add_argument("--critic_lr", type=float, default=ppo_defaults.critic_lr)
    p.add_argument("--clip_eps", type=float, default=ppo_defaults.clip_eps)
    p.add_argument("--entropy_coef", type=float, default=ppo_defaults.entropy_coef)
    p.add_argument("--normalize_rewards", action=argparse.BooleanOptionalAction,
                   default=ppo_defaults.normalize_rewards)
    p.add_argument("--seed", type=int, default=ppo_defaults.seed)
    # Environment extension toggles
    p.add_argument(
        "--use_hf_radar",
        action=argparse.BooleanOptionalAction,
        default=ext_defaults.use_hf_radar,
        help="Enable high-fidelity angular radar model for both comparison runs",
    )
    # Network
    p.add_argument("--actor_hidden", type=int, default=net_defaults.actor_hidden)
    p.add_argument("--critic_hidden", type=int, default=net_defaults.critic_hidden)
    p.add_argument("--depth", type=int, default=net_defaults.depth)
    # Reward weights
    p.add_argument("--target_destroyed", type=float, default=reward_defaults.target_destroyed)
    p.add_argument("--agent_destroyed", type=float, default=reward_defaults.agent_destroyed)
    p.add_argument("--timestep_penalty", type=float, default=reward_defaults.timestep_penalty)
    p.add_argument("--team_spirit", type=float, default=reward_defaults.team_spirit)
    p.add_argument("--striker_progress_scale", type=float, default=reward_defaults.striker_progress_scale)
    p.add_argument("--jammer_progress_scale", type=float, default=reward_defaults.jammer_progress_scale)
    p.add_argument("--jammer_jam_bonus", type=float, default=reward_defaults.jammer_jam_bonus)
    # Save / display
    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--logs_dir", type=str, default="logs")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--no_animate", action="store_true")
    p.add_argument("--n_rollouts", type=int, default=5,
                   help="Number of side-by-side rollout animations to show (default: 5)")
    return p


def _build_configs(args):
    """Build shared config objects from parsed CLI arguments."""
    reward_cfg = RewardConfig(
        target_destroyed=args.target_destroyed,
        agent_destroyed=args.agent_destroyed,
        timestep_penalty=args.timestep_penalty,
        team_spirit=args.team_spirit,
        striker_progress_scale=args.striker_progress_scale,
        jammer_progress_scale=args.jammer_progress_scale,
        jammer_jam_bonus=args.jammer_jam_bonus,
    )
    env_cfg = EnvConfig(
        n_strikers=args.n_strikers,
        n_jammers=args.n_jammers,
        n_known_targets=args.n_known_targets,
        n_unknown_targets=args.n_unknown_targets,
        n_known_radars=args.n_known_radars,
        n_unknown_radars=args.n_unknown_radars,
        max_steps=args.max_steps,
        reward_config=reward_cfg,
    )
    ppo_cfg = PPOConfig(
        num_envs=args.num_envs,
        n_iters=args.n_iters,
        num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        normalize_rewards=args.normalize_rewards,
        seed=args.seed,
    )
    net_cfg = NetworkConfig(
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        depth=args.depth,
    )
    ext_cfg = EnvExtensionsConfig(use_hf_radar=args.use_hf_radar)
    return env_cfg, ppo_cfg, net_cfg, ext_cfg


def _save_checkpoint(path, policy, critic, cfg, logs, reward_normalizer):
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "env_cfg": cfg.env,
            "ppo_cfg": cfg.ppo,
            "net_cfg": cfg.net,
            "fofe_cfg": cfg.fofe,
            "ext_cfg": cfg.ext,
            "logs": logs,
            "reward_normalizer_state_dict": (
                reward_normalizer.state_dict() if reward_normalizer is not None else None
            ),
        },
        path,
    )
    print(f"  Saved checkpoint to: {path}")


def _save_policy(path, policy) -> None:
    torch.save({"policy_state_dict": policy.state_dict()}, path)
    print(f"  Saved policy to: {path}")


def _to_jsonable(obj):
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj


def _save_run_logs(path: Path, run_name: str, cfg: ExperimentConfig, logs: dict) -> None:
    payload = {
        "run_name": run_name,
        "env_cfg": _to_jsonable(vars(cfg.env)),
        "ppo_cfg": _to_jsonable(vars(cfg.ppo)),
        "net_cfg": _to_jsonable(vars(cfg.net)),
        "fofe_cfg": _to_jsonable(vars(cfg.fofe)),
        "ext_cfg": _to_jsonable(vars(cfg.ext)),
        "logs": _to_jsonable(logs),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved logs to: {path}")


def _print_final_metrics(label, logs):
    if not logs["eval_mean_episode_total_reward"]:
        return
    print(f"\n  [{label}] Final metrics:")
    print(f"    train_mean_episode_total_reward = {logs['train_mean_episode_total_reward'][-1]:.4f}")
    print(f"    eval_mean_episode_total_reward  = {logs['eval_mean_episode_total_reward'][-1]:.4f}")
    print(f"    eval_task_completion_rate       = {logs['eval_task_completion_rate'][-1]:.4f}")
    print(f"    eval_survival_rate              = {logs['eval_survival_rate'][-1]:.4f}")
    print(f"    eval_mean_duration              = {logs['eval_mean_duration'][-1]:.4f}")


def main() -> None:
    args = build_parser().parse_args()
    env_cfg, ppo_cfg, net_cfg, ext_cfg = _build_configs(args)
    save_dir = Path(args.save_dir)
    logs_dir = Path(args.logs_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    hf_radar_cfg = ext_cfg.hf_radar if ext_cfg.use_hf_radar else None
    if ext_cfg.use_hf_radar:
        print("HF angular radar model: ENABLED (comparison runs use HF environment)")
    else:
        print("HF angular radar model: DISABLED (comparison runs use standard environment)")

    # ==================================================================
    # Phase 1: Train MAPPO (Legacy) — use_fofe=False
    # ==================================================================
    print("=" * 70)
    print("  Phase 1/2: Training MAPPO (Legacy)  [use_fofe=False]")
    print("=" * 70)

    legacy_fofe_cfg = FOFEConfig(use_fofe=False)
    legacy_cfg = ExperimentConfig(
        env=copy.deepcopy(env_cfg),
        ppo=copy.deepcopy(ppo_cfg),
        net=copy.deepcopy(net_cfg),
        fofe=legacy_fofe_cfg,
        ext=copy.deepcopy(ext_cfg),
    ).finalize()

    print(f"  FOFE config: use_fofe={legacy_cfg.fofe.use_fofe}")
    print(f"  Env config:  use_fofe={legacy_cfg.env.use_fofe}")

    legacy_env, legacy_policy, legacy_critic, legacy_logs, legacy_rn = train_mappo(
        legacy_cfg.env, legacy_cfg.ppo, legacy_cfg.net,
        fofe_cfg=legacy_cfg.fofe,
        hf_radar_cfg=hf_radar_cfg,
    )

    _save_checkpoint(save_dir / "comparison_legacy.pt",
                     legacy_policy, legacy_critic, legacy_cfg,
                     legacy_logs, legacy_rn)
    _save_policy(save_dir / "comparison_legacy_policy.pt", legacy_policy)
    _save_run_logs(logs_dir / "comparison_legacy_logs.json", "mappo_legacy", legacy_cfg, legacy_logs)
    _print_final_metrics("MAPPO (Legacy)", legacy_logs)

    # ── Free Phase 1 GPU resources before Phase 2 ────────────────────
    # Move legacy artefacts to CPU so GPU VRAM is available for the
    # (larger) FOFE networks.  We keep the objects alive for the
    # side-by-side rollout animations at the end.
    legacy_policy.cpu()
    legacy_critic.cpu()
    del legacy_env, legacy_critic, legacy_rn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  Freed Phase 1 GPU resources.\n")

    # ==================================================================
    # Phase 2: Train FOFE-MAPPO — use_fofe=True
    # ==================================================================
    print("=" * 70)
    print("  Phase 2/2: Training FOFE-MAPPO  [use_fofe=True]")
    print("=" * 70)

    fofe_fofe_cfg = FOFEConfig(use_fofe=True)
    fofe_cfg = ExperimentConfig(
        env=copy.deepcopy(env_cfg),
        ppo=copy.deepcopy(ppo_cfg),
        net=copy.deepcopy(net_cfg),
        fofe=fofe_fofe_cfg,
        ext=copy.deepcopy(ext_cfg),
    ).finalize()

    print(f"  FOFE config: use_fofe={fofe_cfg.fofe.use_fofe}")
    print(f"  Env config:  use_fofe={fofe_cfg.env.use_fofe}")

    fofe_env, fofe_policy, fofe_critic, fofe_logs, fofe_rn = train_mappo(
        fofe_cfg.env, fofe_cfg.ppo, fofe_cfg.net,
        fofe_cfg=fofe_cfg.fofe,
        hf_radar_cfg=hf_radar_cfg,
    )

    _save_checkpoint(save_dir / "comparison_fofe.pt",
                     fofe_policy, fofe_critic, fofe_cfg,
                     fofe_logs, fofe_rn)
    _save_policy(save_dir / "comparison_fofe_policy.pt", fofe_policy)
    _save_run_logs(logs_dir / "comparison_fofe_logs.json", "fofe_mappo", fofe_cfg, fofe_logs)
    _print_final_metrics("FOFE-MAPPO", fofe_logs)

    # ==================================================================
    # Phase 3: Comparison Dashboard
    # ==================================================================
    print("\n" + "=" * 70)
    print("  Generating comparison dashboard")
    print("=" * 70)

    if not args.no_plot:
        try:
            plot_comparison(legacy_logs, fofe_logs)
        except Exception as exc:
            print(f"plot_comparison warning (continuing): {type(exc).__name__}: {exc}")

    # ==================================================================
    # Phase 4: Side-by-side rollout animations
    # ==================================================================
    if not args.no_animate:
        print("\n" + "=" * 70)
        print(f"  Generating {args.n_rollouts} side-by-side rollout animation(s)")
        print("=" * 70)
        # Move legacy policy back to GPU for rollout (was moved to CPU to free VRAM)
        rollout_device = fofe_cfg.ppo.device
        legacy_policy.to(rollout_device)
        try:
            for r in range(args.n_rollouts):
                seed = 999 + r
                if ext_cfg.use_hf_radar:
                    legacy_tester = HFTestRunner(
                        legacy_policy,
                        env_cfg=legacy_cfg.env,
                        hf_cfg=legacy_cfg.ext.hf_radar,
                        device=rollout_device,
                        seed=seed,
                    )
                    fofe_tester = HFTestRunner(
                        fofe_policy,
                        env_cfg=fofe_cfg.env,
                        hf_cfg=fofe_cfg.ext.hf_radar,
                        device=rollout_device,
                        seed=seed,
                    )
                else:
                    legacy_tester = TestRunner(
                        legacy_policy,
                        env_cfg=legacy_cfg.env,
                        device=rollout_device,
                        seed=seed,
                    )
                    fofe_tester = TestRunner(
                        fofe_policy,
                        env_cfg=fofe_cfg.env,
                        device=rollout_device,
                        seed=seed,
                    )
                legacy_frames = legacy_tester.rollout()
                fofe_frames = fofe_tester.rollout()
                animate_comparison_rollout(
                    legacy_frames, fofe_frames,
                    legacy_tester.env, fofe_tester.env,
                )
                print(f"  Rollout {r+1}/{args.n_rollouts}: "
                      f"legacy={len(legacy_frames)} frames, "
                      f"fofe={len(fofe_frames)} frames")
        except Exception as exc:
            print(f"animate_comparison_rollout warning (continuing): "
                  f"{type(exc).__name__}: {exc}")

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
