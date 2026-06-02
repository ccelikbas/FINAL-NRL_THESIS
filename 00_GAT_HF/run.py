""""
To run from prev checkpoint: 
python run.py --load_checkpoint ../runs/curriculum_mappo.pt
"""


from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force Inductor to compile Triton kernels in the main process instead of a
# subprocess pool. Works around a Triton bug where worker subprocesses return
# an asm dict missing the 'cubin' key on some Linux + CUDA toolchains, which
# raises `KeyError: 'cubin'` during the first PPO update.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

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

from .config import EnvConfig, EnvExtensionsConfig, ExperimentConfig, FOFEConfig, GATConfig, NetworkConfig, PPOConfig
from .trainer import train_mappo
from .rewards import RewardConfig
from .visualization import TestRunner, animate_rollout, plot_training
from .HF_visualization import HFTestRunner, hf_animate_rollout

import torch


def build_parser() -> argparse.ArgumentParser:
    env_defaults = EnvConfig()
    ext_defaults = EnvExtensionsConfig()
    ppo_defaults = PPOConfig()
    net_defaults = NetworkConfig()
    fofe_defaults = FOFEConfig()
    reward_defaults = RewardConfig()

    p = argparse.ArgumentParser(description="FOFE-MAPPO for Strike-EA")
    # Counts
    p.add_argument("--n_strikers", type=int, default=env_defaults.n_strikers)
    p.add_argument("--n_jammers", type=int, default=env_defaults.n_jammers)
    p.add_argument("--n_targets", type=int, default=env_defaults.n_targets)
    p.add_argument("--n_radars", type=int, default=env_defaults.n_radars)
    p.add_argument("--n_known_targets", type=int, default=env_defaults.n_known_targets)
    p.add_argument("--n_unknown_targets", type=int, default=env_defaults.n_unknown_targets)
    p.add_argument("--n_known_radars", type=int, default=env_defaults.n_known_radars)
    p.add_argument("--n_unknown_radars", type=int, default=env_defaults.n_unknown_radars)
    p.add_argument("--radar_min_sep", type=float, default=env_defaults.radar_min_sep)
    p.add_argument(
        "--scenario", type=str, choices=("S1", "S2"), default=env_defaults.scenario,
        help="S1=protected targets (radars guard targets); S2=defensive line (radars between agents and targets)",
    )
    p.add_argument("--s2_radar_min_sep", type=float, default=env_defaults.s2_radar_min_sep)
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
    p.add_argument("--normalize_rewards", action=argparse.BooleanOptionalAction, default=ppo_defaults.normalize_rewards)
    p.add_argument("--seed", type=int, default=ppo_defaults.seed)
    # Network
    p.add_argument("--actor_hidden", type=int, default=net_defaults.actor_hidden)
    p.add_argument("--critic_hidden", type=int, default=net_defaults.critic_hidden)
    p.add_argument("--depth", type=int, default=net_defaults.depth)
    # Observation-encoder selector — single source of truth. When omitted,
    # falls back to the legacy --use_fofe flag for back-compat (True→'fofe',
    # False→'flat'); when set explicitly, --encoder_type wins.
    p.add_argument(
        "--encoder_type",
        type=str,
        choices=("flat", "fofe", "gat"),
        default=None,
        help="Observation-encoder selector. If unset, derived from --use_fofe.",
    )
    p.add_argument(
        "--use_fofe",
        action=argparse.BooleanOptionalAction,
        default=fofe_defaults.use_fofe,
        help="(Legacy) Enable/disable FOFE encoder. Ignored if --encoder_type is given.",
    )
    # Reward weights
    p.add_argument("--target_destroyed", type=float, default=reward_defaults.target_destroyed)
    p.add_argument("--agent_destroyed", type=float, default=reward_defaults.agent_destroyed)
    p.add_argument("--timestep_penalty", type=float, default=reward_defaults.timestep_penalty)
    p.add_argument("--team_spirit", type=float, default=reward_defaults.team_spirit)
    p.add_argument("--striker_progress_scale", type=float, default=reward_defaults.striker_progress_scale)
    p.add_argument("--jammer_progress_scale", type=float, default=reward_defaults.jammer_progress_scale)
    p.add_argument("--jammer_jam_bonus", type=float, default=reward_defaults.jammer_jam_bonus)
    # Save / load
    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--save_name", type=str, default="fofe_mappo.pt")
    p.add_argument("--load_checkpoint", type=str, default=None)
    # HF radar model
    p.add_argument(
        "--use_hf_radar",
        action=argparse.BooleanOptionalAction,
        default=ext_defaults.use_hf_radar,
        help="Enable high-fidelity angular radar model (replaces simple binary jam/kill)",
    )
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--no_animate", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
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
        radar_min_sep=args.radar_min_sep,
        scenario=args.scenario,
        s2_radar_min_sep=args.s2_radar_min_sep,
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
    fofe_cfg = FOFEConfig(use_fofe=args.use_fofe)
    gat_cfg = GATConfig()
    ext_cfg = EnvExtensionsConfig(use_hf_radar=args.use_hf_radar)
    cfg = ExperimentConfig(
        env=env_cfg, ppo=ppo_cfg, net=net_cfg,
        fofe=fofe_cfg, gat=gat_cfg, ext=ext_cfg,
        encoder_type=args.encoder_type,
    ).finalize()

    print(f"Observation encoder: {cfg.encoder_type.upper()}")

    hf_radar_cfg = cfg.ext.hf_radar if cfg.ext.use_hf_radar else None
    if cfg.ext.use_hf_radar:
        print("HF angular radar model: ENABLED")
    else:
        print("HF angular radar model: DISABLED (simple binary jam/kill)")

    checkpoint = None
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location=cfg.ppo.device, weights_only=False)
        print(f"Loaded checkpoint from: {args.load_checkpoint}")

    print(
        f"Config: scenario={cfg.env.scenario}, strikers={cfg.env.n_strikers}, jammers={cfg.env.n_jammers}, "
        f"targets={cfg.env.n_targets} (known={cfg.env.n_known_targets}/unknown={cfg.env.n_unknown_targets}), "
        f"radars={cfg.env.n_radars} (known={cfg.env.n_known_radars}/unknown={cfg.env.n_unknown_radars}), "
        f"iters={cfg.ppo.n_iters}, envs={cfg.ppo.num_envs}, max_steps={cfg.env.max_steps}"
    )

    base_env, policy, critic, logs, reward_normalizer = train_mappo(
        cfg.env, cfg.ppo, cfg.net,
        fofe_cfg=cfg.fofe,
        gat_cfg=cfg.gat,
        encoder_type=cfg.encoder_type,
        checkpoint=checkpoint,
        hf_radar_cfg=hf_radar_cfg,
    )

    # ── Save checkpoint ──────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / args.save_name
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "env_cfg": cfg.env,
            "ppo_cfg": cfg.ppo,
            "net_cfg": cfg.net,
            "fofe_cfg": cfg.fofe,
            "gat_cfg": cfg.gat,
            "encoder_type": cfg.encoder_type,
            "ext_cfg": cfg.ext,
            "logs": logs,
            "reward_normalizer_state_dict": (
                reward_normalizer.state_dict() if reward_normalizer is not None else None
            ),
        },
        save_path,
    )
    print(f"\nSaved checkpoint to: {save_path}")

    # ── Print final metrics ──────────────────────────────────────────
    if logs["eval_mean_episode_total_reward"]:
        print("Last metrics:")
        print(f"  train_mean_episode_total_reward = {logs['train_mean_episode_total_reward'][-1]:.4f}")
        print(f"  eval_mean_episode_total_reward  = {logs['eval_mean_episode_total_reward'][-1]:.4f}")
        print(f"  eval_task_completion_rate       = {logs['eval_task_completion_rate'][-1]:.4f}")
        print(f"  eval_survival_rate              = {logs['eval_survival_rate'][-1]:.4f}")
        print(f"  eval_mean_duration              = {logs['eval_mean_duration'][-1]:.4f}")

    # Output directories live alongside the project package, not the cwd from
    # which the script was launched. This keeps artifacts grouped with the
    # run code regardless of where you invoke it from.
    project_dir = Path(__file__).resolve().parent
    plots_dir = project_dir / "plots"
    vis_dir = project_dir / "visualisations"

    if not args.no_plot:
        try:
            plot_training(logs, save_dir=str(plots_dir))
        except Exception as exc:
            print(f"plot_training warning (continuing): {type(exc).__name__}: {exc}")

    if not args.no_animate:
        try:
            vis_dir.mkdir(parents=True, exist_ok=True)
            n_rollouts = 10
            for i in range(n_rollouts):
                # Different seed per rollout → domain-randomised env layout.
                # Without this all 10 visualisations would be the same scene.
                rollout_seed = 999 + i
                gif_path = vis_dir / f"rollout_{i + 1:02d}.gif"
                if cfg.ext.use_hf_radar:
                    tester = HFTestRunner(
                        policy, env_cfg=cfg.env, hf_cfg=cfg.ext.hf_radar,
                        device=cfg.ppo.device, seed=rollout_seed,
                    )
                    frames = tester.rollout()
                    hf_animate_rollout(frames, tester.env, save_path=str(gif_path))
                else:
                    tester = TestRunner(policy, env_cfg=cfg.env, device=cfg.ppo.device, seed=rollout_seed)
                    frames = tester.rollout()
                    animate_rollout(frames, tester.env, save_path=str(gif_path))
                print(f"Visualised rollout {i + 1}/{n_rollouts} ({len(frames)} frames) → {gif_path.name}")
        except Exception as exc:
            print(f"animate_rollout warning (continuing): {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

