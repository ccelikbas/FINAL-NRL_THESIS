from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from centralized_strike_ppo.config import EnvConfig, ExperimentConfig, NetworkConfig, PPOConfig
    from centralized_strike_ppo.trainer import train_centralized_ppo
    from centralized_strike_ppo.rewards import RewardConfig
    from centralized_strike_ppo.visualization import TestRunner, animate_rollout, plot_training
else:
    from .config import EnvConfig, ExperimentConfig, NetworkConfig, PPOConfig
    from .trainer import train_centralized_ppo
    from .rewards import RewardConfig
    from .visualization import TestRunner, animate_rollout, plot_training

import torch


def build_parser() -> argparse.ArgumentParser:
    env_defaults = EnvConfig()
    ppo_defaults = PPOConfig()
    net_defaults = NetworkConfig()
    reward_defaults = RewardConfig()

    p = argparse.ArgumentParser(description="Simple centralized PPO for Strike-EA")
    # Counts
    p.add_argument("--n_strikers", type=int, default=env_defaults.n_strikers)
    p.add_argument("--n_jammers", type=int, default=env_defaults.n_jammers)
    p.add_argument("--n_targets", type=int, default=env_defaults.n_targets)
    p.add_argument("--n_radars", type=int, default=env_defaults.n_radars)
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
    p.add_argument("--seed", type=int, default=ppo_defaults.seed)
    # Network
    p.add_argument("--actor_hidden", type=int, default=net_defaults.actor_hidden)
    p.add_argument("--critic_hidden", type=int, default=net_defaults.critic_hidden)
    p.add_argument("--depth", type=int, default=net_defaults.depth)
    # A few reward weights that are often changed during experimentation
    p.add_argument("--target_destroyed", type=float, default=reward_defaults.target_destroyed)
    p.add_argument("--agent_destroyed", type=float, default=reward_defaults.agent_destroyed)
    p.add_argument("--timestep_penalty", type=float, default=reward_defaults.timestep_penalty)
    p.add_argument("--team_spirit", type=float, default=reward_defaults.team_spirit)
    p.add_argument("--striker_progress_scale", type=float, default=reward_defaults.striker_progress_scale)
    p.add_argument("--jammer_progress_scale", type=float, default=reward_defaults.jammer_progress_scale)
    p.add_argument("--jammer_jam_bonus", type=float, default=reward_defaults.jammer_jam_bonus)
    # Save
    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--save_name", type=str, default="centralized_actor.pt")
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
        n_targets=args.n_targets,
        n_radars=args.n_radars,
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
        seed=args.seed,
    )
    net_cfg = NetworkConfig(
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        depth=args.depth,
    )
    cfg = ExperimentConfig(env=env_cfg, ppo=ppo_cfg, net=net_cfg).finalize()
    base_env, actor, critic, logs = train_centralized_ppo(cfg.env, cfg.ppo, cfg.net)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / args.save_name
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "env_cfg": cfg.env,
            "ppo_cfg": cfg.ppo,
            "net_cfg": cfg.net,
            "logs": logs,
        },
        save_path,
    )
    print(f"\nSaved checkpoint to: {save_path}")
    if logs["eval_mean_episode_total_reward"]:
        print("Last metrics:")
        print(f"  eval_mean_episode_total_reward = {logs['eval_mean_episode_total_reward'][-1]:.4f}")
        print(f"  eval_task_completion_rate      = {logs['eval_task_completion_rate'][-1]:.4f}")
        print(f"  eval_survival_rate             = {logs['eval_survival_rate'][-1]:.4f}")
        print(f"  eval_mean_duration             = {logs['eval_mean_duration'][-1]:.4f}")
        print(f"  clip_ratio                = {logs['clip_ratio'][-1]:.4f}")

    if not args.no_plot:
        try:
            plot_training(logs)
        except Exception as exc:
            print(f"plot_training warning (continuing): {type(exc).__name__}: {exc}")

    if not args.no_animate:
        try:
            tester = TestRunner(actor, env_cfg=cfg.env, device=cfg.ppo.device, seed=999)
            frames = tester.rollout()
            animate_rollout(frames, tester.env)
            print(f"Visualized rollout with {len(frames)} frames")
        except Exception as exc:
            print(f"animate_rollout warning (continuing): {type(exc).__name__}: {exc}")

if __name__ == "__main__":
    main()
