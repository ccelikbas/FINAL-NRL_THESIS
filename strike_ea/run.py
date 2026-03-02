#!/usr/bin/env python
"""
run.py – Experiment entry point.

Usage examples
--------------
# Default run
python run.py

# Use a named preset
python run.py --preset fast
python run.py --preset high_kill
python run.py --preset big_team

# Override individual params on the fly (additive on top of preset)
python run.py --preset default --n_iters 50 --lr 1e-4

# Sensitivity sweep over kill reward
python run.py --sweep kill_reward

# Play/test mode: run a saved policy without training
python run.py --play --policy_path path/to/policy.pt

# Play mode with a preset (uses random initial policy or loads if available)
python run.py --play --preset default

# Override radar kill probability (0=never kill, 1=instant kill)
python run.py --radar_kill_probability 0.5

Available presets:  fast | default | high_kill | strong_jam | big_team | hard_radar | no_step_pen
Available sweeps:   kill_reward | jam_weight | radar_range | entropy
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import replace
from pathlib import Path

# Add repo root to path so imports work
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch

from strike_ea.config import EnvConfig, TrainConfig, NetworkConfig, get_preset, PRESETS
from strike_ea.env.rewards import RewardConfig
from strike_ea.training import train_mappo
from strike_ea.evaluation.runner import TestRunner
from strike_ea.evaluation.visualize import animate_rollout, plot_training


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity sweeps
# ─────────────────────────────────────────────────────────────────────────────

SWEEPS = {
    # name → list of (label, env_cfg, train_cfg, net_cfg)
    "kill_reward": [
        (f"kill={v}", replace(EnvConfig(), reward_config=replace(RewardConfig(), target_destroyed=v)), TrainConfig(), NetworkConfig())
        for v in [2.0, 5.0, 10.0, 20.0]
    ],
    "jam_weight": [
        (f"jam={v}", replace(EnvConfig(), reward_config=replace(RewardConfig(), jamming=v)), TrainConfig(), NetworkConfig())
        for v in [0.1, 0.5, 1.0, 3.0]
    ],
    "radar_range": [
        (f"radar={v}", replace(EnvConfig(), radar_range=v), TrainConfig(), NetworkConfig())
        for v in [0.10, 0.15, 0.20, 0.30]
    ],
    "entropy": [
        (f"ent={v}", EnvConfig(), replace(TrainConfig(), entropy_coef=v), NetworkConfig())
        for v in [0.0, 1e-4, 1e-3, 5e-3]
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Policy saving/loading utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_actor(actor, save_path: str):
    """Save actor network state dict."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), save_path)
    print(f"✓ Policy saved to: {save_path}")


def load_actor(actor, load_path: str):
    """Load actor network state dict."""
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Policy file not found: {load_path}")
    actor.load_state_dict(torch.load(load_path, weights_only=True))
    print(f"✓ Policy loaded from: {load_path}")
    return actor


def run_single(env_cfg: EnvConfig, train_cfg: TrainConfig, net_cfg: NetworkConfig,
               label: str = "run", animate: bool = True, save_dir: str = None,
               save_policy: str = None):
    print(f"\n{'='*60}")
    print(f"  Run: {label}")
    print(f"{'='*60}")

    base_env, actor, critic, logs = train_mappo(train_cfg, env_cfg, net_cfg)
    plot_training(logs, save_dir=save_dir)

    # Save policy if requested
    if save_policy:
        save_actor(actor, save_policy)

    if animate:
        tester = TestRunner(actor, device=train_cfg.device, max_steps=220,
                            seed=999, env_cfg=env_cfg)
        frames = tester.rollout()
        animate_rollout(frames, tester.env)

    return logs


def run_sweep(sweep_name: str, animate: bool = False):
    if sweep_name not in SWEEPS:
        raise ValueError(f"Unknown sweep '{sweep_name}'. Available: {sorted(SWEEPS.keys())}")

    results = {}
    for label, env_cfg, train_cfg, net_cfg in SWEEPS[sweep_name]:
        logs = run_single(env_cfg, train_cfg, net_cfg, label=label, animate=animate)
        results[label] = logs

    # Compare episode rewards across all sweep variants
    import matplotlib.pyplot as plt
    plt.figure()
    for label, logs in results.items():
        plt.plot(logs["episode_reward_mean"], label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Mean episode reward")
    plt.title(f"Sensitivity sweep: {sweep_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_play(env_cfg: EnvConfig, train_cfg: TrainConfig, net_cfg: NetworkConfig, 
             policy_path: str = None):
    """Load and test a policy without training."""
    print(f"\n{'='*60}")
    print(f"  Play Mode: Testing Policy")
    print(f"{'='*60}")
    
    from strike_ea.models.actor import make_actor
    from strike_ea.env.environment import StrikeEA2DEnv
    
    device = train_cfg.device
    
    # Create environment with unpacked config
    env = StrikeEA2DEnv(
        num_envs      = train_cfg.num_envs,
        max_steps     = train_cfg.max_steps,
        device        = device,
        seed          = train_cfg.seed,
        # pass through every env_cfg field
        n_strikers    = env_cfg.n_strikers,
        n_jammers     = env_cfg.n_jammers,
        n_targets     = env_cfg.n_targets,
        n_radars      = env_cfg.n_radars,
        dt            = env_cfg.dt,
        world_bounds  = env_cfg.world_bounds,
        v_max         = env_cfg.v_max,
        dpsi_max      = env_cfg.dpsi_max,
        R_obs         = env_cfg.R_obs,
        radar_range   = env_cfg.radar_range,
        border_thresh = env_cfg.border_thresh,
        reward_config = env_cfg.reward_config,
    )
    
    actor = make_actor(env, hidden=net_cfg.hidden)
    
    # Load policy if provided, otherwise use random initialization
    if policy_path:
        actor = load_actor(actor, policy_path)
    else:
        print("✓ Using random initial policy (no checkpoint loaded)")
    
    # Run test
    tester = TestRunner(actor, device=device, max_steps=train_cfg.max_steps, 
                        seed=999, env_cfg=env_cfg)
    frames = tester.rollout()
    animate_rollout(frames, tester.env)
    print(f"\n✓ Test complete! Rollout length: {len(frames)} steps")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Strike–EA MARL experiment runner")
    p.add_argument("--play",      action="store_true", help="Play mode: test a policy without training")
    p.add_argument("--policy_path", default=None,      help="Path to saved policy checkpoint (.pt file)")
    p.add_argument("--preset",    default="default", help="Named config preset")
    p.add_argument("--sweep",     default=None,      help="Run a sensitivity sweep instead of a single experiment")
    p.add_argument("--no_animate", action="store_true", help="Skip rollout animation after training")
    p.add_argument("--save_dir",  default=None,      help="Directory to save plots")
    p.add_argument("--save_policy", default=None,    help="Path to save trained policy checkpoint")

    # Per-run overrides (applied on top of the selected preset)
    # Training
    p.add_argument("--n_iters",        type=int,   default=None, help="Number of training iterations")
    p.add_argument("--num_envs",       type=int,   default=None, help="Number of parallel environments")
    p.add_argument("--max_steps",      type=int,   default=None, help="Max steps per episode")
    p.add_argument("--lr",             type=float, default=None, help="Learning rate")
    p.add_argument("--entropy_coef",   type=float, default=None, help="Entropy regularization weight")
    p.add_argument("--hidden",         type=int,   default=None, help="Hidden layer width for networks")
    
    # Environment - kinematics
    p.add_argument("--v_max",          type=float, default=None, help="Maximum velocity")
    p.add_argument("--accel_magnitude", type=float, default=None, help="Velocity acceleration per action")
    p.add_argument("--dpsi_max_deg",   type=float, default=None, help="Max heading rate (degrees/step)")
    p.add_argument("--h_accel_fraction", type=float, default=None, help="Heading accel fraction of dpsi_max")
    
    # Environment - agent capabilities
    p.add_argument("--striker_engage_range", type=float, default=None, help="Striker engagement range")
    p.add_argument("--striker_engage_fov", type=float, default=None, help="Striker FOV (degrees)")
    p.add_argument("--jammer_jam_radius", type=float, default=None, help="Jammer range")
    p.add_argument("--jammer_jam_effect", type=float, default=None, help="Radar range reduction per jammer")
    
    # Environment - radar
    p.add_argument("--radar_range",    type=float, default=None, help="Radar detection range")
    p.add_argument("--radar_kill_probability", type=float, default=None, help="Radar kill probability [0, 1]")
    
    # Rewards
    p.add_argument("--kill_reward",    type=float, default=None, help="Target destroyed reward weight")
    p.add_argument("--jam_reward",     type=float, default=None, help="Jamming reward weight")
    return p.parse_args()


def apply_overrides(args, env_cfg: EnvConfig, train_cfg: TrainConfig, net_cfg: NetworkConfig):
    """Patch config objects with any CLI overrides."""
    # Training config
    if args.n_iters      is not None: train_cfg = replace(train_cfg, n_iters=args.n_iters)
    if args.num_envs     is not None: train_cfg = replace(train_cfg, num_envs=args.num_envs)
    if args.max_steps    is not None: train_cfg = replace(train_cfg, max_steps=args.max_steps)
    if args.lr           is not None: train_cfg = replace(train_cfg, lr=args.lr)
    if args.entropy_coef is not None: train_cfg = replace(train_cfg, entropy_coef=args.entropy_coef)
    
    # Network config
    if args.hidden       is not None: net_cfg   = replace(net_cfg,   hidden=args.hidden)
    
    # Environment config - kinematics
    if args.v_max           is not None: env_cfg = replace(env_cfg, v_max=args.v_max)
    if args.accel_magnitude is not None: env_cfg = replace(env_cfg, accel_magnitude=args.accel_magnitude)
    if args.dpsi_max_deg    is not None: env_cfg = replace(env_cfg, dpsi_max=math.radians(args.dpsi_max_deg))
    if args.h_accel_fraction is not None: env_cfg = replace(env_cfg, h_accel_magnitude_fraction=args.h_accel_fraction)
    
    # Environment config - agent capabilities
    if args.striker_engage_range is not None: env_cfg = replace(env_cfg, striker_engage_range=args.striker_engage_range)
    if args.striker_engage_fov is not None: env_cfg = replace(env_cfg, striker_engage_fov=args.striker_engage_fov)
    if args.jammer_jam_radius is not None: env_cfg = replace(env_cfg, jammer_jam_radius=args.jammer_jam_radius)
    if args.jammer_jam_effect is not None: env_cfg = replace(env_cfg, jammer_jam_effect=args.jammer_jam_effect)
    
    # Environment config - radar
    if args.radar_range  is not None: env_cfg = replace(env_cfg, radar_range=args.radar_range)
    if args.radar_kill_probability is not None: env_cfg = replace(env_cfg, radar_kill_probability=args.radar_kill_probability)

    # Reward config
    rw = env_cfg.reward_config
    if args.kill_reward is not None: rw = replace(rw, target_destroyed=args.kill_reward)
    if args.jam_reward  is not None: rw = replace(rw, jamming=args.jam_reward)
    env_cfg = replace(env_cfg, reward_config=rw)

    return env_cfg, train_cfg, net_cfg

    return env_cfg, train_cfg, net_cfg


def main():
    args = parse_args()

    if args.play:
        # Play mode: test a policy without training
        env_cfg, train_cfg, net_cfg = get_preset(args.preset)
        env_cfg, train_cfg, net_cfg = apply_overrides(args, env_cfg, train_cfg, net_cfg)
        run_play(env_cfg, train_cfg, net_cfg, policy_path=args.policy_path)
        return

    if args.sweep:
        run_sweep(args.sweep, animate=not args.no_animate)
        return

    env_cfg, train_cfg, net_cfg = get_preset(args.preset)
    env_cfg, train_cfg, net_cfg = apply_overrides(args, env_cfg, train_cfg, net_cfg)

    run_single(
        env_cfg, train_cfg, net_cfg,
        label=args.preset,
        animate=not args.no_animate,
        save_dir=args.save_dir,
        save_policy=args.save_policy,
    )


if __name__ == "__main__":
    main()


# PLAY MODE:
# & "c:/Users/celikbas/Documents/REPO GIT NLR/.venv/Scripts/python.exe" strike_ea/run.py --play --preset default

