#!/usr/bin/env python
"""
run.py – Experiment entry point.

Usage examples
--------------
# Default run
python run.py

# Use a named preset
python run.py --preset fast

# Override individual params on the fly
python run.py --preset default --n_iters 50 --lr 1e-4

# Play/test mode
python run.py --play --preset default
python run.py --play --policy_path path/to/policy.pt

Available presets:  fast | default | high_kill | strong_jam | big_team | hard_radar | no_step_pen
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch

from strike_ea.config import EnvConfig, TrainConfig, NetworkConfig, get_preset
from strike_ea.env.rewards import RewardConfig
from strike_ea.training import train_mappo
from strike_ea.evaluation.runner import TestRunner
from strike_ea.evaluation.visualize import animate_rollout, plot_training


# ─────────────────────────────────────────────────────────────────────────────
# Policy saving / loading
# ─────────────────────────────────────────────────────────────────────────────

def save_actor(actor, save_path: str):
    """Save actor network state dict. Creates parent directories if needed."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), save_path)
    print(f"Policy saved to: {save_path}")


def get_timestamped_policy_path(preset_name: str = "default", policy_dir: str = "saved_policies") -> str:
    """Generate timestamped policy save path.
    
    Example: saved_policies/default/2026-03-03_17-22-46.pt
    
    This allows you to easily track when each policy was trained and compare
    multiple training runs. Timestamp format: YYYY-MM-DD_HH-MM-SS
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(policy_dir) / preset_name / f"{timestamp}.pt"
    return str(path)


def load_actor(actor, load_path: str):
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Policy file not found: {load_path}")
    actor.load_state_dict(torch.load(load_path, weights_only=True))
    print(f"Policy loaded from: {load_path}")
    return actor


# ─────────────────────────────────────────────────────────────────────────────
# Run modes
# ─────────────────────────────────────────────────────────────────────────────

def run_single(env_cfg, train_cfg, net_cfg, *, label="run", animate=True, save_dir=None, save_policy=None):
    print(f"\n{'='*60}\n  Run: {label}\n{'='*60}")
    base_env, actor, critic, logs = train_mappo(train_cfg, env_cfg, net_cfg)
    plot_training(logs, save_dir=save_dir)
    
    if save_policy:
        save_actor(actor, save_policy)
    
    if animate:
        tester = TestRunner(actor, device=train_cfg.device, max_steps=220, seed=999, env_cfg=env_cfg)
        frames = tester.rollout()
        animate_rollout(frames, tester.env)
    return logs


def run_play(env_cfg, train_cfg, net_cfg, policy_path=None):
    print(f"\n{'='*60}\n  Play Mode\n{'='*60}")
    from strike_ea.models.actor import make_actor
    from strike_ea.env.environment import StrikeEA2DEnv

    device = train_cfg.device
    env = StrikeEA2DEnv(
        num_envs=1, max_steps=train_cfg.max_steps, device=device, seed=train_cfg.seed,
        n_strikers=env_cfg.n_strikers, n_jammers=env_cfg.n_jammers,
        n_targets=env_cfg.n_targets, n_radars=env_cfg.n_radars,
        dt=env_cfg.dt, world_bounds=env_cfg.world_bounds,
        v_max=env_cfg.v_max, accel_magnitude=env_cfg.accel_magnitude,
        dpsi_max=env_cfg.dpsi_max, h_accel_magnitude_fraction=env_cfg.h_accel_magnitude_fraction,
        R_obs=env_cfg.R_obs,
        striker_engage_range=env_cfg.striker_engage_range, striker_engage_fov=env_cfg.striker_engage_fov,
        striker_v_min=env_cfg.striker_v_min,
        jammer_jam_radius=env_cfg.jammer_jam_radius, jammer_jam_effect=env_cfg.jammer_jam_effect,
        jammer_v_min=env_cfg.jammer_v_min,
        radar_range=env_cfg.radar_range, radar_kill_probability=env_cfg.radar_kill_probability,
        border_thresh=env_cfg.border_thresh, reward_config=env_cfg.reward_config,
    )

    actor = make_actor(env, hidden=net_cfg.hidden)
    if policy_path:
        actor = load_actor(actor, policy_path)
    else:
        print("Using random initial policy (no checkpoint loaded)")

    tester = TestRunner(actor, device=device, max_steps=train_cfg.max_steps, seed=999, env_cfg=env_cfg)
    frames = tester.rollout()
    animate_rollout(frames, tester.env)
    print(f"\nTest complete! Rollout length: {len(frames)} steps")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Strike-EA MARL experiment runner")
    p.add_argument("--play",          action="store_true")
    p.add_argument("--policy_path",   default=None)
    p.add_argument("--preset",        default="default")
    p.add_argument("--no_animate",    action="store_true")
    p.add_argument("--save_dir",      default=None, help="Directory to save training plots")
    p.add_argument("--save_policy",   default=None, help="Explicit policy save path (default: auto-timestamped)")
    p.add_argument("--policy_dir",    default="saved_policies", help="Directory for auto-saved policies (default: saved_policies)")
    p.add_argument("--no_save_policy", action="store_true", help="Disable automatic policy saving")

    # Training overrides
    p.add_argument("--n_iters",        type=int,   default=None)
    p.add_argument("--num_envs",       type=int,   default=None)
    p.add_argument("--max_steps",      type=int,   default=None)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--entropy_coef",   type=float, default=None)
    p.add_argument("--hidden",         type=int,   default=None)

    # Kinematics
    p.add_argument("--v_max",            type=float, default=None)
    p.add_argument("--accel_magnitude",  type=float, default=None)
    p.add_argument("--dpsi_max_deg",     type=float, default=None)
    p.add_argument("--h_accel_fraction", type=float, default=None)

    # Agent capabilities
    p.add_argument("--striker_engage_range", type=float, default=None)
    p.add_argument("--striker_engage_fov",   type=float, default=None)
    p.add_argument("--jammer_jam_radius",    type=float, default=None)
    p.add_argument("--jammer_jam_effect",    type=float, default=None)

    # Radar
    p.add_argument("--radar_range",            type=float, default=None)
    p.add_argument("--radar_kill_probability", type=float, default=None)

    # Reward overrides (maps to RewardConfig fields)
    p.add_argument("--kill_reward",        type=float, default=None, help="target_destroyed weight")
    p.add_argument("--timestep_penalty",   type=float, default=None)
    p.add_argument("--border_penalty",     type=float, default=None)
    p.add_argument("--jam_reward",         type=float, default=None, help="jammer_jamming weight")
    p.add_argument("--striker_proximity",  type=float, default=None)
    return p.parse_args()


def apply_overrides(args, env_cfg, train_cfg, net_cfg):
    """Patch config objects with any CLI overrides."""
    if args.n_iters      is not None: train_cfg = replace(train_cfg, n_iters=args.n_iters)
    if args.num_envs     is not None: train_cfg = replace(train_cfg, num_envs=args.num_envs)
    if args.max_steps    is not None: train_cfg = replace(train_cfg, max_steps=args.max_steps)
    if args.lr           is not None: train_cfg = replace(train_cfg, lr=args.lr)
    if args.entropy_coef is not None: train_cfg = replace(train_cfg, entropy_coef=args.entropy_coef)
    if args.hidden       is not None: net_cfg   = replace(net_cfg,   hidden=args.hidden)

    if args.v_max            is not None: env_cfg = replace(env_cfg, v_max=args.v_max)
    if args.accel_magnitude  is not None: env_cfg = replace(env_cfg, accel_magnitude=args.accel_magnitude)
    if args.dpsi_max_deg     is not None: env_cfg = replace(env_cfg, dpsi_max=math.radians(args.dpsi_max_deg))
    if args.h_accel_fraction is not None: env_cfg = replace(env_cfg, h_accel_magnitude_fraction=args.h_accel_fraction)

    if args.striker_engage_range is not None: env_cfg = replace(env_cfg, striker_engage_range=args.striker_engage_range)
    if args.striker_engage_fov   is not None: env_cfg = replace(env_cfg, striker_engage_fov=args.striker_engage_fov)
    if args.jammer_jam_radius    is not None: env_cfg = replace(env_cfg, jammer_jam_radius=args.jammer_jam_radius)
    if args.jammer_jam_effect    is not None: env_cfg = replace(env_cfg, jammer_jam_effect=args.jammer_jam_effect)

    if args.radar_range            is not None: env_cfg = replace(env_cfg, radar_range=args.radar_range)
    if args.radar_kill_probability is not None: env_cfg = replace(env_cfg, radar_kill_probability=args.radar_kill_probability)

    # Reward overrides
    rw = env_cfg.reward_config
    if args.kill_reward       is not None: rw = replace(rw, target_destroyed=args.kill_reward)
    if args.timestep_penalty  is not None: rw = replace(rw, timestep_penalty=args.timestep_penalty)
    if args.border_penalty    is not None: rw = replace(rw, border_penalty=args.border_penalty)
    if args.jam_reward        is not None: rw = replace(rw, jammer_jamming=args.jam_reward)
    if args.striker_proximity is not None: rw = replace(rw, striker_proximity=args.striker_proximity)
    env_cfg = replace(env_cfg, reward_config=rw)

    return env_cfg, train_cfg, net_cfg


def main():
    args = parse_args()

    if args.play:
        env_cfg, train_cfg, net_cfg = get_preset(args.preset)
        env_cfg, train_cfg, net_cfg = apply_overrides(args, env_cfg, train_cfg, net_cfg)
        run_play(env_cfg, train_cfg, net_cfg, policy_path=args.policy_path)
        return

    env_cfg, train_cfg, net_cfg = get_preset(args.preset)
    env_cfg, train_cfg, net_cfg = apply_overrides(args, env_cfg, train_cfg, net_cfg)

    # Handle policy saving: explicit > auto-timestamped > disabled
    if args.no_save_policy:
        save_policy = None
    else:
        save_policy = args.save_policy or get_timestamped_policy_path(
            preset_name=args.preset,
            policy_dir=args.policy_dir
        )

    run_single(
        env_cfg, train_cfg, net_cfg,
        label=args.preset,
        animate=not args.no_animate,
        save_dir=args.save_dir,
        save_policy=save_policy,
    )


if __name__ == "__main__":
    main()