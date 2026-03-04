#!/usr/bin/env python
r"""
run.py – Experiment entry point.

Usage examples
--------------
# TRAINING (auto-saves policy + shows visualization after training)
python run.py                       # Train with default config
python run.py --preset fast         # Fast training (fewer iters, smaller net)
python run.py --preset default --n_iters 50 --lr 1e-4  # Override params
python run.py --no_animate          # Train + save policy, skip visualization
python run.py --n_env_layouts 1     # Train on a single fixed radar layout
python run.py --n_env_layouts 50    # Train on 50 distinct radar layouts

# TEST / VISUALIZE A SAVED POLICY (no training)
.\.venv\Scripts\python.exe .\strike_ea\run.py --test --policy_path .\saved_policies\default\2026-03-04_11-53-53.pt
python run.py --test --preset default  # Test random (untrained) policy

# LIST saved policies
python run.py --list

# If python not in PATH:
.venv\Scripts\python.exe run.py --test --policy_path saved_policies\default\2026-03-04_11-43-32.pt

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

def list_saved_policies(policy_dir: str = "saved_policies"):
    """Print all saved policy files, organized by preset."""
    policy_path = Path(policy_dir)
    if not policy_path.exists():
        print(f"No saved policies found in {policy_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Saved Policies in {policy_dir}:")
    print(f"{'='*70}")
    
    presets = sorted([p for p in policy_path.iterdir() if p.is_dir()])
    if not presets:
        print("  (none)")
        return
    
    for preset_dir in presets:
        print(f"\n  📁 {preset_dir.name}/")
        policies = sorted([p for p in preset_dir.glob("*.pt")])
        if not policies:
            print(f"     (empty)")
        for policy in policies:
            size_mb = policy.stat().st_size / (1024 * 1024)
            print(f"     • {policy.name}  ({size_mb:.1f} MB)")
    
    print(f"\n{'='*70}\n")


def save_actor(actor, save_path: str, env_cfg=None, net_cfg=None, preset_name=None):
    """Save actor network state dict + config metadata. Creates parent directories if needed.
    
    Saves a dict with:
      - 'state_dict': network weights
      - 'env_cfg': EnvConfig used during training (for correct env reconstruction)
      - 'net_cfg': NetworkConfig (hidden size, etc.)
      - 'preset': preset name used
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": actor.state_dict(),
        "env_cfg": env_cfg,
        "net_cfg": net_cfg,
        "preset": preset_name,
    }
    torch.save(checkpoint, save_path)
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
    """Load actor weights from a checkpoint file.
    
    Supports both new format (dict with 'state_dict' + metadata) and
    legacy format (raw state_dict).
    
    Returns: (actor, env_cfg_or_None, net_cfg_or_None)
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Policy file not found: {load_path}")
    checkpoint = torch.load(load_path, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        actor.load_state_dict(checkpoint["state_dict"])
        env_cfg = checkpoint.get("env_cfg")
        net_cfg = checkpoint.get("net_cfg")
        # Backward compat: old checkpoints may lack n_env_layouts
        if env_cfg is not None and not hasattr(env_cfg, 'n_env_layouts'):
            object.__setattr__(env_cfg, 'n_env_layouts', 0)
        preset = checkpoint.get("preset", "unknown")
        print(f"  Policy loaded from: {load_path} (preset: {preset})")
        return actor, env_cfg, net_cfg
    else:
        # Legacy format: raw state_dict (no config metadata)
        try:
            actor.load_state_dict(checkpoint)
        except (ValueError, RuntimeError) as e:
            raise RuntimeError(
                f"Cannot load legacy checkpoint: {load_path}\n"
                f"  Error: {e}\n"
                f"  This checkpoint was saved without config metadata.\n"
                f"  The network dimensions don't match current config.\n"
                f"  Fix: retrain with current code (auto-saves metadata),\n"
                f"  or specify matching --preset / env params on the CLI."
            ) from e
        print(f"  Policy loaded from: {load_path} (legacy format, no metadata)")
        return actor, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Run modes
# ─────────────────────────────────────────────────────────────────────────────

def run_single(env_cfg, train_cfg, net_cfg, *, label="run", animate=True, save_dir=None, save_policy=None):
    """Train, save policy, and optionally visualize a test rollout."""
    print(f"\n{'='*60}\n  Training: {label}\n{'='*60}")
    base_env, actor, critic, logs = train_mappo(train_cfg, env_cfg, net_cfg)
    plot_training(logs, save_dir=save_dir)

    if save_policy:
        save_actor(actor, save_policy, env_cfg=env_cfg, net_cfg=net_cfg, preset_name=label)

    if animate:
        print(f"\n{'='*60}\n  Visualizing trained policy\n{'='*60}")
        tester = TestRunner(actor, device=train_cfg.device, max_steps=train_cfg.max_steps, seed=999, env_cfg=env_cfg)
        frames = tester.rollout()
        animate_rollout(frames, tester.env)
        print(f"Test rollout: {len(frames)} steps")
    return logs


def run_play(env_cfg, train_cfg, net_cfg, policy_path=None, animate=True):
    """Test mode: load a saved policy and visualize a rollout (no training).

    If the policy file contains saved config metadata, those configs are used
    automatically (so you don't need to specify --preset to match training).
    """
    print(f"\n{'='*60}\n  Test Mode (Visualization)\n{'='*60}")
    from strike_ea.models.actor import make_actor

    device = train_cfg.device

    # Load config from checkpoint if available
    if policy_path:
        ckpt_path = Path(policy_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Policy file not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            if checkpoint.get("env_cfg") is not None:
                env_cfg = checkpoint["env_cfg"]
                # Backward compat: old checkpoints may lack n_env_layouts
                if not hasattr(env_cfg, 'n_env_layouts'):
                    object.__setattr__(env_cfg, 'n_env_layouts', 0)
                print(f"  Using env config from checkpoint")
            if checkpoint.get("net_cfg") is not None:
                net_cfg = checkpoint["net_cfg"]
                print(f"  Using network config (hidden={net_cfg.hidden})")
            print(f"  Preset: {checkpoint.get('preset', 'unknown')}")

    # Create test environment and build actor from its specs
    tester = TestRunner(device=device, max_steps=train_cfg.max_steps, seed=999, env_cfg=env_cfg)
    actor = make_actor(tester.env, hidden=net_cfg.hidden)
    if policy_path:
        actor, _, _ = load_actor(actor, policy_path)
    else:
        print("  No policy loaded — using random initial policy")
    tester.policy = actor.eval()

    # Run rollout
    frames = tester.rollout()
    if animate:
        animate_rollout(frames, tester.env)
    print(f"\nTest complete! Rollout: {len(frames)} steps")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Strike-EA MARL experiment runner")
    p.add_argument("--test",          action="store_true", help="Test mode: load + visualize a saved policy (no training)")
    p.add_argument("--play",          action="store_true", help="(alias for --test)")
    p.add_argument("--list",          action="store_true", help="List all saved policies and exit")
    p.add_argument("--policy_path",   default=None, help="Path to saved policy file for testing")
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
    p.add_argument("--min_turn_radius",  type=float, default=None, help="Minimum turn radius (0.05 = 50 km)")

    # Environment layout
    p.add_argument("--n_env_layouts",  type=int, default=None, help="Pre-generated env layouts (0=random, 1=fixed, N=N scenarios)")

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
    p.add_argument("--agent_destroyed",   type=float, default=None, help="Death penalty (e.g. -100)")
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
    if args.min_turn_radius  is not None: env_cfg = replace(env_cfg, min_turn_radius=args.min_turn_radius)
    if args.n_env_layouts    is not None: env_cfg = replace(env_cfg, n_env_layouts=args.n_env_layouts)

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
    if args.agent_destroyed   is not None: rw = replace(rw, agent_destroyed=args.agent_destroyed)
    env_cfg = replace(env_cfg, reward_config=rw)

    return env_cfg, train_cfg, net_cfg


def main():
    args = parse_args()

    # List saved policies and exit
    if args.list:
        list_saved_policies(args.policy_dir)
        return

    # --- Test mode: load a saved policy and visualize (no training) ---
    if args.play or args.test:
        env_cfg, train_cfg, net_cfg = get_preset(args.preset)
        env_cfg, train_cfg, net_cfg = apply_overrides(args, env_cfg, train_cfg, net_cfg)
        run_play(env_cfg, train_cfg, net_cfg, policy_path=args.policy_path, animate=not args.no_animate)
        return

    # --- Training mode: train + save + (optional) visualize ---
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

