"""
config.py – Centralized configuration for all Strike-EA parameters.

This is the single source of truth for experiment parameters. Edit values here,
then use run.py to execute experiments with these settings.

COORDINATE SCALING:
  Environment operates in normalized coordinates (0, 1) representing realistic dimensions:
  • Spatial: 1 unit = 1000 km (map is 0-1000 km x 0-1000 km)
  • Temporal: 1 step = 60 seconds (1 minute per simulation step)
  • Velocity: 0.02 units/step ≈ 20 km/min ≈ Mach 0.95
  • Velocity: 0.01 units/step ≈ 10 km/min ≈ Mach 0.5

Usage:
    python run.py                           # Use default config
    python run.py --preset fast             # Use a preset
    python run.py --lr 1e-4                 # Override specific parameter
    python run.py --preset default --n_iters 50  # Combine preset + overrides

Three main config categories:
  • EnvConfig     – Environment, agents, sensors, rewards
  • TrainConfig   – Training algorithm, optimization, logging
  • NetworkConfig – Neural network architecture
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from strike_ea.env.rewards import RewardConfig


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    """Environment parameters: world, agents, sensors, physics, and rewards.
    
    All distances are in normalized coordinates where 1 unit = 1000 km.
    All times are in simulation steps where 1 step = 60 seconds (1 minute).
    """

    # ─── Team Composition ────────────────────────────────────────────────────
    n_strikers: int = 2                     # Number of striker agents (kinetic strike capability)
    n_jammers: int = 2                      # Number of jammer agents (electronic attack)
    n_targets: int = 2                      # Number of target entities to destroy
    n_radars: int = 2                       # Number of radar entities (detect agents)

    # ─── World / Dynamics ────────────────────────────────────────────────────
    world_bounds: Tuple[float, float] = (0.0, 1.0)  # [min, max] normalized world bounds (0-1000 km)
    dt: float = 1.0                         # Time step = 60 seconds (1 minute per simulation step)

    # ─── Agent Kinematics ────────────────────────────────────────────────────
    # All agents cruise at Mach 0.5-0.95 (~10-20 km/min)
    v_max: float = 0.02                     # Max velocity: 20 km/min ≈ Mach 0.95
    accel_magnitude: float = 0.01           # Acceleration per discrete action: 10 km/min per step
    dpsi_max: float = math.radians(12.0)   # Max heading rate: 12 deg/min (realistic cruise-speed pivot)
    h_accel_magnitude_fraction: float = 0.1  # Angular accel = dpsi_max * 0.1 = 1.2 deg/min per ±1 action

    # ─── Sensor / Observation ────────────────────────────────────────────────
    R_obs: float = 0.30                     # Observation radius: 300 km (realistic airborne sensor range)
    
    # ─── Striker Capabilities (kinetic interceptors) ──────────────────────────
    striker_engage_range: float = 0.10      # Engagement range: 100 km (kinetic standoff)
    striker_engage_fov: float = 60.0        # Field-of-view cone for engagement (degrees)
    striker_v_min: float = 0.01             # Minimum velocity: 10 km/min ≈ Mach 0.5
    
    # ─── Jammer Capabilities (electronic attack) ────────────────────────────
    jammer_jam_radius: float = 0.25         # Jamming coverage radius: 150 km
    jammer_jam_effect: float = 0.10         # Range reduction from jamming: 100 km
    jammer_v_min: float = 0.01              # Minimum velocity: 10 km/min ≈ Mach 0.5
    
    # ─── Radar Threat ───────────────────────────────────────────────────────
    radar_range: float = 0.20                # Baseline detection range: 200 km
    radar_kill_probability: float = 1.0     # Probability [0, 1] agent is killed per step if detected

    # ─── Reward Shaping ─────────────────────────────────────────────────────
    border_thresh: float = 0.05             # Boundary penalty zone: 50 km from edge
    reward_config: RewardConfig = field(default_factory=RewardConfig)  # All reward weights


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Training algorithm, optimization, and data collection parameters.
    
    Note: max_steps = 160 steps * 60 sec/step = 9600 seconds ≈ 160 minutes ≈ 2.7 hours per episode.
    """

    # ─── Rollout / Environment ──────────────────────────────────────────────
    num_envs: int = 256                     # Number of parallel environments
    max_steps: int = 160                    # Maximum steps per episode (≈160 minutes real-world time)

    # ─── Data Collection ────────────────────────────────────────────────────
    frames_per_batch: int = 8_192           # Frames collected per training iteration
    n_iters: int = 10                       # Number of collection → update cycles

    # ─── PPO Algorithm ──────────────────────────────────────────────────────
    num_epochs: int = 10                    # Epochs per minibatch update
    minibatch_size: int = 1_024             # Minibatch size for gradient updates
    clip_eps: float = 0.2                   # PPO clipping range
    gamma: float = 0.99                     # Discount factor
    lmbda: float = 0.95                     # GAE lambda (advantage estimation)
    entropy_coef: float = 1e-3              # Entropy regularization weight

    # ─── Optimization ───────────────────────────────────────────────────────
    lr: float = 3e-4                        # Learning rate
    max_grad_norm: float = 1.0              # Gradient clipping threshold

    # ─── Logging / Misc ─────────────────────────────────────────────────────
    seed: int = 0                           # Random seed
    log_every: int = 10                     # Print summary every N iterations
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )                                       # Training device (GPU/CPU)


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NetworkConfig:
    """Neural network architecture parameters."""

    hidden: int = 256                       # Hidden layer width for actor and critic MLPs


# ─────────────────────────────────────────────────────────────────────────────
# NAMED PRESETS
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: dict[str, tuple[EnvConfig, TrainConfig, NetworkConfig]] = {

    # ── Smoke test (runs in seconds) ──────────────────────────────────────────
    "fast": (
        EnvConfig(),
        TrainConfig(
            num_envs=32, frames_per_batch=512, n_iters=3,
            num_epochs=2, minibatch_size=128, max_steps=40
        ),
        NetworkConfig(hidden=64),
    ),

    # ── Default full training run ─────────────────────────────────────────────
    "default": (
        EnvConfig(),
        TrainConfig(),
        NetworkConfig(),
    ),

    # ── Aggressive target destruction ─────────────────────────────────────────
    "high_kill": (
        EnvConfig(reward_config=RewardConfig(target_destroyed=25.0, agent_destroyed=-5.0)),
        TrainConfig(),
        NetworkConfig(),
    ),

    # ── Strong electronic warfare focus ───────────────────────────────────────
    "strong_jam": (
        EnvConfig(reward_config=RewardConfig(jamming=5.0, move_closer=0.3)),
        TrainConfig(),
        NetworkConfig(),
    ),

    # ── Larger teams ──────────────────────────────────────────────────────────
    "big_team": (
        EnvConfig(n_strikers=4, n_jammers=4, n_targets=4, n_radars=4),
        TrainConfig(num_envs=128, frames_per_batch=4_096),
        NetworkConfig(hidden=512),
    ),

    # ── Tight radar defense ───────────────────────────────────────────────────
    "hard_radar": (
        EnvConfig(
            radar_range=0.35,
            border_thresh=0.08,
            reward_config=RewardConfig(border=-2.0)
        ),
        TrainConfig(n_iters=20),
        NetworkConfig(),
    ),

    # ── No step penalty variant ───────────────────────────────────────────────
    "no_step_pen": (
        EnvConfig(reward_config=RewardConfig(small_step=0.0)),
        TrainConfig(),
        NetworkConfig(),
    ),
}


def get_preset(name: str) -> tuple[EnvConfig, TrainConfig, NetworkConfig]:
    """Fetch a named preset, raising a clear error if unknown."""
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {sorted(PRESETS.keys())}"
        )
    return PRESETS[name]
