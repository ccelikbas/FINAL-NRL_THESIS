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

import copy
import math
from dataclasses import dataclass, field, replace
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
    n_strikers: int = 2
    n_jammers: int = 2
    n_targets: int = 2
    n_radars: int = 2

    # ─── World / Dynamics ────────────────────────────────────────────────────
    world_bounds: Tuple[float, float] = (0.0, 1.0)
    dt: float = 1.0

    # ─── Agent Kinematics ────────────────────────────────────────────────────
    v_max: float = 0.02
    accel_magnitude: float = 0.01
    dpsi_max: float = math.radians(12.0)
    h_accel_magnitude_fraction: float = 0.1

    # ─── Sensor / Observation ────────────────────────────────────────────────
    R_obs: float = 0.30

    # ─── Striker Capabilities ────────────────────────────────────────────────
    striker_engage_range: float = 0.10
    striker_engage_fov: float = 60.0
    striker_v_min: float = 0.01

    # ─── Jammer Capabilities ────────────────────────────────────────────────
    jammer_jam_radius: float = 0.25
    jammer_jam_effect: float = 0.10
    jammer_v_min: float = 0.01

    # ─── Radar Threat ───────────────────────────────────────────────────────
    radar_range: float = 0.20
    radar_kill_probability: float = 1.0

    # ─── Reward Shaping ─────────────────────────────────────────────────────
    border_thresh: float = 0.05
    reward_config: RewardConfig = field(default_factory=RewardConfig)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Training algorithm, optimization, and data collection parameters."""

    # ─── Rollout / Environment ──────────────────────────────────────────────
    num_envs: int = 256
    max_steps: int = 160

    # ─── Data Collection ────────────────────────────────────────────────────
    frames_per_batch: int = 8_192
    n_iters: int = 100

    # ─── PPO Algorithm ──────────────────────────────────────────────────────
    num_epochs: int = 10
    minibatch_size: int = 1_024
    clip_eps: float = 0.1
    gamma: float = 0.99
    lmbda: float = 0.95
    entropy_coef: float = 1e-3

    # ─── Optimization ───────────────────────────────────────────────────────
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    # ─── Logging / Misc ─────────────────────────────────────────────────────
    seed: int = 0
    log_every: int = 10
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NetworkConfig:
    """Neural network architecture parameters."""
    hidden: int = 256


# ─────────────────────────────────────────────────────────────────────────────
# PRESETS
# ─────────────────────────────────────────────────────────────────────────────

def get_preset(name: str) -> Tuple[EnvConfig, TrainConfig, NetworkConfig]:
    """Return (EnvConfig, TrainConfig, NetworkConfig) for a named preset."""
    _presets = {
        "default": lambda: (
            EnvConfig(),
            TrainConfig(),
            NetworkConfig(),
        ),
        "fast": lambda: (
            EnvConfig(),
            TrainConfig(n_iters=20, num_envs=64, frames_per_batch=2048),
            NetworkConfig(hidden=128),
        ),
        "high_kill": lambda: (
            EnvConfig(reward_config=RewardConfig(target_destroyed=20.0)),
            TrainConfig(),
            NetworkConfig(),
        ),
        "strong_jam": lambda: (
            EnvConfig(reward_config=RewardConfig(jammer_jamming=3.0)),
            TrainConfig(),
            NetworkConfig(),
        ),
        "big_team": lambda: (
            EnvConfig(n_strikers=4, n_jammers=4, n_targets=4, n_radars=4),
            TrainConfig(),
            NetworkConfig(),
        ),
        "hard_radar": lambda: (
            EnvConfig(radar_range=0.30, radar_kill_probability=1.0),
            TrainConfig(),
            NetworkConfig(),
        ),
        "no_step_pen": lambda: (
            EnvConfig(reward_config=RewardConfig(timestep_penalty=0.0)),
            TrainConfig(),
            NetworkConfig(),
        ),
    }
    if name not in _presets:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(_presets.keys())}")
    return _presets[name]()


PRESETS = list(get_preset.__doc__ or "")  # just for import compat; use get_preset()