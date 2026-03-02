"""
config.py – Single source of truth for every experiment parameter.

Edit values here, then run:
    python run.py                   # quick default training
    python run.py --preset fast     # tiny run for smoke-testing
    python run.py --preset sweep    # sensitivity sweep over reward weights

All tunable knobs live in three dataclasses:
  ─ EnvConfig     world / sensor / reward-shaping settings
  ─ TrainConfig   PPO / collector hyper-parameters
  ─ NetworkConfig architecture settings
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from strike_ea.env.rewards import RewardConfig


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    # --- agents / entities ---
    n_strikers: int   = 2
    n_jammers:  int   = 2
    n_targets:  int   = 2
    n_radars:   int   = 2

    # --- world ---
    world_bounds: Tuple[float, float] = (0.0, 1.0)
    dt:           float = 1.0

    # --- kinematics ---
    v_max:    float = 0.02
    dpsi_max: float = math.radians(20.0)   # max heading-rate per step (rad)

    # --- sensors ---
    R_obs:       float = 0.50   # observation radius
    radar_range: float = 0.20   # baseline radar kill range
    radar_kill_probability: float = 1.0  # probability [0, 1] that an agent in radar range is killed per step

    # --- reward shaping ---
    border_thresh: float = 0.05  # agents within this distance of border get penalised

    # --- reward weights (override any individual field below) ---
    reward_config: RewardConfig = field(default_factory=RewardConfig)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # --- environment / rollout ---
    num_envs:         int   = 256
    max_steps:        int   = 160    # max steps per episode

    # --- data collection ---
    frames_per_batch: int   = 8_192
    n_iters:          int   = 10     # number of collection → update cycles

    # --- PPO ---
    num_epochs:       int   = 10
    minibatch_size:   int   = 1_024
    clip_eps:         float = 0.2
    gamma:            float = 0.99
    lmbda:            float = 0.95   # GAE lambda
    entropy_coef:     float = 1e-3

    # --- optimiser ---
    lr:               float = 3e-4
    max_grad_norm:    float = 1.0

    # --- misc ---
    seed:             int   = 0
    log_every:        int   = 10     # print a summary line every N iterations
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Network architecture
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NetworkConfig:
    hidden: int = 256   # hidden layer width for both actor and critic MLP


# ─────────────────────────────────────────────────────────────────────────────
# Named presets  (edit or add new ones freely)
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: dict[str, tuple[EnvConfig, TrainConfig, NetworkConfig]] = {

    # ── Minimal smoke-test (runs in seconds) ─────────────────────────────────
    "fast": (
        EnvConfig(),
        TrainConfig(num_envs=32, frames_per_batch=512, n_iters=3,
                    num_epochs=2, minibatch_size=128, max_steps=40),
        NetworkConfig(hidden=64),
    ),

    # ── Default full training run ─────────────────────────────────────────────
    "default": (
        EnvConfig(),
        TrainConfig(),
        NetworkConfig(),
    ),

    # ── Aggressive kill rewards (are strikers more decisive?) ─────────────────
    "high_kill": (
        EnvConfig(reward_config=RewardConfig(target_destroyed=25.0, agent_destroyed=-5.0)),
        TrainConfig(),
        NetworkConfig(),
    ),

    # ── Strong jamming incentive (do jammers prioritise suppression?) ─────────
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

    # ── Tight radar net (harder env) ─────────────────────────────────────────
    "hard_radar": (
        EnvConfig(radar_range=0.35, border_thresh=0.08,
                  reward_config=RewardConfig(border=-2.0)),
        TrainConfig(n_iters=20),
        NetworkConfig(),
    ),

    # ── Sensitivity: step penalty off ────────────────────────────────────────
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
