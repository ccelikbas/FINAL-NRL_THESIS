from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from .rewards import RewardConfig


@dataclass
class EnvConfig:
    # Team composition
    n_strikers: int = 2
    n_jammers: int = 2
    n_targets: int = 2
    n_radars: int = 2

    # World / episode
    world_bounds: Tuple[float, float] = (0.0, 1.0)
    dt: float = 1.0
    max_steps: int = 100
    n_env_layouts: int = 0
    target_spawn_angle_range: Tuple[float, float] = (0, 360)

    # Kinematics
    v_max: float = 0.02
    accel_magnitude: float = 0.01
    dpsi_max: float = math.radians(12.0)
    h_accel_magnitude_fraction: float = 0.1
    min_turn_radius: float = 0.001

    # Sensors
    R_obs: float = 1.0

    # Strikers
    striker_engage_range: float = 0.10
    striker_engage_fov: float = 60.0
    striker_v_min: float = 0.005

    # Jammers
    jammer_jam_radius: float = 0.35
    jammer_jam_effect: float = 0.15
    jammer_v_min: float = 0.005

    # Threats
    radar_range: float = 0.20
    radar_kill_probability: float = 1

    # Rewards
    border_thresh: float = 0.05
    reward_config: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class PPOConfig:
    num_envs: int = 256
    n_iters: int = 100
    frames_per_batch: Optional[int] = None
    num_epochs: int = 10
    minibatch_size: int = 2048

    gamma: float = 0.99
    lmbda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0 #1e-3

    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    max_grad_norm: float = 1.0

    seed: int = 0
    log_every: int = 10
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __post_init__(self):
        self.num_envs = int(self.num_envs)
        self.n_iters = int(self.n_iters)
        self.num_epochs = int(self.num_epochs)
        self.minibatch_size = int(self.minibatch_size)


@dataclass
class NetworkConfig:
    actor_hidden: int = 256
    critic_hidden: int = 256
    depth: int = 3


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    net: NetworkConfig = field(default_factory=NetworkConfig)

    def finalize(self):
        if self.ppo.frames_per_batch is None:
            self.ppo.frames_per_batch = int(self.ppo.num_envs * self.env.max_steps)
        return self

