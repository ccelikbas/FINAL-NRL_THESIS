from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from .rewards import RewardConfig


# ======================================================================
# FOFE architecture configuration
# ======================================================================

@dataclass
class FOFEConfig:
    """Configuration for Flexible Observation Feature Encoding (FOFE).

    FOFE replaces fixed top-K slot encoding with a permutation-invariant
    set encoder.  Each observation channel (agents, targets, radars) passes
    through its own FOFE block:

        entities [n, d_in] → SEE stack → per-entity MLP → global maxpool → [D_fofe]

    The self-observation bypasses FOFE and uses a small MLP.
    All channel outputs are concatenated and fused by a fusion MLP.

    Dimension trace example (2 SEE layers, h=[96,128], mlp=[512,96]):
        Input:          [n, d_in]   e.g. [n, 4] for agents
        SEE layer 1:    [n, 2×96]  = [n, 192]
        SEE layer 2:    [n, 2×128] = [n, 256]
        Per-entity MLP: [n, 512] → [n, 96]
        MaxPool:        [96]              (fixed regardless of n)

    Parameters
    ----------
    use_fofe : bool
        Master switch.  False = legacy flat-obs MLP.  True = FOFE encoder.

    agents_see_dims : tuple of int
        Hidden dim per SEE layer for agents channel.  Length = N_SEE.
        Each SEE layer i projects [d_prev → h_i], outputs [n, 2*h_i].
        Wang (2026) uses (96, 128).

    targets_see_dims, radars_see_dims : tuple of int
        Same as agents_see_dims but for targets / radars channels.

    fofe_mlp_dims : tuple of int
        Per-entity MLP after the SEE stack, applied to each row.
        Input dim = 2 * last_see_dim.  Output = fofe_mlp_dims[-1] = D_fofe.
        Wang uses (512, 96).

    self_mlp_dims : tuple of int
        MLP for the fixed-size self-obs (6 floats).
        Output dim = self_mlp_dims[-1].

    fusion_mlp_dims : tuple of int
        Fusion MLP after concatenating all channel outputs.
        Input = self_out + 3 × D_fofe.  Wang uses (256, 256).

    critic_agents_see_dims, critic_targets_see_dims, critic_radars_see_dims :
        SEE dims for the critic FOFE (processes global state as entity sets).
        Defaults to same as actor.

    critic_fofe_mlp_dims : tuple of int
        Per-entity MLP dims for critic FOFE blocks.

    critic_self_mlp_dims : tuple of int
        Self-obs MLP dims for critic.  Critic self = concatenation of all
        agent self-features (7 per agent), so input is larger than actor.

    critic_fusion_mlp_dims : tuple of int
        Fusion MLP dims for critic.
    """
    use_fofe: bool = False

    # --- Actor FOFE dims ---
    agents_see_dims:   Tuple[int, ...] = (96, 128)
    targets_see_dims:  Tuple[int, ...] = (96, 128)
    radars_see_dims:   Tuple[int, ...] = (96, 128)
    fofe_mlp_dims:     Tuple[int, ...] = (512, 96)
    self_mlp_dims:     Tuple[int, ...] = (64, 64)
    fusion_mlp_dims:   Tuple[int, ...] = (256, 256)

    # --- Critic FOFE dims (default = same as actor) ---
    critic_agents_see_dims:   Tuple[int, ...] = (96, 128)
    critic_targets_see_dims:  Tuple[int, ...] = (96, 128)
    critic_radars_see_dims:   Tuple[int, ...] = (96, 128)
    critic_fofe_mlp_dims:     Tuple[int, ...] = (512, 96)
    critic_fusion_mlp_dims:   Tuple[int, ...] = (256, 256)


# ======================================================================
# Environment configuration
# ======================================================================

@dataclass
class EnvConfig:
    # Team composition
    n_strikers: int = 1
    n_jammers: int = 1
    n_known_targets: int = 1
    n_unknown_targets: int = 1
    n_known_radars: int = 1
    n_unknown_radars: int = 1
    n_targets: int = 0
    n_radars: int = 0

    # World / episode
    world_bounds: Tuple[float, float] = (0.0, 1.0)
    dt: float = 1.0
    max_steps: int = 150
    n_env_layouts: int = 0
    target_spawn_angle_range: Tuple[float, float] = (0, 360)

    # Kinematics
    v_max: float = 0.02
    accel_magnitude: float = 0.01
    dpsi_max: float = math.radians(12.0)
    h_accel_magnitude_fraction: float = 0.1
    min_turn_radius: float = 0.001

    # Sensors
    R_obs: float = 0.4
    R_comm: float = 0.6

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

    # FOFE mode (set by ExperimentConfig.finalize)
    use_fofe: bool = False

    def __post_init__(self):
        self.n_known_targets = int(self.n_known_targets)
        self.n_unknown_targets = int(self.n_unknown_targets)
        self.n_known_radars = int(self.n_known_radars)
        self.n_unknown_radars = int(self.n_unknown_radars)

        if self.n_known_targets < 0 or self.n_unknown_targets < 0:
            raise ValueError("n_known_targets and n_unknown_targets must be >= 0")
        if self.n_known_radars < 0 or self.n_unknown_radars < 0:
            raise ValueError("n_known_radars and n_unknown_radars must be >= 0")

        if self.n_known_targets == 0 and self.n_unknown_targets == 0:
            self.n_targets = int(self.n_targets)
            self.n_known_targets = int(self.n_targets)
            self.n_unknown_targets = 0
        else:
            self.n_targets = int(self.n_known_targets + self.n_unknown_targets)

        if self.n_known_radars == 0 and self.n_unknown_radars == 0:
            self.n_radars = int(self.n_radars)
            self.n_known_radars = int(self.n_radars)
            self.n_unknown_radars = 0
        else:
            self.n_radars = int(self.n_known_radars + self.n_unknown_radars)


@dataclass
class PPOConfig:
    """Shared PPO hyperparameters for both striker and jammer MAPPO."""
    num_envs: int = 512
    n_iters: int = 200
    frames_per_batch: Optional[int] = None
    num_epochs: int = 10
    minibatch_size: int = 2048

    gamma: float = 0.99
    lmbda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    normalize_rewards: bool = True

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 1.0

    seed: int = 0
    log_every: int = 20
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
    fofe: FOFEConfig = field(default_factory=FOFEConfig)

    def finalize(self):
        if self.ppo.frames_per_batch is None:
            self.ppo.frames_per_batch = int(self.ppo.num_envs * self.env.max_steps)
        # Propagate FOFE flag into EnvConfig so the env knows to emit FOFE obs
        self.env.use_fofe = self.fofe.use_fofe
        return self
