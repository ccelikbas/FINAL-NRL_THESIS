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
    """
    Environment parameters: world, agents, sensors, physics, and rewards.
    
    These parameters define the MARL (Multi-Agent RL) task:
    - Observation space: what agents can see
    - Action space: what agents can do
    - Reward signal: what behavior is encouraged (critical for MAPPO convergence)
    
    All distances in normalized coordinates where 1 unit = 1000 km.
    All times in simulation steps where 1 step = 60 seconds (1 minute).
    """

    # ─── Team Composition ────────────────────────────────────────────────────
    # Multi-agent team structure affects network architecture and training stability
    n_strikers: int = 1  # Agents with kinetic (offensive) capability; share policy parameters
    n_jammers: int = 0   # Agents with electronic (defensive) capability; share policy parameters
    n_targets: int = 1   # Objectives to destroy (define episode success criteria)
    n_radars: int = 1    # Environmental threats (create intrinsic difficulty)

    # ─── World / Dynamics ──────────────────────────────────────────────────
    world_bounds: Tuple[float, float] = (0.0, 1.0)  # Normalized coordinate space (0-1000 km)
    dt: float = 1.0      # Time step = 1 minute. Larger dt = discretized actions have bigger effect

    # ─── Agent Kinematics (Physical Model) ───────────────────────────────────
    # Kinematics define action semantics: 2 discrete dimensions (accel, angular_accel), each Categorical(7)
    # Action values: {-1, -0.5, -0.1, 0, +0.1, +0.5, +1} × magnitude
    v_max: float = 0.02
    # Maximum velocity: 20 km/min ≈ Mach 0.95 aircraft cruise speed
    # Affects episode time scale and how far agents can travel
    
    accel_magnitude: float = 0.01
    # Discrete acceleration magnitude per action step
    # Determines maneuverability: how quickly agents can change speed
    
    dpsi_max: float = math.radians(12.0)
    # Maximum heading (yaw) rate: 12°/min at cruise. Realistic turning rates
    # Lower = agents turn slowly (less agile); Higher = more responsive
    
    h_accel_magnitude_fraction: float = 0.1
    # Angular acceleration = dpsi_max × this fraction = 1.2°/min per action
    # Controls yaw acceleration (how quickly agents change turn rate)
    
    min_turn_radius: float = 0.001
    # Minimum turn radius: 50 km. Prevents agents from spinning in tight circles
    # Turn rate is limited by: omega_max = speed / min_turn_radius
    # At v_min (0.01 = 10 km/min): max 11.5°/step turn rate
    # At v_max (0.02 = 20 km/min): max 23°/step (but dpsi_max caps at 12°)
    # Larger = wider turns (less agile); Smaller = tighter turns (more agile)

    # ─── Sensor / Observation (State Representation for MAPPO) ────────────────
    R_obs: float = 1
    # Observation radius: 300 km. Agents see allies/threats within this range
    # Affects network input size and partial observability (policy must infer unseen info)
    # Larger R_obs = more info → easier task; Smaller = harder (agent must predict)

    # ─── Striker Capabilities (Offense dynamics) ─────────────────────────────
    striker_engage_range: float = 0.10
    # Kinetic kill range: 100 km. Strikers destroy targets within this distance
    # Defines sub-goal: agents must navigate close to targets to win
    
    striker_engage_fov: float = 60.0
    # Engagement cone: targets must be in ±30° arc ahead. Refines targeting
    # Requires agents to orient their heading correctly, not just distance
    
    striker_v_min: float = 0.005
    # Minimum cruise speed for strikers: 10 km/min ≈ Mach 0.5
    # Can't hover; must keep moving. Affects energy/momentum constraints

    # ─── Jammer Capabilities (Defense dynamics) ───────────────────────────────
    jammer_jam_radius: float = 0.35
    # Electronic warfare coverage: 150 km. Suppresses radar effectiveness nearby
    # Enables team coordination: jammers create "safe zones" for strikers
    
    jammer_jam_effect: float = 0.15
    # Radar range reduction when jammed: 100 km reduction (0.20 - 0.10 = 0.10 effective)
    # Partially negates threat; doesn't eliminate it (creates strategic depth)
    
    jammer_v_min: float = 0.005
    # Minimum cruise speed (same as strikers)

    # ─── Radar Threat (Environmental pressure) ──────────────────────────────
    radar_range: float = 0.20
    # Baseline radar detection range: 200 km. Agents killed if detected for 1 step
    # Main source of episode failure; drives jamming necessity
    
    radar_kill_probability: float = 0
    # Probability [0,1] of kill per detected step. 1.0 = instant kill if seen
    # Lower values make radars less threatening, easier for learner

    # ─── Reward Shaping (Most critical for MAPPO convergence) ────────────────
    border_thresh: float = 0.05
    # Boundary penalty zone: 50 km from edge. Keeps agents in play area
    # Prevents agents from fleeing to margins to avoid radar

    # ─── Target Spawn Control ─────────────────────────────────────────────
    target_spawn_angle_range: Tuple[float, float] = (270, 360)
    # Angular range (degrees, clockwise from "east" / +x axis) within which
    # targets are spawned around their assigned radar.
    # (0, 360) = full circle (default, legacy behaviour).
    # (180, 360) = bottom half only (angles from south-west to south-east),
    #   which keeps targets away from the top border.
    # The range wraps: (350, 10) means from 350° through 0° to 10°.

    # ─── Environment Layout Control ──────────────────────────────────────────
    n_env_layouts: int = 0
    # Number of pre-generated environment layouts (radar positions).
    # 0 = fully random (new random radar positions every episode reset)
    # 1 = single fixed scenario (same radar positions in every episode)
    # N = N distinct scenarios, cycled across parallel environments
    # Use n_env_layouts > 0 for controlled, reproducible training scenarios

    reward_config: RewardConfig = field(default_factory=RewardConfig)
    # Reward weights. Carefully tuned for MAPPO convergence:
    # - target_destroyed:      sparse team objective (cooperation)
    # - team_spirit:            blends team vs individual reward distribution
    # - striker_approach:       piecewise lin-exp reward toward targets
    # - jammer_approach:        piecewise lin-exp reward toward radars
    # - striker/jammer_nearest_only: reward nearest entity only or all
    # - striker/jammer_progress_scale: (legacy potential-based, 0 = off)
    # - jammer_jam_bonus:       (legacy per-step jamming bonus, 0 = off)
    # - striker/jammer_formation_scale: cross-role cohesion (striker↔jammer only)
    # - border_penalty, radar_avoidance, timestep_penalty: constraints


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """
    Training algorithm, optimization, and data collection parameters for MAPPO.

    ┌─ EPISODE CALCULATION ──────────────────────────────────────────────────┐
    │ frames_per_batch is AUTO-DERIVED: num_envs × max_steps                │
    │   → each iteration collects exactly one full episode per environment  │
    │ Total episodes ≈ num_envs × n_iters                                   │
    │ Change max_steps or num_envs → frames_per_batch updates automatically │
    │ Override by passing an explicit int to frames_per_batch               │
    └────────────────────────────────────────────────────────────────────────┘
    """

    # ─── Rollout / Environment ──────────────────────────────────────────────
    # MAPPO collects transitions from multiple parallel environments for sample efficiency
    num_envs: int = 256
    # [Frames per batch / num_envs] = steps collected from each environment per iteration
    # (frames_per_batch / 256 = max_steps steps per env per iteration)
    
    max_steps: int = 50
    # Maximum steps per episode (100 steps × 1 min/step = 100 minutes per episode)
    # Episodes can terminate early if all targets destroyed or all agents dead

    # ─── Data Collection (Off-policy → On-policy conversion) ────────────────
    # MAPPO uses on-policy learning: collect rollout data, then discard after update
    frames_per_batch: Optional[int] = None
    # Total transitions collected across all num_envs before one policy update.
    # DEFAULT: None → auto-set to num_envs × max_steps in __post_init__.
    #   This means each iteration collects exactly one full episode per environment.
    # OVERRIDE: set an explicit int to control collection size manually.
    
    n_iters: int = 30
    # Number of collect→update cycles. Each cycle collects frames_per_batch transitions
    # Higher = longer training, potential for better convergence

    # ─── PPO Clipping & Advantage Estimation ───────────────────────────────
    # PPO Objective: min( rt * Ât, clip(rt, 1-ε, 1+ε) * Ât )
    # where rt = π_new(a|s) / π_old(a|s), Ât = advantage estimate
    num_epochs: int = 3
    # Number of repeat passes over collected data before next rollout
    # 5 epochs × 12 minibatches = 60 gradient steps per iteration
    
    minibatch_size: int = 1_024
    # Minibatch size for gradient descent within each epoch
    # Smaller = more noisy gradients but faster updates; Larger = fewer updates per epoch
    
    clip_eps: float = 0.2
    # PPO clipping range ε: prevents policy from changing too rapidly
    # clip_eps=0.2 means policy probability ratio clamped to [0.8, 1.2]
    # Standard value; was 0.1 which was too conservative for fast convergence
    
    gamma: float = 0.99
    # Discount factor for returns: R_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...
    # Closer to 1.0 = values distant future rewards (longer horizon, higher variance)
    # Closer to 0.0 = only immediate rewards matter (lower variance, less far-sighted)
    
    lmbda: float = 0.95
    # GAE (Generalized Advantage Estimation) lambda parameter
    # Advantage = λ * TD_error + λ² * TD_error_t+1 + ... (blend of n-step returns)
    # Higher λ → higher bias, lower variance; Lower λ → lower bias, higher variance
    
    entropy_coef: float = 0.02
    # Coefficient for entropy bonus in loss: loss_total = loss_policy + loss_value - entropy_coef * entropy
    # Higher entropy bonus encourages broader exploration of the action space,
    # critical for discrete 7×7 actions with double-integrator dynamics

    # ─── Optimization (Gradient Step Control) ───────────────────────────────
    lr: float = 1e-4
    # Legacy single learning rate (used if actor_lr/critic_lr are None)
    # Typical range: 1e-5 to 1e-3 for policy learning

    actor_lr: float = 5e-4
    # Actor (policy) learning rate. Lower = more conservative policy updates.
    # Typical range: 1e-5 to 5e-4.
    # The actor should update slowly to stay within the PPO trust region.

    critic_lr: float = 1e-4
    # Critic (value function) learning rate. Higher = faster value fitting.
    # Increased from 1e-4 to help critic converge faster → better advantages.
    
    max_grad_norm: float = 1
    # Gradient clipping: if ||∇loss|| > max_grad_norm, rescale to this norm
    # Prevents exploding gradients and stabilizes training
    # Typical values: 0.5 to 1.0

    # ─── Logging / Misc ─────────────────────────────────────────────────────
    seed: int = 0
    # Random seed for reproducibility (RNG for environment, policy init, sampling)
    
    log_every: int = 10
    # Print training stats every N iterations (useful for monitoring convergence)

    eval_episodes: int = 10
    # Number of test-rollout episodes to run after training completes.
    # Results include task completion rate, survival rate, mean reward, and per-step
    # reward component breakdowns.  Set to 0 to skip post-training evaluation.

    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # Device for training: "cuda" (GPU) for speed, "cpu" for debugging

    def __post_init__(self):
        # Auto-derive frames_per_batch so changing max_steps or num_envs is enough.
        if self.frames_per_batch is None:
            self.frames_per_batch = int(self.num_envs * self.max_steps)
        # Ensure integer types (guards against accidental float literals like 200/4)
        self.max_steps         = int(self.max_steps)
        self.frames_per_batch  = int(self.frames_per_batch)


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NetworkConfig:
    """
    Neural network architecture parameters for actor and critic networks.
    
    Both actor (decentralised) and critic (centralised) use MLPs with this hidden layer width.
    """
    hidden: int = 256
    # Hidden layer width for all MLP layers in actor/critic networks
    # Larger = more expressive (can learn complex policies) but slower training & more memory
    # Typical range: 64-512. Increase if agent struggles to learn; Decrease if training is slow


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
            TrainConfig(n_iters=20, num_envs=64),
            NetworkConfig(hidden=128),
        ),
        "high_kill": lambda: (
            EnvConfig(reward_config=RewardConfig(target_destroyed=20.0)),
            TrainConfig(),
            NetworkConfig(),
        ),
        "strong_jam": lambda: (
            EnvConfig(reward_config=RewardConfig(jammer_approach_w_lin=0.5, jammer_approach_w_exp=1.0)),
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