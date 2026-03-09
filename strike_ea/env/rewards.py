from dataclasses import dataclass


@dataclass
class RewardConfig:
    """
    Reward weight configuration for MAPPO.

    Dense rewards use a **piecewise linear-exponential** shaping function:

        f(d) = ┌ 0                                               if d ≥ d_max
               │ w_lin × (d_max − d) / (d_max − d_knee)          if d_knee ≤ d < d_max
               └ w_lin + w_exp × (e^{α(1 − d/d_knee)} − 1)      if d < d_knee

    • Linear region: gentle gradient at large distances.
    • Exponential region: steep gradient near the entity of interest.
    • Continuous at d_knee (exponential term = 0 at the boundary).

    Sign convention:
      Approach rewards (striker→target, jammer→radar) are *positive*.
      Avoidance penalties (border, radar zone) are *negative* (−f).

    Tuning guide:
    - w_lin : overall magnitude in the linear region
    - w_exp : additional magnitude in the exponential region
    - d_knee: distance where the exponential kicks in (smaller = sharper)
    - alpha : steepness of the exponential (higher = steeper)
    - d_max : distance beyond which shaping is zero (approach rewards use
              the map diagonal; border uses border_thresh; radar zone uses
              radar_avoid_d_max)
    """

    # ─── SPARSE TEAM REWARDS ─────────────────────────────────────────────────
    target_destroyed: float = 200
    # Shared equally among alive agents when a target is killed.

    timestep_penalty: float = -0.05
    # Small per-step cost for every alive agent (encourages finishing fast).

    agent_destroyed: float = -100
    # One-time penalty applied to an agent the step it is killed by a radar.

    # ─── BORDER AVOIDANCE  (piecewise lin-exp penalty, per alive agent) ──────
    # d = distance from nearest map edge.  d_max = border_thresh (EnvConfig).
    border_w_lin:  float = 0.5
    border_w_exp:  float = 1.0
    border_d_knee: float = 0.01   # 20 km from edge → exponential kicks in
    border_alpha:  float = 3.0

    # ─── RADAR ZONE AVOIDANCE  (piecewise lin-exp penalty, ALL agents) ───────
    # d = distance from nearest radar detection-zone boundary (clamped ≥ 0).
    # Agents inside the zone get killed and receive agent_destroyed instead.
    radar_avoid_w_lin:  float = 0.1
    radar_avoid_w_exp:  float = 0.5
    radar_avoid_d_max:  float = 0.1  # penalty starts 100 km from zone edge
    radar_avoid_d_knee: float = 0.1  # 30 km from zone → exponential
    radar_avoid_alpha:  float = 0.3

    # ─── STRIKER PROGRESS  (potential-based: reward ∝ reduction in distance to best target) ──
    # Each step: reward += striker_progress_scale × max_j(prev_dist_j − curr_dist_j)
    # Positive when striker moves closer to any alive target, negative when moving away.
    # At max speed (~0.02 units/step) × scale=5 → up to ~0.10 reward/step.
    striker_progress_scale: float = 5.0

    # ─── JAMMER ACTIVE-JAMMING  (binary reward per step of active jamming) ─────
    # Reward given each step a jammer is within jam_radius of at least one radar.
    # Replaces proximity shaping: jammers are only rewarded for actually jamming.
    jammer_active_reward: float = 0.2




'''
RUN: Visualisation of the piecewise linear-exponential shaping function using current paramters:
'''
from typing import Tuple
import torch
import matplotlib.pyplot as plt
import numpy as np


def _piecewise_lin_exp(d: torch.Tensor, d_max: float, d_knee: float,
                       w_lin: float, w_exp: float, alpha: float) -> torch.Tensor:
    """Piecewise linear-exponential shaping."""
    t_lin = ((d_max - d) / (d_max - d_knee + 1e-8)).clamp(0.0, 1.0)
    lin_val = w_lin * t_lin

    t_exp = ((d_knee - d) / (d_knee + 1e-8)).clamp(0.0, 1.0)
    exp_val = w_exp * (torch.exp(alpha * t_exp) - 1.0)

    return lin_val + exp_val


def plot_reward_functions(reward_config: RewardConfig, distance_range: Tuple[float, float]):
    """
    Plot reward shaping functions for avoidance components.

    Note: Striker reward is now potential-based progress (prev_dist - curr_dist),
    and jammer reward is binary per-step active-jamming — neither is a function
    of distance alone, so they are not plotted here.
    """
    d = torch.linspace(distance_range[0], distance_range[1], 1000)

    # Radar avoidance penalty
    radar = -_piecewise_lin_exp(
        d,
        d_max=reward_config.radar_avoid_d_max,
        d_knee=reward_config.radar_avoid_d_knee,
        w_lin=reward_config.radar_avoid_w_lin,
        w_exp=reward_config.radar_avoid_w_exp,
        alpha=reward_config.radar_avoid_alpha,
    )

    # Border avoidance penalty
    border_dmax = distance_range[1]
    border = -_piecewise_lin_exp(
        d,
        d_max=border_dmax,
        d_knee=reward_config.border_d_knee,
        w_lin=reward_config.border_w_lin,
        w_exp=reward_config.border_w_exp,
        alpha=reward_config.border_alpha,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(d.numpy(), radar.numpy(), label="Radar Avoidance")
    plt.plot(d.numpy(), border.numpy(), label="Border Avoidance")
    plt.axhline(0)
    plt.xlabel("Distance")
    plt.ylabel("Reward / Penalty")
    plt.title("Reward Shaping Functions (avoidance only)")
    plt.legend()
    plt.grid(True)
    plt.show()

# reward_config = RewardConfig()
# plot_reward_functions(reward_config, distance_range=(0, 0.5))