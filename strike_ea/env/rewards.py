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
    target_destroyed: float = 10.0
    # Shared equally among alive agents when a target is killed.
    # Scaled to be ~1-2× total dense reward per episode.

    timestep_penalty: float = -0.02
    # Small per-step cost for every alive agent (encourages finishing fast).
    # Budget: 200 steps × 2 agents × 0.02 = 8 total. Moderate pressure.

    agent_destroyed: float = -5.0
    # One-time penalty applied to an agent the step it is killed by a radar.
    # Comparable to ~50 steps of progress reward.

    # ─── BORDER AVOIDANCE  (piecewise lin-exp penalty, per alive agent) ──────
    # d = distance from nearest map edge.  d_max = border_thresh (EnvConfig).
    border_w_lin:  float = 0.3
    border_w_exp:  float = 0.5
    border_d_knee: float = 0.01   # 10 km from edge → exponential kicks in
    border_alpha:  float = 3.0

    # ─── RADAR ZONE AVOIDANCE  (piecewise lin-exp penalty, ALL agents) ───────
    # d = distance from *effective* radar zone boundary (adapts when jammed).
    # Agents inside the zone get killed and receive agent_destroyed instead.
    radar_avoid_w_lin:  float = 0.2
    radar_avoid_w_exp:  float = 0.5
    radar_avoid_d_max:  float = 0.10   # penalty starts 100 km outside zone edge
    radar_avoid_d_knee: float = 0.03   # 30 km from zone → exponential
    radar_avoid_alpha:  float = 3.0

    # ─── STRIKER PROGRESS  (potential-based toward best alive target) ─────────
    # Each step: reward = scale × max_j(prev_dist_j − curr_dist_j)
    # At v_max (0.02/step) × scale=5 → up to ~0.10 reward/step.
    striker_progress_scale: float = 5.0

    # ─── JAMMER PROGRESS  (potential-based toward nearest radar) ──────────────
    # Each step: reward = scale × (prev_nearest_dist − curr_nearest_dist)
    # Gives jammer a gradient to approach radars even when far away.
    jammer_progress_scale: float = 5.0

    # ─── JAMMER ACTIVE-JAMMING BONUS  (per step actively suppressing) ────────
    # Extra reward each step a jammer is within jam_radius of ≥ 1 radar.
    # Stacks on top of jammer progress; rewards actually performing the mission.
    jammer_jam_bonus: float = 0.1

    # ─── FORMATION COHESION  (proximity to nearest alive ally) ───────────────
    # Reward = scale × max(0, 1 − d_nearest_ally / ref_dist) per alive agent.
    # Encourages striker and jammer to move as a pair / close formation.
    formation_scale:    float = 0.02
    formation_ref_dist: float = 0.15   # 150 km — reward decays to 0 beyond this




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
    Plot reward shaping functions for all visualisable components.

    Dense potential-based rewards (striker progress, jammer progress) are
    proportional to Δdist per step and cannot be drawn as a distance curve.
    This plot shows the avoidance penalty profiles.
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

    # Formation proximity reward (static, not potential-based)
    form_rew = reward_config.formation_scale * (
        1.0 - d / reward_config.formation_ref_dist
    ).clamp(min=0.0)

    plt.figure(figsize=(10, 6))
    plt.plot(d.numpy(), radar.numpy(), label="Radar Avoidance", color="#9467bd")
    plt.plot(d.numpy(), border.numpy(), label="Border Avoidance", color="#d62728")
    plt.plot(d.numpy(), form_rew.numpy(), label="Formation Cohesion", color="#8c564b")
    plt.axhline(0, color="gray", lw=0.5)
    plt.xlabel("Distance")
    plt.ylabel("Reward / Penalty")
    plt.title("Reward Shaping Functions")
    plt.legend()
    plt.grid(True)
    plt.show()

# reward_config = RewardConfig()
# plot_reward_functions(reward_config, distance_range=(0, 0.5))