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
    target_destroyed: float = 1
    # Reward when a target is killed.
    # Distribution controlled by team_spirit parameter (see below).

    timestep_penalty: float = -0.1
    # Per-step cost per alive agent. Kept low relative to approach reward
    # so the per-step signal is clearly positive when approaching.
    # Over 50 steps × 2 agents = −5 total (10% of kill reward).

    agent_destroyed: float = -1
    # Penalty applied when an agent is killed by a radar.
    # Distribution controlled by team_spirit parameter (see below).

    # ─── TEAM vs INDIVIDUAL REWARD DISTRIBUTION ─────────────────────────────
    team_spirit: float = 0
    # Controls how team rewards (target_destroyed, agent_destroyed) are
    # distributed among agents:
    #   1.0 = fully shared equally across all alive agents (cooperative)
    #   0.0 = only the individual agent responsible (the striker that killed
    #          the target, or the agent that was destroyed)
    #   0.5 = 50% shared + 50% individual
    # Formula: reward_i = team_spirit × (team_avg) + (1 - team_spirit) × (individual_i)
    # 0.8: jammer receives 80% of kill credit → incentivises it to enable the striker,
    # which is the primary driver for agents flying together toward the objective.
    # timestep_penalty is always per alive agent (not affected by team_spirit).

    # ─── BORDER AVOIDANCE  (piecewise lin-exp penalty, per alive agent) ──────
    # d = distance from nearest map edge.  d_max = border_thresh (EnvConfig).
    border_w_lin:  float = 0   # gentle early-warning ramp (50 km → 30 km from edge)
    border_w_exp:  float = 0
    border_d_knee: float = 0   # 30 km from edge → exponential kicks in
    border_alpha:  float = 0

    # ─── RADAR ZONE AVOIDANCE  (piecewise lin-exp penalty, ALL agents) ───────
    # d = distance from *effective* radar zone boundary (adapts when jammed).
    # Agents inside the zone get killed and receive agent_destroyed instead.
    radar_avoid_w_lin:  float = 0  # reduced so approach reward dominates
    radar_avoid_w_exp:  float = 0
    radar_avoid_d_max:  float = 0   # penalty starts 200 km outside zone edge
    radar_avoid_d_knee: float = 0   # 30 km from zone → exponential
    radar_avoid_alpha:  float = 0

    # ─── STRIKER APPROACH  (piecewise lin-exp reward toward targets) ──────────
    # Positive reward that increases as striker gets closer to alive targets.
    # d_max=1.0 covers the full map width so the approach reward is nonzero
    # from ANY starting position. w_lin=0.5 ensures the linear-region reward
    # exceeds the timestep penalty even at mid-map distances:
    #   f(0.5) = 0.5 × (1.0−0.5)/(1.0−0.15) = 0.29 > 0.05 ✓
    striker_approach_w_lin:  float = 0.1
    striker_approach_w_exp:  float = 0    # strong pull into engage range
    striker_approach_d_max:  float = 1    # spans FULL map width (was 0.5)
    striker_approach_d_knee: float = 0   # exponential onset near engage range (0.10)
    striker_approach_alpha:  float = 0
    striker_nearest_only:    bool  = True
    # True  = reward based only on distance to nearest alive target
    # False = reward = mean over all alive targets (encourages approaching all)

    # ─── JAMMER APPROACH  (piecewise lin-exp reward toward radars) ────────────
    # Positive reward that increases as jammer gets closer to radars.
    # Matched to striker approach scale so both agents have equally strong
    # gradients pulling them toward their respective objectives.
    jammer_approach_w_lin:  float = 0.1    # matched to striker
    jammer_approach_w_exp:  float = 0    # moderate exponential near jam range
    jammer_approach_d_max:  float = 1    # spans full map (was 0.5)
    jammer_approach_d_knee: float = 0    # exponential onset inside jam_radius (0.35)
    jammer_approach_alpha:  float = 0
    jammer_nearest_only:    bool  = True
    # True  = reward based only on distance to nearest radar
    # False = reward = mean over all radars (encourages approaching all)

    # ─── POTENTIAL-BASED PROGRESS  (per-step velocity signal) ────────────────
    # Reward = scale × (prev_dist − curr_dist). At v_max=0.02, each step of
    # direct approach yields scale × 0.02. With scale=5: +0.10 per step —
    # a clear "you moved in the right direction" signal that helps the policy
    # gradient identify which actions reduce distance (especially important
    # with double-integrator heading dynamics).
    striker_progress_scale: float = 0
    jammer_progress_scale:  float = 0

    # ─── JAMMER ACTIVE-JAMMING BONUS  (deactivated by default) ───────────────
    # Per-step bonus when a jammer is within jam_radius of ≥ 1 radar.
    jammer_jam_bonus: float = 0   # Deactivated

    # ─── FORMATION COHESION  (striker ↔ jammer cross-role proximity) ──────────
    # Each striker is rewarded for being close to the nearest *alive jammer*,
    # and each jammer for being close to the nearest *alive striker*.
    # Same-role proximity (striker↔striker, jammer↔jammer) is NOT rewarded.
    # Works for any ns × nj configuration (1+1, 1+2, 2+2, …).
    #
    # Reward = scale × max(0, 1 − d_nearest_cross_role / ref_dist)
    # Set scale to 0.0 to disable for that role independently.
    striker_formation_scale:    float = 0.   # reward to each striker for being near a jammer
    striker_formation_ref_dist: float = 0    # distance (map units) beyond which reward = 0

    jammer_formation_scale:     float = 0   # reward to each jammer for being near a striker
    jammer_formation_ref_dist:  float = 0    # distance (map units) beyond which reward = 0




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
    Plot reward shaping functions for all visualisable distance-based components.

    Shows the piecewise lin-exp curves for:
      - Striker approach (positive, toward targets)
      - Jammer approach  (positive, toward radars)
      - Radar avoidance  (negative, from radar zone boundary)
      - Border avoidance  (negative, from map edge)
      - Formation cohesion (positive, toward nearest ally)
    """
    d = torch.linspace(distance_range[0], distance_range[1], 1000)

    # Striker approach reward (positive)
    striker_app = _piecewise_lin_exp(
        d,
        d_max=reward_config.striker_approach_d_max,
        d_knee=reward_config.striker_approach_d_knee,
        w_lin=reward_config.striker_approach_w_lin,
        w_exp=reward_config.striker_approach_w_exp,
        alpha=reward_config.striker_approach_alpha,
    )

    # Jammer approach reward (positive)
    jammer_app = _piecewise_lin_exp(
        d,
        d_max=reward_config.jammer_approach_d_max,
        d_knee=reward_config.jammer_approach_d_knee,
        w_lin=reward_config.jammer_approach_w_lin,
        w_exp=reward_config.jammer_approach_w_exp,
        alpha=reward_config.jammer_approach_alpha,
    )

    # Radar avoidance penalty (negative)
    radar = -_piecewise_lin_exp(
        d,
        d_max=reward_config.radar_avoid_d_max,
        d_knee=reward_config.radar_avoid_d_knee,
        w_lin=reward_config.radar_avoid_w_lin,
        w_exp=reward_config.radar_avoid_w_exp,
        alpha=reward_config.radar_avoid_alpha,
    )

    # Border avoidance penalty (negative)
    border_dmax = distance_range[1]
    border = -_piecewise_lin_exp(
        d,
        d_max=border_dmax,
        d_knee=reward_config.border_d_knee,
        w_lin=reward_config.border_w_lin,
        w_exp=reward_config.border_w_exp,
        alpha=reward_config.border_alpha,
    )

    # Formation proximity reward — striker side (linear decay, cross-role)
    striker_form = reward_config.striker_formation_scale * (
        1.0 - d / reward_config.striker_formation_ref_dist
    ).clamp(min=0.0)
    # Formation proximity reward — jammer side (linear decay, cross-role)
    jammer_form = reward_config.jammer_formation_scale * (
        1.0 - d / reward_config.jammer_formation_ref_dist
    ).clamp(min=0.0)

    plt.figure(figsize=(12, 7))
    plt.plot(d.numpy(), striker_app.numpy(), label="Striker Approach", color="#1f77b4", linewidth=2)
    plt.plot(d.numpy(), jammer_app.numpy(), label="Jammer Approach", color="#17becf", linewidth=2)
    plt.plot(d.numpy(), radar.numpy(), label="Radar Avoidance", color="#9467bd", linewidth=2)
    plt.plot(d.numpy(), border.numpy(), label="Border Avoidance", color="#d62728", linewidth=2)
    plt.plot(d.numpy(), striker_form.numpy(), label="Striker Formation (↔ jammer)", color="#8c564b", linewidth=2)
    plt.plot(d.numpy(), jammer_form.numpy(),  label="Jammer Formation (↔ striker)",  color="#bcbd22", linewidth=2, linestyle="--")
    plt.axhline(0, color="gray", lw=0.5)
    plt.xlabel("Distance")
    plt.ylabel("Reward / Penalty")
    plt.title("Reward Shaping Functions (piecewise lin-exp)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    reward_config = RewardConfig()
    plot_reward_functions(reward_config, distance_range=(0, 0.5))