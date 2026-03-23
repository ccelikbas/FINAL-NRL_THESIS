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
            Approach shaping (striker→target, jammer→radar) is flipped into a
            distance penalty: it is **0 at d = 0** and becomes more negative as
            distance increases.
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
    target_destroyed: float = 3
    # Reward when a target is killed.
    # Distribution controlled by team_spirit parameter (see below).

    timestep_penalty: float = -0.01
    # Per-step cost per alive agent. Kept low relative to approach reward
    # so the per-step signal is clearly positive when approaching.
    # Over 50 steps × 2 agents = −5 total (10% of kill reward).

    agent_destroyed: float = -3
    # Penalty applied when an agent is killed by a radar.
    # Distribution controlled by team_spirit parameter (see below).

    # ─── TEAM vs INDIVIDUAL REWARD DISTRIBUTION ─────────────────────────────
    team_spirit: float = 0.5
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
    border_d_max:  float = 0.05
    border_w_lin:  float = 0.05   # gentle early-warning ramp (50 km → 30 km from edge)
    border_w_exp:  float = 0
    border_d_knee: float = 0   # 30 km from edge → exponential kicks in
    border_alpha:  float = 0

    # ─── RADAR ZONE AVOIDANCE  (piecewise lin-exp penalty, ALL agents) ───────
    # d = distance from a FIXED jammed radar boundary (non-adaptive).
    # This shaping does not switch with live jamming state; it always uses
    # the jammed effective range as the avoidance boundary.
    # Agents inside lethal radar range are still handled by agent_destroyed.
    radar_avoid_w_lin:  float = 0.01  # reduced so approach reward dominates
    radar_avoid_w_exp:  float = 0
    radar_avoid_d_max:  float = 0.2   # penalty starts 200 km outside zone edge
    radar_avoid_d_knee: float = 0   # 30 km from zone → exponential
    radar_avoid_alpha:  float = 0

    # ─── STRIKER APPROACH  (piecewise lin-exp distance penalty toward targets) ──────────
    # Flipped from a positive approach reward into a penalty that is 0 at d=0
    # and becomes more negative as the striker gets farther from alive targets.
    # The same shape and scale are kept; only the sign/baseline are flipped.
    striker_approach_w_lin:  float = 0.1
    striker_approach_w_exp:  float = 0    # same scale as before, now as penalty
    striker_approach_d_max:  float = 1    # spans FULL map width (was 0.5)
    striker_approach_d_knee: float = 0    # exponential onset near engage range (0.10)
    striker_approach_alpha:  float = 0
    striker_nearest_only:    bool  = True
    # True  = penalty based only on distance to nearest alive target
    # False = penalty = mean over all alive targets (same aggregation as before)

    # ─── JAMMER APPROACH  (piecewise lin-exp distance penalty toward radars) ────────────
    # Flipped from a positive approach reward into a penalty that is 0 at d=0
    # and becomes more negative as the jammer gets farther from radars.
    # Matched to striker approach scale so both agents have equally strong
    # gradients, but now in the away-from-target direction.
    jammer_approach_w_lin:  float = 0    # matched to striker
    jammer_approach_w_exp:  float = 0    # same scale as before, now as penalty
    jammer_approach_d_max:  float = 0    # spans full map (was 0.5)
    jammer_approach_d_knee: float = 0    # exponential onset inside jam_radius (0.35)
    jammer_approach_alpha:  float = 0
    jammer_nearest_only:    bool  = False
    # True  = penalty based only on distance to nearest radar
    # False = penalty = mean over all radars (same aggregation as before)

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
    # Each striker/jammer receives a distance penalty for being far from the
    # nearest alive cross-role teammate.  Flipped sign convention (same as
    # striker approach): 0 at d=0, becomes more negative as distance grows.
    # Penalty = scale × (max(0, 1 − d / ref_dist) − 1)
    #         = 0 at d=0,  −scale at d ≥ ref_dist.
    # Works for any ns × nj configuration (1+1, 1+2, 2+2, …).
    # Set scale to 0.0 to disable for that role independently.
    striker_formation_scale:    float = 0   # reward to each striker for being near a jammer
    striker_formation_ref_dist: float = 0    # distance (map units) beyond which reward = 0

    jammer_formation_scale:     float = 0.1   # reward to each jammer for being near a striker
    jammer_formation_ref_dist:  float = 0.5    # distance (map units) beyond which reward = 0

    # ─── OPTIONAL PAPER-STYLE MISSION REWARD ────────────────────────────────
    # R_mission = -Reward_fn(n_targets_alive, n_targets_initial)
    #             +Reward_fn(n_agents_alive, n_agents_initial)
    # Reward is distributed to all currently alive agents each timestep.
    use_paper_mission_reward: bool = False
    mission_reward_weight: float = 0.02

    # ─── SAME-ROLE SEPARATION PENALTY  (piecewise lin-exp penalty, per agent) ──────────
    # d = distance from each agent to its nearest alive same-role teammate.
    # Penalty = −f(d) where f is the piecewise lin-exp shaping function.
    # Agents with no teammates (ns=1 or nj=1, or all same-role teammates dead) receive 0 penalty.
    # Use this to discourage clustering among strikers (or jammers), promoting spatial diversity
    # and reducing mutual interference. Set all parameters to 0.0 to disable.
    # Typical d_max: 0.2–0.3 (map units, e.g., 2–3× typical agent spacing).
    # Typical w_lin, w_exp: 0.01–0.05 (gentle penalty relative to approach rewards).
    striker_sep_d_max:  float = 0.1
    striker_sep_d_knee: float = 0.0
    striker_sep_w_lin:  float = 0.05
    striker_sep_w_exp:  float = 0.0
    striker_sep_alpha:  float = 0.0

    jammer_sep_d_max:  float = 0.1
    jammer_sep_d_knee: float = 0.0
    jammer_sep_w_lin:  float = 0.05
    jammer_sep_w_exp:  float = 0.0
    jammer_sep_alpha:  float = 0.0

    # ─── CONTROL EFFORT PENALTY  (per alive agent, per step) ───────────────
    # Penalises large control actions to encourage smooth trajectories.
    # Penalty = −accel_effort_scale × accel² − angular_effort_scale × angular_accel²
    # where accel and angular_accel are the discrete multipliers in [-1, 1].
    # Set both scales to 0.0 to disable.
    accel_effort_scale:   float = 0.01   # weight on velocity-acceleration squared
    angular_effort_scale: float = 0.01   # weight on angular-acceleration squared



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

    # Striker approach distance penalty (0 at d=0, negative as d increases)
    striker_app = _piecewise_lin_exp(
        d,
        d_max=reward_config.striker_approach_d_max,
        d_knee=reward_config.striker_approach_d_knee,
        w_lin=reward_config.striker_approach_w_lin,
        w_exp=reward_config.striker_approach_w_exp,
        alpha=reward_config.striker_approach_alpha,
    )
    striker_app = striker_app - _piecewise_lin_exp(
        torch.zeros((), device=d.device),
        d_max=reward_config.striker_approach_d_max,
        d_knee=reward_config.striker_approach_d_knee,
        w_lin=reward_config.striker_approach_w_lin,
        w_exp=reward_config.striker_approach_w_exp,
        alpha=reward_config.striker_approach_alpha,
    )

    # Jammer approach distance penalty (0 at d=0, negative as d increases)
    jammer_app = _piecewise_lin_exp(
        d,
        d_max=reward_config.jammer_approach_d_max,
        d_knee=reward_config.jammer_approach_d_knee,
        w_lin=reward_config.jammer_approach_w_lin,
        w_exp=reward_config.jammer_approach_w_exp,
        alpha=reward_config.jammer_approach_alpha,
    )
    jammer_app = jammer_app - _piecewise_lin_exp(
        torch.zeros((), device=d.device),
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

    border = -_piecewise_lin_exp(
        d,
        d_max=reward_config.border_d_max,
        d_knee=reward_config.border_d_knee,
        w_lin=reward_config.border_w_lin,
        w_exp=reward_config.border_w_exp,
        alpha=reward_config.border_alpha,
    )

    # Formation distance penalty — striker side (flipped: 0 at d=0, −scale at d≥ref_dist)
    striker_form = reward_config.striker_formation_scale * (
        (1.0 - d / reward_config.striker_formation_ref_dist).clamp(min=0.0) - 1.0
    )
    # Formation distance penalty — jammer side (flipped: 0 at d=0, −scale at d≥ref_dist)
    jammer_form = reward_config.jammer_formation_scale * (
        (1.0 - d / reward_config.jammer_formation_ref_dist).clamp(min=0.0) - 1.0
    )

    plt.figure(figsize=(12, 7))
    plt.plot(d.numpy(), striker_app.numpy(), label="Striker Distance Penalty", color="#1f77b4", linewidth=2)
    plt.plot(d.numpy(), jammer_app.numpy(), label="Jammer Distance Penalty", color="#17becf", linewidth=2)
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



    
# import torch
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np
# from typing import TYPE_CHECKING, Tuple

# if TYPE_CHECKING:
#     try:
#         from .config import EnvConfig
#     except ImportError:
#         from config import EnvConfig


# def _piecewise_lin_exp(d: torch.Tensor, d_max: float, d_knee: float,
#                        w_lin: float, w_exp: float, alpha: float) -> torch.Tensor:
#     """Piecewise linear-exponential shaping."""
#     t_lin = ((d_max - d) / (d_max - d_knee + 1e-8)).clamp(0.0, 1.0)
#     lin_val = w_lin * t_lin

#     t_exp = ((d_knee - d) / (d_knee + 1e-8)).clamp(0.0, 1.0)
#     exp_val = w_exp * (torch.exp(alpha * t_exp) - 1.0)

#     return lin_val + exp_val


# def plot_jammer_and_striker_reward_landscapes(
#     env_config: "EnvConfig",
#     reward_config: RewardConfig,
#     distance_range: Tuple[float, float] = (0.0, 0.6),
#     n_points: int = 1000,
# ):
#     """
#     Two-panel figure:
#       Left  – Jammer:  reward components vs distance to RADAR
#       Right – Striker: reward components vs distance to TARGET

#     Vertical reference lines mark operational radii (jam range, engage range,
#     radar kill zone, jammed kill zone).
#     """
#     d = torch.linspace(distance_range[0], distance_range[1], n_points)

#     # ── Key distances from EnvConfig ──
#     radar_kill_radius = env_config.radar_range                          # 0.20
#     jammed_kill_radius = env_config.radar_range * env_config.jammer_jam_effect  # 0.20 * 0.15 = 0.03
#     jam_radius = env_config.jammer_jam_radius                           # 0.35
#     engage_range = env_config.striker_engage_range                      # 0.10

#     fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)

#     # =====================================================================
#     #  PANEL 1 — JAMMER (distance to radar)
#     # =====================================================================
#     ax = axes[0]

#     # 1a. Jammer approach reward (positive, increases as d→0)
#     jammer_approach = _piecewise_lin_exp(
#         d,
#         d_max=reward_config.jammer_approach_d_max,
#         d_knee=reward_config.jammer_approach_d_knee,
#         w_lin=reward_config.jammer_approach_w_lin,
#         w_exp=reward_config.jammer_approach_w_exp,
#         alpha=reward_config.jammer_approach_alpha,
#     )

#     # 1b. Radar avoidance penalty (negative, increases as d→0 from zone boundary)
#     #     For the jammer, distance to zone boundary = d - radar_kill_radius
#     #     (penalty is about proximity to the UNJAMMED kill zone)
#     dist_to_boundary = (d - radar_kill_radius).clamp(min=0.0)
#     radar_avoid = -_piecewise_lin_exp(
#         dist_to_boundary,
#         d_max=reward_config.radar_avoid_d_max,
#         d_knee=reward_config.radar_avoid_d_knee,
#         w_lin=reward_config.radar_avoid_w_lin,
#         w_exp=reward_config.radar_avoid_w_exp,
#         alpha=reward_config.radar_avoid_alpha,
#     )
#     # Inside kill zone: mark as death penalty zone (no shaping, just agent_destroyed)
#     inside_kill = d < radar_kill_radius
#     radar_avoid[inside_kill] = float('nan')

#     # 1c. Jammer jam bonus (flat per-step if within jam_radius)
#     jam_bonus = torch.where(
#         d <= jam_radius,
#         torch.full_like(d, reward_config.jammer_jam_bonus),
#         torch.zeros_like(d),
#     )

#     # 1d. Timestep penalty (constant)
#     timestep = torch.full_like(d, reward_config.timestep_penalty)

#     # 1e. Net reward
#     net_jammer = jammer_approach + radar_avoid + jam_bonus + timestep
#     net_jammer[inside_kill] = float('nan')

#     # Plot components
#     ax.plot(d.numpy(), jammer_approach.numpy(), label="Approach reward", color="#2ca02c", lw=2)
#     ax.plot(d.numpy(), radar_avoid.numpy(), label="Radar avoidance penalty", color="#9467bd", lw=2)
#     ax.plot(d.numpy(), jam_bonus.numpy(), label="Jam bonus", color="#ff7f0e", lw=2, ls="--")
#     ax.plot(d.numpy(), timestep.numpy(), label="Timestep penalty", color="gray", lw=1, ls=":")
#     ax.plot(d.numpy(), net_jammer.numpy(), label="NET reward", color="black", lw=2.5)

#     # Kill zone shading
#     ax.axvspan(distance_range[0], radar_kill_radius, alpha=0.15, color="red", label="Kill zone (unjammed)")
#     ax.axvspan(distance_range[0], jammed_kill_radius, alpha=0.25, color="darkred", label=f"Kill zone (jammed, ×{env_config.jammer_jam_effect})")

#     # Vertical reference lines
#     ax.axvline(radar_kill_radius, color="red", ls="--", lw=1.5, label=f"Radar kill radius = {radar_kill_radius}")
#     ax.axvline(jam_radius, color="#ff7f0e", ls="-.", lw=1.5, label=f"Jam radius = {jam_radius}")
#     ax.axvline(jammed_kill_radius, color="darkred", ls="--", lw=1, label=f"Jammed kill radius = {jammed_kill_radius:.3f}")

#     # Death penalty annotation
#     ax.annotate(
#         f"agent_destroyed = {reward_config.agent_destroyed}",
#         xy=(radar_kill_radius * 0.5, reward_config.agent_destroyed * 0.3),
#         fontsize=9, color="red", ha="center",
#         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
#     )

#     ax.axhline(0, color="gray", lw=0.5)
#     ax.set_xlabel("Distance to Radar", fontsize=12)
#     ax.set_ylabel("Reward per step", fontsize=12)
#     ax.set_title("JAMMER — Reward Landscape vs Distance to Radar", fontsize=13, fontweight="bold")
#     ax.legend(loc="upper right", fontsize=8)
#     ax.grid(True, alpha=0.3)
#     ax.set_xlim(distance_range)

#     # =====================================================================
#     #  PANEL 2 — STRIKER (distance to target)
#     # =====================================================================
#     ax = axes[1]

#     # 2a. Striker approach reward
#     striker_approach = _piecewise_lin_exp(
#         d,
#         d_max=reward_config.striker_approach_d_max,
#         d_knee=reward_config.striker_approach_d_knee,
#         w_lin=reward_config.striker_approach_w_lin,
#         w_exp=reward_config.striker_approach_w_exp,
#         alpha=reward_config.striker_approach_alpha,
#     )

#     # 2b. Radar avoidance penalty for striker
#     #     Assumption: target is co-located with or near a radar.
#     #     We show the penalty as if the striker must pass through the radar zone
#     #     to reach the target. Distance to radar boundary = d - radar_kill_radius.
#     dist_to_boundary_striker = (d - radar_kill_radius).clamp(min=0.0)
#     radar_avoid_striker = -_piecewise_lin_exp(
#         dist_to_boundary_striker,
#         d_max=reward_config.radar_avoid_d_max,
#         d_knee=reward_config.radar_avoid_d_knee,
#         w_lin=reward_config.radar_avoid_w_lin,
#         w_exp=reward_config.radar_avoid_w_exp,
#         alpha=reward_config.radar_avoid_alpha,
#     )
#     radar_avoid_striker[inside_kill] = float('nan')

#     # 2b-alt. Radar avoidance when JAMMED (smaller kill zone)
#     dist_to_jammed_boundary = (d - jammed_kill_radius).clamp(min=0.0)
#     radar_avoid_jammed = -_piecewise_lin_exp(
#         dist_to_jammed_boundary,
#         d_max=reward_config.radar_avoid_d_max,
#         d_knee=reward_config.radar_avoid_d_knee,
#         w_lin=reward_config.radar_avoid_w_lin,
#         w_exp=reward_config.radar_avoid_w_exp,
#         alpha=reward_config.radar_avoid_alpha,
#     )
#     inside_jammed_kill = d < jammed_kill_radius
#     radar_avoid_jammed[inside_jammed_kill] = float('nan')

#     # 2c. Timestep penalty
#     timestep_s = torch.full_like(d, reward_config.timestep_penalty)

#     # 2d. Net reward — UNJAMMED scenario
#     net_striker_unjammed = striker_approach + radar_avoid_striker + timestep_s
#     net_striker_unjammed[inside_kill] = float('nan')

#     # 2e. Net reward — JAMMED scenario
#     net_striker_jammed = striker_approach + radar_avoid_jammed + timestep_s
#     net_striker_jammed[inside_jammed_kill] = float('nan')

#     # Plot components
#     ax.plot(d.numpy(), striker_approach.numpy(), label="Approach reward", color="#2ca02c", lw=2)
#     ax.plot(d.numpy(), radar_avoid_striker.numpy(), label="Radar avoidance (unjammed)", color="#9467bd", lw=2)
#     ax.plot(d.numpy(), radar_avoid_jammed.numpy(), label="Radar avoidance (jammed)", color="#9467bd", lw=2, ls="--")
#     ax.plot(d.numpy(), timestep_s.numpy(), label="Timestep penalty", color="gray", lw=1, ls=":")
#     ax.plot(d.numpy(), net_striker_unjammed.numpy(), label="NET reward (unjammed)", color="black", lw=2.5)
#     ax.plot(d.numpy(), net_striker_jammed.numpy(), label="NET reward (jammed)", color="blue", lw=2.5, ls="--")

#     # Kill zone shading
#     ax.axvspan(distance_range[0], radar_kill_radius, alpha=0.15, color="red", label="Kill zone (unjammed)")
#     ax.axvspan(distance_range[0], jammed_kill_radius, alpha=0.25, color="darkred", label=f"Kill zone (jammed)")

#     # Vertical reference lines
#     ax.axvline(radar_kill_radius, color="red", ls="--", lw=1.5, label=f"Radar kill radius = {radar_kill_radius}")
#     ax.axvline(jammed_kill_radius, color="darkred", ls="--", lw=1, label=f"Jammed kill radius = {jammed_kill_radius:.3f}")
#     ax.axvline(engage_range, color="#2ca02c", ls="-.", lw=1.5, label=f"Engage range = {engage_range}")

#     # Highlight the "safe corridor" when jammed
#     if jammed_kill_radius < engage_range:
#         ax.axvspan(jammed_kill_radius, engage_range, alpha=0.1, color="green",
#                    label=f"Safe engage corridor (jammed)")
#         ax.annotate(
#             "Safe engage\ncorridor",
#             xy=((jammed_kill_radius + engage_range) / 2, 0),
#             fontsize=9, color="green", ha="center", va="bottom",
#             fontweight="bold",
#             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),
#         )

#     # Death penalty annotation
#     ax.annotate(
#         f"agent_destroyed = {reward_config.agent_destroyed}",
#         xy=(radar_kill_radius * 0.5, reward_config.agent_destroyed * 0.3),
#         fontsize=9, color="red", ha="center",
#         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
#     )

#     # Engage reward annotation
#     ax.annotate(
#         f"target_destroyed = +{reward_config.target_destroyed}",
#         xy=(engage_range, striker_approach[d <= engage_range][-1].item() if (d <= engage_range).any() else 0),
#         xytext=(engage_range + 0.05, 0.15),
#         fontsize=9, color="#2ca02c",
#         arrowprops=dict(arrowstyle="->", color="#2ca02c"),
#         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", alpha=0.8),
#     )

#     ax.axhline(0, color="gray", lw=0.5)
#     ax.set_xlabel("Distance to Target (≈ distance to radar)", fontsize=12)
#     ax.set_ylabel("Reward per step", fontsize=12)
#     ax.set_title("STRIKER — Reward Landscape vs Distance to Target", fontsize=13, fontweight="bold")
#     ax.legend(loc="upper right", fontsize=8)
#     ax.grid(True, alpha=0.3)
#     ax.set_xlim(distance_range)

#     plt.tight_layout()
#     plt.savefig("reward_landscape_jammer_striker.png", dpi=150, bbox_inches="tight")
#     plt.show()


# if __name__ == "__main__":
#     try:
#         from .config import ExperimentConfig
#     except ImportError:
#         from config import ExperimentConfig

#     config = ExperimentConfig()
#     config.finalize()

#     plot_jammer_and_striker_reward_landscapes(
#         env_config=config.env,
#         reward_config=config.env.reward_config,
#         distance_range=(0.0, 1),
#     )