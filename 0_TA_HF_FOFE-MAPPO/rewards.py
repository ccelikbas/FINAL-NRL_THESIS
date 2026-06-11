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

    terminal_bonus: float = 1
    # One-time bonus granted when all targets are destroyed (mission complete).
    # Applied to every agent in the terminal transition.

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
    border_w_lin:  float = 0.05  # gentle early-warning ramp (50 km → 30 km from edge)
    border_w_exp:  float = 0
    border_d_knee: float = 0   # 30 km from edge → exponential kicks in
    border_alpha:  float = 0

    # ─── RADAR ZONE AVOIDANCE  (piecewise lin-exp penalty, ALL agents) ───────
    # d = distance from a FIXED jammed radar boundary (non-adaptive).
    # This shaping does not switch with live jamming state; it always uses
    # the jammed effective range as the avoidance boundary.
    # Agents inside lethal radar range are still handled by agent_destroyed.
    radar_avoid_w_lin:  float = 0  # reduced so approach reward dominates
    radar_avoid_w_exp:  float = 0
    radar_avoid_d_max:  float = 0   # penalty starts 200 km outside zone edge
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
    # True  = hard-nearest: use distance to nearest alive target.
    # False = soft-nearest: compute shaped_dist = Σ_i (w_i * dist_i) over alive targets,
    #         with w_i = (1 / (dist_i + eps)) / Σ_j (1 / (dist_j + eps)), eps=1e-6.
    #         This smooths target-switching and reduces jumps when a target dies.

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

    # ─── HF RADAR MARGIN REWARD  (HF env only, per alive agent, per radar) ──
    # Discrete reward keyed on the signs of the HF margins exposed in the
    # actor's per-(agent, radar) obs:
    #     m_range = (d − R_unc)  (sign only is used here)
    #     m_angle = θ_s − δ*     (sign only; protection requires the radar
    #                              to be actively jammed by an alive jammer)
    # Three mutually-exclusive cases per (agent, radar) pair, summed over
    # all radars and applied to alive agents of BOTH roles:
    #   1) m_range < 0, m_angle < 0  →  exposed inside R_unc
    #        contributes hf_margin_exposed_penalty   (large negative)
    #   2) m_range < 0, m_angle > 0  →  inside R_unc but inside an
    #        actively jammed side-lobe cone
    #        contributes hf_margin_protected_penalty (small negative)
    #   3) m_range > 0                →  outside R_unc (safe)
    #        contributes hf_margin_outside_bonus     (small positive)
    # Defaults are 0.0 so existing training runs are unaffected unless these
    # weights are explicitly set. Suggested starting magnitudes when enabling:
    # exposed=-0.05, protected=-0.005, outside=0.001.
    hf_margin_exposed_penalty:   float = -0.005
    hf_margin_protected_penalty: float = -0.001
    hf_margin_outside_bonus:     float = 0

    # ─── JAMMER ACTIVE-JAMMING BONUS  (deactivated by default) ───────────────
    # Per-step bonus when a jammer is within jam_radius of ≥ 1 radar.
    jammer_jam_bonus: float = 0   # Deactivated

    # ─── JAMMER BEAM-ON-RADAR BONUS  (HF directional-jammer model only) ──────
    # Per-step bonus given to a jammer when ANY alive radar lies inside the
    # jammer's directional cone (full beam width, not just the centerline).
    # Binary signal: useful as a sparse 'finished aligning' bonus that can
    # stack on top of the smooth alignment shaping below.
    # No effect in the legacy (non-HF) jammer model, which has no cone.
    # Set to 0.0 to disable.
    jammer_beam_on_radar_bonus: float = 0.005

    # ─── JAMMER BEAM ALIGNMENT SHAPING  (HF directional-jammer model only) ───
    # Smooth angular shaping toward the *physically nearest* alive radar.
    # For each alive jammer:
    #     a = | wrap(angle_jammer_to_radar - (heading + jammer_bearing)) |
    #     penalty = -jammer_beam_alignment_scale * (a / pi)
    # so a = 0 (beam directly on the radar) → 0, and a = pi (beam pointing
    # 180° away) → -jammer_beam_alignment_scale. Linear in |a| gives a
    # uniform gradient across the whole angular range and avoids the
    # discontinuity of a binary in/out-of-cone bonus.
    # Applied unconditionally every step (no in-cone gate).
    # Set to 0.0 to disable.
    jammer_beam_alignment_scale: float = 0.010 # deze kleiner maken xxx

    # ─── JAMMER COALITION COVERAGE  (HF directional-jammer model only) ───────
    # Penalises pairs of nearby jammers whose beams overlap too much, so the
    # coalition is pushed toward spatial diversity (joint angular coverage).
    # A small overlap is tolerated (jamming is not perfectly additive, so a
    # little beam overlap is acceptable / even beneficial at the seams).
    #
    # Coalition: any pair of *alive* jammers whose Euclidean distance is
    # ≤ jammer_coalition_d_max are considered to be in the same coalition.
    # Uses the cached pairwise agent-distance tensor (_c_dist_aa), so the
    # coalition test is essentially free.
    #
    # Geometry: each beam has half-width main_lobe_rad/2, so two beams stop
    # overlapping once their separation Δ ∈ [0, π] reaches main_lobe_rad.
    # The overlap angle is therefore (main_lobe_rad − Δ). A margin of
    # jammer_coalition_overlap_margin_deg of overlap is tolerated for free;
    # excess overlap below the onset Δ_on = main_lobe_rad − margin is penalised.
    #
    # Per-pair shaping (penalty-only, symmetric to R_min):
    #     r_pair(Δ) = −R_min · clamp((Δ_on − Δ) / Δ_on, 0, 1)
    # Properties:
    #   Δ ≥ Δ_on  (overlap ≤ margin) → r = 0       (tolerated, no penalty)
    #   Δ < Δ_on                     → r < 0       (excess-overlap penalty)
    #   Δ = 0     (fully co-aligned) → r = −R_min  (max penalty)
    # Aggregation: each jammer receives the SUM of r_pair over all coalition
    # partners (symmetric, so both jammers in a pair receive the same term).
    # Dead jammers receive 0 and do not form coalitions.
    # Set jammer_coalition_R_min = 0.0 to disable. Setting the margin ≥ the
    # full main lobe also disables the penalty (Δ_on ≤ 0 → no penalty region).
    jammer_coalition_d_max: float = 0
    jammer_coalition_R_min: float = 0
    jammer_coalition_overlap_margin_deg: float = 0

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

    jammer_formation_scale:     float = 0.05   # prev: 0.1 # reward to each jammer for being near a striker
    jammer_formation_ref_dist:  float = 0.5    # distance (map units) beyond which reward = 0

    # Per-striker jammer capacity for the jammer formation reward.
    # A jammer only earns proximity reward from a striker if it is among that
    # striker's K nearest *alive* jammers. Redundant jammers fall through to the
    # nearest striker that still has capacity; jammers beyond total capacity
    # (nj > K * ns) earn 0 formation reward. This discourages more than K
    # jammers escorting the same striker and pushes the coalition to spread
    # across strikers. Set <= 0 to disable the cap (every jammer is pulled to
    # its nearest striker — the legacy behaviour, equivalent to K >= nj).
    # Only affects the jammer side; striker_formation is unchanged.
    jammer_formation_k:         int   = 2

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
    striker_sep_d_max:  float = 0
    striker_sep_d_knee: float = 0
    striker_sep_w_lin:  float = 0
    striker_sep_w_exp:  float = 0
    striker_sep_alpha:  float = 0

    jammer_sep_d_max:  float = 0
    jammer_sep_d_knee: float = 0
    jammer_sep_w_lin:  float = 0
    jammer_sep_w_exp:  float = 0
    jammer_sep_alpha:  float = 0

    # ─── CONTROL EFFORT PENALTY  (per alive agent, per step) ───────────────
    # Penalises large control actions to encourage smooth trajectories.
    # Penalty = −accel_effort_scale × accel² − angular_effort_scale × angular_accel²
    # where accel and angular_accel are the discrete multipliers in [-1, 1].
    # Set both scales to 0.0 to disable.
    accel_effort_scale:   float = 0.01   # weight on velocity-acceleration squared
    angular_effort_scale: float = 0.01   # weight on angular-acceleration squared

    # ─── BEAM CONTROL EFFORT PENALTY  (HF directional-jammer model only) ────
    # Applied only to jammers, on action dim 2 (beam angular acceleration).
    # Penalty = −beam_accel_effort_scale × beam_accel² (per alive jammer).
    # Set to 0.0 to disable.
    beam_accel_effort_scale: float = 0.005


'''
RUN: Visualisation of the piecewise linear-exponential shaping function using current paramters:
'''
from typing import Tuple
import torch
import matplotlib.pyplot as plt
import numpy as np

try:
    from .nlr_style import (
        apply_nlr_style,
        NLR_PRIMARY,
        NLR_SECONDARY,
        NLR_ACCENT,
        NLR_GRAY,
        NLR_DARKGRAY,
        NLR_LIGHTBLUE_50,
        NLR_TERRA_50,
    )
except ImportError:  # standalone script execution (python rewards.py)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nlr_style import (  # type: ignore[no-redef]
        apply_nlr_style,
        NLR_PRIMARY,
        NLR_SECONDARY,
        NLR_ACCENT,
        NLR_GRAY,
        NLR_DARKGRAY,
        NLR_LIGHTBLUE_50,
        NLR_TERRA_50,
    )

apply_nlr_style()


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
    plt.plot(d.numpy(), striker_app.numpy(), label="Striker Distance Penalty", color=NLR_PRIMARY, linewidth=2)
    plt.plot(d.numpy(), jammer_app.numpy(), label="Jammer Distance Penalty", color=NLR_SECONDARY, linewidth=2)
    plt.plot(d.numpy(), radar.numpy(), label="Radar Avoidance", color=NLR_ACCENT, linewidth=2)
    plt.plot(d.numpy(), border.numpy(), label="Border Avoidance", color=NLR_TERRA_50, linewidth=2)
    plt.plot(d.numpy(), striker_form.numpy(), label="Striker Formation (↔ jammer)", color=NLR_DARKGRAY, linewidth=2)
    plt.plot(d.numpy(), jammer_form.numpy(),  label="Jammer Formation (↔ striker)",  color=NLR_LIGHTBLUE_50, linewidth=2, linestyle="--")
    plt.axhline(0, color=NLR_GRAY, lw=0.5)
    plt.xlabel("Distance")
    plt.ylabel("Reward / Penalty")
    plt.title("Reward Shaping Functions (piecewise lin-exp)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    reward_config = RewardConfig()
    print('printing rewards')
    plot_reward_functions(reward_config, distance_range=(0, 0.5))

