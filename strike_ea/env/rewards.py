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

    timestep_penalty: float = -0.04
    # Small per-step cost for every alive agent (encourages finishing fast).

    agent_destroyed: float = -10.0
    # One-time penalty applied to an agent the step it is killed by a radar.

    # ─── BORDER AVOIDANCE  (piecewise lin-exp penalty, per alive agent) ──────
    # d = distance from nearest map edge.  d_max = border_thresh (EnvConfig).
    border_w_lin:  float = 1.0
    border_w_exp:  float = 2.0
    border_d_knee: float = 0.02   # 20 km from edge → exponential kicks in
    border_alpha:  float = 3.0

    # ─── RADAR ZONE AVOIDANCE  (piecewise lin-exp penalty, ALL agents) ───────
    # d = distance from nearest radar detection-zone boundary (clamped ≥ 0).
    # Agents inside the zone get killed and receive agent_destroyed instead.
    radar_avoid_w_lin:  float = 0.3
    radar_avoid_w_exp:  float = 1.0
    radar_avoid_d_max:  float = 0.10  # penalty starts 100 km from zone edge
    radar_avoid_d_knee: float = 0.03  # 30 km from zone → exponential
    radar_avoid_alpha:  float = 3.0

    # ─── STRIKER APPROACH  (piecewise lin-exp reward toward nearest target) ──
    # d = distance to nearest alive target.  d_max = map diagonal.
    striker_w_lin:  float = 0.5
    striker_w_exp:  float = 0.5
    striker_d_knee: float = 0.30   # 200 km → exponential kicks in
    striker_alpha:  float = 3.0

    # ─── JAMMER APPROACH  (piecewise lin-exp reward toward nearest radar) ────
    # d = distance to nearest radar (outside detection zone).  d_max = map diag.
    jammer_w_lin:  float = 0.3
    jammer_w_exp:  float = 0.5
    jammer_d_knee: float = 0.30   # 300 km → exponential kicks in
    jammer_alpha:  float = 3.0
