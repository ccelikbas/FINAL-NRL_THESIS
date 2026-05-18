"""
High-fidelity angular radar model extension of the Strike-EA 2D environment.

Replaces the simple binary jam/kill radar model with a physics-based model
using Jammer-to-Signal Ratio (JSR) in the main-lobe and side-lobe directions.

Each alive jammer creates an angular "notch" in every radar's detection range.
The effective detection range is per-(agent, radar) — agents at different
bearings from the radar see different effective ranges depending on whether
they fall in a jammer's main-lobe, side-lobe, or uncovered sector.

Activated when EnvExtensionsConfig.use_hf_radar == True.
"""

from __future__ import annotations

import math
import time
from math import radians
from typing import Dict, Optional, Tuple

import torch
from tensordict import TensorDict

from .config import HFRadarConfig
from .environment import StrikeEA2DEnv
from .rewards import RewardConfig


class HFStrikeEA2DEnv(StrikeEA2DEnv):
    """Strike-EA environment with high-fidelity angular radar/jammer model.

    Inherits all behaviour from StrikeEA2DEnv and overrides only:
      - __init__  : accepts HFRadarConfig, precomputes BT constants
      - _step     : replaces the jam + kill section with the HF radar model
      - _compute_hf_radar_eff_range : new method computing per-agent ranges
    """

    def __init__(
        self,
        *,
        hf_cfg: HFRadarConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hf_cfg = hf_cfg

        def _cfg_float(
            new_name: str,
            legacy_name: Optional[str] = None,
            default: Optional[float] = None,
        ) -> float:
            if hasattr(hf_cfg, new_name):
                val = getattr(hf_cfg, new_name)
                if val is not None:
                    return float(val)
            if legacy_name is not None and hasattr(hf_cfg, legacy_name):
                return float(getattr(hf_cfg, legacy_name))
            if default is not None:
                return float(default)
            raise AttributeError(
                f"Missing HF radar config parameter: '{new_name}'"
                + (f" (or legacy '{legacy_name}')" if legacy_name else "")
            )

        def _db_to_linear(db_value: float) -> float:
            return float(10.0 ** (float(db_value) / 10.0))

        # Read radar equation parameters (supporting old config field names too).
        radar_tx_power = _cfg_float("radar_tx_power", "P_t")
        radar_tx_gain_db = _cfg_float("radar_tx_gain", None)
        radar_rx_gain_db_raw = getattr(hf_cfg, "radar_rx_gain", None)
        radar_rx_gain_db = float(radar_tx_gain_db if radar_rx_gain_db_raw is None else radar_rx_gain_db_raw)
        wavelength = _cfg_float("wavelength", None, 0.03)
        target_rcs = _cfg_float("target_rcs", "sigma")
        system_temperature = _cfg_float("system_temperature", None, 290.0)
        receiver_bandwidth = _cfg_float("receiver_bandwidth", None, 1e6)
        system_losses_db = _cfg_float("system_losses", None, 0.0)
        boltzmann_constant = _cfg_float("boltzmann_constant", None, 1.380649e-23)
        radar_side_lobe_gain_db = _cfg_float("radar_side_lobe_gain", "G_S")
        jammer_tx_power = _cfg_float("jammer_tx_power", "P_J")
        jammer_gain_db = _cfg_float("jammer_gain", "G_J")
        meters_per_world_unit = _cfg_float("meters_per_world_unit", None, 1_000_000.0)
        normalized_range_scale = _cfg_float("normalized_range_scale", None, 1.0)
        target_unc_world = getattr(hf_cfg, "target_unconstrained_range_world", None)

        # dB -> linear conversion for all power-ratio terms.
        radar_tx_gain = _db_to_linear(radar_tx_gain_db)
        radar_rx_gain = _db_to_linear(radar_rx_gain_db)
        system_losses = _db_to_linear(system_losses_db)
        radar_side_lobe_gain = _db_to_linear(radar_side_lobe_gain_db)
        jammer_gain = _db_to_linear(jammer_gain_db)

        # Precompute angle half-widths in radians
        self._theta_main_half = radians(hf_cfg.theta_main_deg / 2)
        self._theta_side_half = radians(hf_cfg.theta_side_deg / 2)
        # Directional-jammer cone half-width (radians). A radar is only
        # jammed by jammer j if its bearing (from jammer j) lies within
        # ±_jammer_main_lobe_half of jammer j's pointing direction.
        self._jammer_main_lobe_half = radians(
            float(getattr(hf_cfg, "jammer_main_lobe_deg", 30.0)) / 2.0
        )

        # Unconstrained range from radar SNR equation (SI meters).
        # Gains/losses are entered in dB and converted to linear above.
        # R_unc_m = (P_t G_t G_r lambda^2 sigma / ((4pi)^3 k T0 B_n L))^(1/4)
        snr_num = (
            radar_tx_power
            * radar_tx_gain
            * radar_rx_gain
            * (wavelength * wavelength)
            * target_rcs
        )
        snr_den = (
            (4.0 * math.pi) ** 3
            * boltzmann_constant
            * system_temperature
            * receiver_bandwidth
            * system_losses
        )
        self._meters_per_world_unit = meters_per_world_unit
        self.radar_range_unconstrained_m_raw = math.pow(max(snr_num / max(snr_den, 1e-30), 0.0), 0.25)

        # Convert physical SI range (meters) into normalized world units.
        # Optional calibration supports matching a target world-space range
        # without changing RF parameters.
        world_unc_raw = self.radar_range_unconstrained_m_raw / self._meters_per_world_unit
        if target_unc_world is not None:
            self._range_world_scale = float(target_unc_world) / max(world_unc_raw, 1e-12)
        else:
            self._range_world_scale = normalized_range_scale

        self.radar_range_unconstrained_m = self.radar_range_unconstrained_m_raw * self._range_world_scale
        self.radar_range_unconstrained = self.radar_range_unconstrained_m / self._meters_per_world_unit

        # Precompute burn-through constants in meters (JSR = 1 model).
        # Main-lobe: R_main = ((P_t*G_t*sigma/(4*pi*P_J*G_J)) * R_J^2)^(1/4)
        self._r_main_bt_coeff = (
            (radar_tx_power * radar_tx_gain * target_rcs)
            / max(4.0 * math.pi * jammer_tx_power * jammer_gain, 1e-30)
        )

        # Side-lobe: R_side = ((sigma/(4*pi)) * ((P_t*G_t)/(P_J*G_J)) * ((G_t*R_J^2)/G_S))^(1/4)
        # Collect R_J-independent multiplier C so R_side = (C * R_J^2)^(1/4).
        self._r_side_bt_coeff = (
            (target_rcs / (4.0 * math.pi))
            * ((radar_tx_power * radar_tx_gain) / max(jammer_tx_power * jammer_gain, 1e-30))
            * (radar_tx_gain / max(radar_side_lobe_gain, 1e-30))
        )

        # Override radar_range so existing reward / obs code uses R_unconstrained
        self.radar_range = self.radar_range_unconstrained

        # Re-initialise radar_eff_range with unconstrained range
        B, R = self.num_envs, self.n_radars
        self.radar_eff_range = torch.full(
            (B, R), self.radar_range, device=self.device
        )

        # Per-agent effective range buffer [B, A, R]
        self.radar_eff_range_per_agent = torch.full(
            (B, self.n_agents, R), self.radar_range, device=self.device
        )

        # Kept for compatibility with HF visualization helper.
        self._gt_over_gs = radar_tx_gain / max(radar_side_lobe_gain, 1e-30)

        # ------------------------------------------------------------------
        # Directional-jammer action extension
        # ------------------------------------------------------------------
        # The HF env adds a 3rd action dimension that selects the jammer
        # bearing (relative to the jammer's own heading). Strikers also
        # receive this dimension but IGNORE it — see CombinedPolicy where
        # the striker's bearing logits are masked to a constant index so
        # the dimension is a no-op for the striker policy.
        #
        # Motion dims (0, 1) keep the original 7-value discrete table.
        # Bearing dim (2) uses a separate evenly-spaced table over (-pi, pi].
        # The shared Categorical n is the max of the two so a single tensor
        # spec covers both — out-of-range indices on either dim are masked
        # to near-zero probability at the policy level.
        self._n_motion_choices = int(self.n_choices)               # 7
        self._n_bearing_choices = int(getattr(hf_cfg, "jammer_bearing_n_choices", 16))
        self.n_choices = max(self._n_motion_choices, self._n_bearing_choices)
        self.act_dim = 3
        self._bearing_table = torch.linspace(
            -math.pi, math.pi, self._n_bearing_choices + 1, device=self.device,
        )[:-1]  # [n_bearings] in radians, in (-pi, pi]
        # Per-jammer current bearing (radians, relative to jammer heading).
        # Overwritten every _step from action[..., 2]; zero before first step.
        self.jammer_bearing = torch.zeros(self.num_envs, self.n_jammers, device=self.device)
        # Rebuild specs so action_spec reflects the new act_dim/n_choices.
        self._make_specs()

        # --- Fine-grained step profiling (off by default) ---
        self._profile_active: bool = False
        self._profile_buckets: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Fine-grained profiling helpers (used only when set_profile_active(True))
    # ------------------------------------------------------------------

    def set_profile_active(self, active: bool) -> None:
        """Enable/disable fine-grained timing of _step sub-sections.
        When enabled, each _step call accumulates per-section wall time
        into self._profile_buckets (keys prefixed with 'env_').
        """
        self._profile_active = bool(active)
        if self._profile_active:
            self._profile_buckets = {}

    def pop_profile_buckets(self) -> Dict[str, float]:
        """Return accumulated timings and reset the buckets."""
        out = self._profile_buckets
        self._profile_buckets = {}
        return out

    def _prof_tic(self):
        if not self._profile_active:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _prof_lap(self, name: str, t):
        if t is None:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        self._profile_buckets[name] = self._profile_buckets.get(name, 0.0) + (now - t)
        return now

    # ------------------------------------------------------------------
    # HF radar model
    # ------------------------------------------------------------------

    def _compute_hf_radar_eff_range(self) -> None:
        """Compute per-(agent, radar) effective detection range.

        Uses burn-through (BT) sector cuts derived directly from JSR = 1:

            R_main = ((P_t*G_t*sigma/(4*pi*P_J*G_J)) * R_J^2)^(1/4)
            R_side = ((sigma/(4*pi)) * ((P_t*G_t)/(P_J*G_J))
                      * ((G_t*R_J^2)/G_S))^(1/4)

        Here R_unc is first computed in physical meters from the radar SNR
        equation, then converted to world units.

        For each (agent, radar) pair the effective range is determined by the
        angular offset between the agent's bearing and each jammer's bearing
        (both from the radar's perspective).  The deepest cut across all
        active jammers wins (minimum R_eff).

        Updates
        -------
        self.radar_eff_range_per_agent : [B, A, R]
        self.radar_eff_range           : [B, R]  (min across agents, for obs)
        """
        B  = self.num_envs
        A  = self.n_agents
        R  = self.n_radars
        J  = self.n_jammers
        ns = self.n_strikers
        R_unc_m = self.radar_range_unconstrained_m

        # Fast path: no jammers → unconstrained everywhere
        if J == 0:
            self.radar_eff_range_per_agent.fill_(self.radar_range_unconstrained)
            self.radar_eff_range.fill_(self.radar_range_unconstrained)
            return

        # ----- distances -----
        dist_jr_m = self._c_dist_ar[:, ns:, :] * self._meters_per_world_unit  # [B, J, R]

        # ----- absolute bearing angles from radar to entity -----
        # _c_rel_ar = radar_pos - agent_pos  →  radar-to-agent = -_c_rel_ar
        neg_rel = -self._c_rel_ar                                          # [B, A, R, 2]
        abs_angle_ra = torch.atan2(neg_rel[..., 1], neg_rel[..., 0])      # [B, A, R]
        abs_angle_rj = abs_angle_ra[:, ns:, :]                             # [B, J, R]

        # ----- jammer alive mask -----
        jammer_alive = self.agent_alive[:, ns:]                            # [B, J]

        # ----- BT ranges per (jammer, radar): [B, J, R] -----
        # R_main depends on jammer-radar distance R_J.
        R_main_jr = torch.pow(
            torch.clamp(self._r_main_bt_coeff * dist_jr_m * dist_jr_m, min=0.0),
            0.25,
        )

        # R_side depends on jammer-radar distance R_J.
        R_side_jr = torch.pow(
            torch.clamp(self._r_side_bt_coeff * dist_jr_m * dist_jr_m, min=0.0),
            0.25,
        )

        # Clamp to unconstrained (far-away jammers have no net effect)
        R_main_jr = torch.clamp(R_main_jr, max=R_unc_m)
        R_side_jr = torch.clamp(R_side_jr, max=R_unc_m)

        # ----- angular delta [B, J, A, R] -----
        # Broadcast: angle_ra [B, 1, A, R]  -  angle_rj [B, J, 1, R]
        delta = abs_angle_ra.unsqueeze(1) - abs_angle_rj.unsqueeze(2)     # [B, J, A, R]
        # Wrap to [0, pi]
        delta = delta % (2.0 * math.pi)
        delta = torch.min(delta, 2.0 * math.pi - delta)

        # ----- per-(jammer, agent, radar) effective range [B, J, A, R] -----
        R_main_exp = R_main_jr.unsqueeze(2)                                # [B, J, 1, R]
        R_side_exp = R_side_jr.unsqueeze(2)                                # [B, J, 1, R]

        R_eff_jar = torch.where(
            delta <= self._theta_main_half,
            R_main_exp,
            torch.where(
                delta <= self._theta_side_half,
                R_side_exp,
                R_unc_m,  # scalar broadcasts
            ),
        )                                                                  # [B, J, A, R]

        # --------------------------------------------------------------
        # Directional-jammer gate.
        # The physics above is radar-centric and unchanged. The only
        # difference for a directional jammer is that it now only emits
        # power inside its chosen cone. So for every (jammer, radar)
        # pair where the radar lies OUTSIDE the jammer's cone, that
        # jammer has zero effect on that radar -> R_eff = R_unc.
        # --------------------------------------------------------------
        # World-frame pointing direction of each jammer (heading + bearing).
        jammer_pointing = self.agent_heading[:, ns:] + self.jammer_bearing  # [B, J]
        # Bearing from each jammer to each radar in the world frame.
        rel_jr_vec = self.radar_pos[:, None, :, :] - self.agent_pos[:, ns:, None, :]  # [B, J, R, 2]
        angle_j_to_r = torch.atan2(rel_jr_vec[..., 1], rel_jr_vec[..., 0])  # [B, J, R]
        # Wrap (angle_j_to_r - pointing) to [-pi, pi] for an absolute-value test.
        delta_point = angle_j_to_r - jammer_pointing[:, :, None]
        delta_point = torch.atan2(torch.sin(delta_point), torch.cos(delta_point))
        in_cone_jr = delta_point.abs() <= self._jammer_main_lobe_half       # [B, J, R]
        R_eff_jar = torch.where(
            in_cone_jr[:, :, None, :],   # broadcast over A
            R_eff_jar,
            torch.full_like(R_eff_jar, R_unc_m),
        )

        # Dead jammers have no effect → set to R_unc
        alive_mask = jammer_alive[:, :, None, None]                        # [B, J, 1, 1]
        R_eff_jar = torch.where(alive_mask, R_eff_jar, R_unc_m)

        # ----- min across jammers (deepest cut wins) → [B, A, R] in meters -----
        R_eff_ar_m = R_eff_jar.min(dim=1).values

        # Convert back to normalized world units for env state/obs consumers.
        self.radar_eff_range_per_agent = R_eff_ar_m / self._meters_per_world_unit

        # Aggregate for observations: min across agents → [B, R]
        self.radar_eff_range = self.radar_eff_range_per_agent.min(dim=1).values

    # ------------------------------------------------------------------
    # Step override
    # ------------------------------------------------------------------

    def _step(self, tensordict: TensorDict) -> TensorDict:
        _t = self._prof_tic()
        # Action layout in HF mode is [B, A, 3]:
        #   dim 0 (motion accel)  : discrete index in {0..n_motion_choices-1}, all agents
        #   dim 1 (heading accel) : discrete index in {0..n_motion_choices-1}, all agents
        #   dim 2 (jammer bearing): discrete index in {0..n_bearing_choices-1}
        #                           — strikers IGNORE this dim entirely (it is
        #                             masked to a constant index by the policy
        #                             and never read here for strikers).
        action = tensordict.get(self._action_key)  # [B, A, 3] discrete
        B, A, _ = action.shape
        rp = self.reward_params

        # Decode motion dims via the original 7-value table.
        motion_idx = action[..., :2].long().clamp(0, self._n_motion_choices - 1)
        acc = self._act_table[motion_idx]  # [B, A, 2]

        # Decode jammer bearing (dim 2): convert index → radians via _bearing_table.
        # Only the jammer slice is consumed; the striker slice is discarded.
        bearing_idx = action[..., 2].long().clamp(0, self._n_bearing_choices - 1)  # [B, A]
        if self.n_jammers > 0:
            self.jammer_bearing = self._bearing_table[bearing_idx[:, self.n_strikers:]]  # [B, J]

        alive = self.agent_alive
        alive_before_kill = alive
        alive_float = alive.float()
        _t = self._prof_lap("env_decode_action", _t)

        # ---- Velocity dynamics ----
        v_accel = acc[..., 0]
        self.agent_speed = (
            self.agent_speed + v_accel * self.accel_magnitude
        ).clamp(0.0, self.v_max)
        self.agent_speed = self.agent_speed * alive_float

        v_min_per_agent = torch.zeros_like(self.agent_speed)
        v_min_per_agent[:, :self.n_strikers] = self.striker.v_min
        v_min_per_agent[:, self.n_strikers:] = self.jammer.v_min
        self.agent_speed = torch.where(
            alive, torch.max(self.agent_speed, v_min_per_agent), self.agent_speed
        )

        # ---- Heading dynamics ----
        h_accel = acc[..., 1]
        self.agent_heading_rate = (
            self.agent_heading_rate + h_accel * self.h_accel_magnitude
        ).clamp(-self.dpsi_max, self.dpsi_max)
        self.agent_heading_rate = self.agent_heading_rate * alive_float

        if self.min_turn_radius > 0:
            max_omega = (self.agent_speed / self.min_turn_radius).clamp(max=self.dpsi_max)
            self.agent_heading_rate = torch.max(
                torch.min(self.agent_heading_rate, max_omega), -max_omega
            )

        self.agent_heading = (self.agent_heading + self.agent_heading_rate) % (2.0 * math.pi)

        dx = self.agent_speed * torch.cos(self.agent_heading) * self.dt
        dy = self.agent_speed * torch.sin(self.agent_heading) * self.dt
        self.agent_pos = (self.agent_pos + torch.stack([dx, dy], dim=-1)).clamp(self.low, self.high)
        _t = self._prof_lap("env_dynamics", _t)
        self._update_geometry_cache()
        _t = self._prof_lap("env_geom_cache", _t)

        # ================================================================
        # HF radar model (replaces simple jam + kill logic)
        # ================================================================
        jammer_idx = torch.arange(self.n_strikers, self.n_agents, device=self.device)

        # Compute per-(agent, radar) effective ranges from radar equations
        self._compute_hf_radar_eff_range()

        # jam_active: in the HF model every alive jammer always jams
        if jammer_idx.numel() > 0:
            jam_active = self.agent_alive[:, self.n_strikers:]             # [B, nj]
        else:
            jam_active = torch.zeros(B, 0, dtype=torch.bool, device=self.device)
        _t = self._prof_lap("env_hf_radar", _t)

        # ---- radar kills (probabilistic, per-agent effective range) ----
        dist_ar = self._c_dist_ar                                          # [B, A, R]
        in_radar = dist_ar <= self.radar_eff_range_per_agent               # [B, A, R]

        kill_samples = torch.rand(B, A, self.n_radars, device=self.device, generator=self._rng)
        kill_prob = self.radar.kill_probability
        kills_from_radar = in_radar & (kill_samples < kill_prob)
        killed = kills_from_radar.any(dim=-1) & alive
        self.agent_alive = self.agent_alive & (~killed)

        # ---- striker kinetic kills ----
        kill_t = torch.zeros(B, self.n_targets, dtype=torch.bool, device=self.device)
        striker_idx = torch.arange(0, self.n_strikers, device=self.device)

        if striker_idx.numel() > 0:
            rel_st = self.target_pos[:, None, :, :] - self.agent_pos[:, striker_idx, None, :]
            can = self.striker.can_engage(rel_st, self.agent_heading[:, striker_idx][:, :, None])
            can = can & alive[:, striker_idx, None] & self.target_alive[:, None, :]
            kill_t = can.any(dim=1)
            self.target_alive = self.target_alive & (~kill_t)

        alive = self.agent_alive
        alive_float = alive.float()
        _t = self._prof_lap("env_kills", _t)

        # ==================================================================
        # Reward computation  (identical to parent — copied verbatim)
        # ==================================================================
        reward = torch.zeros(B, A, device=self.device)
        max_dist = math.hypot(self.high - self.low, self.high - self.low)
        ts = float(rp.team_spirit)

        # 1. Target destroyed
        n_killed = kill_t.float().sum(dim=-1)
        n_alive = self.agent_alive.float().sum(dim=-1).clamp_min(1.0)
        target_destroyed_full = torch.zeros(B, A, device=self.device)
        if float(rp.target_destroyed) != 0.0 and bool(kill_t.any().item()):
            team_share = (n_killed * float(rp.target_destroyed) / n_alive)
            team_comp = team_share.unsqueeze(-1) * alive_float

            indiv_comp = torch.zeros(B, A, device=self.device)
            if striker_idx.numel() > 0:
                engaged_kills = can & kill_t[:, None, :]
                n_engaged_per_target = engaged_kills.float().sum(dim=1).clamp_min(1.0)
                credit_per_pair = engaged_kills.float() * (float(rp.target_destroyed) / n_engaged_per_target[:, None, :])
                indiv_comp[:, :self.n_strikers] = credit_per_pair.sum(dim=-1)

            target_rew = ts * team_comp + (1.0 - ts) * indiv_comp
            reward += target_rew
            target_destroyed_full = target_rew

        # 1b. Terminal bonus
        terminal_bonus_full = torch.zeros(B, A, device=self.device)
        all_targets_done_now = (~self.target_alive).all(dim=-1)
        if float(rp.terminal_bonus) != 0.0 and bool(all_targets_done_now.any().item()):
            terminal_bonus_full[all_targets_done_now] = float(rp.terminal_bonus)
            reward += terminal_bonus_full

        # 2. Border avoidance
        pos = self.agent_pos
        dist_bord = torch.stack([
            pos[..., 0] - self.low,
            self.high - pos[..., 0],
            pos[..., 1] - self.low,
            self.high - pos[..., 1],
        ], dim=-1).min(dim=-1).values
        border_pen = -self._piecewise_lin_exp(
            dist_bord,
            d_max=rp.border_d_max,
            d_knee=rp.border_d_knee,
            w_lin=rp.border_w_lin,
            w_exp=rp.border_w_exp,
            alpha=rp.border_alpha,
        ) * alive_float
        reward += border_pen

        # 3. Timestep penalty
        timestep_rew = float(rp.timestep_penalty) * alive_float
        reward += timestep_rew

        # 4. Radar zone avoidance
        jammed_zone_range = max(self.radar_range - self.jammer.delta_range, 0.0)
        d_zone = dist_ar - jammed_zone_range
        d_zone_min = d_zone.min(dim=-1).values.clamp(min=0.0)
        radar_pen = -self._piecewise_lin_exp(
            d_zone_min,
            d_max=rp.radar_avoid_d_max,
            d_knee=rp.radar_avoid_d_knee,
            w_lin=rp.radar_avoid_w_lin,
            w_exp=rp.radar_avoid_w_exp,
            alpha=rp.radar_avoid_alpha,
        )
        reward += radar_pen

        # 5. Striker approach
        striker_approach_full = torch.zeros(B, A, device=self.device)
        if striker_idx.numel() > 0 and self.n_targets > 0:
            dist_st = self._c_dist_at[:, :self.n_strikers, :]
            mask_t = self.target_alive[:, None, :].expand(-1, self.n_strikers, -1)

            if rp.striker_nearest_only:
                big_dist = torch.where(mask_t, dist_st, torch.full_like(dist_st, 1e6))
                shaped_dist = big_dist.min(dim=-1).values
            else:
                eps = 1e-6
                inv_dist = torch.where(mask_t, 1.0 / (dist_st + eps), torch.zeros_like(dist_st))
                weight_sum = inv_dist.sum(dim=-1, keepdim=True).clamp_min(eps)
                weights = inv_dist / weight_sum
                shaped_dist = (weights * dist_st).sum(dim=-1)

            striker_app = self._piecewise_lin_exp(
                shaped_dist,
                d_max=rp.striker_approach_d_max,
                d_knee=rp.striker_approach_d_knee,
                w_lin=rp.striker_approach_w_lin,
                w_exp=rp.striker_approach_w_exp,
                alpha=rp.striker_approach_alpha,
            )
            striker_zero = self._piecewise_lin_exp(
                torch.zeros((), device=self.device, dtype=dist_st.dtype),
                d_max=rp.striker_approach_d_max,
                d_knee=rp.striker_approach_d_knee,
                w_lin=rp.striker_approach_w_lin,
                w_exp=rp.striker_approach_w_exp,
                alpha=rp.striker_approach_alpha,
            )
            striker_app = striker_app - striker_zero

            any_alive = mask_t.any(dim=-1)
            striker_app = torch.where(any_alive, striker_app, torch.zeros_like(striker_app))

            striker_alive_f = alive[:, :self.n_strikers].float()
            striker_app = striker_app * striker_alive_f
            reward[:, :self.n_strikers] += striker_app
            striker_approach_full[:, :self.n_strikers] = striker_app

        # 6. Jammer approach
        jammer_approach_full = torch.zeros(B, A, device=self.device)
        if jammer_idx.numel() > 0 and self.n_radars > 0:
            dist_jr = self._c_dist_ar[:, self.n_strikers:, :]
            jammer_alive_f = alive[:, self.n_strikers:].float()

            app_vals_j = self._piecewise_lin_exp(
                dist_jr,
                d_max=rp.jammer_approach_d_max,
                d_knee=rp.jammer_approach_d_knee,
                w_lin=rp.jammer_approach_w_lin,
                w_exp=rp.jammer_approach_w_exp,
                alpha=rp.jammer_approach_alpha,
            )

            if rp.jammer_nearest_only:
                nearest_idx_j = dist_jr.argmin(dim=-1, keepdim=True)
                jammer_app = app_vals_j.gather(-1, nearest_idx_j).squeeze(-1)
            else:
                jammer_app = app_vals_j.mean(dim=-1)

            jammer_zero = self._piecewise_lin_exp(
                torch.zeros((), device=self.device, dtype=dist_jr.dtype),
                d_max=rp.jammer_approach_d_max,
                d_knee=rp.jammer_approach_d_knee,
                w_lin=rp.jammer_approach_w_lin,
                w_exp=rp.jammer_approach_w_exp,
                alpha=rp.jammer_approach_alpha,
            )
            jammer_app = jammer_app - jammer_zero

            jammer_app = jammer_app * jammer_alive_f
            reward[:, self.n_strikers:] += jammer_app
            jammer_approach_full[:, self.n_strikers:] = jammer_app

        # 7. Potential-based progress
        jammer_progress_full = torch.zeros(B, A, device=self.device)
        jammer_jam_bonus_full = torch.zeros(B, A, device=self.device)
        if jammer_idx.numel() > 0 and self.n_radars > 0:
            jammer_alive_f = alive[:, self.n_strikers:].float()

            if float(rp.jammer_progress_scale) > 0:
                dist_jr_min_curr, _ = dist_jr.min(dim=-1)
                dist_jr_min_prev, _ = self._jammer_prev_dist.min(dim=-1)
                progress_j = dist_jr_min_prev - dist_jr_min_curr
                jammer_prog = float(rp.jammer_progress_scale) * progress_j * jammer_alive_f
                reward[:, self.n_strikers:] += jammer_prog
                jammer_progress_full[:, self.n_strikers:] = jammer_prog

            if float(rp.jammer_jam_bonus) > 0:
                jam_bonus = float(rp.jammer_jam_bonus) * jam_active.float() * jammer_alive_f
                reward[:, self.n_strikers:] += jam_bonus
                jammer_jam_bonus_full[:, self.n_strikers:] = jam_bonus

            self._jammer_prev_dist = dist_jr.detach()

        striker_progress_full = torch.zeros(B, A, device=self.device)
        if striker_idx.numel() > 0 and self.n_targets > 0:
            if float(rp.striker_progress_scale) > 0:
                dist_st = self._c_dist_at[:, :self.n_strikers, :]
                mask_t_p = self.target_alive[:, None, :].expand(-1, self.n_strikers, -1)
                any_alive_p = mask_t_p.any(dim=-1)
                progress = self._striker_prev_dist - dist_st
                progress = torch.where(mask_t_p, progress, torch.full_like(progress, -1e6))
                progress_max, _ = progress.max(dim=-1)
                progress_max = torch.where(any_alive_p, progress_max.clamp(min=-max_dist),
                                           torch.zeros_like(progress_max))
                striker_alive_f_p = alive[:, :self.n_strikers].float()
                striker_contrib = float(rp.striker_progress_scale) * progress_max * striker_alive_f_p
                reward[:, :self.n_strikers] += striker_contrib
                striker_progress_full[:, :self.n_strikers] = striker_contrib

            self._striker_prev_dist = self._c_dist_at[:, :self.n_strikers, :].detach()

        # 8. Formation cohesion
        formation_full = torch.zeros(B, A, device=self.device)
        ns, nj = self.n_strikers, self.n_jammers

        if ns > 0 and nj > 0:
            striker_pos = self.agent_pos[:, :ns, :]
            jammer_pos = self.agent_pos[:, ns:, :]
            d_sj = torch.linalg.norm(
                striker_pos[:, :, None, :] - jammer_pos[:, None, :, :], dim=-1
            )

            if float(rp.striker_formation_scale) > 0:
                dead_j = ~self.agent_alive[:, ns:].unsqueeze(1).expand(B, ns, nj)
                d_near_j = d_sj.masked_fill(dead_j, float('inf')).min(dim=-1).values
                striker_form = float(rp.striker_formation_scale) * (
                    (1.0 - d_near_j / float(rp.striker_formation_ref_dist)).clamp(min=0.0) - 1.0
                ) * alive[:, :ns].float()
                reward[:, :ns] += striker_form
                formation_full[:, :ns] = striker_form

            if float(rp.jammer_formation_scale) > 0:
                d_js = d_sj.transpose(1, 2)
                dead_s = ~self.agent_alive[:, :ns].unsqueeze(1).expand(B, nj, ns)
                d_near_s = d_js.masked_fill(dead_s, float('inf')).min(dim=-1).values
                jammer_form = float(rp.jammer_formation_scale) * (
                    (1.0 - d_near_s / float(rp.jammer_formation_ref_dist)).clamp(min=0.0) - 1.0
                ) * alive[:, ns:].float()
                reward[:, ns:] += jammer_form
                formation_full[:, ns:] = jammer_form

        # 9. Agent destruction penalty
        death_pen_full = torch.zeros(B, A, device=self.device)
        n_killed_agents = killed.float().sum(dim=-1)
        if float(rp.agent_destroyed) != 0.0 and bool(killed.any().item()):
            team_death = (n_killed_agents * float(rp.agent_destroyed) / n_alive)
            team_death_comp = team_death.unsqueeze(-1) * alive_float
            indiv_death_comp = killed.float() * float(rp.agent_destroyed)
            death_pen = ts * team_death_comp + (1.0 - ts) * indiv_death_comp
            reward += death_pen
            death_pen_full = death_pen

        # 10. Paper-style mission reward
        mission_reward_full = torch.zeros(B, A, device=self.device)
        if bool(rp.use_paper_mission_reward):
            n_targets_alive = self.target_alive.float().sum(dim=-1)
            n_agents_alive = self.agent_alive.float().sum(dim=-1)
            n_targets_initial = int(self.n_targets)
            n_agents_initial = int(self.n_agents)

            def _paper_reward_fn(a: torch.Tensor, b: int) -> torch.Tensor:
                if b <= 0:
                    return torch.zeros_like(a)
                b_f = float(b)
                raw = (-torch.exp(-a / b_f) - math.exp(-1.0)) / (1.0 - math.exp(-1.0))
                clipped = torch.maximum(raw, torch.full_like(raw, -10.0))
                return torch.where(a < b_f, clipped, torch.zeros_like(clipped))

            mission_scalar = -_paper_reward_fn(n_targets_alive, n_targets_initial) + _paper_reward_fn(n_agents_alive, n_agents_initial)
            mission_contrib = float(rp.mission_reward_weight) * mission_scalar.unsqueeze(-1) * self.agent_alive.float()
            reward += mission_contrib
            mission_reward_full = mission_contrib

        # 11. Same-role separation penalty
        separation_pen_full = torch.zeros(B, A, device=self.device)

        if ns > 1 and float(rp.striker_sep_d_max) > 0.0:
            striker_pos = self.agent_pos[:, :ns, :]
            d_ss = torch.cdist(striker_pos, striker_pos)
            eye_ss = torch.eye(ns, dtype=torch.bool, device=self.device).unsqueeze(0)
            striker_alive = self.agent_alive[:, :ns]
            valid_ss = striker_alive.unsqueeze(-1) & striker_alive.unsqueeze(-2) & (~eye_ss)
            d_ss_valid = d_ss.masked_fill(~valid_ss, float("inf"))
            d_ss_nearest = d_ss_valid.min(dim=-1).values
            has_neighbor_ss = torch.isfinite(d_ss_nearest)
            striker_sep = -self._piecewise_lin_exp(
                d_ss_nearest,
                d_max=rp.striker_sep_d_max,
                d_knee=rp.striker_sep_d_knee,
                w_lin=rp.striker_sep_w_lin,
                w_exp=rp.striker_sep_w_exp,
                alpha=rp.striker_sep_alpha,
            )
            striker_sep = torch.where(has_neighbor_ss & striker_alive, striker_sep, torch.zeros_like(striker_sep))
            reward[:, :ns] += striker_sep
            separation_pen_full[:, :ns] += striker_sep

        if nj > 1 and float(rp.jammer_sep_d_max) > 0.0:
            jammer_pos = self.agent_pos[:, ns:, :]
            d_jj = torch.cdist(jammer_pos, jammer_pos)
            eye_jj = torch.eye(nj, dtype=torch.bool, device=self.device).unsqueeze(0)
            jammer_alive = self.agent_alive[:, ns:]
            valid_jj = jammer_alive.unsqueeze(-1) & jammer_alive.unsqueeze(-2) & (~eye_jj)
            d_jj_valid = d_jj.masked_fill(~valid_jj, float("inf"))
            d_jj_nearest = d_jj_valid.min(dim=-1).values
            has_neighbor_jj = torch.isfinite(d_jj_nearest)
            jammer_sep = -self._piecewise_lin_exp(
                d_jj_nearest,
                d_max=rp.jammer_sep_d_max,
                d_knee=rp.jammer_sep_d_knee,
                w_lin=rp.jammer_sep_w_lin,
                w_exp=rp.jammer_sep_w_exp,
                alpha=rp.jammer_sep_alpha,
            )
            jammer_sep = torch.where(has_neighbor_jj & jammer_alive, jammer_sep, torch.zeros_like(jammer_sep))
            reward[:, ns:] += jammer_sep
            separation_pen_full[:, ns:] += jammer_sep

        # 12. Control effort penalty
        control_pen_full = torch.zeros(B, A, device=self.device)
        accel_scale = float(rp.accel_effort_scale)
        angular_scale = float(rp.angular_effort_scale)
        if (accel_scale > 0 or angular_scale > 0):
            control_pen = -(accel_scale * acc[..., 0] ** 2
                            + angular_scale * acc[..., 1] ** 2) * alive_float
            reward += control_pen
            control_pen_full = control_pen

        _t = self._prof_lap("env_rewards", _t)

        # Store per-component reward breakdown
        self.last_reward_components = {
            "target_destroyed":   target_destroyed_full.detach(),
            "terminal_bonus":     terminal_bonus_full.detach(),
            "border_penalty":     border_pen.detach(),
            "timestep_penalty":   timestep_rew.detach(),
            "radar_avoidance":    radar_pen.detach(),
            "striker_approach":   striker_approach_full.detach(),
            "jammer_approach":    jammer_approach_full.detach(),
            "striker_progress":   striker_progress_full.detach(),
            "jammer_progress":    jammer_progress_full.detach(),
            "jammer_jam_bonus":   jammer_jam_bonus_full.detach(),
            "formation":          formation_full.detach(),
            "agent_destroyed":    death_pen_full.detach(),
            "paper_mission":      mission_reward_full.detach(),
            "separation_penalty": separation_pen_full.detach(),
            "control_effort":     control_pen_full.detach(),
        }

        reward = reward.unsqueeze(-1).contiguous()  # [B, A, 1]

        step_team_reward = reward.squeeze(-1).sum(dim=-1)
        self._episode_team_reward += step_team_reward
        for comp_key, comp_tensor in self.last_reward_components.items():
            self._episode_component_reward[comp_key] += comp_tensor.sum(dim=-1)
        _t = self._prof_lap("env_reward_accum", _t)

        # ---- done flags ----
        self.step_count += 1
        all_targets_done = (~self.target_alive).all(dim=-1, keepdim=True)
        all_agents_dead = (~self.agent_alive).all(dim=-1, keepdim=True)
        timeout = self.step_count >= self.max_steps

        terminated = all_targets_done | all_agents_dead
        done = terminated | timeout

        next_td = TensorDict({}, batch_size=[B], device=self.device)
        next_td.set(self._reward_key, reward)
        next_td.set("done", done.to(torch.bool))
        next_td.set("terminated", terminated.to(torch.bool))
        _t = self._prof_lap("env_done_flags", _t)
        self._update_comm_cache()
        _t = self._prof_lap("env_comm_cache", _t)
        next_td.set(self._obs_key, self._build_local_obs())
        _t = self._prof_lap("env_build_local_obs", _t)
        next_td.set("state", self._build_global_state())
        _t = self._prof_lap("env_build_state", _t)
        if self.use_fofe:
            for k, v in self._build_fofe_obs().items():
                next_td.set(("agents", k), v)
            _t = self._prof_lap("env_build_fofe_obs", _t)
            for k, v in self._build_fofe_critic_state().items():
                next_td.set(k, v)
            _t = self._prof_lap("env_build_fofe_critic", _t)

        if bool(done.any().item()):
            for b in range(B):
                if done[b, 0].item():
                    tgt_frac = float((~self.target_alive[b]).float().mean().item())
                    surv_frac = float(self.agent_alive[b].float().mean().item())
                    self._completed_episodes.append({
                        "env_idx": b,
                        "mission_complete": bool(all_targets_done[b, 0].item()),
                        "targets_frac": tgt_frac,
                        "survival_frac": surv_frac,
                        "duration": int(self.step_count[b, 0].item()),
                        "episode_total_reward": float(self._episode_team_reward[b].item()),
                        "episode_component_reward": {
                            comp_key: float(self._episode_component_reward[comp_key][b].item())
                            for comp_key in self._episode_component_reward
                        },
                    })

            self._episode_team_reward[done.squeeze(-1)] = 0.0
            for comp_key in self._episode_component_reward:
                self._episode_component_reward[comp_key][done.squeeze(-1)] = 0.0
        _t = self._prof_lap("env_done_loop", _t)

        return next_td
