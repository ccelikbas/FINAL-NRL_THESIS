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
        # Per-jammer in-cone radar capacity. None → unlimited (every
        # in-cone radar gets jammed by this jammer). Positive int K →
        # each jammer only jams its K closest in-cone radars.
        _cap = getattr(hf_cfg, "jammer_max_jammed_radars", None)
        self._jammer_max_jammed_radars: Optional[int] = (
            None if _cap is None else int(_cap)
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
        # Directional-jammer action extension (kinematic beam model)
        # ------------------------------------------------------------------
        # The HF env adds a 3rd action dimension that drives the jammer
        # beam's angular acceleration (radians/step²). Strikers also
        # receive this dimension but the env discards it — see
        # CombinedPolicy where the striker's dim-2 logits are pinned to
        # the zero-acceleration index so the dimension is a no-op for the
        # striker policy.
        #
        # Action tables (decoded independently per dim):
        #   motion dims 0,1 → 7-value table _act_table  =
        #                     [-1, -0.5, -0.1, 0, +0.1, +0.5, +1]
        #   beam   dim  2  → 9-value table _beam_act_table  =
        #                     [-1, -0.2, -0.1, -0.05, 0, +0.05, +0.1, +0.2, +1]
        #
        # The beam table has finer resolution near zero so the policy can
        # apply small angular accelerations and keep the beam stable.
        # The shared Categorical n is the max of the two so a single
        # action spec covers both — out-of-range indices on the motion
        # dims are masked to near-zero probability at the policy level.
        #
        # The beam itself is a 2D kinematic state per jammer:
        #   jammer_bearing : current beam angle relative to jammer heading
        #                    (radians, wrapped to (-pi, pi]).
        #   beam_rate      : current beam angular velocity (rad/step).
        #
        # On each step:  beam_rate ← clamp(beam_rate + beam_act*beam_h_accel,
        #                                  ±beam_dpsi_max)
        #                jammer_bearing ← wrap(jammer_bearing + beam_rate)
        self._n_motion_choices = int(self.n_choices)               # 7
        self._beam_act_table = torch.tensor(
            [-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5],
            device=self.device,
        )
        self._n_beam_choices = int(self._beam_act_table.numel())   # 9
        # Index of the zero-acceleration entry in the beam table — the
        # striker policy pins dim 2 to this index so its (discarded) beam
        # action contributes nothing to the entropy budget.
        self._beam_zero_idx = int(
            torch.argmin(self._beam_act_table.abs()).item()
        )
        self.n_choices = max(self._n_motion_choices, self._n_beam_choices)
        self.act_dim = 3
        # Beam kinematic constants (rad/step and rad/step²).
        self._beam_dpsi_max = float(getattr(hf_cfg, "beam_dpsi_max", math.pi))
        self._beam_h_accel_magnitude = (
            self._beam_dpsi_max
            * float(getattr(hf_cfg, "beam_h_accel_magnitude_fraction", 0.1))
        )
        # Per-jammer current beam state (radians / rad-per-step).
        self.jammer_bearing = torch.zeros(self.num_envs, self.n_jammers, device=self.device)
        self.beam_rate = torch.zeros(self.num_envs, self.n_jammers, device=self.device)
        # Register HF-only reward components so per-episode accumulators
        # and the visualizer's reward subplot pick them up automatically.
        self._episode_component_reward["jammer_beam_on_radar_bonus"] = torch.zeros(
            self.num_envs, device=self.device,
        )
        self._episode_component_reward["jammer_beam_alignment"] = torch.zeros(
            self.num_envs, device=self.device,
        )
        self._episode_component_reward["beam_control_effort"] = torch.zeros(
            self.num_envs, device=self.device,
        )
        self._episode_component_reward["hf_margin_reward"] = torch.zeros(
            self.num_envs, device=self.device,
        )
        # Rebuild specs so action_spec reflects the new act_dim/n_choices.
        self._make_specs()

        # --- Fine-grained step profiling (off by default) ---
        self._profile_active: bool = False
        self._profile_buckets: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Self-obs extension: expose beam state to the jammer policy
    # ------------------------------------------------------------------

    def _self_extra_dim(self) -> int:
        # Two extra floats per agent: (beam_angle / pi, beam_rate / beam_dpsi_max).
        # Both are zero for strikers (no beam).
        return 2

    def _build_self_extra(self, B: int, A: int) -> torch.Tensor:
        extra = torch.zeros(B, A, 2, device=self.device, dtype=torch.float32)
        if self.n_jammers > 0:
            beam_dpsi = max(float(getattr(self, "_beam_dpsi_max", math.pi)), 1e-8)
            extra[:, self.n_strikers:, 0] = self.jammer_bearing / math.pi
            extra[:, self.n_strikers:, 1] = self.beam_rate / beam_dpsi
        return extra

    # ------------------------------------------------------------------
    # Radar-obs extension: per-(agent, radar) beam-to-radar angle (signed)
    # and a beam-on-radar jammed flag instead of effective-range degradation.
    # ------------------------------------------------------------------

    def _radar_extra_dim(self) -> int:
        # Three extra floats per (agent, radar):
        #   col 0 — signed beam→radar angle / pi (jammer-only steering)
        #   col 1 — range margin to R_unc, signed, in [-1, +1]
        #           (positive = striker outside R_unc = safe; both roles)
        #   col 2 — angular margin to the jammed cone, signed, in [-1, +1]
        #           (positive = inside the cone = safe; zero when radar
        #           not jammed; both roles)
        return 3

    def _build_radar_extra(self, B: int, A: int, R: int) -> torch.Tensor:
        """[B, A, R, 3] — three per-(agent, radar) extras for the actor.

        Column 0 (jammer-only): signed (beam_pointing − radar_bearing)
            wrapped to [-pi, pi] and normalised to [-1, 1]. Strikers and
            dead jammers contribute zero. For multiple jammers we expose
            the value for the SMALLEST |angle| (the jammer whose beam is
            most on-axis with this radar).

        Column 1 (both roles): signed range margin against R_unc.
            m_range = (d − R_unc)/R_unc           if d ≤ R_unc   (→ [-1, 0])
                      (d − R_unc)/(D − R_unc)     if d  > R_unc   (→ [ 0,+1])
            with d = ‖agent − radar‖ in world units and D = world
            diagonal. Positive = striker beyond R_unc = safe.

        Column 2 (both roles): signed angular margin against the side-lobe
            outer cone boundary (±θ_s = ±θ_side/2 from the radar→jammer
            direction), evaluated against the DEEPEST-CUT jammer at this
            (agent, radar) pair (cached by _compute_hf_radar_eff_range).
            With δ* the wrapped abs angle in [0, π]:
                raw_θ = θ_s − δ*
                inside  (raw_θ ≥ 0):  raw_θ / θ_s        → [ 0,+1]
                outside (raw_θ < 0):  raw_θ / (π − θ_s)  → [-1, 0]
            Multiplied by `_radar_jammed_flag` so the column is identically
            zero on radars no alive jammer is currently pointing at.
        """
        extra = torch.zeros(B, A, R, 3, device=self.device, dtype=torch.float32)
        if R == 0:
            return extra

        # ----- Column 0: signed beam→radar angle (jammer-only) -----
        if self.n_jammers > 0:
            # Recompute the signed delta inline so this works during _reset
            # too (jammer_bearing is post-reset and we want the same value
            # _build_radar_extra has always returned, independent of caches).
            jammer_pointing = self.agent_heading[:, self.n_strikers:] + self.jammer_bearing  # [B, J]
            rel_jr = self.radar_pos[:, None, :, :] - self.agent_pos[:, self.n_strikers:, None, :]  # [B, J, R, 2]
            angle_j_to_r = torch.atan2(rel_jr[..., 1], rel_jr[..., 0])                        # [B, J, R]
            delta_signed = angle_j_to_r - jammer_pointing[:, :, None]
            delta_signed = torch.atan2(torch.sin(delta_signed), torch.cos(delta_signed))      # [-pi, pi]

            jammer_alive = self.agent_alive[:, self.n_strikers:]                              # [B, J]
            abs_delta = delta_signed.abs().masked_fill(~jammer_alive[:, :, None], float("inf"))  # [B, J, R]
            best_jammer = abs_delta.argmin(dim=1, keepdim=True)                               # [B, 1, R]
            best_delta = delta_signed.gather(1, best_jammer).squeeze(1)                       # [B, R]

            any_alive = jammer_alive.any(dim=1, keepdim=True).expand_as(best_delta)
            best_delta = torch.where(any_alive, best_delta, torch.zeros_like(best_delta))

            extra[:, self.n_strikers:, :, 0] = (
                best_delta[:, None, :].expand(B, self.n_jammers, R) / math.pi
            )

        # ----- Column 1: range margin to R_unc (both roles) -----
        # _c_dist_ar is refreshed by _update_geometry_cache, so this value
        # is consistent with the rest of the obs even in the reset slice.
        eps = 1e-8
        dist_ar = self._c_dist_ar                                              # [B, A, R] world units
        R_unc = float(self.radar_range_unconstrained)
        D = math.hypot(self.high - self.low, self.high - self.low)             # world diagonal
        inside_norm = max(R_unc, eps)
        outside_norm = max(D - R_unc, eps)
        raw_d = dist_ar - R_unc
        m_in = (raw_d / inside_norm).clamp(min=-1.0, max=0.0)
        m_out = (raw_d / outside_norm).clamp(min=0.0, max=1.0)
        extra[..., 1] = torch.where(raw_d <= 0.0, m_in, m_out)

        # ----- Column 2: angular margin to jammed cone (both roles) -----
        # Zero by construction when there are no jammers or the cache is
        # empty; gated to zero on radars no alive jammer is covering.
        if self.n_jammers > 0 and self._delta_jar.shape[1] > 0:
            # Gather δ* for each (a, r) using the cached deepest-cut jammer
            # index. _delta_jar: [B, J, A, R];  _deepest_jammer_idx: [B, A, R].
            deepest_idx = self._deepest_jammer_idx.unsqueeze(1)                # [B, 1, A, R]
            delta_a = self._delta_jar.gather(1, deepest_idx).squeeze(1)        # [B, A, R] in [0, π]
            theta_s = float(self._theta_side_half)
            theta_in_norm = max(theta_s, eps)
            theta_out_norm = max(math.pi - theta_s, eps)
            raw_t = theta_s - delta_a                                          # positive inside, negative outside
            ang_in = (raw_t / theta_in_norm).clamp(min=0.0, max=1.0)
            ang_out = (raw_t / theta_out_norm).clamp(min=-1.0, max=0.0)
            m_angle = torch.where(raw_t >= 0.0, ang_in, ang_out)
            jammed = self._radar_jammed_flag()                                 # [B, R] in {0, 1}
            extra[..., 2] = m_angle * jammed[:, None, :]

        return extra

    # ------------------------------------------------------------------
    # Critic-side per-radar extras: jamming-cone centre-axis direction
    # ------------------------------------------------------------------

    def _critic_radar_extra_dim(self) -> int:
        # Two extra floats per radar: (sin θ*, cos θ*) of the radar→jammer
        # bearing for the deepest-cut alive in-cone jammer. Both zero when
        # no alive jammer is currently covering the radar.
        return 2

    def _build_critic_radar_extra(self, B: int, R: int) -> torch.Tensor:
        """[B, R, 2] — (sin θ*, cos θ*) of the cone centre-axis bearing.

        θ*(b, r) = atan2(jy − ry, jx − rx) for the deepest-cut jammer at
        radar r, defined as the alive in-cone jammer closest to the radar
        (smallest R_J, which monotonically gives the smallest R_main /
        R_side). Both columns are multiplied by `_radar_jammed_flag`, so
        the feature is identically zero when no alive jammer covers the
        radar (including n_jammers == 0).
        """
        extra = torch.zeros(B, R, 2, device=self.device, dtype=torch.float32)
        if self.n_jammers == 0 or R == 0:
            return extra

        ns = self.n_strikers
        # Jammer→radar geometry in world frame.
        rel_jr = self.agent_pos[:, ns:, None, :] - self.radar_pos[:, None, :, :]  # [B, J, R, 2]
        dist_jr = self._c_dist_ar[:, ns:, :]                                       # [B, J, R]
        bearing_jr = torch.atan2(rel_jr[..., 1], rel_jr[..., 0])                   # [B, J, R]

        # Restrict the argmin to (alive ∧ in_cone) jammers; everyone else
        # gets +inf so they lose the race. `_jammer_in_cone` is cached by
        # _compute_hf_radar_eff_range (called via _update_geometry_cache,
        # so it is fresh on both step and reset-slice paths).
        jammer_alive = self.agent_alive[:, ns:]                                    # [B, J]
        in_cone = getattr(self, "_jammer_in_cone", None)
        if in_cone is None or in_cone.shape != (B, self.n_jammers, R):
            in_cone = torch.zeros(B, self.n_jammers, R, dtype=torch.bool, device=self.device)
        valid = in_cone & jammer_alive[:, :, None]                                 # [B, J, R]
        dist_masked = dist_jr.masked_fill(~valid, float("inf"))
        deepest_idx = dist_masked.argmin(dim=1, keepdim=True)                      # [B, 1, R]
        chosen_bearing = bearing_jr.gather(1, deepest_idx).squeeze(1)              # [B, R]

        jammed = self._radar_jammed_flag()                                         # [B, R] in {0, 1}
        extra[..., 0] = torch.sin(chosen_bearing) * jammed
        extra[..., 1] = torch.cos(chosen_bearing) * jammed
        return extra

    def _radar_jammed_flag(self) -> torch.Tensor:
        """[B, R] float in {0, 1}: True when at least one alive jammer's
        beam main lobe currently covers the radar. Recomputed inline so it
        is correct on reset slices too.
        """
        B = self.num_envs
        R = self.n_radars
        if self.n_jammers == 0 or R == 0:
            return torch.zeros(B, R, device=self.device)

        jammer_pointing = self.agent_heading[:, self.n_strikers:] + self.jammer_bearing  # [B, J]
        rel_jr = self.radar_pos[:, None, :, :] - self.agent_pos[:, self.n_strikers:, None, :]  # [B, J, R, 2]
        angle_j_to_r = torch.atan2(rel_jr[..., 1], rel_jr[..., 0])                        # [B, J, R]
        delta = angle_j_to_r - jammer_pointing[:, :, None]
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))                            # [-pi, pi]

        in_cone = delta.abs() <= self._jammer_main_lobe_half                               # [B, J, R]
        jammer_alive = self.agent_alive[:, self.n_strikers:]                              # [B, J]
        in_cone = in_cone & jammer_alive[:, :, None]
        # Apply the K-closest restriction so this flag matches the
        # active-jamming set used by _compute_hf_radar_eff_range. Use the
        # cached jammer→radar distance kept fresh by _update_geometry_cache.
        dist_jr = self._c_dist_ar[:, self.n_strikers:, :]                                 # [B, J, R]
        in_cone = self._restrict_to_closest_in_cone_radars(in_cone, dist_jr)
        return in_cone.any(dim=1).float()                                                 # [B, R]

    # ------------------------------------------------------------------
    # Reset hook — zero the beam state on episode reset
    # ------------------------------------------------------------------

    def _reset(self, tensordict=None, **kwargs):
        # Zero beam state for the envs being reset BEFORE building obs
        # so the first observation post-reset reflects the cleared beam.
        # Times the whole reset under the 'env_reset' profile bucket so the
        # FINE profile can show the gap between rollout total and env_sum.
        _t_total = self._prof_tic()
        reset_mask = self._extract_reset_mask(tensordict)
        reset_idx = reset_mask.nonzero(as_tuple=False).squeeze(-1)
        if reset_idx.numel() > 0 and self.n_jammers > 0:
            self.jammer_bearing[reset_idx] = 0.0
            self.beam_rate[reset_idx] = 0.0
        result = super()._reset(tensordict, **kwargs)
        self._prof_lap("env_reset", _t_total)
        return result

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
    # Per-jammer "K closest in-cone radars" restriction
    # ------------------------------------------------------------------

    def _restrict_to_closest_in_cone_radars(
        self,
        in_cone_jr: torch.Tensor,    # [B, J, R] bool — directional gate (alive optional)
        dist_jr: torch.Tensor,        # [B, J, R] float — jammer→radar distance (any unit)
    ) -> torch.Tensor:
        """Restrict `in_cone_jr` to each jammer's K closest in-cone radars.

        K = `self._jammer_max_jammed_radars`. If None or K >= R, returns
        `in_cone_jr` unchanged. Out-of-cone entries always stay False.

        Uses `torch.topk(largest=False)` on a distance tensor where
        out-of-cone entries are pushed to +inf, then AND-masks with the
        original in-cone mask so that when fewer than K radars are
        in-cone the extra topk picks (which sit at +inf) are dropped.
        """
        K = self._jammer_max_jammed_radars
        if K is None:
            return in_cone_jr
        R = in_cone_jr.shape[-1]
        if R == 0 or K >= R:
            return in_cone_jr
        dist_masked = dist_jr.masked_fill(~in_cone_jr, float("inf"))
        _, top_idx = dist_masked.topk(K, dim=-1, largest=False, sorted=False)  # [B, J, K]
        keep = torch.zeros_like(in_cone_jr)
        keep.scatter_(-1, top_idx, True)
        return in_cone_jr & keep

    # ------------------------------------------------------------------
    # Geometry-cache override
    # ------------------------------------------------------------------

    def _update_geometry_cache(self) -> None:
        """Refresh base geometry + HF jamming geometry in one pass.

        The base method populates pairwise distance / bearing caches. HF
        downstream consumers (radar_eff_range_per_agent, _delta_jar,
        _deepest_jammer_idx, …) also need to stay in sync with current
        positions — including reset-slice rebuilds where _step never runs.
        Calling _compute_hf_radar_eff_range here keeps both paths
        consistent without duplicating geometry code in _build_radar_extra.
        """
        super()._update_geometry_cache()
        self._compute_hf_radar_eff_range()

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
            # Empty cone-membership / angular-delta tensors so reward code
            # can index them safely.
            self._jammer_in_cone = torch.zeros(
                B, 0, R, dtype=torch.bool, device=self.device,
            )
            self._jammer_abs_angular_delta = torch.zeros(
                B, 0, R, device=self.device,
            )
            # Empty caches for the angular-margin obs feature.
            self._delta_jar = torch.zeros(B, 0, A, R, device=self.device)
            self._deepest_jammer_idx = torch.zeros(
                B, A, R, dtype=torch.long, device=self.device,
            )
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
        abs_delta_point = delta_point.abs()                                 # [B, J, R] in [0, pi]
        in_cone_jr = abs_delta_point <= self._jammer_main_lobe_half         # [B, J, R]
        # Capacity restriction: each jammer can only actively jam its K
        # closest in-cone radars (K = self._jammer_max_jammed_radars).
        # Other in-cone radars are demoted to "not actively jammed by this
        # jammer", so R_eff for that (j, *, r) pair falls back to R_unc.
        in_cone_jr = self._restrict_to_closest_in_cone_radars(
            in_cone_jr, dist_jr_m
        )
        R_eff_jar = torch.where(
            in_cone_jr[:, :, None, :],   # broadcast over A
            R_eff_jar,
            torch.full_like(R_eff_jar, R_unc_m),
        )
        # Cache for reward computation:
        #   _jammer_in_cone           — bool gate for the binary bonus.
        #   _jammer_abs_angular_delta — |beam-to-radar| (rad, in [0, pi])
        #                               for the smooth alignment shaping.
        # Dead jammers will be masked out by alive_mask at the reward site.
        self._jammer_in_cone = in_cone_jr
        self._jammer_abs_angular_delta = abs_delta_point

        # Cache for the angular-margin actor feature:
        #   _delta_jar — |angle(radar→agent) − angle(radar→jammer)| wrapped
        #                to [0, π]; the same `delta` tensor used for R_eff
        #                above.
        self._delta_jar = delta

        # Dead jammers have no effect → set to R_unc
        alive_mask = jammer_alive[:, :, None, None]                        # [B, J, 1, 1]
        R_eff_jar = torch.where(alive_mask, R_eff_jar, R_unc_m)

        # ----- min across jammers (deepest cut wins) → [B, A, R] in meters -----
        _min_result = R_eff_jar.min(dim=1)
        R_eff_ar_m = _min_result.values
        # Cache deepest-cut jammer index for the angular-margin actor feature.
        self._deepest_jammer_idx = _min_result.indices                     # [B, A, R]

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
        #   dim 0 (motion accel)   : discrete index in {0..n_motion_choices-1}, all agents
        #   dim 1 (heading accel)  : discrete index in {0..n_motion_choices-1}, all agents
        #   dim 2 (beam ang accel) : discrete index in {0..n_motion_choices-1}, jammers only
        #                            — strikers IGNORE this dim entirely (it is
        #                              pinned to the zero-accel index by the
        #                              policy and never read here for strikers).
        action = tensordict.get(self._action_key)  # [B, A, 3] discrete
        B, A, _ = action.shape
        rp = self.reward_params

        # Decode motion dims (0, 1) via the 7-value motion table and the
        # beam dim (2) via the 9-value beam table. The shared Categorical
        # n is max(motion, beam) so we clamp each dim into its own valid
        # range before looking up the multiplier. Out-of-range indices on
        # the motion dims are already masked by the policy, so the clamp
        # is just a defensive no-op there.
        motion_idx = action[..., :2].long().clamp(0, self._n_motion_choices - 1)
        acc = self._act_table[motion_idx]                                    # [B, A, 2]
        beam_idx = action[..., 2].long().clamp(0, self._n_beam_choices - 1)  # [B, A]
        beam_acc_all = self._beam_act_table[beam_idx]                        # [B, A] in [-1, 1]

        # Integrate the jammer beam (2D kinematics) for jammers only.
        # Strikers' dim-2 entries are discarded.
        if self.n_jammers > 0:
            jammer_alive_f = self.agent_alive[:, self.n_strikers:].float()
            beam_acc = beam_acc_all[:, self.n_strikers:]  # [B, J]
            self.beam_rate = (
                self.beam_rate + beam_acc * self._beam_h_accel_magnitude
            ).clamp(-self._beam_dpsi_max, self._beam_dpsi_max)
            self.beam_rate = self.beam_rate * jammer_alive_f
            new_bearing = self.jammer_bearing + self.beam_rate
            # Wrap to (-pi, pi]
            self.jammer_bearing = torch.atan2(torch.sin(new_bearing), torch.cos(new_bearing))

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

        # _compute_hf_radar_eff_range already ran inside _update_geometry_cache
        # above (overridden in HF env so reset slices stay consistent).

        # jam_active: in the HF model every alive jammer always jams
        if jammer_idx.numel() > 0:
            jam_active = self.agent_alive[:, self.n_strikers:]             # [B, nj]
        else:
            jam_active = torch.zeros(B, 0, dtype=torch.bool, device=self.device)
        _t = self._prof_lap("env_hf_radar", _t)

        # ---- radar kills (probabilistic, per-agent effective range) ----
        dist_ar = self._c_dist_ar                                          # [B, A, R]
        in_radar = (dist_ar <= self.radar_eff_range_per_agent) & self.radar_present[:, None, :]  # [B, A, R]

        kill_samples = torch.rand(B, A, self.n_radars, device=self.device, generator=self._rng)
        # Per-env kill probability ([B,1]→[B,1,1]); scalar fill when DR is off.
        kills_from_radar = in_radar & (kill_samples < self.radar_kill_prob.unsqueeze(-1))
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
        # Branchless: when kill_t has no Trues, all terms below evaluate to
        # zero. Removing .item() avoids a per-step GPU→CPU sync.
        if float(rp.target_destroyed) != 0.0:
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
        # Branchless: when no env hit the terminal, the masked write is a no-op
        # and `reward += zeros` is a no-op. Removing .item() avoids a GPU sync.
        if float(rp.terminal_bonus) != 0.0:
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

        # 4b. HF margin reward (per-(agent, radar), summed over radars, both roles)
        # Uses already-cached fields: _c_dist_ar, radar_range_unconstrained,
        # _delta_jar, _deepest_jammer_idx, _theta_side_half, _radar_jammed_flag.
        hf_margin_full = torch.zeros(B, A, device=self.device)
        exposed_w   = float(rp.hf_margin_exposed_penalty)
        protected_w = float(rp.hf_margin_protected_penalty)
        outside_w   = float(rp.hf_margin_outside_bonus)
        if (
            (exposed_w != 0.0 or protected_w != 0.0 or outside_w != 0.0)
            and self.n_radars > 0
        ):
            inside_range_ar = dist_ar < float(self.radar_range_unconstrained)  # [B, A, R]

            if self.n_jammers > 0 and self._delta_jar.shape[1] > 0:
                deepest_idx = self._deepest_jammer_idx.unsqueeze(1)            # [B, 1, A, R]
                delta_a = self._delta_jar.gather(1, deepest_idx).squeeze(1)    # [B, A, R] in [0, π]
                inside_cone_geom = delta_a < float(self._theta_side_half)      # [B, A, R]
                jammed_ar = self._radar_jammed_flag().bool()                   # [B, R]
                protected_ar = inside_cone_geom & jammed_ar[:, None, :]
            else:
                protected_ar = torch.zeros(
                    B, A, self.n_radars, dtype=torch.bool, device=self.device,
                )

            exposed_ar = inside_range_ar & (~protected_ar)
            in_cone_ar = inside_range_ar & protected_ar
            outside_ar = ~inside_range_ar

            per_radar = (
                exposed_w   * exposed_ar.float()
                + protected_w * in_cone_ar.float()
                + outside_w   * outside_ar.float()
            )                                                                   # [B, A, R]
            hf_margin = per_radar.sum(dim=-1) * alive_float                     # [B, A]
            reward += hf_margin
            hf_margin_full = hf_margin

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
        jammer_beam_on_radar_full = torch.zeros(B, A, device=self.device)
        jammer_beam_alignment_full = torch.zeros(B, A, device=self.device)
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

            # Beam-on-radar bonus: reward a jammer whenever ANY alive radar
            # lies inside its directional cone (full beam width). Uses
            # _jammer_in_cone (set by _compute_hf_radar_eff_range earlier
            # this step) so the geometry stays consistent with the physics
            # gate. Only alive radars count.
            beam_scale = float(getattr(rp, "jammer_beam_on_radar_bonus", 0.0))
            if beam_scale != 0.0:
                # _jammer_in_cone: [B, J, R] — radar inside this jammer's cone.
                # radar_present excludes masked-out (absent) radars under DR;
                # it is all-True when DR is off.
                any_in_cone = (
                    self._jammer_in_cone & self.radar_present[:, None, :]
                ).any(dim=-1).float()                                              # [B, J]
                beam_bonus = beam_scale * any_in_cone * jammer_alive_f
                reward[:, self.n_strikers:] += beam_bonus
                jammer_beam_on_radar_full[:, self.n_strikers:] = beam_bonus

            # Beam alignment shaping: smooth angular penalty toward the
            # physically nearest radar. Uses _jammer_abs_angular_delta
            # (cached by _compute_hf_radar_eff_range) so the geometry
            # stays consistent with the physics gate.
            #     penalty = -scale * |angle_to_nearest_radar| / pi
            # 0 when the beam points exactly at the nearest radar, and
            # -scale at the worst case (180° away). Applied unconditionally
            # (no in-cone gate) so the gradient is smooth everywhere.
            align_scale = float(getattr(rp, "jammer_beam_alignment_scale", 0.0))
            if align_scale != 0.0:
                # dist_jr already in scope from approach block above — used
                # to pick the *physically nearest* radar per jammer.
                nearest_radar_idx = dist_jr.argmin(dim=-1, keepdim=True)           # [B, J, 1]
                nearest_abs_angle = self._jammer_abs_angular_delta.gather(
                    -1, nearest_radar_idx
                ).squeeze(-1)                                                      # [B, J] in [0, pi]
                alignment_pen = -align_scale * (nearest_abs_angle / math.pi) * jammer_alive_f
                reward[:, self.n_strikers:] += alignment_pen
                jammer_beam_alignment_full[:, self.n_strikers:] = alignment_pen

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
        # Branchless: when no agent died, all terms below are zero. Skipping
        # .item() avoids a per-step GPU→CPU sync.
        if float(rp.agent_destroyed) != 0.0:
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

        # 12. Control effort penalty (motion: all agents; beam: jammers only)
        control_pen_full = torch.zeros(B, A, device=self.device)
        beam_control_pen_full = torch.zeros(B, A, device=self.device)
        accel_scale = float(rp.accel_effort_scale)
        angular_scale = float(rp.angular_effort_scale)
        if (accel_scale > 0 or angular_scale > 0):
            control_pen = -(accel_scale * acc[..., 0] ** 2
                            + angular_scale * acc[..., 1] ** 2) * alive_float
            reward += control_pen
            control_pen_full = control_pen

        beam_scale_eff = float(getattr(rp, "beam_accel_effort_scale", 0.0))
        if beam_scale_eff > 0 and self.n_jammers > 0:
            jammer_alive_f = alive[:, self.n_strikers:].float()
            beam_pen = -beam_scale_eff * (beam_acc_all[:, self.n_strikers:] ** 2) * jammer_alive_f
            reward[:, self.n_strikers:] += beam_pen
            beam_control_pen_full[:, self.n_strikers:] = beam_pen

        _t = self._prof_lap("env_rewards", _t)

        # Store per-component reward breakdown
        self.last_reward_components = {
            "target_destroyed":           target_destroyed_full.detach(),
            "terminal_bonus":             terminal_bonus_full.detach(),
            "border_penalty":             border_pen.detach(),
            "timestep_penalty":           timestep_rew.detach(),
            "radar_avoidance":            radar_pen.detach(),
            "striker_approach":           striker_approach_full.detach(),
            "jammer_approach":            jammer_approach_full.detach(),
            "striker_progress":           striker_progress_full.detach(),
            "jammer_progress":            jammer_progress_full.detach(),
            "jammer_jam_bonus":           jammer_jam_bonus_full.detach(),
            "jammer_beam_on_radar_bonus": jammer_beam_on_radar_full.detach(),
            "jammer_beam_alignment":      jammer_beam_alignment_full.detach(),
            "formation":                  formation_full.detach(),
            "agent_destroyed":            death_pen_full.detach(),
            "paper_mission":              mission_reward_full.detach(),
            "separation_penalty":         separation_pen_full.detach(),
            "control_effort":             control_pen_full.detach(),
            "beam_control_effort":        beam_control_pen_full.detach(),
            "hf_margin_reward":           hf_margin_full.detach(),
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
        timeout = self.step_count >= self.max_steps_t

        terminated = all_targets_done | all_agents_dead
        done = terminated | timeout

        next_td = TensorDict({}, batch_size=[B], device=self.device)
        next_td.set(self._reward_key, reward)
        next_td.set("done", done.to(torch.bool))
        next_td.set("terminated", terminated.to(torch.bool))
        _t = self._prof_lap("env_done_flags", _t)
        self._update_comm_cache()
        _t = self._prof_lap("env_comm_cache", _t)
        # Build outputs ONCE here, reuse for the returned TD and the cached
        # persistent buffers (Step 4 — partial-reset optimisation).
        _local_obs = self._build_local_obs()
        next_td.set(self._obs_key, _local_obs)
        _t = self._prof_lap("env_build_local_obs", _t)
        _global_state = self._build_global_state()
        next_td.set("state", _global_state)
        _t = self._prof_lap("env_build_state", _t)
        _fofe_obs_dict = None
        _fofe_critic_dict = None
        if self.use_fofe:
            _fofe_obs_dict = self._build_fofe_obs()
            for k, v in _fofe_obs_dict.items():
                next_td.set(("agents", k), v)
            _t = self._prof_lap("env_build_fofe_obs", _t)
            _fofe_critic_dict = self._build_fofe_critic_state()
            for k, v in _fofe_critic_dict.items():
                next_td.set(k, v)
            _t = self._prof_lap("env_build_fofe_critic", _t)
        # Persist the freshly built outputs so the next _reset only has to
        # refresh rows for the envs that actually terminated.
        self._cache_step_outputs_(_local_obs, _global_state, _fofe_obs_dict, _fofe_critic_dict)

        # Track completed episode stats in Python list (immune to auto-reset).
        # Vectorised: gather all per-env stats for done envs into ONE tensor,
        # transfer in a single .cpu().tolist() call, then build dicts. This
        # replaces the previous `for b in range(B): if done[b].item()` loop
        # which produced O(B × n_components) GPU→CPU syncs every step.
        done_flat = done.squeeze(-1)                                # [B] bool
        done_idx_t = done_flat.nonzero(as_tuple=True)[0]            # [N_done]
        if done_idx_t.numel() > 0:
            comp_keys = list(self._episode_component_reward.keys())
            tgt_frac_t, surv_frac_t = self._episode_metric_fracs(done_idx_t)                    # [N,1] each
            miss_t       = all_targets_done[done_idx_t].float()                                 # [N,1]
            dur_t        = self.step_count[done_idx_t].float()                                  # [N,1]
            team_r_t     = self._episode_team_reward[done_idx_t].unsqueeze(-1)                  # [N,1]
            comp_stack_t = torch.stack(
                [self._episode_component_reward[k][done_idx_t] for k in comp_keys], dim=-1
            )                                                                                    # [N, n_comp]
            all_data = torch.cat(
                [tgt_frac_t, surv_frac_t, miss_t, dur_t, team_r_t, comp_stack_t], dim=-1
            )                                                                                    # [N, 5 + n_comp]

            # Two batched GPU→CPU transfers total (vs B × n_components before).
            env_indices = done_idx_t.cpu().tolist()
            rows        = all_data.cpu().tolist()

            for i, env_idx in enumerate(env_indices):
                row = rows[i]
                self._completed_episodes.append({
                    "env_idx": int(env_idx),
                    "targets_frac": row[0],
                    "survival_frac": row[1],
                    "mission_complete": bool(row[2]),
                    "duration": int(row[3]),
                    "episode_total_reward": row[4],
                    "episode_component_reward": {
                        k: row[5 + j] for j, k in enumerate(comp_keys)
                    },
                })

            # Avoid duplicate logging if terminal envs are stepped again before reset
            self._episode_team_reward[done_flat] = 0.0
            for comp_key in self._episode_component_reward:
                self._episode_component_reward[comp_key][done_flat] = 0.0
        _t = self._prof_lap("env_done_loop", _t)

        return next_td
