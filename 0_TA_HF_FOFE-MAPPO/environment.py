"""
Vectorized 2-D Strike–EA multi-agent environment.

Batch dimension: num_envs (all tensors are [B, ...]).
Agent group key: "agents".

TensorDict layout
-----------------
reset() root keys  (time t):
    ("agents", "observation")  [B, A, obs_dim]
    "state"                    [B, state_dim]
    "done" / "terminated"      [B, 1]

step() returns a "next" nested TD (time t+1):
    ("agents", "observation")
    ("agents", "reward")       [B, A, 1]
    "state"
    "done" / "terminated"
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from .agents import JammerAgent, RadarAgent, StrikerAgent
from .rewards import RewardConfig


class StrikeEA2DEnv(EnvBase):
    """Vectorised Strike–EA 2-D environment (MARL, TorchRL)."""

    group: str = "agents"

    def __init__(
        self,
        *,
        # --- world / episode ---
        num_envs:     int   = 128,
        n_strikers:   int   = 2,
        n_jammers:    int   = 2,
        n_targets:    int   = 2,
        n_radars:     int   = 2,
        n_known_targets: int = 0,
        n_unknown_targets: int = 0,
        n_known_radars: int = 0,
        n_unknown_radars: int = 0,
        max_steps:    int   = 200,
        dt:           float = 1.0,
        world_bounds: Tuple[float, float] = (0.0, 1.0),
        # --- kinematics ---
        v_max:        float = 0.02,
        accel_magnitude: float = 0.01,
        dpsi_max:     float = math.radians(20.0),
        h_accel_magnitude_fraction: float = 0.1,
        min_turn_radius: float = 0.05,
        # --- sensors ---
        R_obs:        float = 0.50,
        R_comm:       float = 0.50,
        # --- striker capabilities ---
        striker_engage_range: float = 0.12,
        striker_engage_fov: float = 60.0,
        striker_v_min: float = 0.01,
        # --- jammer capabilities ---
        jammer_jam_radius: float = 0.25,
        jammer_jam_effect: float = 0.10,
        jammer_v_min: float = 0.01,
        # --- radar threat ---
        radar_range:  float = 0.20,
        radar_kill_probability: float = 1.0,
        # --- reward shaping ---
        border_thresh: float = 0.05,
        reward_config: Optional[RewardConfig] = None,
        # --- target spawn control ---
        target_spawn_angle_range: Tuple[float, float] = (0.0, 360.0),
        # --- misc ---
        device: Optional[torch.device] = None,
        seed:   int = 0,
        n_env_layouts: int = 0,
        radar_min_sep: float = 0.5,
        # --- scenario selection ---
        scenario: str = "S1",
        s2_radar_min_sep: float = 0.2,
        s2_target_min_sep: float = 0.2,
        # --- FOFE mode ---
        use_fofe: bool = False,
        # --- per-environment domain randomization (None = disabled) ---
        dr=None,
    ):
        self._device = device if device is not None else torch.device("cpu")
        super().__init__(device=self._device, batch_size=torch.Size([num_envs]))

        # topology
        self.num_envs   = num_envs
        self.n_strikers = n_strikers
        self.n_jammers  = n_jammers
        self.n_agents   = n_strikers + n_jammers
        if n_known_targets == 0 and n_unknown_targets == 0:
            self.n_targets = n_targets
            self.n_known_targets = n_targets
            self.n_unknown_targets = 0
        else:
            self.n_known_targets = int(n_known_targets)
            self.n_unknown_targets = int(n_unknown_targets)
            self.n_targets = self.n_known_targets + self.n_unknown_targets

        if n_known_radars == 0 and n_unknown_radars == 0:
            self.n_radars = n_radars
            self.n_known_radars = n_radars
            self.n_unknown_radars = 0
        else:
            self.n_known_radars = int(n_known_radars)
            self.n_unknown_radars = int(n_unknown_radars)
            self.n_radars = self.n_known_radars + self.n_unknown_radars

        if self.n_known_targets < 0 or self.n_unknown_targets < 0:
            raise ValueError("n_known_targets and n_unknown_targets must be >= 0")
        if self.n_known_radars < 0 or self.n_unknown_radars < 0:
            raise ValueError("n_known_radars and n_unknown_radars must be >= 0")

        # episode / kinematics
        self.max_steps = max_steps
        self.dt        = dt
        self.low, self.high = world_bounds
        self.v_max     = float(v_max)
        self.accel_magnitude = float(accel_magnitude)
        self.dpsi_max  = float(dpsi_max)
        self.h_accel_magnitude = float(dpsi_max) * float(h_accel_magnitude_fraction)
        self.min_turn_radius   = float(min_turn_radius)

        # sensors
        self.R_obs       = float(R_obs)
        self.R_comm      = float(R_comm)
        self.radar_range = float(radar_range)

        # shaping
        self.border_thresh = float(border_thresh)

        # target spawn angle range (store in radians)
        lo_deg, hi_deg = target_spawn_angle_range
        self.target_spawn_angle_lo = math.radians(lo_deg)
        self.target_spawn_angle_hi = math.radians(hi_deg)

        # reward params
        self.reward_params = reward_config if reward_config is not None else RewardConfig()

        # agent capability objects (with config parameters)
        self.striker = StrikerAgent(
            engage_range=float(striker_engage_range),
            engage_fov_deg=float(striker_engage_fov),
            v_min=float(striker_v_min),
        )
        self.jammer  = JammerAgent(
            jam_radius=float(jammer_jam_radius),
            delta_range=float(jammer_jam_effect),
            v_min=float(jammer_v_min),
        )
        self.radar   = RadarAgent(kill_probability=float(radar_kill_probability))
        self._base_kill_probability = float(radar_kill_probability)

        # ── Per-environment domain randomization ─────────────────────────
        # `dr` is a DomainRandomization (duck-typed: any object exposing the
        # same Optional[(lo, hi)] attributes). When active, each env samples
        # its own counts/scalars at reset; tensors are allocated at the max
        # counts (the values passed above) and surplus entities are masked
        # out per-env via the *_present buffers. When None/inactive the env
        # behaves exactly as before (all entities present, scalar kill_prob).
        self._dr = dr
        self._dr_active = bool(dr is not None and dr.active())
        if self._dr_active:
            self._validate_dr_ranges(dr)

        # RNG
        try:
            self._rng = torch.Generator(device=self.device)
        except TypeError:
            self._rng = torch.Generator()
        self._set_seed(seed)

        # Layout control (pre-generated radar positions for reproducible scenarios)
        self.n_env_layouts = n_env_layouts
        self.radar_min_sep = float(radar_min_sep)
        self.s2_radar_min_sep = float(s2_radar_min_sep)
        self.s2_target_min_sep = float(s2_target_min_sep)
        if self.radar_min_sep < 0.0:
            raise ValueError("radar_min_sep must be >= 0")
        if self.s2_radar_min_sep < 0.0:
            raise ValueError("s2_radar_min_sep must be >= 0")
        if self.s2_target_min_sep < 0.0:
            raise ValueError("s2_target_min_sep must be >= 0")

        # Scenario selection (see EnvConfig docstring):
        #   "S1" = protected targets — radars in top band, targets clustered
        #          around radars (legacy behaviour).
        #   "S2" = defensive line — targets in top band, radars in middle band
        #          (independent of targets).
        scenario = str(scenario).upper()
        if scenario not in ("S1", "S2"):
            raise ValueError(f"scenario must be 'S1' or 'S2', got {scenario!r}")
        self.scenario = scenario

        # Per-scenario sampling boxes for the radar layout pool and (S2) the
        # target uniform draw. Stored as plain floats — _pregenerate_layouts
        # and _reset read them. Keeping them attributes (rather than module
        # constants) makes future per-scenario tweaks a single-line change.
        if self.scenario == "S1":
            self._radar_box = (0.2, 0.8, 0.6, 0.8)  # x_lo, x_hi, y_lo, y_hi
            self._radar_min_sep_active = self.radar_min_sep
        else:  # S2
            self._radar_box = (0.1, 0.9, 0.4, 0.9)
            self._radar_min_sep_active = self.s2_radar_min_sep
        # S2 target box (unused when scenario == "S1").
        self._s2_target_box = (0.1, 0.9, 0.6, 0.9)
        # S2 per-radar boxes: pin radars 0 and 1 to the left/right map edges
        # so they form bookends to the defensive line — preventing the agents
        # from going around the borders. Remaining radars fill the wider band.
        self._s2_radar_boxes: Optional[List[Tuple[float, float, float, float]]] = None
        if self.scenario == "S2" and self.n_radars > 0:
            full_band = self._radar_box
            boxes: List[Tuple[float, float, float, float]] = []
            if self.n_radars >= 1:
                boxes.append((0.0, 0.2, full_band[2], full_band[3]))  # left edge
            if self.n_radars >= 2:
                boxes.append((0.8, 1.0, full_band[2], full_band[3]))  # right edge
            for _ in range(max(0, self.n_radars - 2)):
                boxes.append(full_band)
            self._s2_radar_boxes = boxes

        self._layouts = self._pregenerate_layouts() if n_env_layouts > 0 else None

        # FOFE observation mode
        self.use_fofe = use_fofe

        # dimensions
        self.act_dim    = 2   # [acceleration, angular_acceleration]
        self.n_choices  = 7   # per action dim: {-1, -0.5, -0.1, 0, +0.1, +0.5, +1}
        self._act_table = torch.tensor(
            [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0], device=self._device
        )  # maps discrete index → continuous multiplier
        self.n_other_agent_obs_slots = 3
        self.n_radar_obs_slots = 2
        self.n_target_obs_slots = 2
        # Self-observation dimensionality. Base self block is 6 floats
        # (x, y, speed, heading, heading_rate, t_norm). Subclasses can
        # append role-specific features by overriding _self_extra_dim and
        # _build_self_extra (e.g. HF env adds (beam_angle, beam_rate) for
        # jammers — 2 extra floats, zero for strikers).
        self.d_self    = 6 + int(self._self_extra_dim())
        self.obs_dim   = self._compute_obs_dim()
        self.state_dim = self._compute_state_dim()

        # canonical keys
        self._action_key = (self.group, "action")
        self._reward_key = (self.group, "reward")
        self._obs_key    = (self.group, "observation")

        # allocate state buffers
        B, A, T, R = num_envs, self.n_agents, self.n_targets, self.n_radars
        self.agent_pos     = torch.zeros(B, A, 2, device=self.device)
        self.agent_heading = torch.zeros(B, A,    device=self.device)
        self.agent_speed   = torch.zeros(B, A,    device=self.device)  # Current velocity magnitude
        self.agent_heading_rate = torch.zeros(B, A, device=self.device)  # Current angular velocity
        self.agent_alive   = torch.ones(B, A,     dtype=torch.bool, device=self.device)
        self.target_pos    = torch.zeros(B, T, 2, device=self.device)
        self.target_alive  = torch.ones(B, T,     dtype=torch.bool, device=self.device)
        self.target_known  = torch.zeros(B, T,    dtype=torch.bool, device=self.device)
        self.radar_pos     = torch.zeros(B, R, 2, device=self.device)
        self.radar_known   = torch.zeros(B, R,    dtype=torch.bool, device=self.device)
        self.radar_eff_range = torch.full((B, R), self.radar_range, device=self.device)
        self.step_count    = torch.zeros(B, 1, dtype=torch.int64, device=self.device)
        # ── Per-env DR state (default: everything present, scalar kill_prob,
        #    uniform max_steps — so non-DR runs match the original behaviour) ──
        # *_present[b, i] = True  →  entity slot i is a real entity in env b.
        # Absent agents/targets start with alive=False; absent radars are
        # parked far outside the world so distance-based terms ignore them.
        self.agent_present  = torch.ones(B, A, dtype=torch.bool, device=self.device)
        self.target_present = torch.ones(B, T, dtype=torch.bool, device=self.device)
        self.radar_present  = torch.ones(B, R, dtype=torch.bool, device=self.device)
        # Per-env radar kill probability and episode horizon (scalars when DR off).
        self.radar_kill_prob = torch.full((B, 1), self._base_kill_probability, device=self.device)
        self.max_steps_t     = torch.full((B, 1), float(self.max_steps), device=self.device)
        # Previous striker→target distances for progress reward (potential-based shaping)
        self._striker_prev_dist = torch.zeros(B, n_strikers, self.n_targets, device=self._device)
        # Previous jammer→radar distances for progress reward (potential-based shaping)
        self._jammer_prev_dist = torch.zeros(B, n_jammers, self.n_radars, device=self._device)
        # Running team-total reward per env for current episode (sum over agents)
        self._episode_team_reward = torch.zeros(B, device=self.device)
        # Running per-component team-total reward per env for current episode
        self._episode_component_reward = {
            "target_destroyed": torch.zeros(B, device=self.device),
            "terminal_bonus": torch.zeros(B, device=self.device),
            "border_penalty": torch.zeros(B, device=self.device),
            "timestep_penalty": torch.zeros(B, device=self.device),
            "radar_avoidance": torch.zeros(B, device=self.device),
            "striker_approach": torch.zeros(B, device=self.device),
            "jammer_approach": torch.zeros(B, device=self.device),
            "striker_progress": torch.zeros(B, device=self.device),
            "jammer_progress": torch.zeros(B, device=self.device),
            "jammer_jam_bonus": torch.zeros(B, device=self.device),
            "formation": torch.zeros(B, device=self.device),
            "agent_destroyed": torch.zeros(B, device=self.device),
            "paper_mission": torch.zeros(B, device=self.device),
            "separation_penalty": torch.zeros(B, device=self.device),
            "jammer_coalition_coverage": torch.zeros(B, device=self.device),
            "control_effort": torch.zeros(B, device=self.device),
        }

        # Episode outcome tracking (bypasses tensordict auto-reset overwrite)
        self._completed_episodes: list = []

        self._make_specs()

        # Pairwise geometry cache — populated by _update_geometry_cache /
        # _update_comm_cache every step and consumed by reward + obs builders.
        self._c_rel_ar:    Optional[torch.Tensor] = None  # [B, A, R, 2]
        self._c_dist_ar:   Optional[torch.Tensor] = None  # [B, A, R]
        self._c_angle_ar:  Optional[torch.Tensor] = None  # [B, A, R]
        self._c_rel_at:    Optional[torch.Tensor] = None  # [B, A, T, 2]
        self._c_dist_at:   Optional[torch.Tensor] = None  # [B, A, T]
        self._c_angle_at:  Optional[torch.Tensor] = None  # [B, A, T]
        self._c_rel_aa:    Optional[torch.Tensor] = None  # [B, A, A, 2]
        self._c_dist_aa:   Optional[torch.Tensor] = None  # [B, A, A]
        self._c_angle_aa:  Optional[torch.Tensor] = None  # [B, A, A]
        self._c_comm_reach: Optional[torch.Tensor] = None  # [B, A, A]

        # ------------------------------------------------------------------
        # Persistent output buffers (Step 4 — partial-reset optimisation).
        #
        # Idea: _reset is called every time at least one env emits done. Most
        # rollout steps reset only a handful of envs, but the original code
        # rebuilt obs / state / FOFE for ALL B envs on every reset. Those
        # rebuilds are tensor ops over the full batch and cost ~tens of ms
        # per call.
        #
        # We avoid that cost by:
        #   1. At the end of every _step, caching the just-built local_obs,
        #      global_state, and (if use_fofe) FOFE channels into these
        #      persistent buffers. The clone() detaches them from the TD
        #      that the step returned so we can mutate them freely.
        #   2. In _reset, only rebuilding outputs for the rows in reset_idx
        #      via the temp-slice helper (_update_subset_outputs_), and
        #      writing those rows back into the buffers in place.
        #   3. Returning the buffers as the reset TD. The non-reset rows
        #      carry over unchanged from the previous _step, which is exactly
        #      what their obs/state should be (they haven't moved).
        #
        # Lazy allocation: buffers are None until the first build runs, at
        # which point we allocate them at full [B, ...] shape. The first
        # reset (before any _step) falls back to a full-B build to populate
        # them — that one-time cost is identical to the old behaviour.
        # ------------------------------------------------------------------
        self._obs_buf:           Optional[torch.Tensor] = None      # [B, A, obs_dim]
        self._state_buf:         Optional[torch.Tensor] = None      # [B, state_dim]
        self._fofe_obs_buf:      Dict[str, torch.Tensor] = {}       # FOFE actor channels
        self._fofe_critic_buf:   Dict[str, torch.Tensor] = {}       # FOFE critic channels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_seed(self, seed: int):
        self._rng.manual_seed(int(seed))
        return seed

    @staticmethod
    def _sample_spaced_radars(
        n_radars: int,
        x_lo: float, x_hi: float,
        y_lo: float, y_hi: float,
        min_sep: float,
        rng: torch.Generator,
        device: Optional[torch.device] = None,
        max_attempts: int = 500,
        per_radar_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> torch.Tensor:
        """Sample *n_radars* positions with at least *min_sep* between each pair.

        Uses rejection sampling: draw candidates one-by-one, reject if too
        close to any already-placed radar.

        If *per_radar_boxes* is provided (list of length n_radars), each radar
        i is drawn from its own (x_lo, x_hi, y_lo, y_hi) box and the
        single-box arguments are ignored. This is used by S2 to pin the first
        two radars to the left/right map edges while sampling the rest from
        the wider defensive-line band.

        Raises RuntimeError if the requested spacing cannot be achieved within
        max_attempts. This avoids silently violating the separation constraint.
        """
        sample_device = device if device is not None else torch.device("cpu")
        if n_radars <= 0:
            return torch.zeros(0, 2, device=sample_device)

        min_sep = float(min_sep)
        if min_sep < 0.0:
            raise ValueError("min_sep must be >= 0")

        if per_radar_boxes is not None:
            if len(per_radar_boxes) != n_radars:
                raise ValueError(
                    f"per_radar_boxes length {len(per_radar_boxes)} != n_radars {n_radars}"
                )
            boxes: List[Tuple[float, float, float, float]] = [
                (float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in per_radar_boxes
            ]
        else:
            boxes = [(float(x_lo), float(x_hi), float(y_lo), float(y_hi))] * n_radars

        # Quick impossibility check: pairwise distance between two disjoint
        # boxes can be 0 if they overlap, so we bound feasibility by the
        # diameter of the union bounding box.
        if n_radars > 1 and min_sep > 0.0:
            union_x_lo = min(b[0] for b in boxes)
            union_x_hi = max(b[1] for b in boxes)
            union_y_lo = min(b[2] for b in boxes)
            union_y_hi = max(b[3] for b in boxes)
            max_pairwise_dist = math.hypot(union_x_hi - union_x_lo, union_y_hi - union_y_lo)
            if min_sep > max_pairwise_dist + 1e-9:
                raise ValueError(
                    f"min_sep={min_sep:.3f} is impossible across union box "
                    f"x=[{union_x_lo:.3f},{union_x_hi:.3f}], y=[{union_y_lo:.3f},{union_y_hi:.3f}] "
                    f"(max possible pairwise distance={max_pairwise_dist:.3f})."
                )

        def _draw_candidate(box_i: Tuple[float, float, float, float]) -> torch.Tensor:
            bx_lo, bx_hi, by_lo, by_hi = box_i
            cx = bx_lo + (bx_hi - bx_lo) * torch.rand(1, generator=rng, device=sample_device)
            cy = by_lo + (by_hi - by_lo) * torch.rand(1, generator=rng, device=sample_device)
            return torch.tensor([cx.item(), cy.item()], device=sample_device)

        max_restarts = max(int(max_attempts), 1)
        draws_per_radar = 64

        for _ in range(max_restarts):
            placed: List[torch.Tensor] = []
            success = True

            for i in range(n_radars):
                found = False
                for _ in range(draws_per_radar):
                    candidate = _draw_candidate(boxes[i])
                    if all(torch.linalg.norm(candidate - p).item() >= min_sep for p in placed):
                        placed.append(candidate)
                        found = True
                        break
                if not found:
                    success = False
                    break

            if not success:
                continue

            radar_pos = torch.stack(placed, dim=0)  # [R, 2]
            if n_radars > 1 and min_sep > 0.0:
                min_actual_sep = torch.pdist(radar_pos, p=2).min().item()
                if min_actual_sep + 1e-9 < min_sep:
                    raise RuntimeError(
                        f"Internal radar separation check failed: got min "
                        f"{min_actual_sep:.6f}, expected >= {min_sep:.6f}."
                    )
            return radar_pos

        if per_radar_boxes is not None:
            box_desc = "; ".join(
                f"r{i}=([{b[0]:.3f},{b[1]:.3f}]x[{b[2]:.3f},{b[3]:.3f}])"
                for i, b in enumerate(boxes)
            )
        else:
            box_desc = f"([{x_lo:.3f},{x_hi:.3f}]x[{y_lo:.3f},{y_hi:.3f}])"
        raise RuntimeError(
            "Could not sample radar positions satisfying minimum separation: "
            f"n_radars={n_radars}, min_sep={min_sep:.3f}, "
            f"boxes={box_desc}, restarts={max_restarts}, draws_per_radar={draws_per_radar}. "
            "Increase spawn area, reduce n_radars, or lower radar_min_sep."
        )

    def _pregenerate_layouts(self):
        """Pre-generate a pool of valid radar layouts for domain randomisation.

        Builds n_env_layouts distinct radar placements (each respecting
        radar_min_sep) with deterministic per-layout seeds, then stacks the
        whole pool into a single [L, R, 2] tensor on self.device. _reset
        draws random layout indices from this tensor per env per reset, so
        every env sees a different layout each episode while we still pay
        the slow rejection-sampling cost only once at construction time.
        """
        x_lo, x_hi, y_lo, y_hi = self._radar_box
        min_sep = self._radar_min_sep_active
        per_radar_boxes = self._s2_radar_boxes  # None for S1; corner-pinned list for S2
        layouts = []
        for seed_idx in range(self.n_env_layouts):
            rng = torch.Generator()
            rng.manual_seed(seed_idx + 1000)  # Offset to avoid collision with main RNG
            radar_pos = self._sample_spaced_radars(
                self.n_radars, x_lo, x_hi, y_lo, y_hi, min_sep, rng,
                per_radar_boxes=per_radar_boxes,
            )
            layouts.append(radar_pos)
        # Stack onto self.device once so _reset can fetch positions for an
        # arbitrary batch of envs with a single index op.
        return torch.stack(layouts, dim=0).to(self.device)

    def _compute_obs_dim(self) -> int:
        # Fixed-size ego-centric body-frame observations (independent of team sizes):
        #   own state:     [x, y, speed, heading, heading_rate, t_norm, *extra]     = 6 + d_extra
        #   other agents:  top-K slots [dx, dy, dist, heading, role, *agent_extra]  = K_a * (5 + agent_extra)
        #   radars:        top-K slots [dx, dy, dist, jammed, *radar_extra]          = K_r * (4 + radar_extra)
        #   targets:       top-K slots [dx, dy, dist, alive]                          = K_t * 4
        # (dx, dy) are in the observing agent's body frame; dist is the Euclidean
        # distance for convenience (redundant with sqrt(dx^2+dy^2) but cheap and
        # explicit). Unseen or missing entities are zero-padded.
        return (
            self.d_self
            + self.n_other_agent_obs_slots * (5 + int(self._other_agent_extra_dim()))
            + self.n_radar_obs_slots * (4 + int(self._radar_extra_dim()))
            + self.n_target_obs_slots * 4
        )

    # ------------------------------------------------------------------
    # Self-observation extension hooks
    # ------------------------------------------------------------------
    # Subclasses can append role-specific features to the per-agent
    # self-observation by overriding these two methods. The default
    # implementation contributes zero extra dimensions.

    def _self_extra_dim(self) -> int:
        """Number of extra floats appended to each agent's self-obs."""
        return 0

    def _build_self_extra(self, B: int, A: int) -> torch.Tensor:
        """Return the per-agent self-obs extension as a [B, A, d_extra] tensor.

        Called from _build_local_obs and _build_fofe_obs after the base
        kinematic state. d_extra must equal _self_extra_dim().
        """
        return torch.zeros(B, A, 0, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Other-agent observation extension hooks
    # ------------------------------------------------------------------
    # Subclasses can append role-specific per-(observer, other-agent)
    # features to each "other agents" slot (e.g. HF env adds the observed
    # jammer's beam bearing in the observer's body frame). Features are
    # indexed [B, observer, other], so subclasses are responsible for any
    # body-frame rotation. The default contributes zero extra dimensions.

    def _other_agent_extra_dim(self) -> int:
        """Extra per-(observer, other-agent) floats appended to each slot."""
        return 0

    def _build_other_agent_extra(self, B: int, A: int) -> torch.Tensor:
        """Return the per-(observer, other-agent) extension [B, A, A, d_extra]."""
        return torch.zeros(B, A, A, 0, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Radar-observation extension hooks
    # ------------------------------------------------------------------
    # Subclasses can append role-specific per-radar features to the
    # actor's radar slot (e.g. HF env adds the signed beam-to-radar
    # angle for jammers) and can override the "jammed" flag semantics
    # used by both actor obs and critic state (e.g. HF env uses the
    # beam-in-cone test instead of effective-range degradation).

    def _radar_extra_dim(self) -> int:
        """Extra per-(agent, radar) floats appended to each radar slot."""
        return 0

    def _build_radar_extra(self, B: int, A: int, R: int) -> torch.Tensor:
        """Return the per-(agent, radar) feature extension [B, A, R, d_extra]."""
        return torch.zeros(B, A, R, 0, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Critic-side per-radar feature extension hooks
    # ------------------------------------------------------------------
    # Subclasses can append global per-radar features to the centralised
    # critic's `crt_radars_feat` channel (e.g. HF env adds the cone
    # direction (sin, cos) of the deepest-cut jammer).

    def _critic_radar_extra_dim(self) -> int:
        """Extra per-radar floats appended to `crt_radars_feat`."""
        return 0

    def _build_critic_radar_extra(self, B: int, R: int) -> torch.Tensor:
        """Return the per-radar critic-side extension [B, R, d_extra]."""
        return torch.zeros(B, R, 0, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Critic-side per-agent feature extension hooks
    # ------------------------------------------------------------------
    # Subclasses can append global per-agent features to the centralised
    # critic's `crt_agents_feat` channel and to the flat global-state
    # vector (e.g. HF env appends each jammer's beam_bearing and beam_rate
    # so the critic sees beam state alongside agent kinematics).

    def _critic_agent_extra_dim(self) -> int:
        """Extra per-agent floats appended to `crt_agents_feat` (and flat state)."""
        return 0

    def _build_critic_agent_extra(self, B: int, A: int) -> torch.Tensor:
        """Return the per-agent critic-side extension [B, A, d_extra]."""
        return torch.zeros(B, A, 0, device=self.device, dtype=torch.float32)

    def _radar_jammed_flag(self) -> torch.Tensor:
        """Return [B, R] float in {0, 1}: 1 = radar currently jammed.

        Base semantics: any jammer has degraded the radar's effective
        detection range. HF env overrides to "any alive jammer's beam
        cone covers the radar".
        """
        return (self.radar_eff_range < self.radar_range).float()

    def _compute_state_dim(self) -> int:
        A, T, R = self.n_agents, self.n_targets, self.n_radars
        # Global absolute state for centralised critic:
        #   Per agent:  (x, y, v, ψ, ω, role, alive, *agent_extra)  = (7 + E_a) × A
        #   Per target: (x, y, alive)                                = 3 × T
        #   Per radar:  (x, y, jammed)                               = 3 × R
        #   Global:     (t_norm)                                     = 1
        return (7 + int(self._critic_agent_extra_dim())) * A + 3 * T + 3 * R + 1

    def _spawn_targets_in_valid_zones(self, B: int, T: int, R: int, radar_pos: torch.Tensor) -> torch.Tensor:
        """
        Spawn targets in "strategic zones" with even distribution across radars:
        - Targets are distributed roughly evenly across available radars
        - Each target placed at fixed distance from its assigned radar
        - Spawned in random direction around that radar
        - Works with any combination of T targets and R radars
        
        Examples:
          - 2 targets, 2 radars: ~1 target per radar
          - 3 targets, 2 radars: ~1-2 targets distributed randomly
          - 3 targets, 1 radar: all 3 at that radar
        
        Parameters:
        -----------
        B : int
            Batch size
        T : int
            Number of targets
        R : int
            Number of radars
        radar_pos : torch.Tensor
            Radar positions [B, R, 2]
            
        Returns:
        --------
        torch.Tensor
            Target positions [B, T, 2]
        """
        # Constrained range = base range - jamming effect (minimum detectable range when jammed)
        unconstrained_range = self.radar_range

        # Target spawn distance: 90% of detection range (safe annulus)
        spawn_distance = 0.6 * unconstrained_range

        if T == 0:
            return torch.zeros(B, 0, 2, device=self.device)

        # ------------------------------------------------------------------
        # Vectorised version of the old `for b in range(B): for t in range(T)`
        # loop. Two ideas:
        #   (1) The base radar-assignment template (known-group then unknown-
        #       group, each cycled to fill its slot range) is the SAME across
        #       envs. Compute it once.
        #   (2) Per-env shuffling within each group is just argsort(rand(...)),
        #       which gives independent random permutations along dim -1.
        #   (3) Angles and offsets are independent samples per (env, target).
        # Everything below is a single batched tensor op; no Python loops,
        # no .item() syncs. Same outputs distributionally as the old code.
        # ------------------------------------------------------------------
        known_target_count   = min(self.n_known_targets, T)
        unknown_target_count = max(0, T - known_target_count)

        # ---- Build the cross-env "base" radar-index template (length T) ----
        base_parts: List[torch.Tensor] = []
        if known_target_count > 0:
            if self.n_known_radars > 0:
                known_radar_idx = torch.arange(self.n_known_radars, device=self.device)
                base_known = known_radar_idx[
                    torch.arange(known_target_count, device=self.device) % self.n_known_radars
                ]
            else:
                base_known = torch.arange(known_target_count, device=self.device) % max(R, 1)
            base_parts.append(base_known)
        if unknown_target_count > 0:
            n_unk = self.n_radars - self.n_known_radars
            if n_unk > 0:
                unknown_radar_idx = torch.arange(self.n_known_radars, self.n_radars, device=self.device)
                base_unknown = unknown_radar_idx[
                    torch.arange(unknown_target_count, device=self.device) % n_unk
                ]
            else:
                base_unknown = torch.arange(unknown_target_count, device=self.device) % max(R, 1)
            base_parts.append(base_unknown)
        base = torch.cat(base_parts, dim=0)  # [T]

        # ---- Per-env random permutations within each group ----------------
        # argsort(rand(B, count)) gives an independent permutation per row.
        radar_assignments = torch.empty(B, T, dtype=torch.long, device=self.device)
        if known_target_count > 0:
            perm_k = torch.argsort(
                torch.rand(B, known_target_count, device=self.device), dim=-1
            )                                                              # [B, k]
            radar_assignments[:, :known_target_count] = base[:known_target_count][perm_k]
        if unknown_target_count > 0:
            perm_u = torch.argsort(
                torch.rand(B, unknown_target_count, device=self.device), dim=-1
            )                                                              # [B, u]
            radar_assignments[:, known_target_count:] = base[known_target_count:][perm_u]

        # ---- Sample angles and offsets per (env, target) ------------------
        lo = self.target_spawn_angle_lo
        hi = self.target_spawn_angle_hi
        span = (hi - lo) % (2.0 * math.pi)
        if span == 0.0:
            span = 2.0 * math.pi
        angles = lo + span * torch.rand(B, T, device=self.device)          # [B, T]
        offsets = spawn_distance * torch.stack(
            (torch.cos(angles), torch.sin(angles)), dim=-1
        )                                                                   # [B, T, 2]

        # ---- Gather the assigned radar position for each (env, target) ----
        gather_idx = radar_assignments.unsqueeze(-1).expand(B, T, 2)       # [B, T, 2]
        assigned_radars = radar_pos.gather(1, gather_idx)                  # [B, T, 2]

        target_pos = (assigned_radars + offsets).clamp(self.low, self.high)
        return target_pos

    def _spawn_targets_uniform_box(
        self,
        B: int,
        T: int,
        box: Tuple[float, float, float, float],
        min_sep: Optional[float] = None,
        max_tries: int = 100,
    ) -> torch.Tensor:
        """S2 target sampler: uniform random positions inside an axis-aligned box.

        Targets are independent of radar positions. When ``min_sep > 0`` a
        minimum pairwise separation between targets is enforced via batched
        rejection sampling: only the environments whose closest target pair is
        below ``min_sep`` are redrawn, up to ``max_tries`` rounds. If a few
        environments still violate the constraint after the retry budget is
        exhausted (e.g. the box is too small to fit ``T`` targets at
        ``min_sep``), the last draw is accepted rather than crashing.

        ``min_sep=None`` reads ``self.s2_target_min_sep``; pass ``0.0`` to skip
        the constraint and recover the original single-shot vectorised draw.
        """
        if T == 0:
            return torch.zeros(B, 0, 2, device=self.device)
        x_lo, x_hi, y_lo, y_hi = box
        scale = torch.tensor([x_hi - x_lo, y_hi - y_lo], device=self.device)
        offset = torch.tensor([x_lo, y_lo], device=self.device)

        def _draw(n: int) -> torch.Tensor:
            u = torch.rand(n, T, 2, device=self.device)
            return (u * scale + offset).clamp(self.low, self.high)

        pos = _draw(B)

        if min_sep is None:
            min_sep = self.s2_target_min_sep
        # No constraint possible / requested: keep the cheap single-shot draw.
        if min_sep <= 0.0 or T < 2:
            return pos

        def _violates(p: torch.Tensor) -> torch.Tensor:
            # p: [n, T, 2] -> [n] bool, True where any pair is closer than min_sep.
            d = torch.cdist(p, p)                                  # [n, T, T]
            eye = torch.eye(T, dtype=torch.bool, device=self.device)
            d = d.masked_fill(eye, float("inf"))                   # ignore self-distance
            return d.amin(dim=(-2, -1)) < min_sep

        bad = _violates(pos)
        tries = 0
        while bool(bad.any()) and tries < max_tries:
            idx = bad.nonzero(as_tuple=True)[0]
            pos[idx] = _draw(idx.numel())
            bad = _violates(pos)
            tries += 1
        return pos

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------

    def _make_specs(self):
        B = self.batch_size
        A, T, R = self.n_agents, self.n_targets, self.n_radars

        obs_spec   = Unbounded(shape=B + torch.Size([A, self.obs_dim]),   dtype=torch.float32, device=self.device)
        act_spec   = Categorical(n=self.n_choices, shape=B + torch.Size([A, self.act_dim]), dtype=torch.int64, device=self.device)
        rew_spec   = Unbounded(shape=B + torch.Size([A, 1]),              dtype=torch.float32, device=self.device)
        state_spec = Unbounded(shape=B + torch.Size([self.state_dim]),    dtype=torch.float32, device=self.device)
        done_leaf  = Categorical(n=2, shape=B + torch.Size([1]),          dtype=torch.bool,    device=self.device)

        agents_dict = {"observation": obs_spec}
        root_dict = {}

        if self.use_fofe:
            d_radar_slot = 4 + int(self._radar_extra_dim())
            d_agent_slot = 5 + int(self._other_agent_extra_dim())
            # FOFE actor per-channel observations (nested under "agents")
            agents_dict["obs_self"]          = Unbounded(shape=B + torch.Size([A, self.d_self]),    dtype=torch.float32, device=self.device)
            agents_dict["obs_agents_feat"]   = Unbounded(shape=B + torch.Size([A, A, d_agent_slot]), dtype=torch.float32, device=self.device)
            agents_dict["obs_agents_mask"]   = Unbounded(shape=B + torch.Size([A, A]),    dtype=torch.bool,    device=self.device)
            agents_dict["obs_targets_feat"]  = Unbounded(shape=B + torch.Size([A, T, 4]), dtype=torch.float32, device=self.device)
            agents_dict["obs_targets_mask"]  = Unbounded(shape=B + torch.Size([A, T]),    dtype=torch.bool,    device=self.device)
            agents_dict["obs_radars_feat"]   = Unbounded(shape=B + torch.Size([A, R, d_radar_slot]), dtype=torch.float32, device=self.device)
            agents_dict["obs_radars_mask"]   = Unbounded(shape=B + torch.Size([A, R]),    dtype=torch.bool,    device=self.device)
            # FOFE critic global-state channels (root-level keys)
            d_crt_agent_slot = 7 + int(self._critic_agent_extra_dim())
            root_dict["crt_agents_feat"]   = Unbounded(shape=B + torch.Size([A, d_crt_agent_slot]), dtype=torch.float32, device=self.device)
            root_dict["crt_agents_mask"]   = Unbounded(shape=B + torch.Size([A]),    dtype=torch.bool,    device=self.device)
            root_dict["crt_targets_feat"]  = Unbounded(shape=B + torch.Size([T, 3]), dtype=torch.float32, device=self.device)
            root_dict["crt_targets_mask"]  = Unbounded(shape=B + torch.Size([T]),    dtype=torch.bool,    device=self.device)
            d_crt_radar_slot = 3 + int(self._critic_radar_extra_dim())
            root_dict["crt_radars_feat"]   = Unbounded(shape=B + torch.Size([R, d_crt_radar_slot]), dtype=torch.float32, device=self.device)
            root_dict["crt_radars_mask"]   = Unbounded(shape=B + torch.Size([R]),    dtype=torch.bool,    device=self.device)
            root_dict["crt_time_feat"]     = Unbounded(shape=B + torch.Size([1]),    dtype=torch.float32, device=self.device)

        self.observation_spec = Composite(
            agents=Composite(**agents_dict, shape=B, device=self.device),
            state=state_spec, **root_dict, shape=B, device=self.device,
        )
        self.action_spec = Composite(
            agents=Composite(action=act_spec, shape=B, device=self.device),
            shape=B, device=self.device,
        )
        self.reward_spec = Composite(
            agents=Composite(reward=rew_spec, shape=B, device=self.device),
            shape=B, device=self.device,
        )
        self.done_spec = Composite(
            done=done_leaf, terminated=done_leaf.clone(),
            shape=B, device=self.device,
        )

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _extract_reset_mask(self, tensordict: Optional[TensorDict]) -> torch.Tensor:
        """Return boolean reset mask [B] from TorchRL reset tensordict.

        If no reset indicator is found, resets all environments.
        """
        if tensordict is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        candidate_keys = [
            "_reset",
            (self.group, "_reset"),
            ("next", "_reset"),
            ("next", self.group, "_reset"),
        ]

        reset_mask = None
        for key in candidate_keys:
            try:
                value = tensordict.get(key)
                if value is not None:
                    reset_mask = value
                    break
            except Exception:
                continue

        if reset_mask is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        reset_mask = reset_mask.to(self.device)
        if reset_mask.dtype is not torch.bool:
            reset_mask = reset_mask.bool()
        if reset_mask.ndim > 1 and reset_mask.shape[-1] == 1:
            reset_mask = reset_mask.squeeze(-1)
        reset_mask = reset_mask.reshape(-1)

        if reset_mask.numel() == 1:
            if bool(reset_mask.item()):
                return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if reset_mask.numel() != self.num_envs:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        return reset_mask

    # ------------------------------------------------------------------
    # Domain-randomization helpers
    # ------------------------------------------------------------------
    # Coordinate well outside world_bounds where absent radars are parked.
    _ABSENT_RADAR_COORD: float = 100.0

    def _validate_dr_ranges(self, dr) -> None:
        """Validate DR ranges are well-formed and within the allocated maxima."""
        def _chk(name: str, rng, max_val: int) -> None:
            if rng is None:
                return
            lo, hi = int(rng[0]), int(rng[1])
            if lo > hi:
                raise ValueError(f"dr.{name} has lo>hi: {rng}")
            if lo < 0:
                raise ValueError(f"dr.{name} must be >= 0: {rng}")
            if hi > max_val:
                raise ValueError(
                    f"dr.{name} hi={hi} exceeds allocated max {max_val}; set the "
                    f"EnvConfig count to the range maximum so tensors are large enough."
                )
        _chk("n_strikers",        dr.n_strikers,        self.n_strikers)
        _chk("n_jammers",         dr.n_jammers,         self.n_jammers)
        _chk("n_known_targets",   dr.n_known_targets,   self.n_known_targets)
        _chk("n_unknown_targets", dr.n_unknown_targets, self.n_unknown_targets)
        _chk("n_known_radars",    dr.n_known_radars,    self.n_known_radars)
        _chk("n_unknown_radars",  dr.n_unknown_radars,  self.n_unknown_radars)
        _chk("max_steps",         dr.max_steps,         self.max_steps)
        if dr.radar_kill_probability is not None:
            lo, hi = dr.radar_kill_probability
            if float(lo) > float(hi):
                raise ValueError(f"dr.radar_kill_probability lo>hi: {dr.radar_kill_probability}")

    def _dr_sample_int(self, rng, n: int, default_max: int) -> torch.Tensor:
        """Per-env integer count [n]: sampled in the inclusive range, or filled
        with default_max when the range is None (parameter not randomized)."""
        if rng is None:
            return torch.full((n,), int(default_max), dtype=torch.long, device=self.device)
        lo, hi = int(rng[0]), int(rng[1])
        if lo == hi:
            return torch.full((n,), lo, dtype=torch.long, device=self.device)
        return torch.randint(lo, hi + 1, (n,), generator=self._rng, device=self.device)

    def _present_block(self, counts: torch.Tensor, block_size: int) -> torch.Tensor:
        """[n, block_size] bool with the first `counts[i]` entries True per row."""
        ar = torch.arange(block_size, device=self.device)
        return ar.unsqueeze(0) < counts.unsqueeze(1)

    def _episode_metric_fracs(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Present-aware (targets_destroyed_frac, survival_frac), each [N, 1].

        Computed over PRESENT entities only so masked-out (absent) slots never
        count as destroyed targets or dead agents. Reduces to the plain mean
        when DR is off (all entities present)."""
        tp = self.target_present[idx]
        tgt_frac = ((~self.target_alive[idx]) & tp).float().sum(-1, keepdim=True) / \
            tp.float().sum(-1, keepdim=True).clamp_min(1.0)
        ap = self.agent_present[idx]
        surv_frac = (self.agent_alive[idx] & ap).float().sum(-1, keepdim=True) / \
            ap.float().sum(-1, keepdim=True).clamp_min(1.0)
        return tgt_frac, surv_frac

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        B, A, T, R = self.num_envs, self.n_agents, self.n_targets, self.n_radars

        # Optional sub-bucket timing — populated only when the HF subclass
        # (which owns _prof_tic/_prof_lap) is active. getattr falls back to
        # no-op lambdas so the base env still works standalone.
        _prof_tic = getattr(self, "_prof_tic", lambda: None)
        _prof_lap = getattr(self, "_prof_lap", lambda name, t: None)

        reset_mask = self._extract_reset_mask(tensordict)
        reset_idx = reset_mask.nonzero(as_tuple=False).squeeze(-1)
        n_reset = int(reset_idx.numel())

        if n_reset > 0:
            # ── Per-env domain-randomization sampling ────────────────────
            # Vectorised over the reset envs (no Python per-env loop). When
            # DR is inactive every count equals its max and every mask is all
            # True, so the masks/known-blocks reduce to the original behaviour.
            dr = self._dr
            def _rng_of(attr):
                return getattr(dr, attr, None) if dr is not None else None

            ns_cnt  = self._dr_sample_int(_rng_of("n_strikers"),        n_reset, self.n_strikers)
            nj_cnt  = self._dr_sample_int(_rng_of("n_jammers"),         n_reset, self.n_jammers)
            nkt_cnt = self._dr_sample_int(_rng_of("n_known_targets"),   n_reset, self.n_known_targets)
            nut_cnt = self._dr_sample_int(_rng_of("n_unknown_targets"), n_reset, self.n_unknown_targets)
            nkr_cnt = self._dr_sample_int(_rng_of("n_known_radars"),    n_reset, self.n_known_radars)
            nur_cnt = self._dr_sample_int(_rng_of("n_unknown_radars"),  n_reset, self.n_unknown_radars)

            present_s  = self._present_block(ns_cnt,  self.n_strikers)
            present_j  = self._present_block(nj_cnt,  self.n_jammers)
            agent_present_reset = torch.cat([present_s, present_j], dim=1)        # [n_reset, A]

            present_kt = self._present_block(nkt_cnt, self.n_known_targets)
            present_ut = self._present_block(nut_cnt, self.n_unknown_targets)
            target_present_reset = torch.cat([present_kt, present_ut], dim=1)     # [n_reset, T]
            target_known_reset   = torch.cat([present_kt, torch.zeros_like(present_ut)], dim=1)

            present_kr = self._present_block(nkr_cnt, self.n_known_radars)
            present_ur = self._present_block(nur_cnt, self.n_unknown_radars)
            radar_present_reset  = torch.cat([present_kr, present_ur], dim=1)     # [n_reset, R]
            radar_known_reset    = torch.cat([present_kr, torch.zeros_like(present_ur)], dim=1)

            # Per-env scalars: radar kill probability + episode horizon.
            kp_rng = _rng_of("radar_kill_probability")
            if kp_rng is not None:
                kill_prob_reset = torch.empty(n_reset, 1, device=self.device).uniform_(
                    float(kp_rng[0]), float(kp_rng[1]), generator=self._rng,
                )
            else:
                kill_prob_reset = torch.full((n_reset, 1), self._base_kill_probability, device=self.device)

            ms_rng = _rng_of("max_steps")
            if ms_rng is not None and int(ms_rng[0]) != int(ms_rng[1]):
                max_steps_reset = torch.randint(
                    int(ms_rng[0]), int(ms_rng[1]) + 1, (n_reset, 1),
                    generator=self._rng, device=self.device,
                ).float()
            elif ms_rng is not None:
                max_steps_reset = torch.full((n_reset, 1), float(ms_rng[0]), device=self.device)
            else:
                max_steps_reset = torch.full((n_reset, 1), float(self.max_steps), device=self.device)

            self.agent_present[reset_idx]  = agent_present_reset
            self.target_present[reset_idx] = target_present_reset
            self.radar_present[reset_idx]  = radar_present_reset
            self.radar_kill_prob[reset_idx] = kill_prob_reset
            self.max_steps_t[reset_idx]     = max_steps_reset

            # --- Agents spawn at bottom middle of map, facing north ---
            center_x = (self.low + self.high) / 2.0
            bottom_y = self.low + 0.05  # 50 km from bottom edge
            spacing = 0.02  # 20 km between agents

            agent_pos_reset = torch.zeros(n_reset, A, 2, device=self.device)
            for a in range(A):
                offset_x = (a - (A - 1) / 2.0) * spacing
                agent_pos_reset[:, a, 0] = center_x + offset_x
                agent_pos_reset[:, a, 1] = bottom_y
            self.agent_pos[reset_idx] = agent_pos_reset

            self.agent_heading[reset_idx] = math.pi / 2.0  # Face north

            # Initialize at minimum speed (aircraft can't hover)
            speed_reset = torch.zeros(n_reset, A, device=self.device)
            speed_reset[:, :self.n_strikers] = self.striker.v_min
            speed_reset[:, self.n_strikers:] = self.jammer.v_min
            self.agent_speed[reset_idx] = speed_reset
            self.agent_heading_rate[reset_idx] = 0.0
            # Present agents start alive; absent (masked-out) agents start dead
            # so every alive-gated reward/obs/termination term ignores them.
            self.agent_alive[reset_idx] = agent_present_reset

            # --- Radars: top half of map, not too close to borders ---
            # Domain randomisation: every env being reset gets an independently
            # sampled radar configuration. With a pre-generated layout pool
            # (n_env_layouts > 0) we draw random indices into the pool — this
            # used to be `env_i % L`, which deterministically locked each env
            # to a single layout for its entire lifetime (so vis with num_envs=1
            # always saw layout 0). Without a pool we fall back to per-env
            # rejection sampling (slow but already randomised).
            _t_radar = _prof_tic()
            if self._layouts is not None:
                L = self._layouts.shape[0]
                layout_idx = torch.randint(
                    0, L, (n_reset,), generator=self._rng, device=self.device,
                )
                radar_pos_reset = self._layouts[layout_idx]                    # [n_reset, R, 2]
            else:
                x_lo, x_hi, y_lo, y_hi = self._radar_box
                min_sep = self._radar_min_sep_active
                per_radar_boxes = self._s2_radar_boxes
                radar_pos_reset = torch.zeros(n_reset, R, 2, device=self.device)
                for i in range(n_reset):
                    radar_pos_reset[i] = self._sample_spaced_radars(
                        R, x_lo, x_hi, y_lo, y_hi, min_sep, self._rng, device=self.device,
                        per_radar_boxes=per_radar_boxes,
                    ).to(self.device)
            # Park absent radars far outside the world so every distance-based
            # term (threat, avoidance, approach, jamming, visibility) ignores
            # them with no per-step masking. radar_present still gates the few
            # boolean/critic spots where "far" is not sufficient.
            radar_pos_reset = torch.where(
                radar_present_reset.unsqueeze(-1),
                radar_pos_reset,
                torch.full_like(radar_pos_reset, self._ABSENT_RADAR_COORD),
            )
            self.radar_pos[reset_idx] = radar_pos_reset
            _prof_lap("env_reset_radar_spawn", _t_radar)

            # Spawn targets. S1: clustered around their assigned radar
            # (legacy "protected targets"). S2: uniform random in the top band,
            # independent of radar positions ("defensive line").
            _t_target = _prof_tic()
            if self.scenario == "S1":
                target_pos_reset = self._spawn_targets_in_valid_zones(n_reset, T, R, radar_pos_reset)
            else:  # S2
                target_pos_reset = self._spawn_targets_uniform_box(n_reset, T, self._s2_target_box)
            self.target_pos[reset_idx] = target_pos_reset
            _prof_lap("env_reset_target_spawn", _t_target)
            # Present targets start alive; absent targets start dead (so they
            # never appear in obs, are never engageable, and the mission-
            # complete check `(~target_alive).all()` reduces to "all present
            # targets destroyed"). target_known is True only for present known
            # slots.
            self.target_alive[reset_idx] = target_present_reset
            self.target_known[reset_idx] = target_known_reset
            self.radar_known[reset_idx]  = radar_known_reset

            self.radar_eff_range[reset_idx] = self.radar_range
            self.step_count[reset_idx] = 0
            self._episode_team_reward[reset_idx] = 0.0
            for comp_key in self._episode_component_reward:
                self._episode_component_reward[comp_key][reset_idx] = 0.0

            # Initialise previous distances for striker progress reward
            if self.n_strikers > 0 and self.n_targets > 0:
                rel_st_init = self.target_pos[reset_idx, None, :, :] - self.agent_pos[reset_idx, :self.n_strikers, None, :]
                self._striker_prev_dist[reset_idx] = torch.linalg.norm(rel_st_init, dim=-1)  # [n_reset, ns, T]
            else:
                self._striker_prev_dist[reset_idx] = 0.0

            # Initialise previous distances for jammer progress reward
            if self.n_jammers > 0 and self.n_radars > 0:
                rel_jr_init = self.radar_pos[reset_idx, None, :, :] - self.agent_pos[reset_idx, self.n_strikers:, None, :]
                self._jammer_prev_dist[reset_idx] = torch.linalg.norm(rel_jr_init, dim=-1)  # [n_reset, nj, R]
            else:
                self._jammer_prev_dist[reset_idx] = 0.0

        # ── Step 4: partial-reset output update ──────────────────────────
        # The persistent buffers (_obs_buf, _state_buf, _fofe_*_buf) hold
        # the most recent post-step outputs for ALL B envs. The non-reset
        # envs are still valid in those buffers (their state didn't change),
        # so we only have to rebuild rows for reset_idx.
        #
        # Edge case: the very first call (before any _step has run) finds
        # the buffers None; we then do a one-time full-B build to allocate
        # them, which matches the old behaviour.
        if self._obs_buf is None:
            self._ensure_persistent_buffers_()
        elif n_reset > 0:
            self._update_subset_outputs_(reset_idx)

        td = TensorDict({}, batch_size=[B], device=self.device)
        td.set(self._obs_key, self._obs_buf)
        td.set("state",       self._state_buf)
        td.set("done",        torch.zeros(B, 1, dtype=torch.bool, device=self.device))
        td.set("terminated",  torch.zeros(B, 1, dtype=torch.bool, device=self.device))
        # ---- FOFE: emit structured per-channel observations from buffers ----
        if self.use_fofe:
            for k, v in self._fofe_obs_buf.items():
                td.set(("agents", k), v)
            for k, v in self._fofe_critic_buf.items():
                td.set(k, v)
        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict.get(self._action_key)  # [B, A, 2] discrete in {0..6}
        B, A, _ = action.shape
        rp = self.reward_params

        # Map discrete indices to continuous multipliers via lookup table
        # _act_table: [-1, -0.5, -0.1, 0, +0.1, +0.5, +1]
        action_idx = action.long().clamp(0, self.n_choices - 1)
        acc = self._act_table[action_idx]  # [B, A, 2] float in {-1,-0.5,-0.1,0,+0.1,+0.5,+1}

        alive = self.agent_alive  # Keep as bool for bitwise operations
        alive_before_kill = alive
        alive_float = alive.float()  # Use for multiplication

        # ---- Velocity dynamics ----
        v_accel = acc[..., 0]  # [B, A]
        self.agent_speed = (
            self.agent_speed + v_accel * self.accel_magnitude
        ).clamp(0.0, self.v_max)
        self.agent_speed = self.agent_speed * alive_float

        # Enforce minimum speed for alive agents (aircraft can't hover / spin in place)
        v_min_per_agent = torch.zeros_like(self.agent_speed)
        v_min_per_agent[:, :self.n_strikers] = self.striker.v_min
        v_min_per_agent[:, self.n_strikers:] = self.jammer.v_min
        self.agent_speed = torch.where(
            alive, torch.max(self.agent_speed, v_min_per_agent), self.agent_speed
        )

        # ---- Heading dynamics ----
        h_accel = acc[..., 1]  # [B, A]
        self.agent_heading_rate = (
            self.agent_heading_rate + h_accel * self.h_accel_magnitude
        ).clamp(-self.dpsi_max, self.dpsi_max)
        self.agent_heading_rate = self.agent_heading_rate * alive_float

        # Enforce minimum turn radius: max heading rate = speed / R_min
        # Prevents tight circling; at low speed agents must make wider turns
        if self.min_turn_radius > 0:
            max_omega = (self.agent_speed / self.min_turn_radius).clamp(max=self.dpsi_max)
            self.agent_heading_rate = torch.max(
                torch.min(self.agent_heading_rate, max_omega), -max_omega
            )
        
        # Update heading based on heading rate
        self.agent_heading = (self.agent_heading + self.agent_heading_rate) % (2.0 * math.pi)
        
        # Update position based on speed and heading
        dx = self.agent_speed * torch.cos(self.agent_heading) * self.dt
        dy = self.agent_speed * torch.sin(self.agent_heading) * self.dt
        self.agent_pos = (self.agent_pos + torch.stack([dx, dy], dim=-1)).clamp(self.low, self.high)
        self._update_geometry_cache()

        # ---- EA / jamming: Jammers automatically jam radars within range ----
        radar_eff_range = torch.full((B, self.n_radars), self.radar_range, device=self.device)
        jammer_idx = torch.arange(self.n_strikers, self.n_agents, device=self.device)

        if jammer_idx.numel() > 0:
            rel_jr     = self._c_rel_ar[:, self.n_strikers:, :, :]  # [B,nj,R,2] from cache
            jam_mask   = self.jammer.jams_radar(rel_jr) & alive[:, jammer_idx, None]  # [B,nj,R]
            any_jam    = jam_mask.any(dim=1)                                           # [B,R]
            jam_active = jam_mask.any(dim=2)                                           # [B,nj] per-jammer: actively jamming?
            radar_eff_range  = torch.where(
                any_jam,
                (radar_eff_range - self.jammer.delta_range).clamp_min(0.0),
                radar_eff_range,
            )
        else:
            jam_active = torch.zeros(B, 0, dtype=torch.bool, device=self.device)

        self.radar_eff_range = radar_eff_range.clone()

        # ---- radar kills own agents (probabilistic) ----
        dist_ar = self._c_dist_ar                                                  # [B,A,R]
        in_radar = (dist_ar <= radar_eff_range[:, None, :]) & self.radar_present[:, None, :]  # [B,A,R]

        kill_samples = torch.rand(B, A, self.n_radars, device=self.device, generator=self._rng)
        # Per-env kill probability ([B,1]→[B,1,1]); scalar fill when DR is off.
        kills_from_radar = in_radar & (kill_samples < self.radar_kill_prob.unsqueeze(-1))
        killed = kills_from_radar.any(dim=-1) & alive
        self.agent_alive = self.agent_alive & (~killed)

        # ---- striker kinetic kills: Strikers automatically strike if target in engagement zone ----
        kill_t     = torch.zeros(B, self.n_targets, dtype=torch.bool, device=self.device)
        striker_idx = torch.arange(0, self.n_strikers, device=self.device)

        if striker_idx.numel() > 0:
            rel_st     = self.target_pos[:, None, :, :] - self.agent_pos[:, striker_idx, None, :]
            can        = self.striker.can_engage(rel_st, self.agent_heading[:, striker_idx][:, :, None])
            can        = can & alive[:, striker_idx, None] & self.target_alive[:, None, :]
            kill_t     = can.any(dim=1)
            self.target_alive = self.target_alive & (~kill_t)

        # Refresh alive mask after kill updates so reward terms use current alive state.
        alive = self.agent_alive
        alive_float = alive.float()

        # ==================================================================
        # Reward computation  (piecewise linear-exponential shaping)
        # ==================================================================
        reward = torch.zeros(B, A, device=self.device)
        max_dist = math.hypot(self.high - self.low, self.high - self.low)
        ts = float(rp.team_spirit)  # team vs individual mixing coefficient

        # ------------------------------------------------------------------
        # 1. Team reward: target destroyed  (team_spirit blends team ↔ individual)
        # ------------------------------------------------------------------
        n_killed = kill_t.float().sum(dim=-1)                             # [B]
        n_alive  = self.agent_alive.float().sum(dim=-1).clamp_min(1.0)    # [B]
        target_destroyed_full = torch.zeros(B, A, device=self.device)
        # Always run (branchless): when kill_t has no Trues, all per-component
        # terms below collapse to zero. Removing the .item() avoids a per-step
        # GPU→CPU sync.
        if float(rp.target_destroyed) != 0.0:
            # Team component: shared equally among alive agents
            team_share = (n_killed * float(rp.target_destroyed) / n_alive)  # [B]
            team_comp = team_share.unsqueeze(-1) * alive_float              # [B, A]

            # Individual component: credit to strikers that actually engaged
            indiv_comp = torch.zeros(B, A, device=self.device)
            if striker_idx.numel() > 0:
                # can: [B, ns, T] — which striker-target pairs had engagement
                # kill_t: [B, T] — which targets were killed this step
                engaged_kills = can & kill_t[:, None, :]                    # [B, ns, T]
                n_engaged_per_target = engaged_kills.float().sum(dim=1).clamp_min(1.0)  # [B, T]
                # Credit per striker = sum_t (target_destroyed / n_strikers_that_engaged_t)
                credit_per_pair = engaged_kills.float() * (float(rp.target_destroyed) / n_engaged_per_target[:, None, :])
                indiv_comp[:, :self.n_strikers] = credit_per_pair.sum(dim=-1)  # [B, ns]

            # Blend: team_spirit × team + (1 - team_spirit) × individual
            target_rew = ts * team_comp + (1.0 - ts) * indiv_comp
            reward += target_rew
            target_destroyed_full = target_rew

        # ------------------------------------------------------------------
        # 1b. Mission-complete terminal bonus (all targets destroyed)
        # ------------------------------------------------------------------
        terminal_bonus_full = torch.zeros(B, A, device=self.device)
        all_targets_done_now = (~self.target_alive).all(dim=-1)  # [B]
        # Branchless: when no env hit the terminal, the masked write is a no-op
        # and `reward += zeros` is a no-op. Removing .item() avoids a GPU sync.
        if float(rp.terminal_bonus) != 0.0:
            terminal_bonus_full[all_targets_done_now] = float(rp.terminal_bonus)
            reward += terminal_bonus_full

        # ------------------------------------------------------------------
        # 2. Border avoidance  (piecewise lin-exp penalty, d_max = border_thresh)
        # ------------------------------------------------------------------
        pos       = self.agent_pos
        dist_bord = torch.stack([
            pos[..., 0] - self.low,
            self.high - pos[..., 0],
            pos[..., 1] - self.low,
            self.high - pos[..., 1],
        ], dim=-1).min(dim=-1).values                                     # [B, A]
        border_pen = -self._piecewise_lin_exp(
            dist_bord,
            d_max=rp.border_d_max,
            d_knee=rp.border_d_knee,
            w_lin=rp.border_w_lin,
            w_exp=rp.border_w_exp,
            alpha=rp.border_alpha,
        ) * alive_float
        reward += border_pen

        # ------------------------------------------------------------------
        # 3. Timestep penalty (per alive agent — NOT affected by team_spirit)
        # ------------------------------------------------------------------
        timestep_rew = float(rp.timestep_penalty) * alive_float
        reward += timestep_rew

        # ------------------------------------------------------------------
        # 4. Radar zone avoidance  (piecewise lin-exp penalty, ALL agents)
        #    Fixed boundary = jammed effective radar range (non-adaptive).
        #    This shaping term is always computed against the jammed zone,
        #    independent of whether jamming is currently active.
        # ------------------------------------------------------------------
        jammed_zone_range = max(self.radar_range - self.jammer.delta_range, 0.0)
        d_zone = dist_ar - jammed_zone_range                               # [B, A, R]
        d_zone_min = d_zone.min(dim=-1).values.clamp(min=0.0)             # [B, A]
        radar_pen = -self._piecewise_lin_exp(
            d_zone_min,
            d_max=rp.radar_avoid_d_max,
            d_knee=rp.radar_avoid_d_knee,
            w_lin=rp.radar_avoid_w_lin,
            w_exp=rp.radar_avoid_w_exp,
            alpha=rp.radar_avoid_alpha,
        )
        reward += radar_pen

        # ------------------------------------------------------------------
        # 5. Striker approach  (piecewise lin-exp distance penalty toward targets)
        #    Flipped so it is 0 at d=0 and becomes more negative as the
        #    striker moves farther from the target.
        # ------------------------------------------------------------------
        striker_approach_full = torch.zeros(B, A, device=self.device)
        if striker_idx.numel() > 0 and self.n_targets > 0:
            dist_st = self._c_dist_at[:, :self.n_strikers, :]            # [B, ns, T] from cache
            mask_t     = self.target_alive[:, None, :].expand(-1, self.n_strikers, -1)  # [B, ns, T]

            if rp.striker_nearest_only:
                # Nearest alive target only
                big_dist = torch.where(mask_t, dist_st, torch.full_like(dist_st, 1e6))
                shaped_dist = big_dist.min(dim=-1).values                 # [B, ns]
            else:
                # Soft-nearest weighted distance over alive targets:
                # w_i = 1 / (d_i + eps), normalized over alive targets only.
                eps = 1e-6
                inv_dist = torch.where(mask_t, 1.0 / (dist_st + eps), torch.zeros_like(dist_st))
                weight_sum = inv_dist.sum(dim=-1, keepdim=True).clamp_min(eps)
                weights = inv_dist / weight_sum
                shaped_dist = (weights * dist_st).sum(dim=-1)             # [B, ns]

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

            # Zero when all targets dead
            any_alive = mask_t.any(dim=-1)
            striker_app = torch.where(any_alive, striker_app, torch.zeros_like(striker_app))

            striker_alive_f = alive[:, :self.n_strikers].float()
            striker_app = striker_app * striker_alive_f
            reward[:, :self.n_strikers] += striker_app
            striker_approach_full[:, :self.n_strikers] = striker_app

        # ------------------------------------------------------------------
        # 6. Jammer approach  (piecewise lin-exp distance penalty toward radars)
        #    Flipped so it is 0 at d=0 and becomes more negative as the
        #    jammer moves farther from the radar.
        # ------------------------------------------------------------------
        jammer_approach_full = torch.zeros(B, A, device=self.device)
        if jammer_idx.numel() > 0 and self.n_radars > 0:
            dist_jr = self._c_dist_ar[:, self.n_strikers:, :]            # [B, nj, R] from cache
            jammer_alive_f = alive[:, self.n_strikers:].float()

            # Compute shaping value per jammer-radar pair
            app_vals_j = self._piecewise_lin_exp(
                dist_jr,
                d_max=rp.jammer_approach_d_max,
                d_knee=rp.jammer_approach_d_knee,
                w_lin=rp.jammer_approach_w_lin,
                w_exp=rp.jammer_approach_w_exp,
                alpha=rp.jammer_approach_alpha,
            )  # [B, nj, R]

            if rp.jammer_nearest_only:
                nearest_idx_j = dist_jr.argmin(dim=-1, keepdim=True)      # [B, nj, 1]
                jammer_app = app_vals_j.gather(-1, nearest_idx_j).squeeze(-1)  # [B, nj]
            else:
                jammer_app = app_vals_j.mean(dim=-1)                      # [B, nj]

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

        # ------------------------------------------------------------------
        # 7. Potential-based progress  
        # ------------------------------------------------------------------
        jammer_progress_full  = torch.zeros(B, A, device=self.device)
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
                mask_t_p  = self.target_alive[:, None, :].expand(-1, self.n_strikers, -1)
                any_alive_p = mask_t_p.any(dim=-1)
                progress  = self._striker_prev_dist - dist_st
                progress  = torch.where(mask_t_p, progress, torch.full_like(progress, -1e6))
                progress_max, _ = progress.max(dim=-1)
                progress_max = torch.where(any_alive_p, progress_max.clamp(min=-max_dist),
                                           torch.zeros_like(progress_max))
                striker_alive_f_p = alive[:, :self.n_strikers].float()
                striker_contrib = float(rp.striker_progress_scale) * progress_max * striker_alive_f_p
                reward[:, :self.n_strikers] += striker_contrib
                striker_progress_full[:, :self.n_strikers] = striker_contrib

            self._striker_prev_dist = dist_st.detach()

        # ------------------------------------------------------------------
        # 8. Formation cohesion (striker ↔ jammer cross-role proximity reward)
        #    Each striker is rewarded for being near the nearest alive jammer,
        #    and vice versa.  Same-role proximity is NOT rewarded.
        # ------------------------------------------------------------------
        formation_full = torch.zeros(B, A, device=self.device)
        ns, nj = self.n_strikers, self.n_jammers

        if ns > 0 and nj > 0:
            striker_pos = self.agent_pos[:, :ns, :]   # [B, ns, 2]
            jammer_pos  = self.agent_pos[:, ns:, :]   # [B, nj, 2]
            # Raw pairwise distances: d_sj[b, s, j] = dist(striker_s, jammer_j)
            d_sj = torch.linalg.norm(
                striker_pos[:, :, None, :] - jammer_pos[:, None, :, :], dim=-1
            )  # [B, ns, nj]

            # --- Striker formation: distance penalty toward nearest live jammer ---
            #     Flipped: 0 at d=0, −scale at d ≥ ref_dist.
            if float(rp.striker_formation_scale) > 0:
                # Mask dead jammers so they are never selected as "nearest"
                dead_j = ~self.agent_alive[:, ns:].unsqueeze(1).expand(B, ns, nj)  # [B, ns, nj]
                d_near_j = d_sj.masked_fill(dead_j, float('inf')).min(dim=-1).values  # [B, ns]
                striker_form = float(rp.striker_formation_scale) * (
                    (1.0 - d_near_j / float(rp.striker_formation_ref_dist)).clamp(min=0.0) - 1.0
                ) * alive[:, :ns].float()
                reward[:, :ns]        += striker_form
                formation_full[:, :ns] = striker_form

            # --- Jammer formation: distance penalty toward nearest live striker ---
            #     Flipped: 0 at d=0, −scale at d ≥ ref_dist.
            if float(rp.jammer_formation_scale) > 0:
                # Transpose to [B, nj, ns] then mask dead strikers
                d_js   = d_sj.transpose(1, 2)                                          # [B, nj, ns]
                dead_s = ~self.agent_alive[:, :ns].unsqueeze(1).expand(B, nj, ns)      # [B, nj, ns]
                d_near_s = d_js.masked_fill(dead_s, float('inf')).min(dim=-1).values   # [B, nj]
                jammer_form = float(rp.jammer_formation_scale) * (
                    (1.0 - d_near_s / float(rp.jammer_formation_ref_dist)).clamp(min=0.0) - 1.0
                ) * alive[:, ns:].float()
                reward[:, ns:]        += jammer_form
                formation_full[:, ns:] = jammer_form

        # ------------------------------------------------------------------
        # 9. Agent destruction penalty  (team_spirit blends team ↔ individual)
        # ------------------------------------------------------------------
        death_pen_full = torch.zeros(B, A, device=self.device)
        n_killed_agents = killed.float().sum(dim=-1)                      # [B]
        # Branchless: when no agent died, all terms below are zero. Skipping
        # .item() avoids a per-step GPU→CPU sync.
        if float(rp.agent_destroyed) != 0.0:
            # Team component: shared among alive agents
            team_death = (n_killed_agents * float(rp.agent_destroyed) / n_alive)  # [B]
            team_death_comp = team_death.unsqueeze(-1) * alive_float              # [B, A]
            # Individual component: only the killed agent
            indiv_death_comp = killed.float() * float(rp.agent_destroyed)         # [B, A]
            death_pen = ts * team_death_comp + (1.0 - ts) * indiv_death_comp
            reward += death_pen
            death_pen_full = death_pen

        # ------------------------------------------------------------------
        # 10. Optional paper-style mission reward (separate component)
        # ------------------------------------------------------------------
        mission_reward_full = torch.zeros(B, A, device=self.device)
        if bool(rp.use_paper_mission_reward):
            n_targets_alive = self.target_alive.float().sum(dim=-1)  # [B]
            n_agents_alive = self.agent_alive.float().sum(dim=-1)    # [B]
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

        # ------------------------------------------------------------------
        # 11. Same-role separation penalty (striker↔striker, jammer↔jammer)
        # ------------------------------------------------------------------
        separation_pen_full = torch.zeros(B, A, device=self.device)

        if ns > 1 and float(rp.striker_sep_d_max) > 0.0:
            striker_pos = self.agent_pos[:, :ns, :]  # [B, ns, 2]
            d_ss = torch.cdist(striker_pos, striker_pos)  # [B, ns, ns]
            eye_ss = torch.eye(ns, dtype=torch.bool, device=self.device).unsqueeze(0)
            striker_alive = self.agent_alive[:, :ns]
            valid_ss = striker_alive.unsqueeze(-1) & striker_alive.unsqueeze(-2) & (~eye_ss)
            d_ss_valid = d_ss.masked_fill(~valid_ss, float("inf"))
            d_ss_nearest = d_ss_valid.min(dim=-1).values  # [B, ns]
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
            jammer_pos = self.agent_pos[:, ns:, :]  # [B, nj, 2]
            d_jj = torch.cdist(jammer_pos, jammer_pos)  # [B, nj, nj]
            eye_jj = torch.eye(nj, dtype=torch.bool, device=self.device).unsqueeze(0)
            jammer_alive = self.agent_alive[:, ns:]
            valid_jj = jammer_alive.unsqueeze(-1) & jammer_alive.unsqueeze(-2) & (~eye_jj)
            d_jj_valid = d_jj.masked_fill(~valid_jj, float("inf"))
            d_jj_nearest = d_jj_valid.min(dim=-1).values  # [B, nj]
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

        # ------------------------------------------------------------------
        # 12. Control effort penalty  (per alive agent)
        #     -accel_effort_scale × accel² - angular_effort_scale × angular²
        # ------------------------------------------------------------------
        control_pen_full = torch.zeros(B, A, device=self.device)
        accel_scale = float(rp.accel_effort_scale)
        angular_scale = float(rp.angular_effort_scale)
        if (accel_scale > 0 or angular_scale > 0):
            # acc: [B, A, 2] — multipliers in {-1, -0.5, -0.1, 0, +0.1, +0.5, +1}
            control_pen = -(accel_scale * acc[..., 0] ** 2
                            + angular_scale * acc[..., 1] ** 2) * alive_float
            reward += control_pen
            control_pen_full = control_pen

        # Store per-component reward breakdown for diagnostics  (each [B, A])
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
            "jammer_coalition_coverage": torch.zeros_like(separation_pen_full),
            "control_effort":     control_pen_full.detach(),
        }
        
        reward = reward.unsqueeze(-1).contiguous()  # [B, A, 1]

        # Accumulate team-total reward per env for episode-level logging
        step_team_reward = reward.squeeze(-1).sum(dim=-1)  # [B]
        self._episode_team_reward += step_team_reward
        for comp_key, comp_tensor in self.last_reward_components.items():
            self._episode_component_reward[comp_key] += comp_tensor.sum(dim=-1)

        # ---- done flags ----
        self.step_count += 1
        all_targets_done = (~self.target_alive).all(dim=-1, keepdim=True)
        all_agents_dead  = (~self.agent_alive).all(dim=-1, keepdim=True)
        timeout          = self.step_count >= self.max_steps_t

        terminated = all_targets_done | all_agents_dead
        done       = terminated | timeout

        next_td = TensorDict({}, batch_size=[B], device=self.device)
        next_td.set(self._reward_key, reward)
        next_td.set("done",       done.to(torch.bool))
        next_td.set("terminated", terminated.to(torch.bool))
        self._update_comm_cache()
        # Build obs/state/fofe ONCE and reuse for both the returned TD and
        # the persistent buffers (Step 4 — reused by the next _reset).
        _local_obs   = self._build_local_obs()
        _global_state = self._build_global_state()
        next_td.set(self._obs_key, _local_obs)
        next_td.set("state",       _global_state)
        _fofe_obs_dict: Optional[Dict[str, torch.Tensor]] = None
        _fofe_critic_dict: Optional[Dict[str, torch.Tensor]] = None
        if self.use_fofe:
            _fofe_obs_dict    = self._build_fofe_obs()
            _fofe_critic_dict = self._build_fofe_critic_state()
            for k, v in _fofe_obs_dict.items():
                next_td.set(("agents", k), v)
            for k, v in _fofe_critic_dict.items():
                next_td.set(k, v)
        # Cache fresh outputs in persistent buffers — _reset uses these as
        # the "baseline" so it only has to rebuild rows for reset_idx.
        self._cache_step_outputs_(_local_obs, _global_state, _fofe_obs_dict, _fofe_critic_dict)

        # Track completed episode stats in Python list (immune to auto-reset).
        # Vectorised: gather all per-env stats for done envs into ONE tensor,
        # transfer in a single .cpu().tolist() call, then build dicts. This
        # replaces the previous `for b in range(B): if done[b].item()` loop
        # which produced O(B × n_components) GPU→CPU syncs every step.
        done_flat = done.squeeze(-1)                                # [B] bool
        done_idx_t = done_flat.nonzero(as_tuple=True)[0]           # [N_done]
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

        return next_td

    # ------------------------------------------------------------------
    # Episode statistics (for training-time logging)
    # ------------------------------------------------------------------

    def pop_episode_stats(self) -> list:
        """Return and clear accumulated completed-episode statistics.

        Each entry is a dict with keys:
          mission_complete (bool), targets_frac (float),
                    survival_frac (float), duration (int),
                                        episode_total_reward (float),
                                        episode_component_reward (dict[str, float]).
        """
        stats = self._completed_episodes
        self._completed_episodes = []
        return stats

    # ------------------------------------------------------------------
    # Observation / state builders
    # ------------------------------------------------------------------

    def _build_global_state(self) -> torch.Tensor:
        """Global absolute state for the centralised critic.

        Layout (flat vector):
          Per agent i:  (x, y, v, ψ, ω, role, alive, *agent_extra) × A   ((7+E_a)A)
          Per target k: (x, y, alive)                                × T   (3T)
          Per radar r:  (x, y, jammed)                               × R   (3R)
          Global:       (t_norm)                                           (1)

        Total dim = (7+E_a)A + 3T + 3R + 1
        """
        B    = self.num_envs
        A, T, R = self.n_agents, self.n_targets, self.n_radars
        world_range = self.high - self.low

        # --- Agent features: (x, y, v, ψ, ω, role, alive) per agent ---
        pos_norm   = (self.agent_pos - self.low) / world_range                     # [B,A,2]  in [0,1]
        speed_norm = (self.agent_speed / self.v_max).unsqueeze(-1)                 # [B,A,1]
        hdg_norm   = (self.agent_heading / math.pi).unsqueeze(-1)                  # [B,A,1]  in [0,2]
        omega_norm = (self.agent_heading_rate / self.dpsi_max).unsqueeze(-1)       # [B,A,1]  in [-1,1]
        role       = torch.zeros(B, A, 1, device=self.device)
        role[:, :self.n_strikers, 0] = 1.0                                         # strikers=1, jammers=0
        alive_a    = self.agent_alive.float().unsqueeze(-1)                        # [B,A,1]
        agent_base = torch.cat([pos_norm, speed_norm, hdg_norm,
                                omega_norm, role, alive_a], dim=-1)                # [B,A,7]
        agent_extra = self._build_critic_agent_extra(B, A)                         # [B,A,E_a]
        if agent_extra.shape[-1] > 0:
            agent_feat = torch.cat([agent_base, agent_extra], dim=-1)              # [B,A,7+E_a]
        else:
            agent_feat = agent_base
        agent_flat = agent_feat.reshape(B, -1)                                     # [B, (7+E_a)A]

        # --- Target features: (x, y, alive) per target ---
        tgt_pos_norm = (self.target_pos - self.low) / world_range                  # [B,T,2]
        tgt_alive    = self.target_alive.float().unsqueeze(-1)                     # [B,T,1]
        tgt_feat     = torch.cat([tgt_pos_norm, tgt_alive], dim=-1)               # [B,T,3]
        tgt_flat     = tgt_feat.reshape(B, -1)                                     # [B, 3T]

        # --- Radar features: (x, y, jammed) per radar ---
        rdr_pos_norm  = (self.radar_pos - self.low) / world_range                  # [B,R,2]
        rdr_jammed    = self._radar_jammed_flag().float().unsqueeze(-1)           # [B,R,1] 1=jammed
        rdr_feat      = torch.cat([rdr_pos_norm, rdr_jammed], dim=-1)              # [B,R,3]
        rdr_flat      = rdr_feat.reshape(B, -1)                                    # [B, 3R]

        # --- Global normalised time feature: t / max_steps ---
        time_norm = (self.step_count.float() / self.max_steps_t).clamp(0.0, 1.0)  # [B,1]

        return torch.cat([agent_flat, tgt_flat, rdr_flat, time_norm], dim=-1)

    # ------------------------------------------------------------------
    # Ego-centric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _piecewise_lin_exp(d: torch.Tensor, d_max: float, d_knee: float,
                           w_lin: float, w_exp: float, alpha: float) -> torch.Tensor:
        """Piecewise linear-exponential shaping (vectorised).

        Returns a **positive** value that increases as *d* → 0.
        Caller applies sign (positive for approach, negative for avoidance).

        Regions
        -------
        d ≥ d_max          → 0
        d_knee ≤ d < d_max → linear:  w_lin × (d_max − d) / (d_max − d_knee)
        d < d_knee          → w_lin + w_exp × (e^{α·(1 − d/d_knee)} − 1)

        Continuous at *d_knee* (exponential term = 0 at the boundary).
        """
        # Linear region: progress fraction 0 → 1 as d goes from d_max → d_knee
        t_lin = ((d_max - d) / (d_max - d_knee + 1e-8)).clamp(0.0, 1.0)
        lin_val = w_lin * t_lin

        # Exponential bonus: 0 at d_knee, grows steeply toward d = 0
        t_exp = ((d_knee - d) / (d_knee + 1e-8)).clamp(0.0, 1.0)
        exp_val = w_exp * (torch.exp(alpha * t_exp) - 1.0)

        return lin_val + exp_val

    def _relative_polar(self, agent_pos, agent_heading, entity_pos):
        """Compute (distance, relative_angle) from each agent to each entity.

        Parameters
        ----------
        agent_pos     : [B, A, 2]
        agent_heading : [B, A]       heading in radians
        entity_pos    : [B, E, 2]    positions of E entities

        Returns
        -------
        dist      : [B, A, E]   Euclidean distance
        rel_angle : [B, A, E]   angle from agent heading to entity, in [-pi, pi]
        """
        # relative vector: entity - agent  →  [B, A, E, 2]
        rel = entity_pos[:, None, :, :] - agent_pos[:, :, None, :]  # [B,A,E,2]
        dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-8)       # [B,A,E]

        # Absolute angle from agent to entity
        abs_angle = torch.atan2(rel[..., 1], rel[..., 0])           # [B,A,E]

        # Relative angle = abs_angle - heading, wrapped to [-pi, pi]
        heading_exp = agent_heading[:, :, None].expand_as(abs_angle) # [B,A,E]
        rel_angle = abs_angle - heading_exp
        rel_angle = torch.atan2(torch.sin(rel_angle), torch.cos(rel_angle))  # wrap

        return dist, rel_angle

    # ------------------------------------------------------------------
    # Persistent output buffer helpers (Step 4 — partial-reset speed-up)
    # ------------------------------------------------------------------

    def _cache_step_outputs_(
        self,
        local_obs: torch.Tensor,
        global_state: torch.Tensor,
        fofe_obs: Optional[Dict[str, torch.Tensor]] = None,
        fofe_critic: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Copy a fresh _step's outputs into persistent buffers.

        Called from _step at the very end. The .clone() detaches each tensor
        from the TD that _step returned, so subsequent in-place writes in
        _reset don't corrupt the collector's view of the previous step.
        """
        self._obs_buf   = local_obs.detach().clone()
        self._state_buf = global_state.detach().clone()
        if fofe_obs is not None:
            self._fofe_obs_buf = {k: v.detach().clone() for k, v in fofe_obs.items()}
        if fofe_critic is not None:
            self._fofe_critic_buf = {k: v.detach().clone() for k, v in fofe_critic.items()}

    def _update_subset_outputs_(self, idx: torch.Tensor) -> None:
        """Rebuild outputs for env indices in `idx` and update persistent buffers.

        Uses a temp-slice trick: every per-env state tensor whose first dim
        matches num_envs is temporarily swapped for its idx-sliced view, and
        self.num_envs is set to len(idx). The standard builders (which read
        these attributes) then operate naturally on the subset. After the
        builders return, all swapped attributes are restored.

        Why temp-slice instead of subset-aware builders: the build functions
        are several hundred lines of tensor code spread across the base env
        and (potentially) subclasses. Adding an idx parameter to each would
        be invasive and fragile. The swap is local, single-method, and easy
        to reason about.

        Assumes self._obs_buf, self._state_buf, self._fofe_obs_buf,
        self._fofe_critic_buf are already allocated at full [num_envs, ...]
        shape (see _ensure_persistent_buffers_).
        """
        n_sub = int(idx.numel())
        if n_sub == 0:
            return

        # Attributes to slice. Anything whose first dim is num_envs goes here.
        # Geometry cache is included so the temp _update_geometry_cache call
        # below computes subset geometry that the builders then read.
        slice_attrs: Tuple[str, ...] = (
            # Kinematic / role state
            "agent_pos", "agent_heading", "agent_speed", "agent_heading_rate", "agent_alive",
            "target_pos", "target_alive", "target_known",
            "radar_pos", "radar_known", "radar_eff_range",
            "step_count",
            # Per-env DR state (present masks + per-env scalars) — must be
            # sliced with everything else so the subset-rebuild builders see
            # consistent shapes.
            "agent_present", "target_present", "radar_present",
            "radar_kill_prob", "max_steps_t",
            # HF subclass adds these — slice them if present, ignore otherwise.
            "jammer_bearing", "beam_rate",
            "radar_eff_range_per_agent",
            "_jammer_in_cone", "_jammer_abs_angular_delta",
            "_delta_jar", "_deepest_jammer_idx",
            # Geometry & comm caches (recomputed inside the swap)
            "_c_rel_ar",  "_c_dist_ar",  "_c_angle_ar",
            "_c_rel_at",  "_c_dist_at",  "_c_angle_at",
            "_c_rel_aa",  "_c_dist_aa",  "_c_angle_aa",
            "_c_comm_reach",
        )

        saved: Dict[str, torch.Tensor] = {}
        for attr in slice_attrs:
            v = getattr(self, attr, None)
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == self.num_envs:
                saved[attr] = v
                setattr(self, attr, v[idx])
        saved_num_envs = self.num_envs
        self.num_envs = n_sub

        try:
            self._update_geometry_cache()
            self._update_comm_cache()
            sub_obs   = self._build_local_obs()
            sub_state = self._build_global_state()
            sub_fofe_obs    = self._build_fofe_obs()           if self.use_fofe else None
            sub_fofe_critic = self._build_fofe_critic_state()  if self.use_fofe else None
        finally:
            # Restore everything — order doesn't matter, each set is independent.
            self.num_envs = saved_num_envs
            for attr, original in saved.items():
                setattr(self, attr, original)

        # Write the freshly built subset rows into the persistent buffers.
        # The buffers are full-B and the assignment uses advanced indexing.
        self._obs_buf[idx]   = sub_obs
        self._state_buf[idx] = sub_state
        if self.use_fofe:
            for k, v in sub_fofe_obs.items():
                self._fofe_obs_buf[k][idx] = v
            for k, v in sub_fofe_critic.items():
                self._fofe_critic_buf[k][idx] = v

    def _ensure_persistent_buffers_(self) -> None:
        """Allocate _obs_buf etc. on the first reset by doing a one-time full-B
        build. Subsequent resets only update the slice that needs refreshing.
        """
        self._update_geometry_cache()
        self._update_comm_cache()
        full_obs   = self._build_local_obs()
        full_state = self._build_global_state()
        self._obs_buf   = full_obs.detach().clone()
        self._state_buf = full_state.detach().clone()
        if self.use_fofe:
            for k, v in self._build_fofe_obs().items():
                self._fofe_obs_buf[k] = v.detach().clone()
            for k, v in self._build_fofe_critic_state().items():
                self._fofe_critic_buf[k] = v.detach().clone()

    def _update_geometry_cache(self) -> None:
        """Compute all pairwise geometry tensors once after positions are updated.

        Stores results in self._c_* attributes. Called at the top of _step()
        immediately after agent_pos is updated so that every downstream consumer
        (_step reward terms, _build_local_obs, _build_fofe_obs) can read from
        the cache instead of recomputing the same [B, A, E, 2] allocations.
        """
        # ---- Agent → Radar  [B, A, R, *] ----
        self._c_rel_ar   = self.radar_pos[:, None, :, :] - self.agent_pos[:, :, None, :]
        self._c_dist_ar  = torch.linalg.norm(self._c_rel_ar, dim=-1).clamp_min(1e-8)
        _abs_ar          = torch.atan2(self._c_rel_ar[..., 1], self._c_rel_ar[..., 0])
        _hdg_ar          = self.agent_heading[:, :, None].expand_as(_abs_ar)
        self._c_angle_ar = torch.atan2(torch.sin(_abs_ar - _hdg_ar), torch.cos(_abs_ar - _hdg_ar))

        # ---- Agent → Target  [B, A, T, *] ----
        self._c_rel_at   = self.target_pos[:, None, :, :] - self.agent_pos[:, :, None, :]
        self._c_dist_at  = torch.linalg.norm(self._c_rel_at, dim=-1).clamp_min(1e-8)
        _abs_at          = torch.atan2(self._c_rel_at[..., 1], self._c_rel_at[..., 0])
        _hdg_at          = self.agent_heading[:, :, None].expand_as(_abs_at)
        self._c_angle_at = torch.atan2(torch.sin(_abs_at - _hdg_at), torch.cos(_abs_at - _hdg_at))

        # ---- Agent → Agent  [B, A, A, *] ----
        self._c_rel_aa   = self.agent_pos[:, None, :, :] - self.agent_pos[:, :, None, :]
        self._c_dist_aa  = torch.linalg.norm(self._c_rel_aa, dim=-1).clamp_min(1e-8)
        _abs_aa          = torch.atan2(self._c_rel_aa[..., 1], self._c_rel_aa[..., 0])
        _hdg_aa          = self.agent_heading[:, :, None].expand_as(_abs_aa)
        self._c_angle_aa = torch.atan2(torch.sin(_abs_aa - _hdg_aa), torch.cos(_abs_aa - _hdg_aa))

    def _update_comm_cache(self) -> None:
        """Compute transitive communication reachability from cached distances.

        Must be called AFTER kill updates so agent_alive reflects casualties.
        Result stored in self._c_comm_reach [B, A, A].
        """
        A = self.n_agents
        B = self.num_envs
        eye = torch.eye(A, dtype=torch.bool, device=self.device).unsqueeze(0).expand(B, -1, -1)
        comm_adj = (
            (self._c_dist_aa <= self.R_comm)
            & self.agent_alive[:, :, None]
            & self.agent_alive[:, None, :]
        )
        comm_reach = comm_adj | eye
        for _ in range(max(A - 1, 0)):
            comm_reach = comm_reach | (torch.matmul(comm_reach.float(), comm_reach.float()) > 0)
        self._c_comm_reach = comm_reach

    def _build_local_obs(self) -> torch.Tensor:
        """Ego-centric body-frame observation per agent with fixed slot counts.

        Layout (per agent):
          own:     [x, y, speed, heading, heading_rate, t_norm, *self_extra]      (d_self)
          agents:  3 nearest visible others [dx, dy, dist, heading, role, *agent_extra]  (3×(5+E_a))
          radars:  2 nearest visible radars [dx, dy, dist, jammed, *radar_extra]  (2×(4+E))
          targets: 2 nearest visible alive targets [dx, dy, dist, alive]            (2×4)

        All (dx, dy) are expressed in the OBSERVING agent's body frame
        (rotated by -heading). All position-like scalars are normalised
        by the world diagonal.

        Visibility is partial observability via R_obs. Known entities are
        always visible; unknown entities are visible if within R_obs of self
        OR of a teammate reachable via the multi-hop R_comm graph. Unused
        slots are zero-padded.
        """
        B, A, T, R = self.num_envs, self.n_agents, self.n_targets, self.n_radars
        max_dist = math.hypot(self.high - self.low, self.high - self.low)  # for normalisation
        world_range = self.high - self.low

        def _select_nearest_slots(
            dist: torch.Tensor,
            feat: torch.Tensor,
            visible: torch.Tensor,
            k: int,
        ) -> torch.Tensor:
            """Select up to k nearest visible entities into fixed slots.

            dist:    [B, A, E]
            feat:    [B, A, E, F]
            visible: [B, A, E]
            return:  [B, A, k, F]
            """
            E = feat.shape[2]
            F = feat.shape[3]
            if k <= 0:
                return torch.zeros(B, A, 0, F, device=self.device, dtype=feat.dtype)
            if E == 0:
                return torch.zeros(B, A, k, F, device=self.device, dtype=feat.dtype)

            keep = min(k, E)
            inf = torch.full_like(dist, float("inf"))
            masked_dist = torch.where(visible, dist, inf)
            top_vals, top_idx = torch.topk(masked_dist, k=keep, dim=2, largest=False)

            gather_idx = top_idx.unsqueeze(-1).expand(B, A, keep, F)
            gathered = feat.gather(2, gather_idx)
            valid = torch.isfinite(top_vals)
            gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))

            if keep < k:
                pad = torch.zeros(B, A, k - keep, F, device=self.device, dtype=feat.dtype)
                gathered = torch.cat([gathered, pad], dim=2)
            return gathered

        # --- Own kinematic state ---
        pos_norm     = (self.agent_pos - self.low) / world_range                 # [B,A,2] in [0,1]
        speed_norm   = (self.agent_speed / self.v_max).unsqueeze(-1)            # [B,A,1] in ~[0,1]
        heading_norm = (self.agent_heading / math.pi).unsqueeze(-1)             # [B,A,1] in ~[0,2]
        hrate_norm   = (self.agent_heading_rate / self.dpsi_max).unsqueeze(-1)  # [B,A,1] in ~[-1,1]
        time_norm    = (self.step_count.float() / self.max_steps_t).clamp(0.0, 1.0)  # [B,1]
        time_exp     = time_norm.unsqueeze(1).expand(B, A, 1)                    # [B,A,1]
        own_base = torch.cat([pos_norm, speed_norm, heading_norm, hrate_norm, time_exp], dim=-1)  # [B,A,6]
        own_extra = self._build_self_extra(B, A)                                 # [B,A,d_extra]
        own = torch.cat([own_base, own_extra], dim=-1)                            # [B,A,d_self]

        # --- Agent role vector (static): strikers = 1, jammers = 0 ---
        role = torch.zeros(A, device=self.device)
        role[:self.n_strikers] = 1.0
        role = role[None, :].expand(B, A)  # [B, A]

        # Pre-compute the body-frame rotation (cos h, sin h) for each agent
        # so we can rotate world-frame relative vectors into the observing
        # agent's body frame with a single broadcasted multiply.
        cos_h = torch.cos(self.agent_heading)  # [B, A]
        sin_h = torch.sin(self.agent_heading)  # [B, A]

        # --- Other agents (body-frame dx, dy, dist, heading, role) ---
        eye = torch.eye(A, device=self.device, dtype=torch.bool).unsqueeze(0)  # [1,A,A]
        dist_aa  = torch.where(eye, torch.tensor(float('inf'), device=self.device), self._c_dist_aa)

        # _c_rel_aa[b, i, j] = pos[j] - pos[i] in world frame (other - self).
        rel_aa_w = self._c_rel_aa                                              # [B, A, A, 2]
        dx_aa_b = cos_h[:, :, None] * rel_aa_w[..., 0] + sin_h[:, :, None] * rel_aa_w[..., 1]
        dy_aa_b = -sin_h[:, :, None] * rel_aa_w[..., 0] + cos_h[:, :, None] * rel_aa_w[..., 1]

        # Visibility mask: within R_obs AND alive
        visible = (dist_aa <= self.R_obs) & self.agent_alive[:, None, :]  # [B,A,A]

        # Normalised features
        dx_aa_n       = dx_aa_b / max_dist                                          # [B,A,A]
        dy_aa_n       = dy_aa_b / max_dist                                          # [B,A,A]
        dist_aa_norm  = dist_aa / max_dist                                          # [B,A,A]
        heading_other = self.agent_heading[:, None, :].expand(B, A, A) / math.pi    # [B,A,A]
        role_other    = role[:, None, :].expand(B, A, A)                            # [B,A,A]

        other_base = torch.stack(
            [dx_aa_n, dy_aa_n, dist_aa_norm, heading_other, role_other], dim=-1,
        )  # [B,A,A,5]
        # Role-specific per-(observer, other-agent) extras (e.g. HF observed
        # jammer beam bearing in the observer's body frame).
        other_extra = self._build_other_agent_extra(B, A)                    # [B,A,A,E_a]
        if other_extra.shape[-1] > 0:
            other_feat = torch.cat([other_base, other_extra], dim=-1)        # [B,A,A,5+E_a]
        else:
            other_feat = other_base
        other_slots = _select_nearest_slots(
            dist=dist_aa,
            feat=other_feat,
            visible=visible,
            k=self.n_other_agent_obs_slots,
        )  # [B,A,3,5+E_a]
        other_obs = other_slots.reshape(B, A, -1)  # [B,A,3*(5+E_a)]

        # --- Radars (body-frame dx, dy, dist, jammed, *radar_extra) ---
        dist_ar       = self._c_dist_ar                                      # [B,A,R]
        rel_ar_w      = self._c_rel_ar                                       # [B,A,R,2] (radar - agent)
        dx_ar_b = cos_h[:, :, None] * rel_ar_w[..., 0] + sin_h[:, :, None] * rel_ar_w[..., 1]
        dy_ar_b = -sin_h[:, :, None] * rel_ar_w[..., 0] + cos_h[:, :, None] * rel_ar_w[..., 1]
        dx_ar_n = dx_ar_b / max_dist
        dy_ar_n = dy_ar_b / max_dist
        dist_ar_norm = dist_ar / max_dist
        jammed = self._radar_jammed_flag().float()                          # [B,R] 1=jammed
        jammed_exp = jammed[:, None, :].expand(B, A, R)                     # [B,A,R]

        # Communication reachability — pre-computed by _update_comm_cache
        alive_agents = self.agent_alive
        comm_reach   = self._c_comm_reach

        # ------------------------------------------------------------------
        # Radar visibility sets
        #   Known radars are always visible: R_K
        #   Unknown radars are locally detectable only within R_obs
        #   Shared unknown radars are unioned over comm_reach (multi-hop):
        #       R_share(i) = union_{j in component(i)} R_loc(j)
        # Final set used for slots:
        #   radar_visible = R_K OR shared_unknown_radar_obs
        # ------------------------------------------------------------------
        radar_known_mask = self.radar_known[:, None, :].expand(B, A, R)
        local_radar_obs = (dist_ar <= self.R_obs) & alive_agents[:, :, None]
        unknown_radar_mask = (~self.radar_known)[:, None, :].expand(B, A, R)
        local_unknown_radar_obs = local_radar_obs & unknown_radar_mask
        shared_unknown_radar_obs = torch.matmul(
            comm_reach.float(), local_unknown_radar_obs.float()
        ) > 0
        radar_visible = radar_known_mask | shared_unknown_radar_obs          # [B,A,R]
        radar_base = torch.stack(
            [dx_ar_n, dy_ar_n, dist_ar_norm, jammed_exp], dim=-1,
        )  # [B,A,R,4]
        # Role-specific per-(agent, radar) extras (e.g. HF jammer beam-to-radar angle)
        radar_extra = self._build_radar_extra(B, A, R)                       # [B,A,R,E]
        if radar_extra.shape[-1] > 0:
            radar_feat = torch.cat([radar_base, radar_extra], dim=-1)        # [B,A,R,4+E]
        else:
            radar_feat = radar_base
        radar_slots = _select_nearest_slots(
            dist=dist_ar,
            feat=radar_feat,
            visible=radar_visible,
            k=self.n_radar_obs_slots,
        )  # [B,A,K_r,4+E]
        radar_obs = radar_slots.reshape(B, A, -1)                           # [B,A,K_r*(4+E)]

        # --- Targets (body-frame dx, dy, dist, alive) ---
        dist_at       = self._c_dist_at                                      # [B,A,T]
        rel_at_w      = self._c_rel_at                                       # [B,A,T,2] (target - agent)
        dx_at_b = cos_h[:, :, None] * rel_at_w[..., 0] + sin_h[:, :, None] * rel_at_w[..., 1]
        dy_at_b = -sin_h[:, :, None] * rel_at_w[..., 0] + cos_h[:, :, None] * rel_at_w[..., 1]
        dx_at_n = dx_at_b / max_dist
        dy_at_n = dy_at_b / max_dist
        dist_at_norm  = dist_at / max_dist
        alive_t = self.target_alive[:, None, :].expand(B, A, T).float()     # [B,A,T]

        # ------------------------------------------------------------------
        # Target visibility sets (same logic as radars)
        #   Known targets are always visible: T_K
        #   Unknown targets are local only when within R_obs and alive
        #   Shared unknown targets are unioned over comm_reach
        # Final set used for slots:
        #   target_visible = T_K OR shared_unknown_target_obs
        # Note: no persistence/memory is used; unknown visibility is per-step.
        # ------------------------------------------------------------------
        target_known_mask = self.target_known[:, None, :].expand(B, A, T)
        local_target_obs = (dist_at <= self.R_obs) & alive_agents[:, :, None] & self.target_alive[:, None, :]
        unknown_target_mask = (~self.target_known)[:, None, :].expand(B, A, T)
        local_unknown_target_obs = local_target_obs & unknown_target_mask
        shared_unknown_target_obs = torch.matmul(
            comm_reach.float(), local_unknown_target_obs.float()
        ) > 0
        target_visible = target_known_mask | shared_unknown_target_obs        # [B,A,T]
        target_feat = torch.stack(
            [dx_at_n, dy_at_n, dist_at_norm, alive_t], dim=-1,
        )  # [B,A,T,4]
        target_slots = _select_nearest_slots(
            dist=dist_at,
            feat=target_feat,
            visible=target_visible & self.target_alive[:, None, :],
            k=self.n_target_obs_slots,
        )  # [B,A,K_t,4]
        target_obs = target_slots.reshape(B, A, -1)                          # [B,A,K_t*4]

        # --- Concatenate ---
        obs = torch.cat([own, other_obs, radar_obs, target_obs], dim=-1)  # [B,A,obs_dim]
        return obs

    # ------------------------------------------------------------------
    # FOFE observation builders
    # ------------------------------------------------------------------

    def _build_fofe_obs(self) -> dict:
        """Build per-channel entity observations for the FOFE actor encoder.

        Unlike _build_local_obs (which selects top-K nearest), this returns
        ALL entities per channel with boolean visibility masks.  The FOFE
        network handles variable counts via maxpool.

        Visibility follows the same communication-graph logic as _build_local_obs:
          - Known entities are always visible.
          - Unknown entities are visible only if within R_obs of this agent
            OR within R_obs of a teammate reachable via multi-hop R_comm.

        Returns dict with keys matching FOFE_ACTOR_KEYS:
            obs_self          [B, A, d_self]        ego kinematic state
            obs_agents_feat   [B, A, A, 5+E_a]      per-other-agent features (self-row masked)
                                                    layout: dx_body, dy_body, dist, heading, role, *agent_extra
            obs_agents_mask   [B, A, A]             True = visible & alive & not self
            obs_targets_feat  [B, A, T, 4]          per-target features
                                                    layout: dx_body, dy_body, dist, alive
            obs_targets_mask  [B, A, T]             True = visible & alive
            obs_radars_feat   [B, A, R, 4 + E]      per-radar features
                                                    layout: dx_body, dy_body, dist, jammed, *radar_extra
            obs_radars_mask   [B, A, R]             True = visible
        """
        B, A, T, R = self.num_envs, self.n_agents, self.n_targets, self.n_radars
        max_dist = math.hypot(self.high - self.low, self.high - self.low)
        world_range = self.high - self.low

        # ---- Self-observation: [B, A, d_self] ----
        pos_norm = (self.agent_pos - self.low) / world_range
        speed_norm = (self.agent_speed / self.v_max).unsqueeze(-1)
        heading_norm = (self.agent_heading / math.pi).unsqueeze(-1)
        hrate_norm = (self.agent_heading_rate / self.dpsi_max).unsqueeze(-1)
        time_norm = (self.step_count.float() / self.max_steps_t).clamp(0, 1)
        time_exp = time_norm.unsqueeze(1).expand(B, A, 1)
        obs_self_base = torch.cat([pos_norm, speed_norm, heading_norm, hrate_norm, time_exp], dim=-1)
        obs_self_extra = self._build_self_extra(B, A)
        obs_self = torch.cat([obs_self_base, obs_self_extra], dim=-1)

        role = torch.zeros(A, device=self.device)
        role[:self.n_strikers] = 1.0
        role = role[None, :].expand(B, A)

        # ---- Communication reachability — pre-computed by _update_comm_cache ----
        eye       = torch.eye(A, dtype=torch.bool, device=self.device).unsqueeze(0).expand(B, -1, -1)
        comm_reach = self._c_comm_reach

        # Body-frame rotation: rotate world rel vectors by -heading so each
        # observer sees other entities in its own forward/lateral axes.
        cos_h = torch.cos(self.agent_heading)  # [B, A]
        sin_h = torch.sin(self.agent_heading)  # [B, A]

        # ---- Agents channel: [B, A, A, 5] + mask [B, A, A] ----
        dist_aa  = torch.where(eye, torch.tensor(10.0, device=self.device), self._c_dist_aa)
        rel_aa_w = self._c_rel_aa                                              # [B, A, A, 2]
        dx_aa_b = cos_h[:, :, None] * rel_aa_w[..., 0] + sin_h[:, :, None] * rel_aa_w[..., 1]
        dy_aa_b = -sin_h[:, :, None] * rel_aa_w[..., 0] + cos_h[:, :, None] * rel_aa_w[..., 1]
        agents_visible = (dist_aa <= self.R_obs) & self.agent_alive[:, None, :] & ~eye

        agents_base = torch.stack([
            dx_aa_b / max_dist,
            dy_aa_b / max_dist,
            dist_aa / max_dist,
            self.agent_heading[:, None, :].expand(B, A, A) / math.pi,
            role[:, None, :].expand(B, A, A),
        ], dim=-1)  # [B, A, A, 5]
        agents_extra = self._build_other_agent_extra(B, A)                     # [B, A, A, E_a]
        if agents_extra.shape[-1] > 0:
            agents_feat = torch.cat([agents_base, agents_extra], dim=-1)       # [B, A, A, 5+E_a]
        else:
            agents_feat = agents_base
        agents_feat = agents_feat * agents_visible.unsqueeze(-1).float()
        agents_feat = torch.nan_to_num(agents_feat, nan=0.0)

        # ---- Radars channel: [B, A, R, 4 + E] + mask [B, A, R] ----
        dist_ar  = self._c_dist_ar
        rel_ar_w = self._c_rel_ar                                              # [B, A, R, 2]
        dx_ar_b = cos_h[:, :, None] * rel_ar_w[..., 0] + sin_h[:, :, None] * rel_ar_w[..., 1]
        dy_ar_b = -sin_h[:, :, None] * rel_ar_w[..., 0] + cos_h[:, :, None] * rel_ar_w[..., 1]
        jammed = self._radar_jammed_flag().float()[:, None, :].expand(B, A, R)

        # Known/unknown visibility with communication sharing
        radar_known_mask = self.radar_known[:, None, :].expand(B, A, R)
        local_radar_obs = (dist_ar <= self.R_obs) & self.agent_alive[:, :, None]
        local_unknown_radar = local_radar_obs & (~self.radar_known)[:, None, :].expand(B, A, R)
        shared_unknown_radar = torch.matmul(comm_reach.float(), local_unknown_radar.float()) > 0
        radars_visible = radar_known_mask | shared_unknown_radar

        radars_base = torch.stack(
            [dx_ar_b / max_dist, dy_ar_b / max_dist, dist_ar / max_dist, jammed], dim=-1,
        )  # [B, A, R, 4]
        radar_extra = self._build_radar_extra(B, A, R)                         # [B, A, R, E]
        if radar_extra.shape[-1] > 0:
            radars_feat = torch.cat([radars_base, radar_extra], dim=-1)
        else:
            radars_feat = radars_base
        radars_feat = radars_feat * radars_visible.unsqueeze(-1).float()
        radars_feat = torch.nan_to_num(radars_feat, nan=0.0)

        # ---- Targets channel: [B, A, T, 4] + mask [B, A, T] ----
        dist_at  = self._c_dist_at
        rel_at_w = self._c_rel_at                                              # [B, A, T, 2]
        dx_at_b = cos_h[:, :, None] * rel_at_w[..., 0] + sin_h[:, :, None] * rel_at_w[..., 1]
        dy_at_b = -sin_h[:, :, None] * rel_at_w[..., 0] + cos_h[:, :, None] * rel_at_w[..., 1]
        alive_t = self.target_alive[:, None, :].expand(B, A, T).float()

        target_known_mask = self.target_known[:, None, :].expand(B, A, T)
        local_target_obs = (dist_at <= self.R_obs) & self.agent_alive[:, :, None] & self.target_alive[:, None, :]
        local_unknown_target = local_target_obs & (~self.target_known)[:, None, :].expand(B, A, T)
        shared_unknown_target = torch.matmul(comm_reach.float(), local_unknown_target.float()) > 0
        targets_visible = (target_known_mask | shared_unknown_target) & self.target_alive[:, None, :]

        targets_feat = torch.stack(
            [dx_at_b / max_dist, dy_at_b / max_dist, dist_at / max_dist, alive_t], dim=-1,
        )  # [B, A, T, 4]
        targets_feat = targets_feat * targets_visible.unsqueeze(-1).float()
        targets_feat = torch.nan_to_num(targets_feat, nan=0.0)

        return {
            "obs_self":          obs_self,
            "obs_agents_feat":   agents_feat,
            "obs_agents_mask":   agents_visible,
            "obs_targets_feat":  targets_feat,
            "obs_targets_mask":  targets_visible,
            "obs_radars_feat":   radars_feat,
            "obs_radars_mask":   radars_visible,
        }

    def _build_fofe_critic_state(self) -> dict:
        """Build per-channel entity features for the FOFE critic.

        The critic sees the GLOBAL state (all entities, no partial
        observability), decomposed into entity sets:
            agents  [B, A, 7+E_a]   x, y, v, ψ, ω, role, alive, *agent_extra
            targets [B, T, 3]   x, y, alive
            radars  [B, R, 3]   x, y, jammed
            time    [B, 1]      t_norm

        All masks are True (full visibility for centralised critic).

        Returns dict with keys matching FOFE_CRITIC_KEYS.
        """
        B = self.num_envs
        A, T, R = self.n_agents, self.n_targets, self.n_radars
        world_range = self.high - self.low

        # ---- Agent features [B, A, 7] ----
        pos_norm = (self.agent_pos - self.low) / world_range
        speed_norm = (self.agent_speed / self.v_max).unsqueeze(-1)
        hdg_norm = (self.agent_heading / math.pi).unsqueeze(-1)
        omega_norm = (self.agent_heading_rate / self.dpsi_max).unsqueeze(-1)
        role = torch.zeros(B, A, 1, device=self.device)
        role[:, :self.n_strikers, 0] = 1.0
        alive_a = self.agent_alive.float().unsqueeze(-1)
        agent_base = torch.cat([pos_norm, speed_norm, hdg_norm, omega_norm, role, alive_a], dim=-1)
        agent_extra = self._build_critic_agent_extra(B, A)                        # [B, A, E_a]
        if agent_extra.shape[-1] > 0:
            agent_feat = torch.cat([agent_base, agent_extra], dim=-1)             # [B, A, 7+E_a]
        else:
            agent_feat = agent_base

        # ---- Target features [B, T, 3] ----
        tgt_pos_norm = (self.target_pos - self.low) / world_range
        tgt_alive = self.target_alive.float().unsqueeze(-1)
        target_feat = torch.cat([tgt_pos_norm, tgt_alive], dim=-1)

        # ---- Radar features [B, R, 3 + d_crt_radar_extra] ----
        rdr_pos_norm = (self.radar_pos - self.low) / world_range
        rdr_jammed = self._radar_jammed_flag().float().unsqueeze(-1)
        rdr_extra = self._build_critic_radar_extra(B, R)                       # [B, R, E]
        radar_feat = torch.cat([rdr_pos_norm, rdr_jammed, rdr_extra], dim=-1)

        # ---- Time feature [B, 1] ----
        time_feat = (self.step_count.float() / self.max_steps_t).clamp(0, 1)

        # Critic has full global visibility over PRESENT entities. The present
        # masks are all-True when DR is off (identical to the old all-ones), and
        # exclude masked-out entities (dead-from-reset agents/targets, far-parked
        # radars) from the critic's set encoders under DR.
        # Clone the present masks: the persistent buffer for these is the live
        # self.*_present tensor (mutated in-place at reset), so returning a fresh
        # copy per step keeps the rollout collector from aliasing a single tensor
        # across timesteps (matches the old fresh-torch.ones semantics).
        return {
            "crt_agents_feat":  agent_feat,
            "crt_agents_mask":  self.agent_present.clone(),
            "crt_targets_feat": target_feat,
            "crt_targets_mask": self.target_present.clone(),
            "crt_radars_feat":  radar_feat,
            "crt_radars_mask":  self.radar_present.clone(),
            "crt_time_feat":    time_feat,
        }

