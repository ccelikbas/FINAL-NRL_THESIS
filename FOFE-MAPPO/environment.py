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
from typing import List, Optional, Tuple

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
        # --- FOFE mode ---
        use_fofe: bool = False,
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

        # RNG
        try:
            self._rng = torch.Generator(device=self.device)
        except TypeError:
            self._rng = torch.Generator()
        self._set_seed(seed)

        # Layout control (pre-generated radar positions for reproducible scenarios)
        self.n_env_layouts = n_env_layouts
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
            "control_effort": torch.zeros(B, device=self.device),
        }

        # Episode outcome tracking (bypasses tensordict auto-reset overwrite)
        self._completed_episodes: list = []

        self._make_specs()

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
    ) -> torch.Tensor:
        """Sample *n_radars* positions with at least *min_sep* between each pair.

        Uses rejection sampling: draw candidates one-by-one, reject if too
        close to any already-placed radar.  Falls back to uniform (no spacing
        guarantee) after *max_attempts* total draws to avoid infinite loops.
        """
        sample_device = device if device is not None else torch.device("cpu")
        placed: List[torch.Tensor] = []
        attempts = 0
        while len(placed) < n_radars and attempts < max_attempts:
            cx = x_lo + (x_hi - x_lo) * torch.rand(1, generator=rng, device=sample_device)
            cy = y_lo + (y_hi - y_lo) * torch.rand(1, generator=rng, device=sample_device)
            candidate = torch.tensor([cx.item(), cy.item()], device=sample_device)
            ok = True
            for p in placed:
                if torch.linalg.norm(candidate - p).item() < min_sep:
                    ok = False
                    break
            if ok:
                placed.append(candidate)
            attempts += 1

        # Fallback: fill remaining radars without spacing constraint
        while len(placed) < n_radars:
            cx = x_lo + (x_hi - x_lo) * torch.rand(1, generator=rng, device=sample_device)
            cy = y_lo + (y_hi - y_lo) * torch.rand(1, generator=rng, device=sample_device)
            placed.append(torch.tensor([cx.item(), cy.item()], device=sample_device))

        return torch.stack(placed, dim=0)  # [R, 2]

    def _pregenerate_layouts(self):
        """Pre-generate fixed radar positions for n_env_layouts distinct scenarios.

        Radars are placed within the configured spawn zone with minimum spacing.
        Each layout uses a deterministic seed for reproducibility.
        """
        layouts = []
        x_lo, x_hi = 0.2, 0.8
        y_lo, y_hi = 0.4, 0.8
        min_sep = 0.1
        for seed_idx in range(self.n_env_layouts):
            rng = torch.Generator()
            rng.manual_seed(seed_idx + 1000)  # Offset to avoid collision with main RNG
            radar_pos = self._sample_spaced_radars(
                self.n_radars, x_lo, x_hi, y_lo, y_hi, min_sep, rng,
            )
            layouts.append(radar_pos)
        return layouts

    def _compute_obs_dim(self) -> int:
        # Fixed-size ego-centric relative observations (independent of team sizes):
        #   own state:     [x, y, speed, heading, heading_rate, t_norm]             = 6
        #   other agents:  top-K slots [dist, rel_angle, heading, role]             = K_a * 4
        #   radars:        top-K slots [dist, rel_angle, jammed]                    = K_r * 3
        #   targets:       top-K slots [dist, rel_angle, alive]                     = K_t * 3
        # Unseen or missing entities are zero-padded in their slots.
        return 6 + self.n_other_agent_obs_slots * 4 + self.n_radar_obs_slots * 3 + self.n_target_obs_slots * 3

    def _compute_state_dim(self) -> int:
        A, T, R = self.n_agents, self.n_targets, self.n_radars
        # Global absolute state for centralised critic:
        #   Per agent:  (x, y, v, ψ, ω, role, alive)     = 7 × A
        #   Per target: (x, y, alive)                      = 3 × T
        #   Per radar:  (x, y, active, detection_radius)   = 4 × R
        #   Global:     (t_norm)                           = 1
        return 7 * A + 3 * T + 4 * R + 1

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
        spawn_distance = 0.9 * unconstrained_range
        
        target_pos = torch.zeros(B, T, 2, device=self.device)
        
        known_radar_idx = torch.arange(self.n_known_radars, device=self.device)
        unknown_radar_idx = torch.arange(self.n_known_radars, self.n_radars, device=self.device)

        # For each batch, create radar assignments while keeping known/unknown grouped
        for b in range(B):
            radar_assignments = torch.zeros(T, dtype=torch.long, device=self.device)

            def _assign_group(start_idx: int, count: int, group_radar_idx: torch.Tensor):
                if count <= 0:
                    return
                if group_radar_idx.numel() == 0:
                    base = torch.arange(count, device=self.device) % max(R, 1)
                else:
                    base_local = torch.arange(count, device=self.device) % group_radar_idx.numel()
                    base = group_radar_idx[base_local]
                perm_local = torch.randperm(count, device=self.device)
                radar_assignments[start_idx:start_idx + count] = base[perm_local]

            known_target_count = min(self.n_known_targets, T)
            unknown_target_count = max(0, T - known_target_count)

            _assign_group(0, known_target_count, known_radar_idx)
            _assign_group(known_target_count, unknown_target_count, unknown_radar_idx)
            
            # Spawn each target near its assigned radar
            for t in range(T):
                assigned_radar_idx = radar_assignments[t].item()
                radar = radar_pos[b, assigned_radar_idx, :]  # [2]
                
                # Random angle around the radar (within configured range)
                lo = self.target_spawn_angle_lo
                hi = self.target_spawn_angle_hi
                span = (hi - lo) % (2.0 * math.pi)  # handle wrap-around
                if span == 0.0:
                    span = 2.0 * math.pi  # full circle when lo == hi
                angle = lo + span * torch.rand(1, device=self.device).item()
                
                # Offset at fixed distance in random direction
                offset = spawn_distance * torch.tensor([
                    math.cos(angle),
                    math.sin(angle)
                ], device=self.device)
                
                candidate = radar + offset  # [2]
                
                # Clamp to world bounds
                candidate = candidate.clamp(self.low, self.high)
                
                target_pos[b, t, :] = candidate
        
        return target_pos

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
            # FOFE actor per-channel observations (nested under "agents")
            agents_dict["obs_self"]          = Unbounded(shape=B + torch.Size([A, 6]),    dtype=torch.float32, device=self.device)
            agents_dict["obs_agents_feat"]   = Unbounded(shape=B + torch.Size([A, A, 4]), dtype=torch.float32, device=self.device)
            agents_dict["obs_agents_mask"]   = Unbounded(shape=B + torch.Size([A, A]),    dtype=torch.bool,    device=self.device)
            agents_dict["obs_targets_feat"]  = Unbounded(shape=B + torch.Size([A, T, 3]), dtype=torch.float32, device=self.device)
            agents_dict["obs_targets_mask"]  = Unbounded(shape=B + torch.Size([A, T]),    dtype=torch.bool,    device=self.device)
            agents_dict["obs_radars_feat"]   = Unbounded(shape=B + torch.Size([A, R, 3]), dtype=torch.float32, device=self.device)
            agents_dict["obs_radars_mask"]   = Unbounded(shape=B + torch.Size([A, R]),    dtype=torch.bool,    device=self.device)
            # FOFE critic global-state channels (root-level keys)
            root_dict["crt_agents_feat"]   = Unbounded(shape=B + torch.Size([A, 7]), dtype=torch.float32, device=self.device)
            root_dict["crt_agents_mask"]   = Unbounded(shape=B + torch.Size([A]),    dtype=torch.bool,    device=self.device)
            root_dict["crt_targets_feat"]  = Unbounded(shape=B + torch.Size([T, 3]), dtype=torch.float32, device=self.device)
            root_dict["crt_targets_mask"]  = Unbounded(shape=B + torch.Size([T]),    dtype=torch.bool,    device=self.device)
            root_dict["crt_radars_feat"]   = Unbounded(shape=B + torch.Size([R, 4]), dtype=torch.float32, device=self.device)
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

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        B, A, T, R = self.num_envs, self.n_agents, self.n_targets, self.n_radars

        reset_mask = self._extract_reset_mask(tensordict)
        reset_idx = reset_mask.nonzero(as_tuple=False).squeeze(-1)
        n_reset = int(reset_idx.numel())

        if n_reset > 0:
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
            self.agent_alive[reset_idx] = True

            # --- Radars: top half of map, not too close to borders ---
            radar_pos_reset = torch.zeros(n_reset, R, 2, device=self.device)
            if self._layouts is not None:
                # Use pre-generated deterministic layouts (cycle through them)
                for i, env_i in enumerate(reset_idx.tolist()):
                    layout_idx = env_i % len(self._layouts)
                    radar_pos_reset[i] = self._layouts[layout_idx].to(self.device)
            else:
                # Random radar positions with minimum spacing constraint
                x_lo, x_hi = 0.2, 0.8
                y_lo, y_hi = 0.6, 0.8
                min_sep = 0.2
                for i in range(n_reset):
                    radar_pos_reset[i] = self._sample_spaced_radars(
                        R, x_lo, x_hi, y_lo, y_hi, min_sep, self._rng, device=self.device,
                    ).to(self.device)
            self.radar_pos[reset_idx] = radar_pos_reset

            # Spawn targets relative to radars (strategic zone logic)
            target_pos_reset = self._spawn_targets_in_valid_zones(n_reset, T, R, radar_pos_reset)
            self.target_pos[reset_idx] = target_pos_reset
            self.target_alive[reset_idx] = True
            target_known_reset = torch.zeros(n_reset, T, dtype=torch.bool, device=self.device)
            if self.n_known_targets > 0:
                target_known_reset[:, :self.n_known_targets] = True
            self.target_known[reset_idx] = target_known_reset

            radar_known_reset = torch.zeros(n_reset, R, dtype=torch.bool, device=self.device)
            if self.n_known_radars > 0:
                radar_known_reset[:, :self.n_known_radars] = True
            self.radar_known[reset_idx] = radar_known_reset

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

        td = TensorDict({}, batch_size=[B], device=self.device)
        td.set(self._obs_key, self._build_local_obs())
        td.set("state",       self._build_global_state())
        td.set("done",        torch.zeros(B, 1, dtype=torch.bool, device=self.device))
        td.set("terminated",  torch.zeros(B, 1, dtype=torch.bool, device=self.device))
        # ---- FOFE: emit structured per-channel observations ----
        if self.use_fofe:
            for k, v in self._build_fofe_obs().items():
                td.set(("agents", k), v)
            for k, v in self._build_fofe_critic_state().items():
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

        # ---- EA / jamming: Jammers automatically jam radars within range ----
        radar_eff_range = torch.full((B, self.n_radars), self.radar_range, device=self.device)
        jammer_idx = torch.arange(self.n_strikers, self.n_agents, device=self.device)

        if jammer_idx.numel() > 0:
            rel_jr     = self.radar_pos[:, None, :, :] - self.agent_pos[:, jammer_idx, None, :]  # [B,nj,R,2]
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
        rel_ar = self.radar_pos[:, None, :, :] - self.agent_pos[:, :, None, :]   # [B,A,R,2]
        dist_ar = torch.linalg.norm(rel_ar, dim=-1)                               # [B,A,R]
        in_radar = dist_ar <= radar_eff_range[:, None, :]                         # [B,A,R]
        
        kill_samples = torch.rand(B, A, self.n_radars, device=self.device, generator=self._rng)
        kill_prob = self.radar.kill_probability
        kills_from_radar = in_radar & (kill_samples < kill_prob)
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
        if float(rp.target_destroyed) != 0.0 and bool(kill_t.any().item()):
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
        if float(rp.terminal_bonus) != 0.0 and bool(all_targets_done_now.any().item()):
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
            rel_st_all = self.target_pos[:, None, :, :] - self.agent_pos[:, :self.n_strikers, None, :]
            dist_st    = torch.linalg.norm(rel_st_all, dim=-1)            # [B, ns, T]
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
            rel_jr_all = self.radar_pos[:, None, :, :] - self.agent_pos[:, self.n_strikers:, None, :]
            dist_jr    = torch.linalg.norm(rel_jr_all, dim=-1)            # [B, nj, R]
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
            # Reuse dist_jr computed in section 6 (or compute if jammer approach was skipped)
            if 'dist_jr' not in dir():
                rel_jr_all = self.radar_pos[:, None, :, :] - self.agent_pos[:, self.n_strikers:, None, :]
                dist_jr = torch.linalg.norm(rel_jr_all, dim=-1)
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
            # Reuse dist_st computed in section 5 (or compute if striker approach was skipped)
            if 'dist_st' not in dir():
                rel_st_all = self.target_pos[:, None, :, :] - self.agent_pos[:, :self.n_strikers, None, :]
                dist_st = torch.linalg.norm(rel_st_all, dim=-1)

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
        if float(rp.agent_destroyed) != 0.0 and bool(killed.any().item()):
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
        timeout          = self.step_count >= self.max_steps

        terminated = all_targets_done | all_agents_dead
        done       = terminated | timeout

        next_td = TensorDict({}, batch_size=[B], device=self.device)
        next_td.set(self._reward_key, reward)
        next_td.set("done",       done.to(torch.bool))
        next_td.set("terminated", terminated.to(torch.bool))
        next_td.set(self._obs_key, self._build_local_obs())
        next_td.set("state",       self._build_global_state())
        # ---- FOFE: emit structured per-channel observations ----
        if self.use_fofe:
            for k, v in self._build_fofe_obs().items():
                next_td.set(("agents", k), v)
            for k, v in self._build_fofe_critic_state().items():
                next_td.set(k, v)

        # Track completed episode stats in Python list (immune to auto-reset)
        if bool(done.any().item()):
            for b in range(B):
                if done[b, 0].item():
                    tgt_frac = float((~self.target_alive[b]).float().mean().item())
                    surv_frac = float(self.agent_alive[b].float().mean().item())
                    self._completed_episodes.append({
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

            # Avoid duplicate logging if terminal envs are stepped again before reset
            self._episode_team_reward[done.squeeze(-1)] = 0.0
            for comp_key in self._episode_component_reward:
                self._episode_component_reward[comp_key][done.squeeze(-1)] = 0.0

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
          Per agent i:  (x, y, v, ψ, ω, role, alive)      × A   (7A)
          Per target k: (x, y, alive)                       × T   (3T)
          Per radar r:  (x, y, active, detection_radius)    × R   (4R)
                    Global:       (t_norm)                                  (1)

                Total dim = 7A + 3T + 4R + 1
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
        agent_feat = torch.cat([pos_norm, speed_norm, hdg_norm,
                                omega_norm, role, alive_a], dim=-1)                # [B,A,7]
        agent_flat = agent_feat.reshape(B, -1)                                     # [B, 7A]

        # --- Target features: (x, y, alive) per target ---
        tgt_pos_norm = (self.target_pos - self.low) / world_range                  # [B,T,2]
        tgt_alive    = self.target_alive.float().unsqueeze(-1)                     # [B,T,1]
        tgt_feat     = torch.cat([tgt_pos_norm, tgt_alive], dim=-1)               # [B,T,3]
        tgt_flat     = tgt_feat.reshape(B, -1)                                     # [B, 3T]

        # --- Radar features: (x, y, active, detection_radius) per radar ---
        rdr_pos_norm  = (self.radar_pos - self.low) / world_range                  # [B,R,2]
        rdr_active    = (self.radar_eff_range >= self.radar_range).float().unsqueeze(-1)  # [B,R,1] 1=active
        rdr_range_n   = (self.radar_eff_range / self.radar_range).unsqueeze(-1)    # [B,R,1] in [0,1]
        rdr_feat      = torch.cat([rdr_pos_norm, rdr_active, rdr_range_n], dim=-1) # [B,R,4]
        rdr_flat      = rdr_feat.reshape(B, -1)                                    # [B, 4R]

        # --- Global normalised time feature: t / max_steps ---
        time_norm = (self.step_count.float() / float(self.max_steps)).clamp(0.0, 1.0)  # [B,1]

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

    def _build_local_obs(self) -> torch.Tensor:
        """Ego-centric relative observation per agent with fixed slot counts.

        Layout (per agent):
          own:     [x, y, speed, heading, heading_rate, t_norm]                 (6)
          agents:  3 nearest visible other agents [dist, rel_angle, heading, role] (3×4)
          radars:  2 nearest visible radars [dist, rel_angle, jammed]              (2×3)
          targets: 2 nearest visible alive targets [dist, rel_angle, alive]         (2×3)

        Visibility is partial observability via R_obs. If an entity is outside
        observation range or a slot is unused, that slot is all zeros.
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
        time_norm    = (self.step_count.float() / float(self.max_steps)).clamp(0.0, 1.0)  # [B,1]
        time_exp     = time_norm.unsqueeze(1).expand(B, A, 1)                    # [B,A,1]
        own = torch.cat([pos_norm, speed_norm, heading_norm, hrate_norm, time_exp], dim=-1)  # [B,A,6]

        # --- Agent role vector (static): strikers = 1, jammers = 0 ---
        role = torch.zeros(A, device=self.device)
        role[:self.n_strikers] = 1.0
        role = role[None, :].expand(B, A)  # [B, A]

        # --- Other agents (relative distance + relative angle + heading + role) ---
        dist_aa, angle_aa = self._relative_polar(
            self.agent_pos, self.agent_heading, self.agent_pos
        )  # [B,A,A], [B,A,A]

        dist_agents = torch.linalg.norm(
            self.agent_pos[:, :, None, :] - self.agent_pos[:, None, :, :], dim=-1
        )  # [B,A,A]

        # Exclude self by setting diagonal to inf / 0
        eye = torch.eye(A, device=self.device, dtype=torch.bool).unsqueeze(0)  # [1,A,A]
        dist_aa  = torch.where(eye, torch.tensor(float('inf'), device=self.device), dist_aa)
        angle_aa = torch.where(eye, torch.zeros(1, device=self.device), angle_aa)

        # Visibility mask: within R_obs AND alive
        visible = (dist_aa <= self.R_obs) & self.agent_alive[:, None, :]  # [B,A,A]

        # Normalised features
        dist_aa_norm  = dist_aa / max_dist                                          # [B,A,A]
        angle_aa_norm = angle_aa / math.pi                                          # [B,A,A] in [-1,1]
        heading_other = self.agent_heading[:, None, :].expand(B, A, A) / math.pi    # [B,A,A] heading of observed agent
        role_other    = role[:, None, :].expand(B, A, A)                            # [B,A,A] role of observed agent

        other_feat = torch.stack([dist_aa_norm, angle_aa_norm, heading_other, role_other], dim=-1)  # [B,A,A,4]
        other_slots = _select_nearest_slots(
            dist=dist_aa,
            feat=other_feat,
            visible=visible,
            k=self.n_other_agent_obs_slots,
        )  # [B,A,3,4]
        other_obs = other_slots.reshape(B, A, -1)  # [B,A,3*4]

        # --- Radars (relative distance + relative angle + jammed flag) ---
        dist_ar, angle_ar = self._relative_polar(
            self.agent_pos, self.agent_heading, self.radar_pos
        )  # [B,A,R]
        dist_ar_norm  = dist_ar / max_dist
        angle_ar_norm = angle_ar / math.pi
        jammed = (self.radar_eff_range < self.radar_range).float()          # [B,R] 1=jammed
        jammed_exp = jammed[:, None, :].expand(B, A, R)                     # [B,A,R]

        # ------------------------------------------------------------------
        # Communication graph over alive agents (per env in batch)
        #   comm_adj[b, i, j]  = edge(i,j) in G_t   iff dist(i,j) <= R_comm
        #   comm_reach         = transitive closure of G_t (multi-hop reachability)
        # This implements subset-based sharing: agents in same connected component
        # share unknown detections with each other at this timestep.
        # ------------------------------------------------------------------
        alive_agents = self.agent_alive
        eye_agents = torch.eye(A, dtype=torch.bool, device=self.device).unsqueeze(0).expand(B, -1, -1)
        comm_adj = (dist_agents <= self.R_comm) & alive_agents[:, :, None] & alive_agents[:, None, :]
        comm_reach = comm_adj | eye_agents
        for _ in range(max(A - 1, 0)):
            comm_reach = comm_reach | (torch.matmul(comm_reach.float(), comm_reach.float()) > 0)

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
        radar_feat = torch.stack([dist_ar_norm, angle_ar_norm, jammed_exp], dim=-1)  # [B,A,R,3]
        radar_slots = _select_nearest_slots(
            dist=dist_ar,
            feat=radar_feat,
            visible=radar_visible,
            k=self.n_radar_obs_slots,
        )  # [B,A,2,3]
        radar_obs = radar_slots.reshape(B, A, -1)                           # [B,A,2*3]

        # --- Targets (relative distance + relative angle + alive flag) ---
        dist_at, angle_at = self._relative_polar(
            self.agent_pos, self.agent_heading, self.target_pos
        )  # [B,A,T]
        dist_at_norm  = dist_at / max_dist
        angle_at_norm = angle_at / math.pi
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
        target_feat = torch.stack([dist_at_norm, angle_at_norm, alive_t], dim=-1)  # [B,A,T,3]
        target_slots = _select_nearest_slots(
            dist=dist_at,
            feat=target_feat,
            visible=target_visible & self.target_alive[:, None, :],
            k=self.n_target_obs_slots,
        )  # [B,A,2,3]
        target_obs = target_slots.reshape(B, A, -1)                          # [B,A,2*3]

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
            obs_self          [B, A, 6]      ego kinematic state
            obs_agents_feat   [B, A, A, 4]   per-other-agent features (self-row masked)
            obs_agents_mask   [B, A, A]      True = visible & alive & not self
            obs_targets_feat  [B, A, T, 3]   per-target features
            obs_targets_mask  [B, A, T]      True = visible & alive
            obs_radars_feat   [B, A, R, 3]   per-radar features
            obs_radars_mask   [B, A, R]      True = visible
        """
        B, A, T, R = self.num_envs, self.n_agents, self.n_targets, self.n_radars
        max_dist = math.hypot(self.high - self.low, self.high - self.low)
        world_range = self.high - self.low

        # ---- Self-observation: [B, A, 6] ----
        pos_norm = (self.agent_pos - self.low) / world_range
        speed_norm = (self.agent_speed / self.v_max).unsqueeze(-1)
        heading_norm = (self.agent_heading / math.pi).unsqueeze(-1)
        hrate_norm = (self.agent_heading_rate / self.dpsi_max).unsqueeze(-1)
        time_norm = (self.step_count.float() / float(self.max_steps)).clamp(0, 1)
        time_exp = time_norm.unsqueeze(1).expand(B, A, 1)
        obs_self = torch.cat([pos_norm, speed_norm, heading_norm, hrate_norm, time_exp], dim=-1)

        role = torch.zeros(A, device=self.device)
        role[:self.n_strikers] = 1.0
        role = role[None, :].expand(B, A)

        # ---- Communication graph (same as _build_local_obs) ----
        dist_agents = torch.linalg.norm(
            self.agent_pos[:, :, None, :] - self.agent_pos[:, None, :, :], dim=-1
        )
        eye = torch.eye(A, dtype=torch.bool, device=self.device).unsqueeze(0).expand(B, -1, -1)
        comm_adj = (dist_agents <= self.R_comm) & self.agent_alive[:, :, None] & self.agent_alive[:, None, :]
        comm_reach = comm_adj | eye
        for _ in range(max(A - 1, 0)):
            comm_reach = comm_reach | (torch.matmul(comm_reach.float(), comm_reach.float()) > 0)

        # ---- Agents channel: [B, A, A, 4] + mask [B, A, A] ----
        dist_aa, angle_aa = self._relative_polar(self.agent_pos, self.agent_heading, self.agent_pos)
        dist_aa = torch.where(eye, torch.tensor(float('inf'), device=self.device), dist_aa)
        angle_aa = torch.where(eye, torch.zeros(1, device=self.device), angle_aa)
        agents_visible = (dist_aa <= self.R_obs) & self.agent_alive[:, None, :] & ~eye

        agents_feat = torch.stack([
            dist_aa / max_dist,
            angle_aa / math.pi,
            self.agent_heading[:, None, :].expand(B, A, A) / math.pi,
            role[:, None, :].expand(B, A, A),
        ], dim=-1)  # [B, A, A, 4]
        agents_feat = agents_feat * agents_visible.unsqueeze(-1).float()

        # ---- Radars channel: [B, A, R, 3] + mask [B, A, R] ----
        dist_ar, angle_ar = self._relative_polar(self.agent_pos, self.agent_heading, self.radar_pos)
        jammed = (self.radar_eff_range < self.radar_range).float()[:, None, :].expand(B, A, R)

        # Known/unknown visibility with communication sharing
        radar_known_mask = self.radar_known[:, None, :].expand(B, A, R)
        local_radar_obs = (dist_ar <= self.R_obs) & self.agent_alive[:, :, None]
        local_unknown_radar = local_radar_obs & (~self.radar_known)[:, None, :].expand(B, A, R)
        shared_unknown_radar = torch.matmul(comm_reach.float(), local_unknown_radar.float()) > 0
        radars_visible = radar_known_mask | shared_unknown_radar

        radars_feat = torch.stack([dist_ar / max_dist, angle_ar / math.pi, jammed], dim=-1)
        radars_feat = radars_feat * radars_visible.unsqueeze(-1).float()

        # ---- Targets channel: [B, A, T, 3] + mask [B, A, T] ----
        dist_at, angle_at = self._relative_polar(self.agent_pos, self.agent_heading, self.target_pos)
        alive_t = self.target_alive[:, None, :].expand(B, A, T).float()

        target_known_mask = self.target_known[:, None, :].expand(B, A, T)
        local_target_obs = (dist_at <= self.R_obs) & self.agent_alive[:, :, None] & self.target_alive[:, None, :]
        local_unknown_target = local_target_obs & (~self.target_known)[:, None, :].expand(B, A, T)
        shared_unknown_target = torch.matmul(comm_reach.float(), local_unknown_target.float()) > 0
        targets_visible = (target_known_mask | shared_unknown_target) & self.target_alive[:, None, :]

        targets_feat = torch.stack([dist_at / max_dist, angle_at / math.pi, alive_t], dim=-1)
        targets_feat = targets_feat * targets_visible.unsqueeze(-1).float()

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
            agents  [B, A, 7]   x,y,v,ψ,ω,role,alive
            targets [B, T, 3]   x,y,alive
            radars  [B, R, 4]   x,y,active,eff_range
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
        agent_feat = torch.cat([pos_norm, speed_norm, hdg_norm, omega_norm, role, alive_a], dim=-1)

        # ---- Target features [B, T, 3] ----
        tgt_pos_norm = (self.target_pos - self.low) / world_range
        tgt_alive = self.target_alive.float().unsqueeze(-1)
        target_feat = torch.cat([tgt_pos_norm, tgt_alive], dim=-1)

        # ---- Radar features [B, R, 4] ----
        rdr_pos_norm = (self.radar_pos - self.low) / world_range
        rdr_active = (self.radar_eff_range >= self.radar_range).float().unsqueeze(-1)
        rdr_range_n = (self.radar_eff_range / self.radar_range).unsqueeze(-1)
        radar_feat = torch.cat([rdr_pos_norm, rdr_active, rdr_range_n], dim=-1)

        # ---- Time feature [B, 1] ----
        time_feat = (self.step_count.float() / float(self.max_steps)).clamp(0, 1)

        # All masks True — critic has full global visibility
        return {
            "crt_agents_feat":  agent_feat,
            "crt_agents_mask":  torch.ones(B, A, dtype=torch.bool, device=self.device),
            "crt_targets_feat": target_feat,
            "crt_targets_mask": torch.ones(B, T, dtype=torch.bool, device=self.device),
            "crt_radars_feat":  radar_feat,
            "crt_radars_mask":  torch.ones(B, R, dtype=torch.bool, device=self.device),
            "crt_time_feat":    time_feat,
        }

