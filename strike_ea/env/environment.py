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
from typing import Optional, Tuple

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
        # --- misc ---
        device: Optional[torch.device] = None,
        seed:   int = 0,
        n_env_layouts: int = 0,
    ):
        self._device = device if device is not None else torch.device("cpu")
        super().__init__(device=self._device, batch_size=torch.Size([num_envs]))

        # topology
        self.num_envs   = num_envs
        self.n_strikers = n_strikers
        self.n_jammers  = n_jammers
        self.n_agents   = n_strikers + n_jammers
        self.n_targets  = n_targets
        self.n_radars   = n_radars

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
        self.radar_range = float(radar_range)

        # shaping
        self.border_thresh = float(border_thresh)

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

        # dimensions
        self.act_dim    = 2   # [acceleration, angular_acceleration]
        self.n_choices  = 7   # per action dim: {-1, -0.5, -0.1, 0, +0.1, +0.5, +1}
        self._act_table = torch.tensor(
            [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0], device=self._device
        )  # maps discrete index → continuous multiplier
        self.obs_dim   = self._compute_obs_dim()
        self.state_dim = self._compute_state_dim()

        # canonical keys
        self._action_key = (self.group, "action")
        self._reward_key = (self.group, "reward")
        self._obs_key    = (self.group, "observation")

        # allocate state buffers
        B, A, T, R = num_envs, self.n_agents, n_targets, n_radars
        self.agent_pos     = torch.zeros(B, A, 2, device=self.device)
        self.agent_heading = torch.zeros(B, A,    device=self.device)
        self.agent_speed   = torch.zeros(B, A,    device=self.device)  # Current velocity magnitude
        self.agent_heading_rate = torch.zeros(B, A, device=self.device)  # Current angular velocity
        self.agent_alive   = torch.ones(B, A,     dtype=torch.bool, device=self.device)
        self.target_pos    = torch.zeros(B, T, 2, device=self.device)
        self.target_alive  = torch.ones(B, T,     dtype=torch.bool, device=self.device)
        self.radar_pos     = torch.zeros(B, R, 2, device=self.device)
        self.radar_eff_range = torch.full((B, R), self.radar_range, device=self.device)
        self.step_count    = torch.zeros(B, 1, dtype=torch.int64, device=self.device)

        self._make_specs()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_seed(self, seed: int):
        self._rng.manual_seed(int(seed))
        return seed

    def _pregenerate_layouts(self):
        """Pre-generate fixed radar positions for n_env_layouts distinct scenarios.

        Radars are placed in the top half of the map with margin from borders.
        Each layout uses a deterministic seed for reproducibility.
        """
        layouts = []
        margin = 0.1  # 100 km margin from borders
        for seed_idx in range(self.n_env_layouts):
            rng = torch.Generator()
            rng.manual_seed(seed_idx + 1000)  # Offset to avoid collision with main RNG
            radar_x = self.low + margin + (self.high - self.low - 2 * margin) * torch.rand(self.n_radars, generator=rng)
            radar_y = 0.5 + (self.high - 0.5 - margin) * torch.rand(self.n_radars, generator=rng)
            radar_pos = torch.stack([radar_x, radar_y], dim=-1)  # [R, 2]
            layouts.append(radar_pos)
        return layouts

    def _compute_obs_dim(self) -> int:
        # Ego-centric relative observations:
        #   own state:     [speed, heading_rate]                      = 2
        #   other agents:  [dist, rel_angle] per agent (zeroed if not visible) = (n_agents-1)*2
        #   radars:        [dist, rel_angle] per radar                = n_radars*2
        #   targets:       [dist, rel_angle, alive] per target        = n_targets*3
        return 2 + (self.n_agents - 1) * 2 + self.n_radars * 2 + self.n_targets * 3

    def _compute_state_dim(self) -> int:
        A, T, R = self.n_agents, self.n_targets, self.n_radars
        # Per-agent ego-centric relative polar:
        #   (d, θ) to all targets:       2T per agent
        #   (d, θ) to all other agents:  2(A-1) per agent
        #   (d, θ) to all radars:        2R per agent
        per_agent = 2 * T + 2 * (A - 1) + 2 * R
        # Global boolean flags:
        #   agent_alive(A) + target_alive(T) + radar_jammed(R)
        flags = A + T + R
        return A * per_agent + flags

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
        
        # For each batch, create an evenly distributed radar assignment for targets
        for b in range(B):
            # Create cyclic radar assignment: [0, 1, ..., R-1, 0, 1, ..., R-1, ...]
            # This ensures targets are distributed across radars, not clustered at random ones
            radar_assignments = torch.arange(T, device=self.device) % R  # [0, 1, 0, 1, ...] for R=2
            
            # Shuffle the assignment to randomize which radar gets which target each episode
            perm = torch.randperm(T, device=self.device)
            radar_assignments = radar_assignments[perm]
            
            # Spawn each target near its assigned radar
            for t in range(T):
                assigned_radar_idx = radar_assignments[t].item()
                radar = radar_pos[b, assigned_radar_idx, :]  # [2]
                
                # Random angle around the radar
                angle = 2.0 * math.pi * torch.rand(1, device=self.device).item()
                
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

        obs_spec   = Unbounded(shape=B + torch.Size([self.n_agents, self.obs_dim]),   dtype=torch.float32, device=self.device)
        act_spec   = Categorical(n=self.n_choices, shape=B + torch.Size([self.n_agents, self.act_dim]), dtype=torch.int64, device=self.device)
        rew_spec   = Unbounded(shape=B + torch.Size([self.n_agents, 1]),              dtype=torch.float32, device=self.device)
        state_spec = Unbounded(shape=B + torch.Size([self.state_dim]),                dtype=torch.float32, device=self.device)
        done_leaf  = Categorical(n=2, shape=B + torch.Size([1]),                      dtype=torch.bool,    device=self.device)

        self.observation_spec = Composite(
            agents=Composite(observation=obs_spec, shape=B, device=self.device),
            state=state_spec, shape=B, device=self.device,
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

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        B, A, T, R = self.num_envs, self.n_agents, self.n_targets, self.n_radars

        # --- Agents spawn at bottom middle of map, facing north ---
        center_x = (self.low + self.high) / 2.0
        bottom_y = self.low + 0.05  # 50 km from bottom edge
        spacing = 0.02  # 20 km between agents
        self.agent_pos = torch.zeros(B, A, 2, device=self.device)
        for a in range(A):
            offset_x = (a - (A - 1) / 2.0) * spacing
            self.agent_pos[:, a, 0] = center_x + offset_x
            self.agent_pos[:, a, 1] = bottom_y
        self.agent_heading = torch.full((B, A), math.pi / 2.0, device=self.device)  # Face north

        # Initialize at minimum speed (aircraft can't hover)
        self.agent_speed = torch.zeros(B, A, device=self.device)
        self.agent_speed[:, :self.n_strikers] = self.striker.v_min
        self.agent_speed[:, self.n_strikers:] = self.jammer.v_min
        self.agent_heading_rate = torch.zeros(B, A, device=self.device)
        self.agent_alive = torch.ones(B, A, dtype=torch.bool, device=self.device)

        # --- Radars: top half of map, not too close to borders ---
        if self._layouts is not None:
            # Use pre-generated deterministic layouts (cycle through them)
            for b in range(B):
                layout_idx = b % len(self._layouts)
                self.radar_pos[b] = self._layouts[layout_idx].to(self.device)
        else:
            # Fully random radar positions in top half with margin
            margin = 0.1  # 100 km from borders
            self.radar_pos[..., 0] = (self.low + margin
                + (self.high - self.low - 2 * margin)
                * torch.rand(B, R, device=self.device, generator=self._rng))
            self.radar_pos[..., 1] = (0.5
                + (self.high - 0.5 - margin)
                * torch.rand(B, R, device=self.device, generator=self._rng))

        # Spawn targets relative to radars (strategic zone logic)
        self.target_pos = self._spawn_targets_in_valid_zones(B, T, R, self.radar_pos)
        self.target_alive = torch.ones(B, T, dtype=torch.bool, device=self.device)
        self.radar_eff_range = torch.full((B, R), self.radar_range, device=self.device)
        self.step_count.zero_()

        td = TensorDict({}, batch_size=[B], device=self.device)
        td.set(self._obs_key, self._build_local_obs())
        td.set("state",       self._build_global_state())
        td.set("done",        torch.zeros(B, 1, dtype=torch.bool, device=self.device))
        td.set("terminated",  torch.zeros(B, 1, dtype=torch.bool, device=self.device))
        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict.get(self._action_key)  # [B, A, 2] discrete in {0..6}
        B, A, _ = action.shape
        rp = self.reward_params

        # Map discrete indices to continuous multipliers via lookup table
        # _act_table: [-1, -0.5, -0.1, 0, +0.1, +0.5, +1]
        acc = self._act_table[action.long()]  # [B, A, 2] float in {-1,-0.5,-0.1,0,+0.1,+0.5,+1}

        alive = self.agent_alive  # Keep as bool for bitwise operations
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

        # ==================================================================
        # Reward computation (simplified, role-specific shaping)
        # ==================================================================
        reward = torch.zeros(B, A, device=self.device)

        # 1. Team reward: target destroyed — shared equally among alive agents
        n_killed = kill_t.float().sum(dim=-1)                           # [B]
        n_alive  = self.agent_alive.float().sum(dim=-1).clamp_min(1.0)  # [B]
        team_kill = (n_killed * float(rp.target_destroyed) / n_alive)   # [B]
        reward += team_kill.unsqueeze(-1) * alive_float                 # [B, A]

        # 2. Border penalty (per agent, quadratic scaling for stronger repulsion)
        #    Linear fraction t = (border_thresh - dist) / border_thresh  in [0, 1]
        #    Quadratic penalty = t^2 × weight  →  weak far from edge, very strong at edge
        pos       = self.agent_pos
        dist_bord = torch.stack([
            pos[..., 0] - self.low,
            self.high - pos[..., 0],
            pos[..., 1] - self.low,
            self.high - pos[..., 1],
        ], dim=-1).min(dim=-1).values                                   # [B, A]
        border_frac = (
            (self.border_thresh - dist_bord) / self.border_thresh
        ).clamp(0.0, 1.0)
        border_pen = (border_frac ** 2) * float(rp.border_penalty) * alive_float
        reward += border_pen

        # 3. Timestep penalty (per alive agent)
        reward += float(rp.timestep_penalty) * alive_float

        # 4. Jammer shaping: proximity to nearest radar, but NOT within lethal radar range
        #    Reward = (1 - dist/max_dist) × weight, zeroed if inside radar_eff_range (danger zone)
        if jammer_idx.numel() > 0 and self.n_radars > 0:
            rel_jr_all = self.radar_pos[:, None, :, :] - self.agent_pos[:, self.n_strikers:, None, :]
            dist_jr    = torch.linalg.norm(rel_jr_all, dim=-1)         # [B, nj, R]
            # Find nearest radar index per jammer
            dist_jr_min, nearest_r = dist_jr.min(dim=-1)               # [B, nj]
            max_dist_j = math.hypot(self.high - self.low, self.high - self.low)
            # Get effective range of the nearest radar for each jammer
            nearest_eff_range = radar_eff_range.gather(1, nearest_r)   # [B, nj]
            # Proximity reward: closer → higher, but zero inside lethal radar range
            safe_mask  = dist_jr_min > nearest_eff_range               # [B, nj]
            prox_rew_j = (1.0 - dist_jr_min / max_dist_j).clamp_min(0.0) * float(rp.jammer_jamming)
            jammer_alive = alive[:, self.n_strikers:].float()
            reward[:, self.n_strikers:] += prox_rew_j * jammer_alive * safe_mask.float()

        # 5. Striker shaping: proximity to nearest alive target
        #    Uses alive targets only; when all targets dead the reward is zero
        #    (no inf contamination — striker just gets the team kill bonus instead)
        if striker_idx.numel() > 0 and self.n_targets > 0:
            rel_st_all = self.target_pos[:, None, :, :] - self.agent_pos[:, :self.n_strikers, None, :]
            dist_st    = torch.linalg.norm(rel_st_all, dim=-1)         # [B, ns, T]
            mask_t     = self.target_alive[:, None, :].expand(-1, self.n_strikers, -1)
            any_alive  = mask_t.any(dim=-1)                            # [B, ns] — any target still alive?
            dist_masked = torch.where(mask_t, dist_st, torch.full_like(dist_st, float("inf")))
            dist_min, _ = dist_masked.min(dim=-1)                      # [B, ns]
            max_dist    = math.hypot(self.high - self.low, self.high - self.low)
            prox_rew    = (1.0 - dist_min / max_dist).clamp_min(0.0) * float(rp.striker_proximity)
            prox_rew    = prox_rew * any_alive.float()                 # zero shaping when all targets dead
            striker_alive = alive[:, :self.n_strikers].float()
            reward[:, :self.n_strikers] += prox_rew * striker_alive

        # 6. Agent destruction penalty (applied to agents killed by radar this step)
        reward += killed.float() * float(rp.agent_destroyed)

        reward = reward.unsqueeze(-1).contiguous()  # [B, A, 1]

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
        return next_td

    # ------------------------------------------------------------------
    # Observation / state builders
    # ------------------------------------------------------------------

    def _build_global_state(self) -> torch.Tensor:
        """Ego-centric global state for the centralised critic.

        Layout (flat vector):
          For each agent i  (A blocks):
            (d, θ) to each target       → 2T
            (d, θ) to each other agent   → 2(A-1)
            (d, θ) to each radar         → 2R
          Global flags:
            agent_alive   → A
            target_alive  → T
            radar_jammed  → R

        Total dim = A × (2T + 2(A-1) + 2R) + A + T + R
        """
        B    = self.num_envs
        A, T, R = self.n_agents, self.n_targets, self.n_radars
        max_dist = math.hypot(self.high - self.low, self.high - self.low)

        per_agent_parts: list[torch.Tensor] = []

        # --- Per-agent relative polar to targets ---
        dist_at, angle_at = self._relative_polar(
            self.agent_pos, self.agent_heading, self.target_pos
        )  # [B, A, T]
        dist_at_norm  = dist_at / max_dist          # normalise to ~[0,1]
        angle_at_norm = angle_at / math.pi           # normalise to [-1,1]
        # interleave (d, θ) per target: [B, A, 2T]
        at_feat = torch.stack([dist_at_norm, angle_at_norm], dim=-1).reshape(B, A, 2 * T)
        per_agent_parts.append(at_feat)

        # --- Per-agent relative polar to other agents ---
        dist_aa, angle_aa = self._relative_polar(
            self.agent_pos, self.agent_heading, self.agent_pos
        )  # [B, A, A]
        dist_aa_norm  = dist_aa / max_dist
        angle_aa_norm = angle_aa / math.pi
        aa_feat = torch.stack([dist_aa_norm, angle_aa_norm], dim=-1)  # [B, A, A, 2]
        # Remove self-column (diagonal)
        idx = []
        for a in range(A):
            idx.append(torch.cat([torch.arange(0, a, device=self.device),
                                  torch.arange(a + 1, A, device=self.device)]))
        idx = torch.stack(idx, dim=0)  # [A, A-1]
        idx_exp = idx[None, :, :, None].expand(B, A, A - 1, 2)
        aa_feat = aa_feat.gather(2, idx_exp).reshape(B, A, 2 * (A - 1))  # [B, A, 2(A-1)]
        per_agent_parts.append(aa_feat)

        # --- Per-agent relative polar to radars ---
        dist_ar, angle_ar = self._relative_polar(
            self.agent_pos, self.agent_heading, self.radar_pos
        )  # [B, A, R]
        dist_ar_norm  = dist_ar / max_dist
        angle_ar_norm = angle_ar / math.pi
        ar_feat = torch.stack([dist_ar_norm, angle_ar_norm], dim=-1).reshape(B, A, 2 * R)
        per_agent_parts.append(ar_feat)

        # Flatten per-agent features: [B, A, feat_per_agent] → [B, A * feat_per_agent]
        per_agent = torch.cat(per_agent_parts, dim=-1)  # [B, A, 2T+2(A-1)+2R]
        per_agent_flat = per_agent.reshape(B, -1)        # [B, A*(2T+2(A-1)+2R)]

        # --- Global boolean flags ---
        alive_a  = self.agent_alive.float()              # [B, A]
        alive_t  = self.target_alive.float()             # [B, T]
        jammed_r = (self.radar_eff_range < self.radar_range).float()  # [B, R]  1=jammed

        return torch.cat([per_agent_flat, alive_a, alive_t, jammed_r], dim=-1)

    # ------------------------------------------------------------------
    # Ego-centric helpers
    # ------------------------------------------------------------------

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
        """Ego-centric relative observation per agent.

        Layout (per agent):
          own:     [speed, heading_rate]                           (2)
          agents:  [dist, rel_angle] × (n_agents-1)               (2 per other agent; zeroed if outside R_obs or dead)
          radars:  [dist, rel_angle] × n_radars                   (2 per radar)
          targets: [dist, rel_angle, alive] × n_targets            (3 per target)
        Total obs_dim = 2 + 2*(A-1) + 2*R + 3*T
        """
        B, A, T, R = self.num_envs, self.n_agents, self.n_targets, self.n_radars
        max_dist = math.hypot(self.high - self.low, self.high - self.low)  # for normalisation

        # --- Own kinematic state ---
        speed_norm = (self.agent_speed / self.v_max).unsqueeze(-1)            # [B,A,1] in ~[0,1]
        hrate_norm = (self.agent_heading_rate / self.dpsi_max).unsqueeze(-1)  # [B,A,1] in ~[-1,1]
        own = torch.cat([speed_norm, hrate_norm], dim=-1)                     # [B,A,2]

        # --- Other agents (relative distance + relative angle) ---
        dist_aa, angle_aa = self._relative_polar(
            self.agent_pos, self.agent_heading, self.agent_pos
        )  # [B,A,A], [B,A,A]

        # Exclude self by setting diagonal to inf / 0
        eye = torch.eye(A, device=self.device, dtype=torch.bool).unsqueeze(0)  # [1,A,A]
        dist_aa  = torch.where(eye, torch.tensor(float('inf'), device=self.device), dist_aa)
        angle_aa = torch.where(eye, torch.zeros(1, device=self.device), angle_aa)

        # Visibility mask: within R_obs AND alive
        visible = (dist_aa <= self.R_obs) & self.agent_alive[:, None, :]  # [B,A,A]

        # Normalise distance
        dist_aa_norm = dist_aa / max_dist  # [B,A,A]
        angle_aa_norm = angle_aa / math.pi  # [B,A,A] in [-1,1]

        # Stack (dist, angle) and zero out invisible, then drop self-column
        other_obs = torch.stack([dist_aa_norm, angle_aa_norm], dim=-1)  # [B,A,A,2]
        other_obs = other_obs * visible.unsqueeze(-1).float()           # zero out non-visible

        # Remove self-column: gather all-but-diagonal
        # Build index list excluding self for each agent
        idx = []
        for a in range(A):
            idx.append(torch.cat([torch.arange(0, a, device=self.device),
                                  torch.arange(a + 1, A, device=self.device)]))
        idx = torch.stack(idx, dim=0)  # [A, A-1]
        idx_exp = idx[None, :, :, None].expand(B, A, A - 1, 2)  # [B,A,A-1,2]
        other_obs = other_obs.gather(2, idx_exp)                 # [B,A,A-1,2]
        other_obs = other_obs.reshape(B, A, -1)                  # [B,A,(A-1)*2]

        # --- Radars (relative distance + relative angle) ---
        dist_ar, angle_ar = self._relative_polar(
            self.agent_pos, self.agent_heading, self.radar_pos
        )  # [B,A,R]
        dist_ar_norm  = dist_ar / max_dist
        angle_ar_norm = angle_ar / math.pi
        radar_obs = torch.stack([dist_ar_norm, angle_ar_norm], dim=-1)  # [B,A,R,2]
        radar_obs = radar_obs.reshape(B, A, -1)                         # [B,A,R*2]

        # --- Targets (relative distance + relative angle + alive flag) ---
        dist_at, angle_at = self._relative_polar(
            self.agent_pos, self.agent_heading, self.target_pos
        )  # [B,A,T]
        dist_at_norm  = dist_at / max_dist
        angle_at_norm = angle_at / math.pi
        alive_t = self.target_alive[:, None, :].expand(B, A, T).float()  # [B,A,T]
        target_obs = torch.stack([dist_at_norm, angle_at_norm, alive_t], dim=-1)  # [B,A,T,3]
        target_obs = target_obs.reshape(B, A, -1)                                 # [B,A,T*3]

        # --- Concatenate ---
        obs = torch.cat([own, other_obs, radar_obs, target_obs], dim=-1)  # [B,A,obs_dim]
        return obs
