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
        self.act_dim   = 2  # [acceleration, angular_acceleration], each Categorical(3) → {-1, 0, +1}
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
        # own(3) + other_agents_visible(n_agents-1, each with x,y,angle,alive=4) + radars(2 each) + targets(2 each)
        # = 3 + (n_agents-1)*4 + n_radars*2 + n_targets*2
        return 3 + (self.n_agents - 1) * 4 + self.n_radars * 2 + self.n_targets * 2

    def _compute_state_dim(self) -> int:
        A, T, R = self.n_agents, self.n_targets, self.n_radars
        return (2 * A) + (2 * A) + A + (2 * T) + T + (2 * R)

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
        act_spec   = Categorical(n=3, shape=B + torch.Size([self.n_agents, self.act_dim]), dtype=torch.int64, device=self.device)
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
        action = tensordict.get(self._action_key)  # [B, A, 2] discrete actions in {0, 1, 2} -> {-1, 0, +1}
        B, A, _ = action.shape
        rp = self.reward_params
        
        # Convert discrete actions {0, 1, 2} to accelerations {-1, 0, +1}
        # action[..., 0]: velocity (linear) acceleration
        # action[..., 1]: heading (angular) acceleration
        acc = action.float() - 1.0  # Convert {0,1,2} to {-1,0,+1}  [B, A, 2]
        
        alive = self.agent_alive  # Keep as bool for bitwise operations
        alive_float = alive.float()  # Use for multiplication
        
        # ---- Velocity dynamics ----
        v_accel = acc[..., 0]  # [B, A] in {-1, 0, +1}
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
        h_accel = acc[..., 1]  # [B, A] in {-1, 0, +1}
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

        # 2. Border penalty (per agent, proportional to proximity)
        pos       = self.agent_pos
        dist_bord = torch.stack([
            pos[..., 0] - self.low,
            self.high - pos[..., 0],
            pos[..., 1] - self.low,
            self.high - pos[..., 1],
        ], dim=-1).min(dim=-1).values                                   # [B, A]
        border_pen = (
            (self.border_thresh - dist_bord) / self.border_thresh
        ).clamp(0.0, 1.0) * float(rp.border_penalty) * alive_float
        reward += border_pen

        # 3. Timestep penalty (per alive agent)
        reward += float(rp.timestep_penalty) * alive_float

        # 4. Jammer shaping: reward for actively jamming a radar
        if jam_active.numel() > 0:
            reward[:, self.n_strikers:] += jam_active.float() * float(rp.jammer_jamming)

        # 5. Striker shaping: proximity to nearest alive target
        if striker_idx.numel() > 0 and self.n_targets > 0:
            rel_st_all = self.target_pos[:, None, :, :] - self.agent_pos[:, :self.n_strikers, None, :]
            dist_st    = torch.linalg.norm(rel_st_all, dim=-1)         # [B, ns, T]
            mask_t     = self.target_alive[:, None, :].expand(-1, self.n_strikers, -1)
            dist_masked = torch.where(mask_t, dist_st, torch.full_like(dist_st, float("inf")))
            dist_min, _ = dist_masked.min(dim=-1)                      # [B, ns]
            max_dist    = math.hypot(self.high - self.low, self.high - self.low)
            prox_rew    = (1.0 - dist_min / max_dist).clamp_min(0.0) * float(rp.striker_proximity)
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
        B    = self.num_envs
        A, T, R = self.n_agents, self.n_targets, self.n_radars

        pos_a    = self.agent_pos.reshape(B, 2 * A)
        head     = self.agent_heading
        head_sc  = torch.stack([torch.sin(head), torch.cos(head)], dim=-1).reshape(B, 2 * A)
        alive_a  = self.agent_alive.float().reshape(B, A)
        pos_t    = self.target_pos.reshape(B, 2 * T)
        alive_t  = self.target_alive.float().reshape(B, T)
        pos_r    = self.radar_pos.reshape(B, 2 * R)

        return torch.cat([pos_a, head_sc, alive_a, pos_t, alive_t, pos_r], dim=-1)

    def _build_local_obs(self) -> torch.Tensor:
        """
        Observation for each agent:
        - Own state: [x, y, heading] (3 dims)
        - Other agents within obs range: [x, y, heading, alive] for each (n_agents-1, padded with zeros when not visible) (4 dims each)
        - Radar positions: [x, y] for each radar (fixed, known) (2 dims each)
        - Target positions: [x, y] for each target (fixed, known) (2 dims each)
        """
        B, A, T, R = self.num_envs, self.n_agents, self.n_targets, self.n_radars
        
        # --- Own state [x, y, heading] ---
        pos = self.agent_pos  # [B, A, 2]
        h = self.agent_heading  # [B, A]
        hs = torch.sin(h).unsqueeze(-1)  # [B, A, 1]
        hc = torch.cos(h).unsqueeze(-1)  # [B, A, 1]
        own = torch.cat([pos, hs, hc], dim=-1)  # [B, A, 4] -> actually [B,A,3] (x,y,heading encoded)
        # Actually, let's use [x, y, angle] more directly:
        own = torch.cat([pos, h.unsqueeze(-1)], dim=-1)  # [B, A, 3]
        
        # --- Other agents within observation range ---
        # For each agent, include all OTHER agents within R_obs, padded to (A-1)
        other_agents_obs = torch.zeros(B, A, (A-1), 4, dtype=torch.float32, device=self.device)
        
        # Compute distances between all pairs of agents
        rel_aa = self.agent_pos[:, None, :, :] - self.agent_pos[:, :, None, :]  # [B, A, A, 2]
        dist_aa = torch.linalg.norm(rel_aa, dim=-1)  # [B, A, A]
        
        # Mark own position to exclude self
        eye = torch.eye(A, device=self.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        dist_aa_masked = torch.where(eye, float('inf'), dist_aa)  # [B, A, A]
        
        # For each agent, find which other agents are visible
        visible_mask = (dist_aa_masked <= self.R_obs)  # [B, A, A]
        
        # Build padded observation for other agents
        for a in range(A):
            # Get other agents (excluding self)
            other_indices = torch.arange(A, device=self.device)
            other_indices = other_indices[other_indices != a]
            
            # For this agent, get obs of other agents
            for oth_idx, other_a in enumerate(other_indices):
                other_agents_obs[:, a, oth_idx, 0] = self.agent_pos[:, other_a, 0]  # x
                other_agents_obs[:, a, oth_idx, 1] = self.agent_pos[:, other_a, 1]  # y
                other_agents_obs[:, a, oth_idx, 2] = torch.sin(self.agent_heading[:, other_a])  # sin(heading)
                other_agents_obs[:, a, oth_idx, 3] = torch.cos(self.agent_heading[:, other_a])  # cos(heading)
                
                # Zero out if not visible or not alive
                not_visible = ~visible_mask[:, a, other_a] | ~self.agent_alive[:, other_a]
                other_agents_obs[:, a, oth_idx, :] *= (~not_visible).float().unsqueeze(-1)
        
        other_agents_obs = other_agents_obs.reshape(B, A, -1)  # [B, A, (A-1)*4]
        
        # --- Radar positions [x, y] (known, static) ---
        radar_obs = self.radar_pos.unsqueeze(1).expand(B, A, R, 2).reshape(B, A, -1)  # [B, A, R*2]
        
        # --- Target positions [x, y] (known, static) ---
        target_obs = self.target_pos.unsqueeze(1).expand(B, A, T, 2).reshape(B, A, -1)  # [B, A, T*2]
        
        # Concatenate all observations
        obs = torch.cat([own, other_agents_obs, radar_obs, target_obs], dim=-1)  # [B, A, obs_dim]
        
        return obs
