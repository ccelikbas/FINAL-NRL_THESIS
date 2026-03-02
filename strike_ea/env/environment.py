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
        dpsi_max:     float = math.radians(20.0),
        # --- sensors ---
        R_obs:        float = 0.50,
        radar_range:  float = 0.20,
        radar_kill_probability: float = 1.0,
        # --- reward shaping ---
        border_thresh: float = 0.05,
        reward_config: Optional[RewardConfig] = None,
        # --- misc ---
        device: Optional[torch.device] = None,
        seed:   int = 0,
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
        self.dpsi_max  = float(dpsi_max)

        # sensors
        self.R_obs       = float(R_obs)
        self.radar_range = float(radar_range)

        # shaping
        self.border_thresh = float(border_thresh)

        # reward params
        self.reward_params = reward_config if reward_config is not None else RewardConfig()

        # agent capability objects
        self.striker = StrikerAgent()
        self.jammer  = JammerAgent()
        self.radar   = RadarAgent(kill_probability=radar_kill_probability)

        # RNG
        try:
            self._rng = torch.Generator(device=self.device)
        except TypeError:
            self._rng = torch.Generator()
        self._set_seed(seed)

        # dimensions
        self.act_dim   = 6  # 3 velocity accelerations + 3 heading accelerations, each discrete: -1, 0, +1
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

    def _compute_obs_dim(self) -> int:
        # own(3) + other_agents_visible(n_agents-1, each with x,y,angle,alive=4) + radars(2 each) + targets(2 each)
        # = 3 + (n_agents-1)*4 + n_radars*2 + n_targets*2
        return 3 + (self.n_agents - 1) * 4 + self.n_radars * 2 + self.n_targets * 2

    def _compute_state_dim(self) -> int:
        A, T, R = self.n_agents, self.n_targets, self.n_radars
        return (2 * A) + (2 * A) + A + (2 * T) + T + (2 * R)

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

        self.agent_pos     = self.low + (self.high - self.low) * torch.rand(B, A, 2, device=self.device)
        self.agent_heading = (2.0 * math.pi) * torch.rand(B, A, device=self.device)
        self.agent_speed   = torch.zeros(B, A, device=self.device)  # Start with zero velocity
        self.agent_heading_rate = torch.zeros(B, A, device=self.device)  # Start with zero angular velocity
        self.agent_alive   = torch.ones(B, A, dtype=torch.bool, device=self.device)
        self.target_pos    = self.low + (self.high - self.low) * torch.rand(B, T, 2, device=self.device)
        self.target_alive  = torch.ones(B, T, dtype=torch.bool, device=self.device)
        self.radar_pos     = self.low + (self.high - self.low) * torch.rand(B, R, 2, device=self.device)
        self.radar_eff_range = torch.full((B, R), self.radar_range, device=self.device)
        self.step_count.zero_()

        td = TensorDict({}, batch_size=[B], device=self.device)
        td.set(self._obs_key, self._build_local_obs())
        td.set("state",       self._build_global_state())
        td.set("done",        torch.zeros(B, 1, dtype=torch.bool, device=self.device))
        td.set("terminated",  torch.zeros(B, 1, dtype=torch.bool, device=self.device))
        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict.get(self._action_key)  # [B, A, 6] discrete actions in {0, 1, 2} -> {-1, 0, +1}
        B, A, _ = action.shape
        rp = self.reward_params
        
        # Convert discrete actions {0, 1, 2} to accelerations {-1, 0, +1}
        # action[..., 0:3]: velocity accelerations (or used as needed)
        # action[..., 3:6]: heading/angular accelerations (or used as needed)
        acc = action.float() - 1.0  # Convert {0,1,2} to {-1,0,+1}  [B, A, 6]
        
        alive = self.agent_alive  # Keep as bool for bitwise operations
        alive_float = alive.float()  # Use for multiplication
        
        # ---- Velocity dynamics (discrete acceleration model) ----
        # Update velocity based on acceleration (first 3 actions combined or first action as primary)
        # For simplicity, use first action as main velocity acceleration
        v_accel = acc[..., 0]  # [B, A] acceleration in {-1, 0, +1}
        
        # Set acceleration magnitude (you can tune this)
        accel_magnitude = 0.01  # per step acceleration when action is ±1
        
        # Apply acceleration to speed
        self.agent_speed = (self.agent_speed + v_accel * accel_magnitude).clamp(0.0, self.v_max)
        self.agent_speed = self.agent_speed * alive_float  # Dead agents don't move
        
        # ---- Heading dynamics (discrete angular acceleration) ----
        # Use last action as heading angular acceleration
        h_accel = acc[..., -1]  # [B, A] angular acceleration in {-1, 0, +1}
        
        # Angular acceleration magnitude
        h_accel_magnitude = self.dpsi_max * 0.1  # Fraction of max heading rate per step
        
        # Apply angular acceleration
        self.agent_heading_rate = (self.agent_heading_rate + h_accel * h_accel_magnitude).clamp(-self.dpsi_max, self.dpsi_max)
        self.agent_heading_rate = self.agent_heading_rate * alive_float  # Dead agents don't rotate
        
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
            # Jammers are always "on" if within jam radius - no action needed
            rel_jr     = self.radar_pos[:, None, :, :] - self.agent_pos[:, jammer_idx, None, :]  # [B,nj,R,2]
            jam_mask   = self.jammer.jams_radar(rel_jr) & alive[:, jammer_idx, None]  # [B,nj,R]
            any_jam    = jam_mask.any(dim=1)                                           # [B,R]
            jam_active = jam_mask.any(dim=2)                                           # [B,nj]
            jam_active_count = jam_active.sum(dim=1).float()                           # [B]
            team_jam_reward  = jam_active_count * float(rp.jamming)
            radar_eff_range  = torch.where(
                any_jam,
                (radar_eff_range - self.jammer.delta_range).clamp_min(0.0),
                radar_eff_range,
            )
        else:
            team_jam_reward = torch.zeros(B, device=self.device)

        self.radar_eff_range = radar_eff_range.clone()

        # ---- radar kills own agents (probabilistic) ----
        rel_ar = self.radar_pos[:, None, :, :] - self.agent_pos[:, :, None, :]   # [B,A,R,2]
        dist_ar = torch.linalg.norm(rel_ar, dim=-1)                               # [B,A,R]
        in_radar = dist_ar <= radar_eff_range[:, None, :]                         # [B,A,R]
        
        # Probabilistic kill: agent is killed if in radar range AND passes probability check
        # Generate random samples for each agent-radar pair
        kill_samples = torch.rand(B, A, self.n_radars, device=self.device, generator=self._rng)
        kill_prob = self.radar.kill_probability
        kills_from_radar = in_radar & (kill_samples < kill_prob)                  # [B,A,R]
        killed = kills_from_radar.any(dim=-1) & alive                             # [B,A] - killed if any radar kills them
        self.agent_alive = self.agent_alive & (~killed)

        # ---- striker kinetic kills: Strikers automatically strike if target in engagement zone ----
        kill_t     = torch.zeros(B, self.n_targets, dtype=torch.bool, device=self.device)
        striker_idx = torch.arange(0, self.n_strikers, device=self.device)

        if striker_idx.numel() > 0:
            # Strikers automatically strike (no action needed) - always attempt to engage
            rel_st     = self.target_pos[:, None, :, :] - self.agent_pos[:, striker_idx, None, :]  # [B,ns,T,2]
            can        = self.striker.can_engage(rel_st, self.agent_heading[:, striker_idx][:, :, None])
            can        = can & alive[:, striker_idx, None] & self.target_alive[:, None, :]
            kill_t     = can.any(dim=1)
            self.target_alive = self.target_alive & (~kill_t)
        else:
            can = None

        # ---- shaping rewards ----
        pos        = self.agent_pos
        dist_bord  = torch.stack([
            pos[..., 0] - self.low,
            self.high - pos[..., 0],
            pos[..., 1] - self.low,
            self.high - pos[..., 1],
        ], dim=-1).min(dim=-1).values                                             # [B,A]

        border_pen = (
            (self.border_thresh - dist_bord) / self.border_thresh
        ).clamp(0.0, 1.0) * float(rp.border) * alive_float

        rel_at_all = self.target_pos[:, None, :, :] - self.agent_pos[:, :, None, :]   # [B,A,T,2]
        dist_at    = torch.linalg.norm(rel_at_all, dim=-1)
        mask_at    = self.target_alive[:, None, :].expand(-1, self.n_agents, -1)
        dist_masked = torch.where(mask_at, dist_at, torch.full_like(dist_at, float("inf")))
        dist_min, _ = dist_masked.min(dim=-1)
        max_dist    = math.hypot(self.high - self.low, self.high - self.low)
        target_rew  = (1.0 - (dist_min / max_dist)).clamp_min(0.0) * float(rp.move_closer) * alive_float

        per_agent_shaping = target_rew - border_pen   # [B,A]

        # ---- kill / loss attribution ----
        if can is not None:
            killers       = can.float()
            denom         = killers.sum(dim=1, keepdim=True).clamp_min(1.0)
            share         = killers / denom              # [B,ns,T]
            per_agent_kill = torch.zeros(B, A, device=self.device)
            per_agent_kill[:, : self.n_strikers] = share.sum(dim=2)
        else:
            per_agent_kill = torch.zeros(B, A, device=self.device)

        per_agent_kill_rew = per_agent_kill   * float(rp.target_destroyed)
        per_agent_loss_pen = killed.float()   * float(rp.agent_destroyed)
        team_jam_per_agent = (team_jam_reward.unsqueeze(-1).expand(B, A) / float(self.n_agents))

        reward = (
            per_agent_kill_rew
            + per_agent_loss_pen
            + per_agent_shaping
            + team_jam_per_agent
            + float(rp.small_step)
        ).unsqueeze(-1).contiguous()   # [B, A, 1]

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
