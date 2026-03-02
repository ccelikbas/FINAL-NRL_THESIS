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

from .agents import JammerAgent, StrikerAgent
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

        # RNG
        try:
            self._rng = torch.Generator(device=self.device)
        except TypeError:
            self._rng = torch.Generator()
        self._set_seed(seed)

        # dimensions
        self.act_dim   = 3
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
        # own(7) + nearest_target(3) + nearest_radar(2) + nearest_mate(3) = 15
        return 15

    def _compute_state_dim(self) -> int:
        A, T, R = self.n_agents, self.n_targets, self.n_radars
        return (2 * A) + (2 * A) + A + (2 * T) + T + (2 * R)

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------

    def _make_specs(self):
        B = self.batch_size

        obs_spec   = Unbounded(shape=B + torch.Size([self.n_agents, self.obs_dim]),   dtype=torch.float32, device=self.device)
        act_spec   = Bounded(low=-1.0, high=1.0, shape=B + torch.Size([self.n_agents, self.act_dim]), dtype=torch.float32, device=self.device)
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
        action = tensordict.get(self._action_key)  # [B, A, 3]
        B, A, _ = action.shape
        rp = self.reward_params

        # ---- kinematics ----
        throttle  = (action[..., 0].clamp(-1, 1) + 1.0) * 0.5   # [B, A]  0→1
        v_min_vec = torch.zeros(self.n_agents, device=self.device)
        v_min_vec[: self.n_strikers] = float(self.striker.v_min)
        v_min_vec[self.n_strikers:]  = float(self.jammer.v_min)
        v_min  = v_min_vec.unsqueeze(0).expand(B, -1)
        speed  = throttle * (self.v_max - v_min) + v_min
        dpsi   = action[..., 1].clamp(-1, 1) * self.dpsi_max
        mode   = action[..., 2]

        alive  = self.agent_alive
        speed  = speed * alive
        dpsi   = dpsi   * alive

        self.agent_heading = (self.agent_heading + dpsi) % (2.0 * math.pi)
        dx = speed * torch.cos(self.agent_heading) * self.dt
        dy = speed * torch.sin(self.agent_heading) * self.dt
        self.agent_pos = (self.agent_pos + torch.stack([dx, dy], dim=-1)).clamp(self.low, self.high)

        # ---- EA / jamming ----
        radar_eff_range = torch.full((B, self.n_radars), self.radar_range, device=self.device)
        jammer_idx = torch.arange(self.n_strikers, self.n_agents, device=self.device)

        if jammer_idx.numel() > 0:
            jammer_on  = (mode[:, jammer_idx] > 0.0) & alive[:, jammer_idx]
            rel_jr     = self.radar_pos[:, None, :, :] - self.agent_pos[:, jammer_idx, None, :]  # [B,nj,R,2]
            jam_mask   = self.jammer.jams_radar(rel_jr) & jammer_on[:, :, None]                  # [B,nj,R]
            any_jam    = jam_mask.any(dim=1)                                                      # [B,R]
            jam_active = jam_mask.any(dim=2)                                                      # [B,nj]
            jam_active_count = jam_active.sum(dim=1).float()                                      # [B]
            team_jam_reward  = jam_active_count * float(rp.jamming)
            radar_eff_range  = torch.where(
                any_jam,
                (radar_eff_range - self.jammer.delta_range).clamp_min(0.0),
                radar_eff_range,
            )
        else:
            team_jam_reward = torch.zeros(B, device=self.device)

        self.radar_eff_range = radar_eff_range.clone()

        # ---- radar kills own agents ----
        rel_ar = self.radar_pos[:, None, :, :] - self.agent_pos[:, :, None, :]   # [B,A,R,2]
        dist_ar = torch.linalg.norm(rel_ar, dim=-1)                               # [B,A,R]
        in_radar = dist_ar <= radar_eff_range[:, None, :]
        killed   = in_radar.any(dim=-1) & alive
        self.agent_alive = self.agent_alive & (~killed)

        # ---- striker kinetic kills ----
        kill_t     = torch.zeros(B, self.n_targets, dtype=torch.bool, device=self.device)
        striker_idx = torch.arange(0, self.n_strikers, device=self.device)
        can         = None

        if striker_idx.numel() > 0:
            striker_on = (mode[:, striker_idx] > 0.0) & self.agent_alive[:, striker_idx]
            rel_st     = self.target_pos[:, None, :, :] - self.agent_pos[:, striker_idx, None, :]  # [B,ns,T,2]
            can        = self.striker.can_engage(rel_st, self.agent_heading[:, striker_idx][:, :, None])
            can        = can & striker_on[:, :, None] & self.target_alive[:, None, :]
            kill_t     = can.any(dim=1)
            self.target_alive = self.target_alive & (~kill_t)

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
        ).clamp(0.0, 1.0) * float(rp.border) * alive.float()

        rel_at_all = self.target_pos[:, None, :, :] - self.agent_pos[:, :, None, :]   # [B,A,T,2]
        dist_at    = torch.linalg.norm(rel_at_all, dim=-1)
        mask_at    = self.target_alive[:, None, :].expand(-1, self.n_agents, -1)
        dist_masked = torch.where(mask_at, dist_at, torch.full_like(dist_at, float("inf")))
        dist_min, _ = dist_masked.min(dim=-1)
        max_dist    = math.hypot(self.high - self.low, self.high - self.low)
        target_rew  = (1.0 - (dist_min / max_dist)).clamp_min(0.0) * float(rp.move_closer) * alive.float()

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
        B, A = self.num_envs, self.n_agents

        # --- own state (7) ---
        pos   = self.agent_pos
        h     = self.agent_heading
        alive = self.agent_alive.float().unsqueeze(-1)
        hs    = torch.sin(h).unsqueeze(-1)
        hc    = torch.cos(h).unsqueeze(-1)
        type_oh = torch.zeros(B, A, 2, device=self.device)
        type_oh[:, : self.n_strikers, 0] = 1.0
        type_oh[:, self.n_strikers:,  1] = 1.0
        own = torch.cat([pos, hs, hc, alive, type_oh], dim=-1)   # [B,A,7]

        # --- nearest visible target (3) ---
        rel_at  = self.target_pos[:, None, :, :] - self.agent_pos[:, :, None, :]
        dist_at = torch.linalg.norm(rel_at, dim=-1)
        mask_t  = (dist_at <= self.R_obs) & self.target_alive[:, None, :]
        dist_mt = torch.where(mask_t, dist_at, torch.full_like(dist_at, 1e9))
        idx_t   = dist_mt.argmin(dim=-1)
        rel_t   = rel_at.gather(2, idx_t[..., None, None].expand(B, A, 1, 2)).squeeze(2)
        none_t  = dist_mt.min(dim=-1).values >= 1e8
        rel_t   = torch.where(none_t[..., None], torch.zeros_like(rel_t), rel_t)
        t_alv   = self.target_alive.gather(1, idx_t).float()
        t_alv   = torch.where(none_t, torch.zeros_like(t_alv), t_alv).unsqueeze(-1)
        near_t  = torch.cat([rel_t, t_alv], dim=-1)              # [B,A,3]

        # --- nearest visible radar (2) ---
        rel_ar  = self.radar_pos[:, None, :, :] - self.agent_pos[:, :, None, :]
        dist_ar = torch.linalg.norm(rel_ar, dim=-1)
        mask_r  = dist_ar <= self.R_obs
        dist_mr = torch.where(mask_r, dist_ar, torch.full_like(dist_ar, 1e9))
        idx_r   = dist_mr.argmin(dim=-1)
        rel_r   = rel_ar.gather(2, idx_r[..., None, None].expand(B, A, 1, 2)).squeeze(2)
        none_r  = dist_mr.min(dim=-1).values >= 1e8
        rel_r   = torch.where(none_r[..., None], torch.zeros_like(rel_r), rel_r)  # [B,A,2]

        # --- nearest visible teammate (3) ---
        rel_aa  = self.agent_pos[:, None, :, :] - self.agent_pos[:, :, None, :]
        dist_aa = torch.linalg.norm(rel_aa, dim=-1)
        eye     = torch.eye(A, device=self.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        dist_aa = torch.where(eye, torch.full_like(dist_aa, 1e9), dist_aa)
        mask_m  = (dist_aa <= self.R_obs) & self.agent_alive[:, None, :]
        dist_mm = torch.where(mask_m, dist_aa, torch.full_like(dist_aa, 1e9))
        idx_m   = dist_mm.argmin(dim=-1)
        rel_m   = rel_aa.gather(2, idx_m[..., None, None].expand(B, A, 1, 2)).squeeze(2)
        none_m  = dist_mm.min(dim=-1).values >= 1e8
        rel_m   = torch.where(none_m[..., None], torch.zeros_like(rel_m), rel_m)
        m_alv   = self.agent_alive.gather(1, idx_m).float()
        m_alv   = torch.where(none_m, torch.zeros_like(m_alv), m_alv).unsqueeze(-1)
        near_m  = torch.cat([rel_m, m_alv], dim=-1)              # [B,A,3]

        return torch.cat([own, near_t, rel_r, near_m], dim=-1)   # [B,A,15]
