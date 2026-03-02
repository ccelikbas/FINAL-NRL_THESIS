"""
Baseline 2D Strike–EA MARL + MAPPO (TorchRL 0.11.x compatible)

Key conventions:
- reset() returns root keys at time t (no reward)
        border_thresh: float = 0.05,
        border_penalty: float = 0.5,
        target_reward_coef: float = 0.4,
- step() produces a tensordict with a nested "next" containing obs/done/reward at t+1
- reward exists ONLY in ("next", reward_key)
- done/terminated exist in both root and next (collector/env may duplicate)
- Multi-agent data lives under a nested tensordict "agents" with extra agent dimension
"""

from __future__ import annotations

import math
import os
import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

# --- TorchRL ---
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.transforms import RewardSum
from torchrl.envs.utils import check_env_specs

from torchrl.data import Composite, Unbounded, Bounded, Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.modules import ProbabilisticActor, TanhNormal, MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Polygon


# ----------------------------
# Helpers
# ----------------------------


@dataclass
class RewardConfig:
    target_destroyed: float = 10.0
    agent_destroyed: float = -10.0
    move_closer: float = 1.0
    jamming: float = 1.0
    border: float = -1.0
    small_step: float = -0.01


def flat_key(*parts):
    """Flatten nested tuple keys safely."""
    out = []
    for p in parts:
        if isinstance(p, tuple):
            out.extend(p)
        else:
            out.append(p)
    return tuple(out)


def td_get(td: TensorDict, *keyparts):
    return td.get(flat_key(*keyparts))


def td_set(td: TensorDict, *keyparts, value):
    td.set(flat_key(*keyparts), value)


def _safe_check_env_specs(env):
    try:
        check_env_specs(env)
        print("check_env_specs: OK")
    except Exception as e:
        # TorchRL spec checks can be stricter across versions; don't hard-crash training.
        print(f"check_env_specs warning (continuing): {type(e).__name__}: {e}")


def _get_loss_component(loss_td: TensorDict, candidates: List[str]) -> torch.Tensor:
    """Pick whichever loss key exists across TorchRL versions."""
    for k in candidates:
        if k in loss_td.keys():
            return loss_td.get(k)
    raise KeyError(f"None of loss keys found: {candidates}. Got keys={list(loss_td.keys())}")


# ----------------------------
# Agent capability classes
# ----------------------------

@dataclass
class StrikerAgent:
    engage_range: float = 0.12
    engage_fov_deg: float = 60.0
    v_min: float = 0.0

    def can_engage(self, rel_vec: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        """
        rel_vec: (..., 2) striker->target relative vectors
        heading: (...,) or broadcastable to rel_vec[...,0]
        returns: (...,) bool
        """
        dist = torch.linalg.norm(rel_vec, dim=-1)
        in_range = dist <= self.engage_range

        hx = torch.cos(heading)
        hy = torch.sin(heading)
        hvec = torch.stack([hx, hy], dim=-1)  # (..., 2)

        rel_hat = rel_vec / dist.unsqueeze(-1).clamp_min(1e-6)
        cosang = (hvec * rel_hat).sum(dim=-1).clamp(-1.0, 1.0)
        ang = torch.acos(cosang)

        in_fov = ang <= (math.radians(self.engage_fov_deg) * 0.5)
        return in_range & in_fov


@dataclass
class JammerAgent:
    jam_radius: float = 0.25
    delta_range: float = 0.10
    v_min: float = 0.01

    def jams_radar(self, rel_vec: torch.Tensor) -> torch.Tensor:
        dist = torch.linalg.norm(rel_vec, dim=-1)
        return dist <= self.jam_radius


# ----------------------------
# Environment
# ----------------------------

class StrikeEA2DEnv(EnvBase):
    """
    Vectorized baseline Strike–EA environment, batch_size=[num_envs].

    Multi-agent keys (group: "agents"):
      Root at time t:
        ("agents","observation") : [B, A, obs_dim]
        ("agents","action")      : [B, A, act_dim]
        "done"                   : [B, 1]
        "terminated"             : [B, 1]
        "state"                  : [B, state_dim]

      Next at time t+1:
        ("agents","observation")
        ("agents","reward")      : [B, A, 1]
        "done", "terminated"
        "state"
    """

    group: str = "agents"

    def __init__(
        self,
        *,
        num_envs: int = 128,
        n_strikers: int = 2,
        n_jammers: int = 2,
        n_targets: int = 2,
        n_radars: int = 2,
        max_steps: int = 200,
        dt: float = 1.0,
        world_bounds: Tuple[float, float] = (0.0, 1.0),
        v_max: float = 0.02,
        dpsi_max: float = math.radians(20.0),
        R_obs: float = 0.50,
        radar_range: float = 0.20,
        border_thresh: float = 0.05,
        border_penalty: float = 0.5,
        target_reward_coef: float = 0.4,
        kill_reward_coef: float = 3.0,
        jam_reward_coef: float = 0.2,
        reward_config: Optional[RewardConfig] = None,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ):
        self._device = device if device is not None else torch.device("cpu")
        super().__init__(device=self._device, batch_size=torch.Size([num_envs]))

        self.num_envs = num_envs
        self.n_strikers = n_strikers
        self.n_jammers = n_jammers
        self.n_agents = n_strikers + n_jammers
        self.n_targets = n_targets
        self.n_radars = n_radars

        self.max_steps = max_steps
        self.dt = dt
        self.low, self.high = world_bounds
        self.v_max = float(v_max)
        self.dpsi_max = float(dpsi_max)
        self.R_obs = float(R_obs)
        self.radar_range = float(radar_range)
        # proximity / shaping rewards
        self.border_thresh = float(border_thresh)
        self.border_penalty = float(border_penalty)
        self.target_reward_coef = float(target_reward_coef)
        self.kill_reward_coef = float(kill_reward_coef)
        self.jam_reward_coef = float(jam_reward_coef)

        # consolidated reward params (can pass RewardConfig to override defaults)
        if reward_config is None:
            self.reward_params = RewardConfig()
        else:
            self.reward_params = reward_config
        # keep legacy attrs in sync
        self.kill_reward_coef = float(self.reward_params.target_destroyed)
        self.jam_reward_coef = float(self.reward_params.jamming)
        self.target_reward_coef = float(self.reward_params.move_closer)

        self.striker = StrikerAgent()
        self.jammer = JammerAgent()

        # RNG (kept for compatibility; torch.rand calls below don't use it explicitly)
        try:
            self._rng = torch.Generator(device=self.device)
        except TypeError:
            self._rng = torch.Generator()
        self._set_seed(seed)

        # Dimensions
        self.act_dim = 3
        self.obs_dim = self._compute_obs_dim()
        self.state_dim = self._compute_state_dim()

        # Keys
        self._action_key = (self.group, "action")
        self._reward_key = (self.group, "reward")
        self._obs_key = (self.group, "observation")

        # State tensors (batched)
        B, A, T, R = self.num_envs, self.n_agents, self.n_targets, self.n_radars
        self.agent_pos = torch.zeros(B, A, 2, device=self.device)
        self.agent_heading = torch.zeros(B, A, device=self.device)
        self.agent_alive = torch.ones(B, A, dtype=torch.bool, device=self.device)

        self.target_pos = torch.zeros(B, T, 2, device=self.device)
        self.target_alive = torch.ones(B, T, dtype=torch.bool, device=self.device)

        self.radar_pos = torch.zeros(B, R, 2, device=self.device)

        self.step_count = torch.zeros(B, 1, dtype=torch.int64, device=self.device)

        self._make_specs()

    def _set_seed(self, seed: int):
        self._rng.manual_seed(int(seed))
        return seed

    # ---- spec dims ----

    def _compute_obs_dim(self) -> int:
        return 7 + 3 + 2 + 3  # 15

    def _compute_state_dim(self) -> int:
        A, T, R = self.n_agents, self.n_targets, self.n_radars
        return (2 * A) + (2 * A) + A + (2 * T) + T + (2 * R)

    def _make_specs(self):
        B = self.batch_size

        obs_spec = Unbounded(
            shape=B + torch.Size([self.n_agents, self.obs_dim]),
            dtype=torch.float32,
            device=self.device,
        )
        act_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=B + torch.Size([self.n_agents, self.act_dim]),
            dtype=torch.float32,
            device=self.device,
        )
        rew_spec = Unbounded(
            shape=B + torch.Size([self.n_agents, 1]),
            dtype=torch.float32,
            device=self.device,
        )
        state_spec = Unbounded(
            shape=B + torch.Size([self.state_dim]),
            dtype=torch.float32,
            device=self.device,
        )

        done_leaf = Categorical(
            n=2,
            shape=B + torch.Size([1]),
            dtype=torch.bool,
            device=self.device,
        )

        self.observation_spec = Composite(
            agents=Composite(observation=obs_spec, shape=B, device=self.device),
            state=state_spec,
            shape=B,
            device=self.device,
        )

        self.action_spec = Composite(
            agents=Composite(action=act_spec, shape=B, device=self.device),
            shape=B,
            device=self.device,
        )

        self.reward_spec = Composite(
            agents=Composite(reward=rew_spec, shape=B, device=self.device),
            shape=B,
            device=self.device,
        )

        self.done_spec = Composite(
            done=done_leaf,
            terminated=done_leaf.clone(),
            shape=B,
            device=self.device,
        )

    # ---- reset/step ----

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        B, A, T, R = self.num_envs, self.n_agents, self.n_targets, self.n_radars

        self.agent_pos = self.low + (self.high - self.low) * torch.rand(B, A, 2, device=self.device)
        self.agent_heading = (2.0 * math.pi) * torch.rand(B, A, device=self.device)
        self.agent_alive = torch.ones(B, A, dtype=torch.bool, device=self.device)

        self.target_pos = self.low + (self.high - self.low) * torch.rand(B, T, 2, device=self.device)
        self.target_alive = torch.ones(B, T, dtype=torch.bool, device=self.device)

        self.radar_pos = self.low + (self.high - self.low) * torch.rand(B, R, 2, device=self.device)

        # initialize radar effective ranges (per-radar, per-env)
        self.radar_eff_range = torch.full((B, self.n_radars), self.radar_range, device=self.device)

        self.step_count.zero_()

        td = TensorDict({}, batch_size=[B], device=self.device)
        td.set(self._obs_key, self._build_local_obs())
        td.set("state", self._build_global_state())
        td.set("done", torch.zeros(B, 1, dtype=torch.bool, device=self.device))
        td.set("terminated", torch.zeros(B, 1, dtype=torch.bool, device=self.device))
        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict.get(self._action_key)  # [B,A,3]
        B, A, _ = action.shape

        # map throttle input in [-1,1] to per-agent speed in [v_min, v_max]
        throttle = (action[..., 0].clamp(-1, 1) + 1.0) * 0.5  # [B,A]
        # per-agent minimum speeds (strikers first, then jammers)
        v_min_vals = torch.zeros(self.n_agents, device=self.device)
        if self.n_strikers > 0:
            v_min_vals[: self.n_strikers] = float(self.striker.v_min)
        if self.n_jammers > 0:
            v_min_vals[self.n_strikers :] = float(self.jammer.v_min)
        v_min = v_min_vals.unsqueeze(0).expand(B, -1)  # [B,A]
        speed = throttle * (self.v_max - v_min) + v_min
        dpsi = action[..., 1].clamp(-1, 1) * self.dpsi_max
        mode = action[..., 2]

        alive = self.agent_alive
        speed = speed * alive
        dpsi = dpsi * alive

        self.agent_heading = (self.agent_heading + dpsi) % (2.0 * math.pi)

        dx = speed * torch.cos(self.agent_heading) * self.dt
        dy = speed * torch.sin(self.agent_heading) * self.dt
        self.agent_pos = (self.agent_pos + torch.stack([dx, dy], dim=-1)).clamp(self.low, self.high)

        # ---- EA effects ----
        radar_eff_range = torch.full((B, self.n_radars), self.radar_range, device=self.device)

        jammer_idx = torch.arange(self.n_strikers, self.n_agents, device=self.device)
        if jammer_idx.numel() > 0:
            jammer_on = (mode[:, jammer_idx] > 0.0) & alive[:, jammer_idx]
            rel_jr = self.radar_pos[:, None, :, :] - self.agent_pos[:, jammer_idx, None, :]  # [B,nj,R,2]
            jam_mask = self.jammer.jams_radar(rel_jr) & jammer_on[:, :, None]                  # [B,nj,R]
            any_jam = jam_mask.any(dim=1)                                                     # [B,R]
            # jammer activity per-env (how many jammers actively jamming any radar)
            jam_active = jam_mask.any(dim=2)  # [B,nj]
            jam_active_count = jam_active.sum(dim=1).float()  # [B]
            team_jam_reward = jam_active_count * float(self.jam_reward_coef)
            radar_eff_range = torch.where(
                any_jam,
                (radar_eff_range - self.jammer.delta_range).clamp_min(0.0),
                radar_eff_range,
            )
        else:
            team_jam_reward = torch.zeros(B, device=self.device)

        # store last used radar effective ranges for visualization/inspection
        self.radar_eff_range = radar_eff_range.clone()

        # ---- Radar kill ----
        rel_ar = self.radar_pos[:, None, :, :] - self.agent_pos[:, :, None, :]  # [B,A,R,2]
        dist_ar = torch.linalg.norm(rel_ar, dim=-1)                              # [B,A,R]
        in_radar = dist_ar <= radar_eff_range[:, None, :]                        # [B,A,R]
        killed = in_radar.any(dim=-1) & alive                                    # [B,A]
        self.agent_alive = self.agent_alive & (~killed)

        # ---- Kinetic ----
        kill_t = torch.zeros(B, self.n_targets, dtype=torch.bool, device=self.device)
        striker_idx = torch.arange(0, self.n_strikers, device=self.device)
        if striker_idx.numel() > 0:
            striker_on = (mode[:, striker_idx] > 0.0) & self.agent_alive[:, striker_idx]
            rel_st = self.target_pos[:, None, :, :] - self.agent_pos[:, striker_idx, None, :]  # [B,ns,T,2]
            can = self.striker.can_engage(rel_st, self.agent_heading[:, striker_idx][:, :, None])
            can = can & striker_on[:, :, None] & self.target_alive[:, None, :]
            kill_t = can.any(dim=1)
            self.target_alive = self.target_alive & (~kill_t)

        # ---- Proximity shaping: border penalty & target-proximity reward ----
        # dist to nearest border per agent
        pos = self.agent_pos  # [B,A,2]
        dist_left = pos[..., 0] - self.low
        dist_right = self.high - pos[..., 0]
        dist_bottom = pos[..., 1] - self.low
        dist_top = self.high - pos[..., 1]
        dist_border = torch.stack([dist_left, dist_right, dist_bottom, dist_top], dim=-1).min(dim=-1).values  # [B,A]

        border_pen = ((self.border_thresh - dist_border) / self.border_thresh).clamp_min(0.0).clamp_max(1.0) * float(self.reward_params.border)
        border_pen = border_pen * alive.float()

        # nearest target distance per agent (only consider alive targets)
        rel_at_all = self.target_pos[:, None, :, :] - self.agent_pos[:, :, None, :]  # [B,A,T,2]
        dist_at_all = torch.linalg.norm(rel_at_all, dim=-1)  # [B,A,T]
        mask_alive_t = self.target_alive[:, None, :].expand(-1, self.n_agents, -1)
        dist_masked = torch.where(mask_alive_t, dist_at_all, torch.full_like(dist_at_all, float("inf")))
        dist_min, _ = dist_masked.min(dim=-1)  # [B,A]
        max_dist = math.hypot(self.high - self.low, self.high - self.low)
        target_reward = (1.0 - (dist_min / max_dist)).clamp_min(0.0) * float(self.reward_params.move_closer)
        target_reward = target_reward * alive.float()

        per_agent_shaping = target_reward - border_pen  # [B,A]

        # ---- Reward: per-agent kill attribution + team loss + shaping ----
        # per-target 'can' is available for strikers only (shape [B, ns, T])
        if striker_idx.numel() > 0:
            # can: [B,ns,T] -> distribute each killed target's reward among strikers who can engage it
            killers = can.float()
            killers_count = killers.sum(dim=1, keepdim=True)  # [B,1,T]
            denom = killers_count.clone()
            denom[denom == 0] = 1.0
            share = killers / denom  # [B,ns,T]
            per_agent_kills = share.sum(dim=2)  # [B,ns]
            per_agent_kill_full = torch.zeros(B, A, device=self.device)
            per_agent_kill_full[:, : self.n_strikers] = per_agent_kills
        else:
            per_agent_kill_full = torch.zeros(B, A, device=self.device)

        per_agent_kill_reward = per_agent_kill_full * float(self.reward_params.target_destroyed)  # [B,A]

        # per-agent loss penalty when destroyed by radar: use configured agent_destroyed (negative)
        per_agent_loss = killed.float() * float(self.reward_params.agent_destroyed)  # [B,A]

        # per-step small penalty
        small_step_pen = float(self.reward_params.small_step)

        # team jamming: total reward per-env, distribute equally to agents so it's a team bonus
        team_jam_per_agent = (team_jam_reward.unsqueeze(-1).expand(B, A) / float(self.n_agents))

        reward = (per_agent_kill_reward + per_agent_loss + per_agent_shaping + team_jam_per_agent + small_step_pen).unsqueeze(-1).contiguous()  # [B,A,1]

        # ---- Done ----
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
        next_td.set(self._obs_key, self._build_local_obs())
        next_td.set("state", self._build_global_state())
        return next_td

    # ---- obs/state builders ----

    def _build_global_state(self) -> torch.Tensor:
        B = self.num_envs
        A, T, R = self.n_agents, self.n_targets, self.n_radars

        pos_a = self.agent_pos.reshape(B, 2 * A)
        head = self.agent_heading
        head_sc = torch.stack([torch.sin(head), torch.cos(head)], dim=-1).reshape(B, 2 * A)
        alive_a = self.agent_alive.float().reshape(B, A)

        pos_t = self.target_pos.reshape(B, 2 * T)
        alive_t = self.target_alive.float().reshape(B, T)

        pos_r = self.radar_pos.reshape(B, 2 * R)

        return torch.cat([pos_a, head_sc, alive_a, pos_t, alive_t, pos_r], dim=-1)

    def _build_local_obs(self) -> torch.Tensor:
        B, A = self.num_envs, self.n_agents

        pos = self.agent_pos
        h = self.agent_heading
        alive = self.agent_alive.float().unsqueeze(-1)
        hs = torch.sin(h).unsqueeze(-1)
        hc = torch.cos(h).unsqueeze(-1)

        type_oh = torch.zeros(B, A, 2, device=self.device)
        type_oh[:, : self.n_strikers, 0] = 1.0
        type_oh[:, self.n_strikers :, 1] = 1.0
        own = torch.cat([pos, hs, hc, alive, type_oh], dim=-1)  # [B,A,7]

        # nearest target within R_obs
        rel_at = self.target_pos[:, None, :, :] - self.agent_pos[:, :, None, :]  # [B,A,T,2]
        dist_at = torch.linalg.norm(rel_at, dim=-1)                               # [B,A,T]
        mask_t = (dist_at <= self.R_obs) & self.target_alive[:, None, :]
        dist_masked = torch.where(mask_t, dist_at, torch.full_like(dist_at, 1e9))
        idx_t = dist_masked.argmin(dim=-1)
        rel_t = rel_at.gather(2, idx_t[..., None, None].expand(B, A, 1, 2)).squeeze(2)
        none_t = dist_masked.min(dim=-1).values >= 1e8
        rel_t = torch.where(none_t[..., None], torch.zeros_like(rel_t), rel_t)
        t_alive = self.target_alive.gather(1, idx_t).float()
        t_alive = torch.where(none_t, torch.zeros_like(t_alive), t_alive).unsqueeze(-1)
        near_t = torch.cat([rel_t, t_alive], dim=-1)                              # [B,A,3]

        # nearest radar within R_obs
        rel_ar = self.radar_pos[:, None, :, :] - self.agent_pos[:, :, None, :]    # [B,A,R,2]
        dist_ar = torch.linalg.norm(rel_ar, dim=-1)
        mask_r = dist_ar <= self.R_obs
        dist_masked_r = torch.where(mask_r, dist_ar, torch.full_like(dist_ar, 1e9))
        idx_r = dist_masked_r.argmin(dim=-1)
        rel_r = rel_ar.gather(2, idx_r[..., None, None].expand(B, A, 1, 2)).squeeze(2)
        none_r = dist_masked_r.min(dim=-1).values >= 1e8
        rel_r = torch.where(none_r[..., None], torch.zeros_like(rel_r), rel_r)    # [B,A,2]

        # nearest teammate within R_obs (exclude self)
        rel_aa = self.agent_pos[:, None, :, :] - self.agent_pos[:, :, None, :]    # [B,A,A,2]
        dist_aa = torch.linalg.norm(rel_aa, dim=-1)
        eye = torch.eye(A, device=self.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        dist_aa = torch.where(eye, torch.full_like(dist_aa, 1e9), dist_aa)
        mask_m = (dist_aa <= self.R_obs) & self.agent_alive[:, None, :]
        dist_masked_m = torch.where(mask_m, dist_aa, torch.full_like(dist_aa, 1e9))
        idx_m = dist_masked_m.argmin(dim=-1)
        rel_m = rel_aa.gather(2, idx_m[..., None, None].expand(B, A, 1, 2)).squeeze(2)
        none_m = dist_masked_m.min(dim=-1).values >= 1e8
        rel_m = torch.where(none_m[..., None], torch.zeros_like(rel_m), rel_m)
        m_alive = self.agent_alive.gather(1, idx_m).float()
        m_alive = torch.where(none_m, torch.zeros_like(m_alive), m_alive).unsqueeze(-1)
        near_m = torch.cat([rel_m, m_alive], dim=-1)                               # [B,A,3]

        return torch.cat([own, near_t, rel_r, near_m], dim=-1)                     # [B,A,15]


# ----------------------------
# Networks
# ----------------------------

def make_actor(env: StrikeEA2DEnv, hidden: int = 256) -> ProbabilisticActor:
    n_agents = env.n_agents
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    backbone = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=2 * act_dim,
        n_agents=n_agents,
        centralized=False,
        share_params=True,
        depth=3,
        num_cells=hidden,
        activation_class=nn.ReLU,
    ).to(env.device)

    policy_module = TensorDictModule(
        nn.Sequential(backbone, NormalParamExtractor()),
        in_keys=[env._obs_key],
        out_keys=[(env.group, "loc"), (env.group, "scale")],
    ).to(env.device)

    action_spec_leaf = env.action_spec[env.group, "action"]

    actor = ProbabilisticActor(
        module=policy_module,
        spec=action_spec_leaf,
        in_keys=[(env.group, "loc"), (env.group, "scale")],
        out_keys=[env._action_key],
        distribution_class=TanhNormal,
        return_log_prob=True,
        log_prob_key=(env.group, "sample_log_prob"),
    )
    return actor.to(env.device)


def make_critic(env: StrikeEA2DEnv, hidden: int = 256) -> TensorDictModule:
    n_agents = env.n_agents
    state_dim = env.state_dim

    class CriticNet(nn.Module):
        def __init__(self, in_dim: int, hidden: int, n_agents: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_agents),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            out = self.net(state)
            return out.unsqueeze(-1)

    critic_net = CriticNet(state_dim, hidden, n_agents).to(env.device)
    critic = TensorDictModule(
        critic_net,
        in_keys=["state"],
        out_keys=[(env.group, "state_value")],
    )
    return critic.to(env.device)


# ----------------------------
# Training
# ----------------------------

@dataclass
class TrainConfig:
    device: torch.device = torch.device("cpu")
    num_envs: int = 256
    max_steps: int = 160
    frames_per_batch: int = 8192
    n_iters: int = 10
    num_epochs: int = 10
    minibatch_size: int = 1024
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.95
    entropy_coef: float = 1e-3


def _make_ppo_loss(actor, critic, clip_eps, entropy_coef):
    sig = inspect.signature(ClipPPOLoss.__init__)
    params = sig.parameters
    kwargs = dict(clip_epsilon=clip_eps)

    if "entropy_coeff" in params:
        kwargs["entropy_coeff"] = entropy_coef
    elif "entropy_coef" in params:
        kwargs["entropy_coef"] = entropy_coef

    if "actor_network" in params and "critic_network" in params:
        return ClipPPOLoss(actor_network=actor, critic_network=critic, normalize_advantage=False, **kwargs)
    if "actor" in params and "critic" in params:
        return ClipPPOLoss(actor=actor, critic=critic, normalize_advantage=False, **kwargs)

    return ClipPPOLoss(actor_network=actor, critic_network=critic, **kwargs)


def _make_collector(env, actor, cfg):
    sig = inspect.signature(SyncDataCollector.__init__)
    params = sig.parameters

    kwargs = dict(policy=actor, frames_per_batch=cfg.frames_per_batch, total_frames=cfg.frames_per_batch * cfg.n_iters)
    if "env" in params:
        kwargs["env"] = env
    elif "create_env_fn" in params:
        kwargs["create_env_fn"] = env

    if "device" in params:
        kwargs["device"] = cfg.device
    if "storing_device" in params:
        kwargs["storing_device"] = cfg.device

    return SyncDataCollector(**kwargs)


def _call_gae(gae, td, loss_module):
    try:
        return gae(
            td,
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        )
    except Exception:
        return gae(td)


def _prepare_done_keys_for_loss(td: TensorDict, base_env: StrikeEA2DEnv):
    next_reward = td_get(td, "next", base_env._reward_key)
    reward_shape = next_reward.shape

    next_done_root = td_get(td, "next", "done")
    next_term_root = td_get(td, "next", "terminated")

    next_done_exp = next_done_root.unsqueeze(-1).expand(reward_shape)
    next_term_exp = next_term_root.unsqueeze(-1).expand(reward_shape)

    td_set(td, "next", base_env.group, "done", value=next_done_exp)
    td_set(td, "next", base_env.group, "terminated", value=next_term_exp)

    if ("done" in td.keys()) and ("terminated" in td.keys()):
        root_done_exp = td.get("done").unsqueeze(-1).expand(reward_shape)
        root_term_exp = td.get("terminated").unsqueeze(-1).expand(reward_shape)
        td_set(td, base_env.group, "done", value=root_done_exp)
        td_set(td, base_env.group, "terminated", value=root_term_exp)


def train_mappo(cfg: TrainConfig):
    base_env = StrikeEA2DEnv(num_envs=cfg.num_envs, max_steps=cfg.max_steps, device=cfg.device, seed=0)

    env = TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env._reward_key], out_keys=[(base_env.group, "episode_reward")]),
    )

    _safe_check_env_specs(env)

    actor = make_actor(base_env).to(cfg.device)
    critic = make_critic(base_env).to(cfg.device)

    collector = _make_collector(env, actor, cfg)

    loss_module = _make_ppo_loss(actor, critic, cfg.clip_eps, cfg.entropy_coef).to(cfg.device)

    try:
        loss_module.set_keys(
            reward=base_env._reward_key,
            action=base_env._action_key,
            sample_log_prob=(base_env.group, "sample_log_prob"),
            value=(base_env.group, "state_value"),
            done=(base_env.group, "done"),
            terminated=(base_env.group, "terminated"),
        )
    except Exception as e:
        print(f"loss_module.set_keys warning (continuing): {e}")

    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=cfg.gamma, lmbda=cfg.lmbda)
    gae = loss_module.value_estimator

    optimizer = optim.Adam(loss_module.parameters(), lr=cfg.lr)

    logs: Dict[str, List[float]] = {
        "episode_reward_mean": [],
        "loss_total": [],
        "loss_policy": [],
        "loss_value": [],
        "loss_entropy": [],
    }

    for it, td in enumerate(collector):
        # --- CRITICAL FIX: force collector batch onto cfg.device ---
        td = td.to(cfg.device)

        _prepare_done_keys_for_loss(td, base_env)

        with torch.no_grad():
            try:
                critic(td)
            except Exception:
                pass
            try:
                nxt = td.get("next")
                nxt = nxt.to(cfg.device)
                critic(nxt)
                td.set("next", nxt)
            except Exception:
                pass

            _call_gae(gae, td, loss_module)

        # Flatten and keep on device
        data = td.reshape(-1).to(cfg.device)
        n_samples = data.batch_size[0] if len(data.batch_size) else data.numel()

        total_loss_acc = 0.0
        pol_acc = 0.0
        val_acc = 0.0
        ent_acc = 0.0
        n_updates = 0

        for _ in range(cfg.num_epochs):
            perm = torch.randperm(n_samples, device=cfg.device)
            for start in range(0, n_samples, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                if idx.numel() == 0:
                    continue

                # --- CRITICAL FIX: minibatch also forced onto cfg.device ---
                sub = data[idx].to(cfg.device)

                loss_vals = loss_module(sub)

                loss_policy = _get_loss_component(loss_vals, ["loss_objective", "loss_actor"])
                loss_value = _get_loss_component(loss_vals, ["loss_critic", "loss_value"])
                loss_entropy = _get_loss_component(loss_vals, ["loss_entropy"])

                total_loss = loss_policy + loss_value + loss_entropy

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
                optimizer.step()

                total_loss_acc += float(total_loss.item())
                pol_acc += float(loss_policy.item())
                val_acc += float(loss_value.item())
                ent_acc += float(loss_entropy.item())
                n_updates += 1

        try:
            collector.update_policy_weights_()
        except Exception:
            pass

        try:
            done_mask = td_get(td, "next", base_env.group, "done")
            ep_rew = td_get(td, "next", base_env.group, "episode_reward")[done_mask]
        except Exception:
            ep_rew = torch.tensor([], device=cfg.device)

        ep_rew_mean = float(ep_rew.mean().item()) if ep_rew.numel() else float("nan")

        logs["episode_reward_mean"].append(ep_rew_mean)
        logs["loss_total"].append(total_loss_acc / max(1, n_updates))
        logs["loss_policy"].append(pol_acc / max(1, n_updates))
        logs["loss_value"].append(val_acc / max(1, n_updates))
        logs["loss_entropy"].append(ent_acc / max(1, n_updates))

        if (it + 1) % 10 == 0:
            print(f"Iter {it+1:4d}/{cfg.n_iters} | ep_rew_mean {ep_rew_mean: .3f} | loss {logs['loss_total'][-1]:.4f}")

        if it + 1 >= cfg.n_iters:
            break

    try:
        collector.shutdown()
    except Exception:
        pass

    return base_env, actor, critic, logs


def plot_training(logs: Dict[str, List[float]], save_dir: Optional[str] = None):
    fig1 = plt.figure()
    plt.plot(logs["episode_reward_mean"])
    plt.xlabel("Iteration")
    plt.ylabel("Mean episode reward")
    plt.title("Training: Episode reward mean")
    plt.grid(True)

    fig2 = plt.figure()
    plt.plot(logs["loss_total"], label="total")
    plt.plot(logs["loss_policy"], label="policy")
    plt.plot(logs["loss_value"], label="value")
    plt.plot(logs["loss_entropy"], label="entropy")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training: Loss curves")
    plt.legend()
    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(os.path.join(save_dir, "episode_reward_mean.png"), dpi=160, bbox_inches="tight")
        fig2.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=160, bbox_inches="tight")

    plt.show()


# ----------------------------
# Testing + Visualization
# ----------------------------

class TestRunner:
    def __init__(self, policy: ProbabilisticActor, *, device: torch.device, max_steps: int = 220, seed: int = 123):
        self.device = device
        self.policy = policy.eval()
        self.env = StrikeEA2DEnv(num_envs=1, max_steps=max_steps, device=device, seed=seed)

    @torch.no_grad()
    def rollout(self) -> List[Dict[str, torch.Tensor]]:
        td = self.env.reset()
        frames: List[Dict[str, torch.Tensor]] = [self._snapshot()]

        for _ in range(self.env.max_steps):
            td = self.policy(td)
            td = self.env.step(td)
            frames.append(self._snapshot())

            done_flag = False
            try:
                done_flag = bool(td.get(("next", "done")).item())
            except Exception:
                try:
                    done_flag = bool(td.get("done").item())
                except Exception:
                    done_flag = False

            if done_flag:
                break

            td = td.get("next")

        return frames

    def _snapshot(self) -> Dict[str, torch.Tensor]:
        return {
            "agent_pos": self.env.agent_pos[0].detach().cpu(),
            "agent_alive": self.env.agent_alive[0].detach().cpu(),
            "agent_heading": self.env.agent_heading[0].detach().cpu(),
            "target_pos": self.env.target_pos[0].detach().cpu(),
            "target_alive": self.env.target_alive[0].detach().cpu(),
            "radar_pos": self.env.radar_pos[0].detach().cpu(),
            "radar_eff_range": self.env.radar_eff_range[0].detach().cpu(),
        }

    def animate(self, frames: List[Dict[str, torch.Tensor]], interval_ms: int = 70):
        fig, ax = plt.subplots()
        ax.set_xlim(self.env.low, self.env.high)
        ax.set_ylim(self.env.low, self.env.high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Strike–EA Baseline Rollout (Trained Policy)")

        striker_sc = ax.scatter([], [], s=60, marker="^", label="Strikers")
        jammer_sc = ax.scatter([], [], s=60, marker="s", label="Jammers")
        target_sc = ax.scatter([], [], s=80, marker="*", label="Targets")
        radar_sc = ax.scatter([], [], s=80, marker="X", label="Radars")
        # radar range visualizations (one Circle per radar)
        radar_circles = [ax.add_patch(Circle((0.0, 0.0), radius=0.0, fill=False, edgecolor="C3", alpha=0.6, linewidth=2)) for _ in range(self.env.n_radars)]
        # jammer jam-range circles (one per jammer)
        jammer_circles = [ax.add_patch(Circle((0.0, 0.0), radius=0.0, fill=False, edgecolor="C4", alpha=0.5, linewidth=1.5, linestyle='--')) for _ in range(self.env.n_jammers)]
        # striker forward arcs (one polygon per striker showing engage FOV & range)
        striker_arcs = [ax.add_patch(Polygon(np.empty((0, 2)), closed=True, facecolor="C2", alpha=0.18, edgecolor="C2")) for _ in range(self.env.n_strikers)]
        heading_lines = [ax.plot([], [])[0] for _ in range(self.env.n_agents)]

        ax.legend(loc="upper right")
        empty_xy = np.empty((0, 2), dtype=float)

        def init():
            striker_sc.set_offsets(empty_xy)
            jammer_sc.set_offsets(empty_xy)
            target_sc.set_offsets(empty_xy)
            radar_sc.set_offsets(empty_xy)
            for c in radar_circles:
                c.set_visible(False)
            for c in jammer_circles:
                c.set_visible(False)
            for a in striker_arcs:
                a.set_visible(False)
            for ln in heading_lines:
                ln.set_data([], [])
            return [striker_sc, jammer_sc, target_sc, radar_sc, *radar_circles, *heading_lines]

        def update(i):
            fr = frames[i]
            ap = fr["agent_pos"]
            aa = fr["agent_alive"]
            ah = fr["agent_heading"]
            tp = fr["target_pos"]
            ta = fr["target_alive"]
            rp = fr["radar_pos"]
            rr = fr.get("radar_eff_range", None)

            striker_xy = ap[: self.env.n_strikers][aa[: self.env.n_strikers]]
            jammer_xy = ap[self.env.n_strikers:][aa[self.env.n_strikers:]]

            striker_sc.set_offsets(striker_xy.numpy() if striker_xy.numel() else empty_xy)
            jammer_sc.set_offsets(jammer_xy.numpy() if jammer_xy.numel() else empty_xy)

            txy = tp[ta]
            target_sc.set_offsets(txy.numpy() if txy.numel() else empty_xy)
            radar_sc.set_offsets(rp.numpy() if rp.numel() else empty_xy)

            # update radar circles (center + radius)
            if rp.numel() and rr is not None:
                for j in range(self.env.n_radars):
                    try:
                        cx, cy = float(rp[j, 0]), float(rp[j, 1])
                    except Exception:
                        cx, cy = 0.0, 0.0
                    try:
                        radius_val = float(rr[j])
                    except Exception:
                        radius_val = float(self.env.radar_range)
                    c = radar_circles[j]
                    c.set_visible(True)
                    c.set_center((cx, cy))
                    c.set_radius(radius_val)

            # update jammer circles (center + jam radius)
            for j in range(self.env.n_jammers):
                ag_idx = self.env.n_strikers + j
                jc = jammer_circles[j]
                if aa[ag_idx].item():
                    xj, yj = float(ap[ag_idx, 0]), float(ap[ag_idx, 1])
                    jc.set_visible(True)
                    jc.set_center((xj, yj))
                    jc.set_radius(float(self.env.jammer.jam_radius))
                else:
                    jc.set_visible(False)

            # update striker forward arcs (polygon sectors)
            half_fov = 0.5 * float(self.env.striker.engage_fov_deg)
            r_str = float(self.env.striker.engage_range)
            arc_res = 24
            for s in range(self.env.n_strikers):
                sa = striker_arcs[s]
                if aa[s].item():
                    cx, cy = float(ap[s, 0]), float(ap[s, 1])
                    heading_deg = math.degrees(float(ah[s]))
                    th1 = math.radians(heading_deg - half_fov)
                    th2 = math.radians(heading_deg + half_fov)
                    if th2 < th1:
                        th2 += 2 * math.pi
                    angles = np.linspace(th1, th2, arc_res)
                    xs = cx + r_str * np.cos(angles)
                    ys = cy + r_str * np.sin(angles)
                    verts = np.vstack(([cx, cy], np.column_stack((xs, ys))))
                    sa.set_visible(True)
                    sa.set_xy(verts)
                else:
                    sa.set_visible(False)

            for k in range(self.env.n_agents):
                if not aa[k].item():
                    heading_lines[k].set_data([], [])
                    continue
                x, y = ap[k].tolist()
                dx = 0.03 * math.cos(float(ah[k]))
                dy = 0.03 * math.sin(float(ah[k]))
                heading_lines[k].set_data([x, x + dx], [y, y + dy])

            ax.set_xlabel(f"t={i}")
            return [striker_sc, jammer_sc, target_sc, radar_sc, *radar_circles, *jammer_circles, *striker_arcs, *heading_lines]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(frames),
            init_func=init,
            interval=interval_ms,
            blit=True,
            repeat=False,
        )
        plt.show()
        return ani


# ----------------------------
# Main
# ----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(device=device)

    env, actor, critic, logs = train_mappo(cfg)
    plot_training(logs)

    tester = TestRunner(actor, device=device, max_steps=220, seed=999)
    frames = tester.rollout()
    tester.animate(frames, interval_ms=70)


if __name__ == "__main__":
    main()
