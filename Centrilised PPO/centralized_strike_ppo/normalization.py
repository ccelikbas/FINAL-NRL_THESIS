from __future__ import annotations

from typing import Dict, Tuple

import torch
from tensordict import TensorDict


class RunningMeanStd:
    """Welford-style running mean/variance tracker (GPU compatible)."""

    def __init__(self, *, device: torch.device, epsilon: float = 1e-4):
        self.device = device
        self.mean = torch.zeros((), dtype=torch.float32, device=device)
        self.var = torch.ones((), dtype=torch.float32, device=device)
        self.count = torch.tensor(float(epsilon), dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().to(self.device, dtype=torch.float32)
        if x.numel() == 0:
            return
        x = x.reshape(-1)

        batch_count = torch.tensor(float(x.numel()), dtype=torch.float32, device=self.device)
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / total_count)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / total_count)
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = torch.clamp(new_var, min=1e-12)
        self.count = total_count

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "mean": self.mean.detach().clone(),
            "var": self.var.detach().clone(),
            "count": self.count.detach().clone(),
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.mean = state_dict["mean"].detach().to(self.device, dtype=torch.float32)
        self.var = torch.clamp(state_dict["var"].detach().to(self.device, dtype=torch.float32), min=1e-12)
        self.count = torch.clamp(state_dict["count"].detach().to(self.device, dtype=torch.float32), min=1e-8)

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp(self.var, min=1e-12))


class RewardNormalizer:
    """Return-based reward normalization for MAPPO rollouts.

    - Tracks per-env discounted returns.
    - Updates running std from returns.
    - Normalizes rewards by dividing by std only (no mean subtraction).
    """

    def __init__(
        self,
        *,
        num_envs: int,
        gamma: float,
        device: torch.device,
        eps: float = 1e-8,
    ):
        self.device = device
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.returns = torch.zeros(int(num_envs), dtype=torch.float32, device=device)  # [B]
        self.ret_rms = RunningMeanStd(device=device)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "returns": self.returns.detach().clone(),
            "ret_rms": self.ret_rms.state_dict(),
            "gamma": torch.tensor(self.gamma, dtype=torch.float32, device=self.device),
            "eps": torch.tensor(self.eps, dtype=torch.float32, device=self.device),
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.returns = state_dict["returns"].detach().to(self.device, dtype=torch.float32)
        self.ret_rms.load_state_dict(state_dict["ret_rms"])
        if "gamma" in state_dict:
            self.gamma = float(state_dict["gamma"].item())
        if "eps" in state_dict:
            self.eps = float(state_dict["eps"].item())

    @torch.no_grad()
    def _normalize_step(self, reward_t: torch.Tensor, done_t: torch.Tensor) -> torch.Tensor:
        # reward_t: [B, A, 1], done_t: [B, 1]
        reward_t = reward_t.to(self.device)
        done_t = done_t.to(self.device)

        team_reward_t = reward_t.squeeze(-1).sum(dim=-1)  # [B]
        done_mask = done_t.squeeze(-1).to(torch.float32)   # [B]

        self.returns = self.returns * (1.0 - done_mask)
        self.returns = self.gamma * self.returns + team_reward_t
        self.ret_rms.update(self.returns)

        std = self.ret_rms.std.clamp_min(self.eps)
        return reward_t / std

    @torch.no_grad()
    def normalize_rollout_td(
        self,
        td: TensorDict,
        *,
        reward_key: Tuple[str, ...],
        done_key: Tuple[str, ...],
    ) -> Dict[str, float]:
        rewards = td.get(reward_key)  # [T, B, A, 1] (collector) or [B, A, 1]
        dones = td.get(done_key)      # [T, B, 1] or [B, 1]

        raw_rewards = rewards.detach().clone()

        had_time_dim = rewards.ndim >= 4
        if not had_time_dim:
            rewards = rewards.unsqueeze(0)
            dones = dones.unsqueeze(0)

        rewards_norm = rewards.clone()
        T = rewards_norm.shape[0]
        for t in range(T):
            rewards_norm[t] = self._normalize_step(rewards_norm[t], dones[t])

        if not had_time_dim:
            rewards_norm = rewards_norm.squeeze(0)

        td.set(reward_key, rewards_norm)

        raw_mean = float(raw_rewards.mean().item()) if raw_rewards.numel() > 0 else float("nan")
        raw_std = float(raw_rewards.std(unbiased=False).item()) if raw_rewards.numel() > 0 else float("nan")
        norm_mean = float(rewards_norm.mean().item()) if rewards_norm.numel() > 0 else float("nan")
        norm_std = float(rewards_norm.std(unbiased=False).item()) if rewards_norm.numel() > 0 else float("nan")

        return {
            "raw_reward_mean": raw_mean,
            "raw_reward_std": raw_std,
            "normalized_reward_mean": norm_mean,
            "normalized_reward_std": norm_std,
            "running_mean": float(self.ret_rms.mean.item()),
            "running_std": float(self.ret_rms.std.item()),
        }
