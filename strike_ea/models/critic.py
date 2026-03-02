"""Centralised critic for MAPPO (reads global state, outputs per-agent values)."""

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

from strike_ea.env import StrikeEA2DEnv


class _CentralisedCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int, n_agents: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,    hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).unsqueeze(-1)   # [B, A, 1]


def make_critic(env: StrikeEA2DEnv, hidden: int = 256) -> TensorDictModule:
    """Build a centralised critic that takes global state and outputs per-agent values."""
    net = _CentralisedCritic(env.state_dim, hidden, env.n_agents).to(env.device)
    return TensorDictModule(
        net,
        in_keys=["state"],
        out_keys=[(env.group, "state_value")],
    ).to(env.device)
