"""
Dual centralised critic for MAPPO.

Reads global state → outputs per-agent state values.
Separate sub-networks for striker and jammer agent groups so each role's
value function can specialise independently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

from strike_ea.env import StrikeEA2DEnv


class DualCentralisedCritic(nn.Module):
    """Two-headed centralised critic (striker head + jammer head).

    Input  : state  (B, state_dim)
    Output : values (B, n_agents, 1)
    """

    def __init__(self, state_dim: int, hidden: int, n_strikers: int, n_jammers: int):
        super().__init__()
        self.n_strikers = n_strikers
        self.n_jammers  = n_jammers

        self.striker_head = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, n_strikers),
        )
        self.jammer_head = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, n_jammers),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        sv = self.striker_head(state)   # (B, n_strikers)
        jv = self.jammer_head(state)    # (B, n_jammers)
        return torch.cat([sv, jv], dim=-1).unsqueeze(-1)  # (B, n_agents, 1)


def make_critic(env: StrikeEA2DEnv, hidden: int = 256) -> TensorDictModule:
    """Build a dual centralised critic that outputs per-agent values."""
    net = DualCentralisedCritic(
        state_dim=env.state_dim,
        hidden=hidden,
        n_strikers=env.n_strikers,
        n_jammers=env.n_jammers,
    ).to(env.device)
    return TensorDictModule(
        net,
        in_keys=["state"],
        out_keys=[(env.group, "state_value")],
    ).to(env.device)
