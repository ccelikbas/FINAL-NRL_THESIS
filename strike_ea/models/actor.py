"""
Dual discrete actor for Strike–EA MARL.

Two separate policy networks:
  • Striker policy — shared parameters across all strikers
  • Jammer  policy — shared parameters across all jammers

Action space: 2 discrete dimensions per agent:
  dim 0 — linear acceleration   Categorical(3) → {0,1,2} mapped to {-1,0,+1}
  dim 1 — angular acceleration  Categorical(3) → {0,1,2} mapped to {-1,0,+1}
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import MultiAgentMLP, ProbabilisticActor

from strike_ea.env import StrikeEA2DEnv


# ──────────────────────────────────────────────────────────────────────────────
# Custom distribution: independent Categoricals with joint log-prob
# ──────────────────────────────────────────────────────────────────────────────

class MultiCategorical(torch.distributions.Distribution):
    """Multiple independent Categorical distributions with joint log probability.

    logits shape : (..., n_dims, n_choices)
    sample shape : (..., n_dims)
    log_prob     : (...)          — sum of per-dimension log probs
    entropy      : (...)          — sum of per-dimension entropies
    """

    arg_constraints = {}
    has_rsample = False

    def __init__(self, logits: torch.Tensor):
        self._cats = torch.distributions.Categorical(logits=logits)
        batch_shape = logits.shape[:-2]
        event_shape = logits.shape[-2:-1]          # (n_dims,)
        super().__init__(batch_shape=batch_shape, validate_args=False)
        self._event_shape = event_shape

    def sample(self, sample_shape=torch.Size()):
        return self._cats.sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        per_dim = self._cats.log_prob(value)       # (..., n_dims)
        return per_dim.sum(dim=-1)                 # (...)

    def entropy(self) -> torch.Tensor:
        return self._cats.entropy().sum(dim=-1)    # (...)

    @property
    def mode(self) -> torch.Tensor:
        return self._cats.logits.argmax(dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        # Discrete distribution: return mode as "mean" for deterministic eval
        return self.mode.float()

    @property
    def deterministic_sample(self) -> torch.Tensor:
        return self.mode


# ──────────────────────────────────────────────────────────────────────────────
# Dual-role backbone
# ──────────────────────────────────────────────────────────────────────────────

class DualPolicyNet(nn.Module):
    """Two MultiAgentMLPs (striker & jammer) merged into a single forward pass.

    Input  : (B, n_agents, obs_dim)
    Output : (B, n_agents, act_dim, n_choices)   — logits for MultiCategorical
    """

    def __init__(
        self,
        obs_dim:    int,
        act_dim:    int,
        n_choices:  int,
        n_strikers: int,
        n_jammers:  int,
        hidden:     int = 256,
        depth:      int = 3,
    ):
        super().__init__()
        self.n_strikers = n_strikers
        self.n_jammers  = n_jammers
        self.act_dim    = act_dim
        self.n_choices  = n_choices

        self.striker_net = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=act_dim * n_choices,
            n_agents=n_strikers,
            centralized=False,
            share_params=True,
            depth=depth,
            num_cells=hidden,
            activation_class=nn.ReLU,
        )
        self.jammer_net = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=act_dim * n_choices,
            n_agents=n_jammers,
            centralized=False,
            share_params=True,
            depth=depth,
            num_cells=hidden,
            activation_class=nn.ReLU,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, n_agents, obs_dim)
        s_logits = self.striker_net(obs[:, : self.n_strikers])       # (B, ns, act*n)
        j_logits = self.jammer_net(obs[:, self.n_strikers :])        # (B, nj, act*n)
        all_logits = torch.cat([s_logits, j_logits], dim=1)          # (B, A, act*n)
        B, A, _ = all_logits.shape
        return all_logits.view(B, A, self.act_dim, self.n_choices)   # (B, A, 2, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Public factory
# ──────────────────────────────────────────────────────────────────────────────

def make_actor(env: StrikeEA2DEnv, hidden: int = 256) -> ProbabilisticActor:
    """Build a dual-role discrete actor (strikers share params; jammers share params)."""
    n_choices = 3  # each action dim: {0, 1, 2}

    backbone = DualPolicyNet(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        n_choices=n_choices,
        n_strikers=env.n_strikers,
        n_jammers=env.n_jammers,
        hidden=hidden,
    ).to(env.device)

    policy_module = TensorDictModule(
        backbone,
        in_keys=[env._obs_key],
        out_keys=[(env.group, "logits")],
    ).to(env.device)

    actor = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec[env.group, "action"],
        in_keys=[(env.group, "logits")],
        out_keys=[env._action_key],
        distribution_class=MultiCategorical,
        distribution_kwargs={},
        return_log_prob=True,
        log_prob_key=(env.group, "sample_log_prob"),
    )
    return actor.to(env.device)
