"""
Dual-MAPPO model architecture.

Each role (striker, jammer) has its own independent MAPPO instance:
  - RolePolicyNet:  decentralized actor with parameter sharing within the role
  - RoleValueNet:   centralized critic receiving the full global state

CombinedPolicy and CombinedCritic wrap both roles into single callables
that operate on the standard ("agents", ...) TensorDict layout expected by
the environment and TorchRL Collector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchrl.modules import MultiAgentMLP

from .environment import StrikeEA2DEnv


# ======================================================================
# Shared distribution
# ======================================================================

class MultiCategorical(torch.distributions.Distribution):
    """Independent categorical distributions over the two discrete action dimensions.

    Expected logits shape: (..., act_dim, n_choices)
    Sample shape:          (..., act_dim)
    Log-prob shape:        (...)  (sum over action dimensions)
    """

    arg_constraints = {}
    has_rsample = False

    def __init__(self, logits: torch.Tensor):
        self._cats = torch.distributions.Categorical(logits=logits)
        batch_shape = logits.shape[:-2]
        super().__init__(batch_shape=batch_shape, validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        return self._cats.sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._cats.log_prob(value).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self._cats.entropy().sum(dim=-1)

    @property
    def mode(self) -> torch.Tensor:
        return self._cats.logits.argmax(dim=-1)

    @property
    def deterministic_sample(self) -> torch.Tensor:
        return self.mode


# ======================================================================
# Generic building block
# ======================================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int):
        super().__init__()
        layers = []
        cur = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(cur, hidden), nn.ReLU()])
            cur = hidden
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# Per-role policy network
# ======================================================================

class RolePolicyNet(nn.Module):
    """Decentralized policy for one role (striker or jammer).

    Uses TorchRL MultiAgentMLP with parameter sharing among agents of the
    same role (centralized=False, share_params=True).

    Input  : (B, n_role_agents, obs_dim)
    Output : (B, n_role_agents, act_dim, n_choices) — logits for MultiCategorical
    """

    def __init__(
        self,
        obs_dim:   int,
        act_dim:   int,
        n_choices: int,
        n_agents:  int,
        hidden:    int = 256,
        depth:     int = 3,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.n_choices = n_choices

        self.net = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=act_dim * n_choices,
            n_agents=n_agents,
            centralized=False,
            share_params=True,
            depth=depth,
            num_cells=hidden,
            activation_class=nn.ReLU,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, n_agents, obs_dim) → logits: (B, n_agents, act_dim, n_choices)"""
        raw = self.net(obs)  # (B, n_agents, act_dim * n_choices)
        B, A, _ = raw.shape
        return raw.view(B, A, self.act_dim, self.n_choices)


# ======================================================================
# Per-role centralized critic
# ======================================================================

class RoleValueNet(nn.Module):
    """Centralized value head for one role.

    Receives the full global state and outputs a scalar value per agent
    in the role.

    Input  : (B, state_dim)
    Output : (B, n_role_agents, 1)
    """

    def __init__(
        self,
        state_dim: int,
        n_agents:  int,
        hidden:    int = 256,
        depth:     int = 3,
    ):
        super().__init__()
        self.net = MLP(state_dim, n_agents, hidden, depth)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, state_dim) → values: (B, n_agents, 1)"""
        return self.net(state).unsqueeze(-1)


# ======================================================================
# Combined policy (used by the Collector)
# ======================================================================

class CombinedPolicy(nn.Module):
    """Wraps striker and jammer policies into a single callable for data collection.

    Reads  ("agents", "observation")  [B, A, obs_dim]
    Writes ("agents", "action")       [B, A, 2]
           ("agents", "sample_log_prob") [B, A]
           ("agents", "logits")       [B, A, act_dim, n_choices]

    During training the individual role methods (striker_log_prob_entropy,
    jammer_log_prob_entropy) are used to compute per-role PPO losses.
    """

    def __init__(
        self,
        striker_policy: RolePolicyNet,
        jammer_policy:  RolePolicyNet,
        n_strikers:     int,
        n_jammers:      int,
        obs_key:        tuple = ("agents", "observation"),
        action_key:     tuple = ("agents", "action"),
        deterministic:  bool = False,
    ):
        super().__init__()
        self.striker_policy = striker_policy
        self.jammer_policy = jammer_policy
        self.n_strikers = n_strikers
        self.n_jammers = n_jammers
        self.obs_key = obs_key
        self.action_key = action_key
        self.deterministic = deterministic

    def forward(self, td):
        obs = td.get(self.obs_key)  # [B, A, obs_dim]

        s_logits = self.striker_policy(obs[:, : self.n_strikers])  # [B, ns, ad, nc]
        j_logits = self.jammer_policy(obs[:, self.n_strikers :])   # [B, nj, ad, nc]
        all_logits = torch.cat([s_logits, j_logits], dim=1)        # [B, A, ad, nc]

        dist = MultiCategorical(logits=all_logits)
        if self.deterministic:
            action = dist.mode
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        td.set(self.action_key, action)
        td.set(("agents", "sample_log_prob"), log_prob)
        td.set(("agents", "logits"), all_logits)
        return td

    # ------------------------------------------------------------------
    # Per-role helpers for training
    # ------------------------------------------------------------------

    def striker_log_prob_entropy(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: [B, ns, obs_dim], action: [B, ns, 2] → (log_prob [B,ns], entropy [B,ns])"""
        logits = self.striker_policy(obs)
        dist = MultiCategorical(logits=logits)
        return dist.log_prob(action), dist.entropy()

    def jammer_log_prob_entropy(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: [B, nj, obs_dim], action: [B, nj, 2] → (log_prob [B,nj], entropy [B,nj])"""
        logits = self.jammer_policy(obs)
        dist = MultiCategorical(logits=logits)
        return dist.log_prob(action), dist.entropy()


# ======================================================================
# Combined critic
# ======================================================================

class CombinedCritic(nn.Module):
    """Wraps striker and jammer critics.

    Reads  "state"                     [B, state_dim]
    Writes ("agents", "state_value")   [B, A, 1]
    """

    def __init__(
        self,
        striker_critic: RoleValueNet,
        jammer_critic:  RoleValueNet,
        n_strikers:     int,
        n_jammers:      int,
    ):
        super().__init__()
        self.striker_critic = striker_critic
        self.jammer_critic = jammer_critic
        self.n_strikers = n_strikers
        self.n_jammers = n_jammers

    def forward(self, td):
        state = td.get("state")  # [B, state_dim]
        sv = self.striker_critic(state)  # [B, ns, 1]
        jv = self.jammer_critic(state)   # [B, nj, 1]
        td.set(("agents", "state_value"), torch.cat([sv, jv], dim=1))
        return td


# ======================================================================
# Factory helpers
# ======================================================================

def make_combined_policy(env: StrikeEA2DEnv, hidden: int = 256, depth: int = 3) -> CombinedPolicy:
    """Build the dual-role combined policy."""
    striker_net = RolePolicyNet(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        n_choices=env.n_choices,
        n_agents=env.n_strikers,
        hidden=hidden,
        depth=depth,
    ).to(env.device)

    jammer_net = RolePolicyNet(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        n_choices=env.n_choices,
        n_agents=env.n_jammers,
        hidden=hidden,
        depth=depth,
    ).to(env.device)

    return CombinedPolicy(
        striker_policy=striker_net,
        jammer_policy=jammer_net,
        n_strikers=env.n_strikers,
        n_jammers=env.n_jammers,
        obs_key=env._obs_key,
        action_key=env._action_key,
    ).to(env.device)


def make_combined_critic(env: StrikeEA2DEnv, hidden: int = 256, depth: int = 3) -> CombinedCritic:
    """Build the dual-role combined critic."""
    striker_critic = RoleValueNet(
        state_dim=env.state_dim,
        n_agents=env.n_strikers,
        hidden=hidden,
        depth=depth,
    ).to(env.device)

    jammer_critic = RoleValueNet(
        state_dim=env.state_dim,
        n_agents=env.n_jammers,
        hidden=hidden,
        depth=depth,
    ).to(env.device)

    return CombinedCritic(
        striker_critic=striker_critic,
        jammer_critic=jammer_critic,
        n_strikers=env.n_strikers,
        n_jammers=env.n_jammers,
    ).to(env.device)
