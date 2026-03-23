from __future__ import annotations

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import MultiAgentMLP, ProbabilisticActor

from .environment import StrikeEA2DEnv


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
# MAPPO Actor: Dual Policy Network
# ======================================================================

class DualPolicyNet(nn.Module):
    """Two MultiAgentMLPs (striker & jammer) merged into a single forward pass.

    Each role has its own decentralised policy with parameter sharing among
    agents of the same role (centralized=False, share_params=True).

    Input  : (B, n_agents, obs_dim)
    Output : (B, n_agents, act_dim, n_choices)  — logits for MultiCategorical
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
        return all_logits.view(B, A, self.act_dim, self.n_choices)   # (B, A, 2, 7)


# ======================================================================
# MAPPO Critic: Dual Value Network
# ======================================================================

class DualValueNet(nn.Module):
    """Centralised critic with two role-specific value heads.

    Both heads receive the same global state vector.  Each produces
    per-agent values for its role, then the outputs are concatenated
    to match the expected shape [B, n_agents, 1].

    Input  : (B, state_dim)
    Output : (B, n_agents, 1)
    """

    def __init__(
        self,
        state_dim:  int,
        n_strikers: int,
        n_jammers:  int,
        hidden:     int = 256,
        depth:      int = 3,
    ):
        super().__init__()
        self.n_strikers = n_strikers
        self.n_jammers  = n_jammers

        # Separate value heads — one per role
        self.striker_value = MLP(state_dim, n_strikers, hidden, depth)
        self.jammer_value  = MLP(state_dim, n_jammers,  hidden, depth)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: (B, state_dim)
        sv = self.striker_value(state)   # (B, n_strikers)
        jv = self.jammer_value(state)    # (B, n_jammers)
        # Concatenate along agent dimension and add trailing 1
        return torch.cat([sv, jv], dim=-1).unsqueeze(-1)  # (B, n_agents, 1)


# ======================================================================
# Factory helpers
# ======================================================================

def make_actor(env: StrikeEA2DEnv, hidden: int = 256, depth: int = 3) -> ProbabilisticActor:
    actor_net = DualPolicyNet(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        n_choices=env.n_choices,
        n_strikers=env.n_strikers,
        n_jammers=env.n_jammers,
        hidden=hidden,
        depth=depth,
    ).to(env.device)

    module = TensorDictModule(
        actor_net,
        in_keys=[env._obs_key],
        out_keys=[(env.group, "logits")],
    ).to(env.device)

    actor = ProbabilisticActor(
        module=module,
        spec=env.action_spec[env.group, "action"],
        in_keys=[(env.group, "logits")],
        out_keys=[env._action_key],
        distribution_class=MultiCategorical,
        return_log_prob=True,
        log_prob_key=(env.group, "sample_log_prob"),
    )
    return actor.to(env.device)


def make_critic(env: StrikeEA2DEnv, hidden: int = 256, depth: int = 3) -> TensorDictModule:
    critic_net = DualValueNet(
        state_dim=env.state_dim,
        n_strikers=env.n_strikers,
        n_jammers=env.n_jammers,
        hidden=hidden,
        depth=depth,
    ).to(env.device)

    return TensorDictModule(
        critic_net,
        in_keys=["state"],
        out_keys=[(env.group, "state_value")],
    ).to(env.device)
