from __future__ import annotations

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor

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


class CentralizedActorNet(nn.Module):
    """One policy for all controlled agents.

    Input:  all agent observations flattened -> [B, A * obs_dim]
    Output: all agent action logits          -> [B, A, act_dim, n_choices]
    """

    def __init__(self, n_agents: int, obs_dim: int, act_dim: int, n_choices: int, hidden: int, depth: int):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_choices = n_choices
        self.backbone = MLP(n_agents * obs_dim, n_agents * act_dim * n_choices, hidden, depth)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        flat = obs.reshape(batch, self.n_agents * self.obs_dim)
        logits = self.backbone(flat)
        return logits.view(batch, self.n_agents, self.act_dim, self.n_choices)


class CentralizedCriticNet(nn.Module):
    """One centralized value network producing one value per controlled agent."""

    def __init__(self, state_dim: int, n_agents: int, hidden: int, depth: int):
        super().__init__()
        self.n_agents = n_agents
        self.backbone = MLP(state_dim, n_agents, hidden, depth)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.backbone(state).unsqueeze(-1)


def make_actor(env: StrikeEA2DEnv, hidden: int = 256, depth: int = 3) -> ProbabilisticActor:
    actor_net = CentralizedActorNet(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        n_choices=env.n_choices,
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
    critic_net = CentralizedCriticNet(
        state_dim=env.state_dim,
        n_agents=env.n_agents,
        hidden=hidden,
        depth=depth,
    ).to(env.device)

    return TensorDictModule(
        critic_net,
        in_keys=["state"],
        out_keys=[(env.group, "state_value")],
    ).to(env.device)
