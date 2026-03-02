"""Actor (policy) network for Strike–EA MARL."""

import torch.nn as nn
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

from strike_ea.env import StrikeEA2DEnv


def make_actor(env: StrikeEA2DEnv, hidden: int = 256) -> ProbabilisticActor:
    """Build a shared-parameter decentralised actor for all agents."""
    backbone = MultiAgentMLP(
        n_agent_inputs=env.obs_dim,
        n_agent_outputs=2 * env.act_dim,
        n_agents=env.n_agents,
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

    actor = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec[env.group, "action"],
        in_keys=[(env.group, "loc"), (env.group, "scale")],
        out_keys=[env._action_key],
        distribution_class=TanhNormal,
        return_log_prob=True,
        log_prob_key=(env.group, "sample_log_prob"),
    )
    return actor.to(env.device)
