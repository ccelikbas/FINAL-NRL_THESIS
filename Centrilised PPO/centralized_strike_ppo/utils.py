"""Low-level training helpers (version-compatibility shims for TorchRL)."""

from __future__ import annotations

import inspect
from typing import List

import torch
from tensordict import TensorDict
from torchrl.collectors import Collector
from torchrl.objectives import ClipPPOLoss

from .environment import StrikeEA2DEnv


# ------------------------------------------------------------------
# Loss / GAE helpers
# ------------------------------------------------------------------

def make_ppo_loss(actor, critic, clip_eps: float, entropy_coef: float) -> ClipPPOLoss:
    """Instantiate ClipPPOLoss, handling API differences across TorchRL versions."""
    sig    = inspect.signature(ClipPPOLoss.__init__)
    params = sig.parameters
    kwargs: dict = dict(clip_epsilon=clip_eps, normalize_advantage=True)

    if "entropy_coeff" in params:
        kwargs["entropy_coeff"] = entropy_coef
    elif "entropy_coef" in params:
        kwargs["entropy_coef"] = entropy_coef

    # Multi-agent: normalise over batch but keep agent dim independent
    if "normalize_advantage_exclude_dims" in params:
        kwargs["normalize_advantage_exclude_dims"] = [-1]

    if "actor_network" in params and "critic_network" in params:
        return ClipPPOLoss(actor_network=actor, critic_network=critic, **kwargs)
    if "actor" in params and "critic" in params:
        return ClipPPOLoss(actor=actor, critic=critic, **kwargs)
    return ClipPPOLoss(actor_network=actor, critic_network=critic, **kwargs)


def make_collector(env, actor, frames_per_batch: int, n_iters: int, device: torch.device) -> Collector:
    """Instantiate Collector (newer API, replaces deprecated SyncDataCollector)."""
    sig    = inspect.signature(Collector.__init__)
    params = sig.parameters

    # Total frames to collect across all iterations
    total_frames = frames_per_batch * n_iters
    
    kwargs: dict = dict(
        policy=actor, 
        frames_per_batch=frames_per_batch, 
        total_frames=total_frames
    )
    
    # Handle different parameter names across TorchRL versions
    kwargs["env" if "env" in params else "create_env_fn"] = env
    if "device"         in params: kwargs["device"]         = device
    if "storing_device" in params: kwargs["storing_device"] = device

    return Collector(**kwargs)


def get_loss_component(loss_td: TensorDict, candidates: List[str]) -> torch.Tensor:
    """Return whichever loss component key exists (handles TorchRL API evolution)."""
    for k in candidates:
        if k in loss_td.keys():
            return loss_td.get(k)
    raise KeyError(f"None of {candidates} found. Available: {list(loss_td.keys())}")


def call_gae(gae, td: TensorDict, loss_module) -> TensorDict:
    """Call GAE with or without explicit critic params depending on TorchRL version."""
    try:
        return gae(
            td,
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        )
    except Exception:
        return gae(td)


# ------------------------------------------------------------------
# Done-key propagation
# ------------------------------------------------------------------

def _flat(*parts):
    out = []
    for p in parts:
        out.extend(p if isinstance(p, tuple) else [p])
    return tuple(out)


def prepare_done_keys(td: TensorDict, env: StrikeEA2DEnv):
    """
    Propagate scalar done/terminated flags into the agent-group nested TD
    so that ClipPPOLoss can find them at the expected keys.
    """
    reward_shape = td.get(_flat("next", env._reward_key)).shape

    for prefix, src_key in [("next", "done"), ("next", "terminated")]:
        src = td.get(_flat(prefix, src_key))
        exp = src.unsqueeze(-1).expand(reward_shape)
        td.set(_flat(prefix, env.group, src_key), exp)

    if "done" in td.keys() and "terminated" in td.keys():
        for src_key in ("done", "terminated"):
            src = td.get(src_key)
            exp = src.unsqueeze(-1).expand(reward_shape)
            td.set(_flat(env.group, src_key), exp)
