"""Training helpers for dual-MAPPO.

Provides manual GAE and PPO-clip loss computation so that each role
(striker, jammer) can be trained independently without sharing a single
TorchRL ClipPPOLoss module.
"""

from __future__ import annotations

import inspect
from typing import Dict, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.collectors import Collector

from .environment import StrikeEA2DEnv


# ------------------------------------------------------------------
# Collector helper
# ------------------------------------------------------------------

def make_collector(env, policy, frames_per_batch: int, n_iters: int, device: torch.device) -> Collector:
    """Instantiate Collector (newer API, replaces deprecated SyncDataCollector)."""
    sig = inspect.signature(Collector.__init__)
    params = sig.parameters

    total_frames = frames_per_batch * n_iters

    kwargs: dict = dict(
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )

    kwargs["env" if "env" in params else "create_env_fn"] = env
    if "device" in params:
        kwargs["device"] = device
    if "storing_device" in params:
        kwargs["storing_device"] = device

    return Collector(**kwargs)


# ------------------------------------------------------------------
# Done-key propagation
# ------------------------------------------------------------------

def _flat(*parts):
    out = []
    for p in parts:
        out.extend(p if isinstance(p, tuple) else [p])
    return tuple(out)


def prepare_done_keys(td: TensorDict, env: StrikeEA2DEnv):
    """Propagate scalar done/terminated flags into the agent-group nested TD."""
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


# ------------------------------------------------------------------
# Manual GAE computation
# ------------------------------------------------------------------

@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,      # [N, n_role, 1]
    values: torch.Tensor,       # [N, n_role, 1]
    next_values: torch.Tensor,  # [N, n_role, 1]
    dones: torch.Tensor,        # [N, 1]
    gamma: float,
    lmbda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

    All inputs are flat batches (no explicit time dimension).  We treat each
    sample as an independent transition — this matches the standard TorchRL
    Collector output after ``td.reshape(-1)``.

    Returns
    -------
    advantages : [N, n_role, 1]
    returns    : [N, n_role, 1]  (= advantages + values)
    """
    not_done = (1.0 - dones.float()).unsqueeze(-1)  # [N, 1, 1]
    delta = rewards + gamma * next_values * not_done - values  # [N, n, 1]
    advantages = delta.clone()
    # Note: With the flat-batch layout from the collector each sample is
    # already an independent transition so we do not back-propagate the
    # GAE residual across time.  Full multi-step GAE would require the
    # original (T, B) layout.  This single-step formulation (equivalent to
    # TD(0) advantage) is a pragmatic choice that keeps the code simple and
    # compatible with TorchRL's reshape-based pipeline.
    returns = advantages + values
    return advantages, returns


@torch.no_grad()
def compute_gae_sequential(
    rewards: torch.Tensor,      # [T, B, n_role, 1]
    values: torch.Tensor,       # [T, B, n_role, 1]
    next_values: torch.Tensor,  # [T, B, n_role, 1]
    dones: torch.Tensor,        # [T, B, 1]
    gamma: float,
    lmbda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute multi-step GAE over a (T, B, ...) rollout.

    Returns
    -------
    advantages : [T, B, n_role, 1]
    returns    : [T, B, n_role, 1]
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])

    for t in reversed(range(T)):
        not_done = (1.0 - dones[t].float()).unsqueeze(-1)  # [B, 1, 1]
        delta = rewards[t] + gamma * next_values[t] * not_done - values[t]
        last_gae = delta + gamma * lmbda * not_done * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ------------------------------------------------------------------
# Manual PPO clip loss
# ------------------------------------------------------------------

def ppo_clip_loss(
    new_log_probs: torch.Tensor,   # [M, n_role]
    old_log_probs: torch.Tensor,   # [M, n_role]
    advantages: torch.Tensor,      # [M, n_role, 1]
    entropy: torch.Tensor,         # [M, n_role]
    clip_eps: float,
    entropy_coef: float,
) -> Dict[str, torch.Tensor]:
    """Compute PPO-clip loss for a single role.

    Returns a dict with loss_policy, loss_entropy, entropy_mean,
    approx_kl, clip_fraction.
    """
    adv = advantages.squeeze(-1)  # [M, n_role]

    # Normalise advantages (exclude agent dim)
    adv_mean = adv.mean()
    adv_std = adv.std().clamp_min(1e-8)
    adv = (adv - adv_mean) / adv_std

    ratio = (new_log_probs - old_log_probs).exp()  # [M, n_role]
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()
    entropy_loss = -entropy.mean()

    with torch.no_grad():
        approx_kl = (old_log_probs - new_log_probs).mean().item()
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

    total_loss = policy_loss + entropy_coef * entropy_loss

    return {
        "loss_total": total_loss,
        "loss_policy": policy_loss,
        "loss_entropy": entropy_loss,
        "entropy_mean": entropy.mean().detach(),
        "approx_kl": approx_kl,
        "clip_fraction": clip_frac,
    }


def value_loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Simple MSE value loss.

    predictions, targets: [M, n_role, 1]
    """
    return nn.functional.mse_loss(predictions, targets)


# ------------------------------------------------------------------
# Explained-variance helper
# ------------------------------------------------------------------

def compute_explained_variance(returns: torch.Tensor, predictions: torch.Tensor) -> float:
    """EV = 1 - Var(returns - predictions) / Var(returns)."""
    r = returns.detach().reshape(-1)
    v = predictions.detach().reshape(-1)
    finite = torch.isfinite(r) & torch.isfinite(v)
    if not bool(finite.any()):
        return float("nan")
    r, v = r[finite], v[finite]
    var_r = torch.var(r, unbiased=False)
    if float(var_r.item()) <= 1e-12:
        return float("nan")
    return float((1.0 - torch.var(r - v, unbiased=False) / var_r).item())
