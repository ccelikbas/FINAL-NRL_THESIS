"""MAPPO training loop."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import RewardSum
from torchrl.envs.utils import check_env_specs
from torchrl.objectives import ValueEstimators

from strike_ea.config import EnvConfig, TrainConfig, NetworkConfig
from strike_ea.env import StrikeEA2DEnv
from strike_ea.models import make_actor, make_critic
from .utils import call_gae, get_loss_component, make_collector, make_ppo_loss, prepare_done_keys


def _safe_check(env):
    try:
        check_env_specs(env)
        print("check_env_specs: OK")
    except Exception as e:
        print(f"check_env_specs warning (continuing): {type(e).__name__}: {e}")


def train_mappo(
    train_cfg:   TrainConfig,
    env_cfg:     EnvConfig   = None,
    net_cfg:     NetworkConfig = None,
) -> tuple[StrikeEA2DEnv, object, object, Dict[str, List[float]]]:
    """
    Run MAPPO training and return (base_env, actor, critic, logs).

    Parameters
    ----------
    train_cfg:
        All training hyper-parameters.
    env_cfg:
        Environment construction parameters (uses defaults if None).
    net_cfg:
        Network architecture parameters (uses defaults if None).
    """
    if env_cfg   is None: env_cfg   = EnvConfig()
    if net_cfg   is None: net_cfg   = NetworkConfig()

    device = train_cfg.device

    base_env = StrikeEA2DEnv(
        num_envs      = train_cfg.num_envs,
        max_steps     = train_cfg.max_steps,
        device        = device,
        seed          = train_cfg.seed,
        # pass through every env_cfg field
        n_strikers    = env_cfg.n_strikers,
        n_jammers     = env_cfg.n_jammers,
        n_targets     = env_cfg.n_targets,
        n_radars      = env_cfg.n_radars,
        dt            = env_cfg.dt,
        world_bounds  = env_cfg.world_bounds,
        v_max         = env_cfg.v_max,
        dpsi_max      = env_cfg.dpsi_max,
        R_obs         = env_cfg.R_obs,
        radar_range   = env_cfg.radar_range,
        border_thresh = env_cfg.border_thresh,
        reward_config = env_cfg.reward_config,
    )

    env = TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env._reward_key], out_keys=[(base_env.group, "episode_reward")]),
    )
    _safe_check(env)

    actor  = make_actor(base_env,  hidden=net_cfg.hidden)
    critic = make_critic(base_env, hidden=net_cfg.hidden)

    collector    = make_collector(env, actor, train_cfg.frames_per_batch, train_cfg.n_iters, device)
    loss_module  = make_ppo_loss(actor, critic, train_cfg.clip_eps, train_cfg.entropy_coef).to(device)

    try:
        loss_module.set_keys(
            reward          = base_env._reward_key,
            action          = base_env._action_key,
            sample_log_prob = (base_env.group, "sample_log_prob"),
            value           = (base_env.group, "state_value"),
            done            = (base_env.group, "done"),
            terminated      = (base_env.group, "terminated"),
        )
    except Exception as e:
        print(f"loss_module.set_keys warning (continuing): {e}")

    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=train_cfg.gamma, lmbda=train_cfg.lmbda)
    gae       = loss_module.value_estimator
    optimizer = optim.Adam(loss_module.parameters(), lr=train_cfg.lr)

    logs: Dict[str, List[float]] = {
        "episode_reward_mean": [],
        "loss_total": [], "loss_policy": [], "loss_value": [], "loss_entropy": [],
    }

    for it, td in enumerate(collector):
        td = td.to(device)
        prepare_done_keys(td, base_env)

        with torch.no_grad():
            try:
                critic(td)
            except Exception:
                pass
            try:
                nxt = td.get("next").to(device)
                critic(nxt)
                td.set("next", nxt)
            except Exception:
                pass
            call_gae(gae, td, loss_module)

        data     = td.reshape(-1).to(device)
        n_samples = data.batch_size[0] if len(data.batch_size) else data.numel()

        total_acc = pol_acc = val_acc = ent_acc = 0.0
        n_updates = 0

        for _ in range(train_cfg.num_epochs):
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, train_cfg.minibatch_size):
                idx = perm[start : start + train_cfg.minibatch_size]
                if idx.numel() == 0:
                    continue

                sub       = data[idx].to(device)
                loss_vals = loss_module(sub)

                loss_policy  = get_loss_component(loss_vals, ["loss_objective", "loss_actor"])
                loss_value   = get_loss_component(loss_vals, ["loss_critic",    "loss_value"])
                loss_entropy = get_loss_component(loss_vals, ["loss_entropy"])
                total_loss   = loss_policy + loss_value + loss_entropy

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), train_cfg.max_grad_norm)
                optimizer.step()

                total_acc += float(total_loss.item())
                pol_acc   += float(loss_policy.item())
                val_acc   += float(loss_value.item())
                ent_acc   += float(loss_entropy.item())
                n_updates += 1

        try:
            collector.update_policy_weights_()
        except Exception:
            pass

        # episode reward logging
        try:
            done_mask = td.get(("next", base_env.group, "done"))
            ep_rew    = td.get(("next", base_env.group, "episode_reward"))[done_mask]
        except Exception:
            ep_rew = torch.tensor([], device=device)

        ep_rew_mean = float(ep_rew.mean().item()) if ep_rew.numel() else float("nan")
        div = max(1, n_updates)

        logs["episode_reward_mean"].append(ep_rew_mean)
        logs["loss_total"].append(total_acc / div)
        logs["loss_policy"].append(pol_acc / div)
        logs["loss_value"].append(val_acc / div)
        logs["loss_entropy"].append(ent_acc / div)

        if train_cfg.log_every and (it + 1) % train_cfg.log_every == 0:
            print(
                f"Iter {it+1:4d}/{train_cfg.n_iters} | "
                f"ep_rew {ep_rew_mean: .3f} | "
                f"loss {logs['loss_total'][-1]:.4f}"
            )

        if it + 1 >= train_cfg.n_iters:
            break

    try:
        collector.shutdown()
    except Exception:
        pass

    return base_env, actor, critic, logs
