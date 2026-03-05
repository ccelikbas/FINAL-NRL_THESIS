"""MAPPO training loop (dual-policy discrete actors, dual centralised critic)."""

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
    """Run MAPPO training and return (base_env, actor, critic, logs)."""
    if env_cfg   is None: env_cfg   = EnvConfig()
    if net_cfg   is None: net_cfg   = NetworkConfig()

    device = train_cfg.device

    base_env = StrikeEA2DEnv(
        num_envs      = train_cfg.num_envs,
        max_steps     = train_cfg.max_steps,
        device        = device,
        seed          = train_cfg.seed,
        n_strikers    = env_cfg.n_strikers,
        n_jammers     = env_cfg.n_jammers,
        n_targets     = env_cfg.n_targets,
        n_radars      = env_cfg.n_radars,
        dt            = env_cfg.dt,
        world_bounds  = env_cfg.world_bounds,
        v_max         = env_cfg.v_max,
        accel_magnitude = env_cfg.accel_magnitude,
        dpsi_max      = env_cfg.dpsi_max,
        h_accel_magnitude_fraction = env_cfg.h_accel_magnitude_fraction,
        R_obs         = env_cfg.R_obs,
        striker_engage_range = env_cfg.striker_engage_range,
        striker_engage_fov = env_cfg.striker_engage_fov,
        striker_v_min = env_cfg.striker_v_min,
        jammer_jam_radius = env_cfg.jammer_jam_radius,
        jammer_jam_effect = env_cfg.jammer_jam_effect,
        jammer_v_min = env_cfg.jammer_v_min,
        radar_range   = env_cfg.radar_range,
        radar_kill_probability = env_cfg.radar_kill_probability,
        border_thresh = env_cfg.border_thresh,
        reward_config = env_cfg.reward_config,
        min_turn_radius = env_cfg.min_turn_radius,
        n_env_layouts = env_cfg.n_env_layouts,
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
    gae = loss_module.value_estimator

    # Separate optimizers for actor and critic (different learning rates)
    actor_params  = list(actor.parameters())
    critic_params = list(critic.parameters())
    actor_lr  = train_cfg.actor_lr if train_cfg.actor_lr is not None else train_cfg.lr
    critic_lr = train_cfg.critic_lr if train_cfg.critic_lr is not None else train_cfg.lr
    actor_optimizer  = optim.Adam(actor_params,  lr=actor_lr)
    critic_optimizer = optim.Adam(critic_params, lr=critic_lr)

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

                # --- Actor update (policy loss + entropy bonus) ---
                actor_loss = loss_policy + loss_entropy
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(actor_params, train_cfg.max_grad_norm)
                actor_optimizer.step()

                # --- Critic update (value loss only) ---
                critic_optimizer.zero_grad(set_to_none=True)
                loss_value.backward()
                nn.utils.clip_grad_norm_(critic_params, train_cfg.max_grad_norm)
                critic_optimizer.step()

                total_loss = loss_policy + loss_value + loss_entropy

                total_acc += float(total_loss.item())
                pol_acc   += float(loss_policy.item())
                val_acc   += float(loss_value.item())
                ent_acc   += float(loss_entropy.item())
                n_updates += 1

        try:
            collector.update_policy_weights_()
        except Exception:
            pass

        # episode reward logging (filter NaN values for accurate averaging)
        try:
            done_mask = td.get(("next", base_env.group, "done"))
            ep_rew    = td.get(("next", base_env.group, "episode_reward"))[done_mask]
        except Exception:
            ep_rew = torch.tensor([], device=device)

        # Use nanmean to skip NaN values; if no completed episodes, will be NaN
        if ep_rew.numel() > 0:
            ep_rew_mean = float(torch.nanmean(ep_rew).item())
        else:
            ep_rew_mean = float("nan")
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
