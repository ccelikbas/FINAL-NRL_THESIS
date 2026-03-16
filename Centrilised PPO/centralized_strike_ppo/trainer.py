from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import RewardSum
from torchrl.envs.utils import check_env_specs
from torchrl.objectives import ValueEstimators

from .config import EnvConfig, NetworkConfig, PPOConfig
from .environment import StrikeEA2DEnv
from .models import make_actor, make_critic
from .utils import call_gae, get_loss_component, make_collector, make_ppo_loss, prepare_done_keys


def build_env(env_cfg: EnvConfig, ppo_cfg: PPOConfig) -> StrikeEA2DEnv:
    return StrikeEA2DEnv(
        num_envs=ppo_cfg.num_envs,
        max_steps=env_cfg.max_steps,
        device=ppo_cfg.device,
        seed=ppo_cfg.seed,
        n_strikers=env_cfg.n_strikers,
        n_jammers=env_cfg.n_jammers,
        n_targets=env_cfg.n_targets,
        n_radars=env_cfg.n_radars,
        dt=env_cfg.dt,
        world_bounds=env_cfg.world_bounds,
        v_max=env_cfg.v_max,
        accel_magnitude=env_cfg.accel_magnitude,
        dpsi_max=env_cfg.dpsi_max,
        h_accel_magnitude_fraction=env_cfg.h_accel_magnitude_fraction,
        min_turn_radius=env_cfg.min_turn_radius,
        R_obs=env_cfg.R_obs,
        striker_engage_range=env_cfg.striker_engage_range,
        striker_engage_fov=env_cfg.striker_engage_fov,
        striker_v_min=env_cfg.striker_v_min,
        jammer_jam_radius=env_cfg.jammer_jam_radius,
        jammer_jam_effect=env_cfg.jammer_jam_effect,
        jammer_v_min=env_cfg.jammer_v_min,
        radar_range=env_cfg.radar_range,
        radar_kill_probability=env_cfg.radar_kill_probability,
        border_thresh=env_cfg.border_thresh,
        reward_config=env_cfg.reward_config,
        target_spawn_angle_range=env_cfg.target_spawn_angle_range,
        n_env_layouts=env_cfg.n_env_layouts,
    )


def _safe_check(env) -> None:
    try:
        check_env_specs(env)
        print("check_env_specs: OK")
    except Exception as exc:
        print(f"check_env_specs warning (continuing): {type(exc).__name__}: {exc}")


def train_centralized_ppo(
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    net_cfg: NetworkConfig,
) -> Tuple[StrikeEA2DEnv, object, object, Dict[str, List[float]]]:
    device = ppo_cfg.device

    base_env = build_env(env_cfg, ppo_cfg)
    env = TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env._reward_key], out_keys=[(base_env.group, "episode_reward")]),
    )
    _safe_check(env)

    actor = make_actor(base_env, hidden=net_cfg.actor_hidden, depth=net_cfg.depth)
    critic = make_critic(base_env, hidden=net_cfg.critic_hidden, depth=net_cfg.depth)

    collector = make_collector(env, actor, ppo_cfg.frames_per_batch, ppo_cfg.n_iters, device)
    loss_module = make_ppo_loss(actor, critic, ppo_cfg.clip_eps, ppo_cfg.entropy_coef).to(device)

    try:
        loss_module.set_keys(
            reward=base_env._reward_key,
            action=base_env._action_key,
            sample_log_prob=(base_env.group, "sample_log_prob"),
            value=(base_env.group, "state_value"),
            done=(base_env.group, "done"),
            terminated=(base_env.group, "terminated"),
        )
    except Exception as exc:
        print(f"loss_module.set_keys warning (continuing): {exc}")

    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=ppo_cfg.gamma, lmbda=ppo_cfg.lmbda)
    gae = loss_module.value_estimator

    actor_optim = optim.Adam(actor.parameters(), lr=ppo_cfg.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=ppo_cfg.critic_lr)

    logs: Dict[str, List[float]] = {
        "mean_episode_total_reward": [],
        "loss_policy": [],
        "loss_value": [],
        "entropy": [],
        "approx_kl": [],
        "clip_ratio": [],
        "survival_rate": [],
        "mean_duration": [],
    }

    n_agents = base_env.n_agents

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

        data = td.reshape(-1).to(device)
        n_samples = data.batch_size[0] if len(data.batch_size) else data.numel()

        pol_acc = val_acc = ent_acc = kl_acc = clip_acc = 0.0
        n_updates = 0

        for _ in range(ppo_cfg.num_epochs):
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, ppo_cfg.minibatch_size):
                idx = perm[start:start + ppo_cfg.minibatch_size]
                if idx.numel() == 0:
                    continue
                sub = data[idx].to(device)
                loss_vals = loss_module(sub)

                loss_policy = get_loss_component(loss_vals, ["loss_objective", "loss_actor"])
                loss_value = get_loss_component(loss_vals, ["loss_critic", "loss_value"])

                actor_optim.zero_grad(set_to_none=True)
                loss_policy.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(actor.parameters(), ppo_cfg.max_grad_norm)
                actor_optim.step()

                critic_optim.zero_grad(set_to_none=True)
                loss_value.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), ppo_cfg.max_grad_norm)
                critic_optim.step()

                pol_acc += float(loss_policy.item())
                val_acc += float(loss_value.item())
                try:
                    ent_acc += float(loss_vals.get("entropy").mean().item())
                except Exception:
                    pass
                try:
                    kl_acc += float(loss_vals.get("kl_approx").mean().item())
                except Exception:
                    pass
                try:
                    clip_acc += float(loss_vals.get("clip_fraction").mean().item())
                except Exception:
                    pass
                n_updates += 1

        try:
            collector.update_policy_weights_()
        except Exception:
            pass

        ep_stats = base_env.pop_episode_stats()
        if ep_stats:
            ep_total_mean = sum(s["episode_total_reward"] for s in ep_stats) / len(ep_stats)
            survival_rate = sum(s["survival_frac"] for s in ep_stats) / len(ep_stats)
            mean_duration = sum(s["duration"] for s in ep_stats) / len(ep_stats)
        else:
            ep_total_mean = float("nan")
            survival_rate = float("nan")
            mean_duration = float("nan")

        div = max(1, n_updates)
        logs["mean_episode_total_reward"].append(ep_total_mean)
        logs["loss_policy"].append(pol_acc / div)
        logs["loss_value"].append(val_acc / div)
        logs["entropy"].append(ent_acc / div)
        logs["approx_kl"].append(kl_acc / div)
        logs["clip_ratio"].append(clip_acc / div)
        logs["survival_rate"].append(survival_rate)
        logs["mean_duration"].append(mean_duration)

        if ppo_cfg.log_every and (it + 1) % ppo_cfg.log_every == 0:
            print(
                f"Iter {it + 1:4d}/{ppo_cfg.n_iters} | "
                f"ep_return_total {ep_total_mean: .3f} | "
                f"survival {survival_rate:.2f} | "
                f"clip_ratio {logs['clip_ratio'][-1]:.4f} | "
                f"policy {logs['loss_policy'][-1]:.4f} | "
                f"value {logs['loss_value'][-1]:.4f} | "
                f"entropy {logs['entropy'][-1]:.4f}"
            )

        if it + 1 >= ppo_cfg.n_iters:
            break

    try:
        collector.shutdown()
    except Exception:
        pass

    return base_env, actor, critic, logs
