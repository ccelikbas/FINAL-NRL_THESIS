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
        target_spawn_angle_range = env_cfg.target_spawn_angle_range,
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
        "mean_episode_total_reward": [],
        "loss_policy": [], "loss_value": [],
        "entropy": [],
        # Per-role diagnostics
        "entropy_striker": [], "entropy_jammer": [],
        "adv_std_striker": [], "adv_std_jammer": [],
        "approx_kl_striker": [], "approx_kl_jammer": [],
        "clip_frac_striker": [], "clip_frac_jammer": [],
        # Mission metrics
        "completion_rate": [], "survival_rate": [], "mean_duration": [],
        "mean_targets_frac": [],
    }
    n_agents = base_env.n_agents
    ns = base_env.n_strikers

    # --- Verify env identity: ensure Collector uses the same base_env ---
    _collector_env = getattr(collector, 'env', None)
    if _collector_env is not None:
        _inner = getattr(_collector_env, 'base_env', _collector_env)
        if _inner is base_env:
            print("[diag] Collector env identity: OK (same base_env object)")
        else:
            print(f"[diag] WARNING: Collector env is a DIFFERENT object! "
                  f"id(base_env)={id(base_env)}, id(collector_inner)={id(_inner)}")
            print("         pop_episode_stats() will NOT receive data from the Collector.")
            print("         Falling back to alternative metrics from tensordict.")
    else:
        print("[diag] WARNING: Cannot access collector.env — unknown Collector API")

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

        pol_acc = val_acc = 0.0
        entropy_acc = 0.0
        ent_s_acc = ent_j_acc = 0.0
        kl_s_acc = kl_j_acc = 0.0
        cf_s_acc = cf_j_acc = 0.0
        n_updates = 0

        # Advantage std per role (from full batch, before per-minibatch normalisation)
        try:
            adv_key = (base_env.group, "advantage")
            adv_vals = data.get(adv_key) if adv_key in data.keys(True) else data.get("advantage")
            if adv_vals is not None and adv_vals.dim() >= 2:
                adv_sq = adv_vals.squeeze(-1) if adv_vals.dim() == 3 else adv_vals
                adv_std_s = float(adv_sq[:, :ns].std().item())
                adv_std_j = float(adv_sq[:, ns:].std().item())
            elif adv_vals is not None:
                adv_std_s = adv_std_j = float(adv_vals.std().item())
            else:
                adv_std_s = adv_std_j = float("nan")
        except Exception:
            adv_std_s = adv_std_j = float("nan")

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

                # --- Actor update (policy loss only, no entropy) ---
                actor_optimizer.zero_grad(set_to_none=True)
                loss_policy.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(actor_params, train_cfg.max_grad_norm)
                actor_optimizer.step()

                # --- Critic update (value loss only) ---
                critic_optimizer.zero_grad(set_to_none=True)
                loss_value.backward()
                nn.utils.clip_grad_norm_(critic_params, train_cfg.max_grad_norm)
                critic_optimizer.step()

                pol_acc   += float(loss_policy.item())
                val_acc   += float(loss_value.item())

                # --- Joint entropy from loss module ---
                try:
                    entropy_acc += float(loss_vals.get("entropy").mean().item())
                except Exception:
                    pass

                # --- Per-role entropy (from sample log_prob in sub-batch) ---
                try:
                    lp = sub.get((base_env.group, "sample_log_prob"))
                    if lp is not None:
                        if lp.dim() > 2:
                            lp = lp.sum(dim=-1)  # [mb, A]
                        ent_s_acc += float(-lp[:, :ns].mean().item())
                        ent_j_acc += float(-lp[:, ns:].mean().item())
                except Exception:
                    pass

                # --- Per-role KL and clip fraction ---
                try:
                    kl_val = loss_vals.get("kl_approx")
                    if kl_val is not None:
                        if kl_val.dim() >= 2:
                            kl_sq = kl_val.squeeze(-1) if kl_val.dim() == 3 else kl_val
                            kl_s_acc += float(kl_sq[:, :ns].mean().item())
                            kl_j_acc += float(kl_sq[:, ns:].mean().item())
                        else:
                            v = float(kl_val.mean().item())
                            kl_s_acc += v; kl_j_acc += v
                except Exception:
                    pass

                try:
                    cf_val = loss_vals.get("clip_fraction")
                    if cf_val is not None:
                        if cf_val.dim() >= 2:
                            cf_sq = cf_val.squeeze(-1) if cf_val.dim() == 3 else cf_val
                            cf_s_acc += float(cf_sq[:, :ns].mean().item())
                            cf_j_acc += float(cf_sq[:, ns:].mean().item())
                        else:
                            v = float(cf_val.mean().item())
                            cf_s_acc += v; cf_j_acc += v
                except Exception:
                    pass

                n_updates += 1

        try:
            collector.update_policy_weights_()
        except Exception:
            pass

        # Team episode total reward: sum per-agent rewards, mean over episodes
        try:
            done_mask = td.get(("next", base_env.group, "done"))
            ep_rew    = td.get(("next", base_env.group, "episode_reward"))[done_mask]
        except Exception:
            ep_rew = torch.tensor([], device=device)

        if ep_rew.numel() > 0 and ep_rew.numel() % n_agents == 0:
            ep_rew_by_ep = ep_rew.view(-1, n_agents)     # [N_ep, A]
            team_totals  = ep_rew_by_ep.sum(dim=-1)       # [N_ep]
            ep_total_mean = float(torch.nanmean(team_totals).item())
        elif ep_rew.numel() > 0:
            ep_total_mean = float(torch.nanmean(ep_rew).item()) * n_agents
        else:
            ep_total_mean = float("nan")
        div = max(1, n_updates)

        logs["mean_episode_total_reward"].append(ep_total_mean)
        logs["loss_policy"].append(pol_acc / div)
        logs["loss_value"].append(val_acc / div)
        logs["entropy"].append(entropy_acc / div)
        logs["entropy_striker"].append(ent_s_acc / div)
        logs["entropy_jammer"].append(ent_j_acc / div)
        logs["adv_std_striker"].append(adv_std_s)
        logs["adv_std_jammer"].append(adv_std_j)
        logs["approx_kl_striker"].append(kl_s_acc / div)
        logs["approx_kl_jammer"].append(kl_j_acc / div)
        logs["clip_frac_striker"].append(cf_s_acc / div)
        logs["clip_frac_jammer"].append(cf_j_acc / div)

        # --- Mission outcome metrics (from completed episodes stored on env) ---
        ep_stats = base_env.pop_episode_stats()
        if ep_stats:
            completion_rate   = sum(s["mission_complete"] for s in ep_stats) / len(ep_stats)
            survival_rate     = sum(s["survival_frac"]    for s in ep_stats) / len(ep_stats)
            mean_duration     = sum(s["duration"]         for s in ep_stats) / len(ep_stats)
            mean_targets_frac = sum(s["targets_frac"]     for s in ep_stats) / len(ep_stats)
        else:
            completion_rate   = float("nan")
            survival_rate     = float("nan")
            mean_duration     = float("nan")
            mean_targets_frac = float("nan")

        logs["completion_rate"].append(completion_rate)
        logs["survival_rate"].append(survival_rate)
        logs["mean_duration"].append(mean_duration)
        logs["mean_targets_frac"].append(mean_targets_frac)

        if train_cfg.log_every and (it + 1) % train_cfg.log_every == 0:
            print(
                f"Iter {it+1:4d}/{train_cfg.n_iters} | "
                f"ep_return_total {ep_total_mean: .3f} | "
                f"tgt_frac {mean_targets_frac:.2f} | "
                f"compl {completion_rate:.2f} | "
                f"pol_loss {logs['loss_policy'][-1]:.4f} | "
                f"val_loss {logs['loss_value'][-1]:.4f} | "
                f"ent_s {logs['entropy_striker'][-1]:.4f} | "
                f"ent_j {logs['entropy_jammer'][-1]:.4f} | "
                f"kl_s {logs['approx_kl_striker'][-1]:.4f} | "
                f"kl_j {logs['approx_kl_jammer'][-1]:.4f} | "
                f"clip_s {logs['clip_frac_striker'][-1]:.4f} | "
                f"clip_j {logs['clip_frac_jammer'][-1]:.4f} | "
                f"adv_s {logs['adv_std_striker'][-1]:.4f} | "
                f"adv_j {logs['adv_std_jammer'][-1]:.4f}"
            )

        if it + 1 >= train_cfg.n_iters:
            break

    try:
        collector.shutdown()
    except Exception:
        pass

    return base_env, actor, critic, logs
