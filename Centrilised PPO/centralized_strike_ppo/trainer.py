from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, List, Optional, Tuple

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
from .normalization import RewardNormalizer
from .utils import call_gae, get_loss_component, make_collector, make_ppo_loss, prepare_done_keys

from centralized_strike_ppo.visualization import TestRunner, animate_rollout, plot_training


EVAL_REWARD_COMPONENT_KEYS: Tuple[str, ...] = (
    "target_destroyed",
    "border_penalty",
    "timestep_penalty",
    "radar_avoidance",
    "striker_approach",
    "jammer_approach",
    "striker_progress",
    "jammer_progress",
    "jammer_jam_bonus",
    "formation",
    "agent_destroyed",
    "paper_mission",
    "separation_penalty",
    "control_effort",
)


def _finite_mean(values: List[float]) -> float:
    finite_vals = [v for v in values if math.isfinite(v)]
    if not finite_vals:
        return float("nan")
    return float(sum(finite_vals) / len(finite_vals))

try:
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    _EXPLORATION_API = "new"
except Exception:
    try:
        from tensordict.nn import InteractionType, set_interaction_type
        _EXPLORATION_API = "interaction"
    except Exception:
        _EXPLORATION_API = None


def _deterministic_context():
    if _EXPLORATION_API == "new":
        return set_exploration_type(ExplorationType.DETERMINISTIC)
    if _EXPLORATION_API == "interaction":
        return set_interaction_type(InteractionType.DETERMINISTIC)
    return contextlib.nullcontext()


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


def _compute_explained_variance(returns: torch.Tensor, predictions: torch.Tensor) -> float:
    """EV = 1 - Var(returns - predictions) / Var(returns)."""
    r = returns.detach().reshape(-1)
    v = predictions.detach().reshape(-1)
    finite = torch.isfinite(r) & torch.isfinite(v)
    if not bool(finite.any()):
        return float("nan")

    r = r[finite]
    v = v[finite]
    var_r = torch.var(r, unbiased=False)
    if float(var_r.item()) <= 1e-12:
        return float("nan")

    var_err = torch.var(r - v, unbiased=False)
    ev = 1.0 - (var_err / var_r)
    return float(ev.item())


@torch.no_grad()
def evaluate_current_policy(
    actor,
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    n_eval_episodes: int = 10,
) -> Dict[str, float]:
    eval_env = StrikeEA2DEnv(
        num_envs=1,
        max_steps=env_cfg.max_steps,
        device=ppo_cfg.device,
        seed=ppo_cfg.seed + 10_000,
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
    # print("stating evaluation")
    actor.eval()
    ep_total_rewards: List[float] = []
    ep_survival: List[float] = []
    ep_duration: List[float] = []
    ep_completion: List[float] = []
    ep_component_rewards: Dict[str, List[float]] = {
        key: [] for key in EVAL_REWARD_COMPONENT_KEYS
    }

    # with _deterministic_context():
    for _ in range(max(1, int(n_eval_episodes))):
        td = eval_env.reset()
        episode_component_sums = {key: 0.0 for key in EVAL_REWARD_COMPONENT_KEYS}

        for _step in range(env_cfg.max_steps):
            td = actor(td)
            td_next = eval_env.step(td)

            # Per-step team-total component contributions (sum across agents).
            for comp_key in EVAL_REWARD_COMPONENT_KEYS:
                comp_tensor = eval_env.last_reward_components[comp_key]  # [B, A]
                episode_component_sums[comp_key] += float(comp_tensor[0].sum().item())

            done = bool(td_next.get(("next", "done"))[0, 0].item())
            if done:
                break
            td = td_next.get("next")

        stats = eval_env.pop_episode_stats()
        if stats:
            if len(stats) > 1:
                print(f"WARNING evaluate_current_policy: multiple episodes ({len(stats)}) detected in one eval pass; using first entry")
            s = stats[0]
            ep_total_rewards.append(float(s.get("episode_total_reward", float("nan"))))
            ep_survival.append(float(s.get("survival_frac", float("nan"))))
            ep_duration.append(float(s.get("duration", float("nan"))))
            ep_completion.append(1.0 if bool(s.get("mission_complete", False)) else 0.0)
            for comp_key in EVAL_REWARD_COMPONENT_KEYS:
                ep_component_rewards[comp_key].append(episode_component_sums[comp_key])
        else:
            ep_total_rewards.append(float("nan"))
            ep_survival.append(float("nan"))
            ep_duration.append(float("nan"))
            ep_completion.append(float("nan"))
            for comp_key in EVAL_REWARD_COMPONENT_KEYS:
                ep_component_rewards[comp_key].append(float("nan"))
    # print("ending evaluation")

    # tester = TestRunner(actor, env_cfg=cfg.env, device=cfg.ppo.device, seed=999)
    # frames = tester.rollout()
    # animate_rollout(frames, tester.env)

    metrics = {
        "eval_mean_episode_total_reward": float(sum(ep_total_rewards) / len(ep_total_rewards)),
        "eval_survival_rate": float(sum(ep_survival) / len(ep_survival)),
        "eval_mean_duration": float(sum(ep_duration) / len(ep_duration)),
        "eval_task_completion_rate": float(sum(ep_completion) / len(ep_completion)),
    }

    for comp_key in EVAL_REWARD_COMPONENT_KEYS:
        metrics[f"eval_component_{comp_key}"] = _finite_mean(ep_component_rewards[comp_key])

    return metrics


def train_centralized_ppo(
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    net_cfg: NetworkConfig,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> Tuple[StrikeEA2DEnv, object, object, Dict[str, List[float]], Optional[RewardNormalizer]]:
    device = ppo_cfg.device

    base_env = build_env(env_cfg, ppo_cfg)
    env = TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env._reward_key], out_keys=[(base_env.group, "episode_reward")]),
    )
    _safe_check(env)

    # Define the actor and critic networks 
    actor = make_actor(base_env, hidden=net_cfg.actor_hidden, depth=net_cfg.depth)
    critic = make_critic(base_env, hidden=net_cfg.critic_hidden, depth=net_cfg.depth)
    reward_normalizer: Optional[RewardNormalizer] = None
    if bool(ppo_cfg.normalize_rewards):
        reward_normalizer = RewardNormalizer(
            num_envs=ppo_cfg.num_envs,
            gamma=ppo_cfg.gamma,
            device=device,
        )

    if checkpoint is not None:
        try:
            if "actor_state_dict" in checkpoint:
                actor.load_state_dict(checkpoint["actor_state_dict"])
            if "critic_state_dict" in checkpoint:
                critic.load_state_dict(checkpoint["critic_state_dict"])
            if reward_normalizer is not None and checkpoint.get("reward_normalizer_state_dict") is not None:
                reward_normalizer.load_state_dict(checkpoint["reward_normalizer_state_dict"])
        except Exception as exc:
            print(f"checkpoint load warning (continuing): {type(exc).__name__}: {exc}")

    # The collector is responsible for collecting experience from the environment using the current policy (actor) and providing it to the training loop
    collector = make_collector(env, actor, ppo_cfg.frames_per_batch, ppo_cfg.n_iters, device)
    # The loss module is responsible for computing the loss function for the policy and value networks. It takes batches of stored transitions from the collector and computes the PPO loss, which includes the policy loss and value loss
    loss_module = make_ppo_loss(actor, critic, ppo_cfg.clip_eps, ppo_cfg.entropy_coef).to(device)

    try:
        # This is registering a lookup table inside the loss module because of the multi agent setup. The loss module will look for these keys in the data it receives and use them to compute the loss.
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

    # This builds a GAE calculator and attaches it internally to the loss module
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=ppo_cfg.gamma, lmbda=ppo_cfg.lmbda)
    gae = loss_module.value_estimator

    # Build Adam optimizers for the actor and critic networks
    actor_optim = optim.Adam(actor.parameters(), lr=ppo_cfg.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=ppo_cfg.critic_lr)

    logs: Dict[str, List[float]] = {
        "train_mean_episode_total_reward": [],
        "loss_policy": [],
        "loss_value": [],
        "explained_variance": [],
        "entropy": [],
        "approx_kl": [],
        "clip_ratio": [],
        "eval_mean_episode_total_reward": [],
        "eval_survival_rate": [],
        "eval_mean_duration": [],
        "eval_task_completion_rate": [],
        "reward_norm_running_mean": [],
        "reward_norm_running_std": [],
        "raw_reward_mean": [],
        "raw_reward_std": [],
        "normalized_reward_mean": [],
        "normalized_reward_std": [],
    }
    for comp_key in EVAL_REWARD_COMPONENT_KEYS:
        logs[f"train_component_{comp_key}"] = []
    for comp_key in EVAL_REWARD_COMPONENT_KEYS:
        logs[f"eval_component_{comp_key}"] = []

    # MAIN TRAINING LOOP:
    # collect experience (using the collector)
    for it, td in enumerate(collector):
        # Create a TensorDict from the collected experience for observations, actions, rewards, etc. (one itteration of collected experience)
        td = td.to(device)

        # Reward normalization is applied after collection and before GAE.
        if reward_normalizer is not None:
            norm_stats = reward_normalizer.normalize_rollout_td(
                td,
                reward_key=("next",) + base_env._reward_key,
                done_key=("next", "done"),
            )
        else:
            raw_r = td.get(("next",) + base_env._reward_key)
            raw_mean = float(raw_r.mean().item()) if raw_r.numel() > 0 else float("nan")
            raw_std = float(raw_r.std(unbiased=False).item()) if raw_r.numel() > 0 else float("nan")
            norm_stats = {
                "running_mean": float("nan"),
                "running_std": float("nan"),
                "raw_reward_mean": raw_mean,
                "raw_reward_std": raw_std,
                "normalized_reward_mean": raw_mean,
                "normalized_reward_std": raw_std,
            }

        # Helper fucntion: It takes the scalar done flag [B, 1] at the root of the TensorDict and copies it into the agent group as [B, A, 1] so that the PPO loss module can find it at the key it expects
        prepare_done_keys(td, base_env)

        # computes value estimates and advantages for the entire collected rollout before any gradient updates
        with torch.no_grad():
            try:
                # compute value estimates for the current state
                critic(td)
            except Exception as e:
                print(f"WARNING critic(td) failed: {e}")
                pass
            try:
                nxt = td.get("next").to(device)
                # compute value estimates for the next state
                critic(nxt)
                td.set("next", nxt)
            except Exception as e:
                print(f"WARNING critic(nxt) failed: {e}")
                pass
            # compute advantages using td and added value estimates
            call_gae(gae, td, loss_module)

        # reshape data
        data = td.reshape(-1).to(device)
        n_samples = data.batch_size[0] if len(data.batch_size) else data.numel()

        pol_acc = val_acc = ent_acc = kl_acc = clip_acc = 0.0
        n_updates = 0

        # train for the specified number of epochs using the collected data
        for _ in range(ppo_cfg.num_epochs):
            # shuffle the data and split it into minibatches
            perm = torch.randperm(n_samples, device=device)
            # iterate over the minibatches and perform gradient updates
            for start in range(0, n_samples, ppo_cfg.minibatch_size):
                idx = perm[start:start + ppo_cfg.minibatch_size]
                if idx.numel() == 0:
                    continue
                sub = data[idx].to(device)

                # compute the loss for the current minibatch 
                loss_vals = loss_module(sub)
                loss_policy = get_loss_component(loss_vals, ["loss_objective", "loss_actor"])
                loss_value = get_loss_component(loss_vals, ["loss_critic", "loss_value"])

                # update the actor and critic networks using the computed loss
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

        # Update policy weights in the collector so the next iteration of experience collection uses the updated policy
        try:
            collector.update_policy_weights_()
        except Exception:
            print("policy weight update failed (continuing)")
            pass

        # Training rollout mission return (not evaluation):
        # mean over completed episodes in this iteration, where episode reward is the team-sum across all agents.
        train_ep_stats = base_env.pop_episode_stats()
        if train_ep_stats:
            train_mean_episode_total_reward = (
                sum(s["episode_total_reward"] for s in train_ep_stats) / len(train_ep_stats)
            )
        else:
            train_mean_episode_total_reward = float("nan")

        train_component_means = {comp_key: float("nan") for comp_key in EVAL_REWARD_COMPONENT_KEYS}
        if train_ep_stats:
            for comp_key in EVAL_REWARD_COMPONENT_KEYS:
                comp_vals = [
                    float(s.get("episode_component_reward", {}).get(comp_key, float("nan")))
                    for s in train_ep_stats
                ]
                train_component_means[comp_key] = _finite_mean(comp_vals)

        # Algorithm-performance logs (during training, not evaluation). Averaged over all updates in this iteration.
        div = max(1, n_updates)

        try:
            explained_variance = _compute_explained_variance(
                td.get("value_target"),
                td.get((base_env.group, "state_value")),
            )
        except Exception:
            explained_variance = float("nan")

        logs["train_mean_episode_total_reward"].append(train_mean_episode_total_reward)
        logs["loss_policy"].append(pol_acc / div)
        logs["loss_value"].append(val_acc / div)
        logs["explained_variance"].append(explained_variance)
        logs["entropy"].append(ent_acc / div)
        logs["approx_kl"].append(kl_acc / div)
        logs["clip_ratio"].append(clip_acc / div)
        logs["reward_norm_running_mean"].append(norm_stats["running_mean"])
        logs["reward_norm_running_std"].append(norm_stats["running_std"])
        logs["raw_reward_mean"].append(norm_stats["raw_reward_mean"])
        logs["raw_reward_std"].append(norm_stats["raw_reward_std"])
        logs["normalized_reward_mean"].append(norm_stats["normalized_reward_mean"])
        logs["normalized_reward_std"].append(norm_stats["normalized_reward_std"])
        for comp_key in EVAL_REWARD_COMPONENT_KEYS:
            logs[f"train_component_{comp_key}"].append(train_component_means[comp_key])

        # Mission-level evaluation logs: separate from training itterations
        do_eval = bool(ppo_cfg.log_every) and ((it + 1) % ppo_cfg.log_every == 0)
        if do_eval:
            # Run several episodes using current policy (deterministic) in a separate evaluation environment and log mission level metrics
            eval_metrics = evaluate_current_policy(actor, env_cfg, ppo_cfg)
            logs["eval_mean_episode_total_reward"].append(eval_metrics["eval_mean_episode_total_reward"])
            logs["eval_survival_rate"].append(eval_metrics["eval_survival_rate"])
            logs["eval_mean_duration"].append(eval_metrics["eval_mean_duration"])
            logs["eval_task_completion_rate"].append(eval_metrics["eval_task_completion_rate"])
            for comp_key in EVAL_REWARD_COMPONENT_KEYS:
                logs[f"eval_component_{comp_key}"].append(eval_metrics[f"eval_component_{comp_key}"])
        else:
            logs["eval_mean_episode_total_reward"].append(float("nan"))
            logs["eval_survival_rate"].append(float("nan"))
            logs["eval_mean_duration"].append(float("nan"))
            logs["eval_task_completion_rate"].append(float("nan"))
            for comp_key in EVAL_REWARD_COMPONENT_KEYS:
                logs[f"eval_component_{comp_key}"].append(float("nan"))

        # Print logs for this iteration
        if do_eval:
            norm_log = ""
            if bool(ppo_cfg.normalize_rewards):
                norm_log = (
                    f" | raw_σ {norm_stats['raw_reward_std']:.4f}"
                    f" | norm_σ {norm_stats['normalized_reward_std']:.4f}"
                    f" | ret_std {norm_stats['running_std']:.4f}"
                    f" | ret_μ {norm_stats['running_mean']:.4f}"
                )
            print(
                f"Iter {it + 1:4d}/{ppo_cfg.n_iters} | "
                f"train_ep_return_total {logs['train_mean_episode_total_reward'][-1]: .3f} | "
                f"eval_ep_return_total {logs['eval_mean_episode_total_reward'][-1]: .3f} | "
                f"eval_completion {logs['eval_task_completion_rate'][-1]:.2f} | "
                f"eval_survival {logs['eval_survival_rate'][-1]:.2f} | "
                f"eval_duration {logs['eval_mean_duration'][-1]:.1f} | "
                f"clip_ratio {logs['clip_ratio'][-1]:.4f} | "
                f"policy {logs['loss_policy'][-1]:.4f} | "
                f"value {logs['loss_value'][-1]:.4f} | "
                f"ev {logs['explained_variance'][-1]:.4f} | "
                f"entropy {logs['entropy'][-1]:.4f}"
                f"{norm_log}"
            )

        if it + 1 >= ppo_cfg.n_iters:
            break

    # Shutdown the collector
    try:
        collector.shutdown()
    except Exception:
        pass

    return base_env, actor, critic, logs, reward_normalizer
