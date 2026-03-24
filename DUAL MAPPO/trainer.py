"""
Dual-MAPPO training loop.

Two fully independent MAPPO instances (striker, jammer) share one environment
and one data collector.  After each collection batch the data is split by
role and each instance runs its own PPO-clip update with its own actor,
critic, and optimizer.

Logging is split per role so that striker vs jammer policy/value losses,
entropy, KL, etc. are tracked independently.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import RewardSum
from torchrl.envs.utils import check_env_specs

from .config import EnvConfig, NetworkConfig, PPOConfig
from .environment import StrikeEA2DEnv
from .models import (
    CombinedCritic,
    CombinedPolicy,
    make_combined_critic,
    make_combined_policy,
)
from .normalization import RewardNormalizer
from .utils import (
    compute_explained_variance,
    compute_gae_sequential,
    make_collector,
    ppo_clip_loss,
    prepare_done_keys,
    value_loss_fn,
)


EVAL_REWARD_COMPONENT_KEYS: Tuple[str, ...] = (
    "target_destroyed",
    "terminal_bonus",
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


# ------------------------------------------------------------------
# Environment builder
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_current_policy(
    policy: CombinedPolicy,
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    n_eval_episodes: int = 30,
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

    was_training = policy.training
    policy.eval()
    # Set deterministic mode for evaluation
    policy.deterministic = True

    ep_total_rewards: List[float] = []
    ep_survival: List[float] = []
    ep_duration: List[float] = []
    ep_completion: List[float] = []
    ep_component_rewards: Dict[str, List[float]] = {
        key: [] for key in EVAL_REWARD_COMPONENT_KEYS
    }

    for _ in range(max(1, int(n_eval_episodes))):
        td = eval_env.reset()
        episode_component_sums = {key: 0.0 for key in EVAL_REWARD_COMPONENT_KEYS}

        for _step in range(env_cfg.max_steps):
            td = policy(td)
            td_next = eval_env.step(td)

            for comp_key in EVAL_REWARD_COMPONENT_KEYS:
                comp_tensor = eval_env.last_reward_components[comp_key]
                episode_component_sums[comp_key] += float(comp_tensor[0].sum().item())

            done = bool(td_next.get(("next", "done"))[0, 0].item())
            if done:
                break
            td = td_next.get("next")

        stats = eval_env.pop_episode_stats()
        if stats:
            if len(stats) > 1:
                print(f"WARNING evaluate_current_policy: multiple episodes ({len(stats)}) detected; using first")
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

    # Restore training state
    policy.deterministic = False
    if was_training:
        policy.train()

    metrics = {
        "eval_mean_episode_total_reward": float(sum(ep_total_rewards) / len(ep_total_rewards)),
        "eval_survival_rate": float(sum(ep_survival) / len(ep_survival)),
        "eval_mean_duration": float(sum(ep_duration) / len(ep_duration)),
        "eval_task_completion_rate": float(sum(ep_completion) / len(ep_completion)),
    }
    for comp_key in EVAL_REWARD_COMPONENT_KEYS:
        metrics[f"eval_component_{comp_key}"] = _finite_mean(ep_component_rewards[comp_key])
    return metrics


# ------------------------------------------------------------------
# Rollout data reshaping helpers
# ------------------------------------------------------------------

def _infer_time_batch(td: TensorDict, num_envs: int) -> Tuple[int, int, bool]:
    """Return (T, B, transposed) from the rollout TD shape."""
    obs = td.get(("agents", "observation"))
    if obs.ndim == 3:
        # Already flat [N, A, obs_dim]
        return 1, obs.shape[0], False
    # 4-D: either [T, B, A, obs_dim] or [B, T, A, obs_dim]
    if obs.shape[1] == num_envs:
        return obs.shape[0], obs.shape[1], False  # [T, B, ...]
    elif obs.shape[0] == num_envs:
        return obs.shape[1], obs.shape[0], True  # [B, T, ...] → transpose
    else:
        raise ValueError(f"Cannot infer layout: obs shape={obs.shape}, num_envs={num_envs}")


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------

def train_mappo(
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    net_cfg: NetworkConfig,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> Tuple[StrikeEA2DEnv, CombinedPolicy, CombinedCritic, Dict[str, List[float]], Optional[RewardNormalizer]]:
    device = ppo_cfg.device
    ns = env_cfg.n_strikers
    nj = env_cfg.n_jammers

    # ── Build environment ────────────────────────────────────────────
    base_env = build_env(env_cfg, ppo_cfg)
    env = TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env._reward_key], out_keys=[(base_env.group, "episode_reward")]),
    )
    _safe_check(env)

    # ── Build networks ───────────────────────────────────────────────
    policy = make_combined_policy(base_env, hidden=net_cfg.actor_hidden, depth=net_cfg.depth)
    critic = make_combined_critic(base_env, hidden=net_cfg.critic_hidden, depth=net_cfg.depth)

    # ── Reward normalizer ────────────────────────────────────────────
    reward_normalizer: Optional[RewardNormalizer] = None
    if bool(ppo_cfg.normalize_rewards):
        reward_normalizer = RewardNormalizer(
            num_envs=ppo_cfg.num_envs,
            gamma=ppo_cfg.gamma,
            device=device,
        )

    # ── Load checkpoint ──────────────────────────────────────────────
    if checkpoint is not None:
        try:
            if "policy_state_dict" in checkpoint:
                policy.load_state_dict(checkpoint["policy_state_dict"])
            if "critic_state_dict" in checkpoint:
                critic.load_state_dict(checkpoint["critic_state_dict"])
            if reward_normalizer is not None and checkpoint.get("reward_normalizer_state_dict") is not None:
                reward_normalizer.load_state_dict(checkpoint["reward_normalizer_state_dict"])
        except Exception as exc:
            print(f"checkpoint load warning (continuing): {type(exc).__name__}: {exc}")

    # ── Collector ────────────────────────────────────────────────────
    collector = make_collector(env, policy, ppo_cfg.frames_per_batch, ppo_cfg.n_iters, device)

    # ── Per-role optimizers ──────────────────────────────────────────
    striker_actor_optim = optim.Adam(policy.striker_policy.parameters(), lr=ppo_cfg.actor_lr)
    striker_critic_optim = optim.Adam(critic.striker_critic.parameters(), lr=ppo_cfg.critic_lr)
    jammer_actor_optim = optim.Adam(policy.jammer_policy.parameters(), lr=ppo_cfg.actor_lr)
    jammer_critic_optim = optim.Adam(critic.jammer_critic.parameters(), lr=ppo_cfg.critic_lr)

    # ── Logging dict ─────────────────────────────────────────────────
    logs: Dict[str, List[float]] = {
        "train_mean_episode_total_reward": [],
        # Per-role policy losses
        "striker_loss_policy": [],
        "striker_loss_value": [],
        "striker_entropy": [],
        "striker_approx_kl": [],
        "striker_clip_ratio": [],
        "striker_explained_variance": [],
        "jammer_loss_policy": [],
        "jammer_loss_value": [],
        "jammer_entropy": [],
        "jammer_approx_kl": [],
        "jammer_clip_ratio": [],
        "jammer_explained_variance": [],
        # Combined metrics (averages of both roles)
        "loss_policy": [],
        "loss_value": [],
        "entropy": [],
        "clip_ratio": [],
        "explained_variance": [],
        # Eval
        "eval_mean_episode_total_reward": [],
        "eval_survival_rate": [],
        "eval_mean_duration": [],
        "eval_task_completion_rate": [],
        # Reward normalization
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

    # ==================================================================
    # MAIN TRAINING LOOP
    # ==================================================================
    for it, td in enumerate(collector):
        td = td.to(device)

        # ── Reward normalization ─────────────────────────────────────
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

        # ── Propagate done keys ──────────────────────────────────────
        prepare_done_keys(td, base_env)

        # ── Compute values for current and next states ───────────────
        with torch.no_grad():
            try:
                critic(td)
            except Exception as e:
                print(f"WARNING critic(td) failed: {e}")
            try:
                nxt = td.get("next").to(device)
                critic(nxt)
                td.set("next", nxt)
            except Exception as e:
                print(f"WARNING critic(nxt) failed: {e}")

        # ── Extract tensors and compute GAE per role ─────────────────
        obs_all = td.get(("agents", "observation"))          # [*, A, obs_dim]
        actions_all = td.get(("agents", "action"))            # [*, A, 2]
        old_log_probs_all = td.get(("agents", "sample_log_prob"))  # [*, A]
        rewards_all = td.get(("next", "agents", "reward"))    # [*, A, 1]
        values_all = td.get(("agents", "state_value"))        # [*, A, 1]
        next_values_all = td.get(("next", "agents", "state_value"))  # [*, A, 1]
        dones = td.get(("next", "done"))                      # [*, 1]

        # Try to use sequential GAE if the data has a time dimension
        has_time = obs_all.ndim >= 4
        if has_time:
            T, B, transposed = _infer_time_batch(td, ppo_cfg.num_envs)
            if transposed:
                # [B, T, ...] → [T, B, ...]
                obs_all = obs_all.transpose(0, 1)
                actions_all = actions_all.transpose(0, 1)
                old_log_probs_all = old_log_probs_all.transpose(0, 1)
                rewards_all = rewards_all.transpose(0, 1)
                values_all = values_all.transpose(0, 1)
                next_values_all = next_values_all.transpose(0, 1)
                dones = dones.transpose(0, 1)

        # Split by role
        s_obs = obs_all[..., :ns, :]
        j_obs = obs_all[..., ns:, :]
        s_act = actions_all[..., :ns, :]
        j_act = actions_all[..., ns:, :]
        s_old_lp = old_log_probs_all[..., :ns]
        j_old_lp = old_log_probs_all[..., ns:]
        s_rew = rewards_all[..., :ns, :]
        j_rew = rewards_all[..., ns:, :]
        s_val = values_all[..., :ns, :]
        j_val = values_all[..., ns:, :]
        s_nval = next_values_all[..., :ns, :]
        j_nval = next_values_all[..., ns:, :]

        with torch.no_grad():
            if has_time:
                s_adv, s_ret = compute_gae_sequential(s_rew, s_val, s_nval, dones, ppo_cfg.gamma, ppo_cfg.lmbda)
                j_adv, j_ret = compute_gae_sequential(j_rew, j_val, j_nval, dones, ppo_cfg.gamma, ppo_cfg.lmbda)
            else:
                from .utils import compute_gae
                s_adv, s_ret = compute_gae(s_rew, s_val, s_nval, dones, ppo_cfg.gamma, ppo_cfg.lmbda)
                j_adv, j_ret = compute_gae(j_rew, j_val, j_nval, dones, ppo_cfg.gamma, ppo_cfg.lmbda)

        # Flatten everything to [N, ...]
        def _flatten(t):
            return t.reshape(-1, *t.shape[len(t.shape) - t.ndim + (2 if has_time else 1):]) if has_time else t
        # Simpler: just reshape all temporal dims into batch
        s_obs_f = s_obs.reshape(-1, ns, s_obs.shape[-1])
        j_obs_f = j_obs.reshape(-1, nj, j_obs.shape[-1])
        s_act_f = s_act.reshape(-1, ns, s_act.shape[-1])
        j_act_f = j_act.reshape(-1, nj, j_act.shape[-1])
        s_old_lp_f = s_old_lp.reshape(-1, ns)
        j_old_lp_f = j_old_lp.reshape(-1, nj)
        s_adv_f = s_adv.reshape(-1, ns, 1)
        j_adv_f = j_adv.reshape(-1, nj, 1)
        s_ret_f = s_ret.reshape(-1, ns, 1)
        j_ret_f = j_ret.reshape(-1, nj, 1)
        s_val_f = s_val.reshape(-1, ns, 1)
        j_val_f = j_val.reshape(-1, nj, 1)
        state_f = td.get("state").reshape(-1, td.get("state").shape[-1]) if has_time else td.get("state")
        # For state, handle the temporal reshaping
        state_raw = td.get("state")
        if has_time and transposed:
            state_raw = state_raw.transpose(0, 1)
        state_f = state_raw.reshape(-1, state_raw.shape[-1])

        n_samples = s_obs_f.shape[0]

        # ── PPO update for each role ─────────────────────────────────
        s_pol_acc = s_val_acc = s_ent_acc = s_kl_acc = s_clip_acc = 0.0
        j_pol_acc = j_val_acc = j_ent_acc = j_kl_acc = j_clip_acc = 0.0
        n_updates = 0

        for _ in range(ppo_cfg.num_epochs):
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, ppo_cfg.minibatch_size):
                idx = perm[start : start + ppo_cfg.minibatch_size]
                if idx.numel() == 0:
                    continue

                # ── Striker PPO update ───────────────────────────────
                mb_s_obs = s_obs_f[idx]
                mb_s_act = s_act_f[idx]
                mb_s_old_lp = s_old_lp_f[idx]
                mb_s_adv = s_adv_f[idx]
                mb_s_ret = s_ret_f[idx]
                mb_state = state_f[idx]

                s_new_lp, s_entropy = policy.striker_log_prob_entropy(mb_s_obs, mb_s_act)
                s_loss_info = ppo_clip_loss(
                    s_new_lp, mb_s_old_lp, mb_s_adv, s_entropy,
                    ppo_cfg.clip_eps, ppo_cfg.entropy_coef,
                )
                s_pred_val = critic.striker_critic(mb_state)  # [M, ns, 1]
                s_vloss = value_loss_fn(s_pred_val, mb_s_ret)

                striker_actor_optim.zero_grad(set_to_none=True)
                s_loss_info["loss_total"].backward()
                nn.utils.clip_grad_norm_(policy.striker_policy.parameters(), ppo_cfg.max_grad_norm)
                striker_actor_optim.step()

                striker_critic_optim.zero_grad(set_to_none=True)
                s_vloss.backward()
                nn.utils.clip_grad_norm_(critic.striker_critic.parameters(), ppo_cfg.max_grad_norm)
                striker_critic_optim.step()

                s_pol_acc += float(s_loss_info["loss_policy"].item())
                s_val_acc += float(s_vloss.item())
                s_ent_acc += float(s_loss_info["entropy_mean"].item())
                s_kl_acc += s_loss_info["approx_kl"]
                s_clip_acc += s_loss_info["clip_fraction"]

                # ── Jammer PPO update ────────────────────────────────
                mb_j_obs = j_obs_f[idx]
                mb_j_act = j_act_f[idx]
                mb_j_old_lp = j_old_lp_f[idx]
                mb_j_adv = j_adv_f[idx]
                mb_j_ret = j_ret_f[idx]

                j_new_lp, j_entropy = policy.jammer_log_prob_entropy(mb_j_obs, mb_j_act)
                j_loss_info = ppo_clip_loss(
                    j_new_lp, mb_j_old_lp, mb_j_adv, j_entropy,
                    ppo_cfg.clip_eps, ppo_cfg.entropy_coef,
                )
                j_pred_val = critic.jammer_critic(mb_state)  # [M, nj, 1]
                j_vloss = value_loss_fn(j_pred_val, mb_j_ret)

                jammer_actor_optim.zero_grad(set_to_none=True)
                j_loss_info["loss_total"].backward()
                nn.utils.clip_grad_norm_(policy.jammer_policy.parameters(), ppo_cfg.max_grad_norm)
                jammer_actor_optim.step()

                jammer_critic_optim.zero_grad(set_to_none=True)
                j_vloss.backward()
                nn.utils.clip_grad_norm_(critic.jammer_critic.parameters(), ppo_cfg.max_grad_norm)
                jammer_critic_optim.step()

                j_pol_acc += float(j_loss_info["loss_policy"].item())
                j_val_acc += float(j_vloss.item())
                j_ent_acc += float(j_loss_info["entropy_mean"].item())
                j_kl_acc += j_loss_info["approx_kl"]
                j_clip_acc += j_loss_info["clip_fraction"]

                n_updates += 1

        # ── Update collector policy weights ──────────────────────────
        try:
            collector.update_policy_weights_()
        except Exception:
            print("policy weight update failed (continuing)")

        # ── Compute explained variance per role ──────────────────────
        s_ev = compute_explained_variance(s_ret_f, s_val_f)
        j_ev = compute_explained_variance(j_ret_f, j_val_f)

        # ── Training episode stats ───────────────────────────────────
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

        # ── Record logs ──────────────────────────────────────────────
        div = max(1, n_updates)

        logs["train_mean_episode_total_reward"].append(train_mean_episode_total_reward)

        # Per-role logs
        logs["striker_loss_policy"].append(s_pol_acc / div)
        logs["striker_loss_value"].append(s_val_acc / div)
        logs["striker_entropy"].append(s_ent_acc / div)
        logs["striker_approx_kl"].append(s_kl_acc / div)
        logs["striker_clip_ratio"].append(s_clip_acc / div)
        logs["striker_explained_variance"].append(s_ev)

        logs["jammer_loss_policy"].append(j_pol_acc / div)
        logs["jammer_loss_value"].append(j_val_acc / div)
        logs["jammer_entropy"].append(j_ent_acc / div)
        logs["jammer_approx_kl"].append(j_kl_acc / div)
        logs["jammer_clip_ratio"].append(j_clip_acc / div)
        logs["jammer_explained_variance"].append(j_ev)

        # Combined (averaged) for backward-compatible summary
        logs["loss_policy"].append((s_pol_acc + j_pol_acc) / (2.0 * div))
        logs["loss_value"].append((s_val_acc + j_val_acc) / (2.0 * div))
        logs["entropy"].append((s_ent_acc + j_ent_acc) / (2.0 * div))
        logs["clip_ratio"].append((s_clip_acc + j_clip_acc) / (2.0 * div))
        ev_avg = float("nan")
        if math.isfinite(s_ev) and math.isfinite(j_ev):
            ev_avg = (s_ev + j_ev) / 2.0
        elif math.isfinite(s_ev):
            ev_avg = s_ev
        elif math.isfinite(j_ev):
            ev_avg = j_ev
        logs["explained_variance"].append(ev_avg)

        # Reward normalization logs
        logs["reward_norm_running_mean"].append(norm_stats["running_mean"])
        logs["reward_norm_running_std"].append(norm_stats["running_std"])
        logs["raw_reward_mean"].append(norm_stats["raw_reward_mean"])
        logs["raw_reward_std"].append(norm_stats["raw_reward_std"])
        logs["normalized_reward_mean"].append(norm_stats["normalized_reward_mean"])
        logs["normalized_reward_std"].append(norm_stats["normalized_reward_std"])

        for comp_key in EVAL_REWARD_COMPONENT_KEYS:
            logs[f"train_component_{comp_key}"].append(train_component_means[comp_key])

        # ── Evaluation ───────────────────────────────────────────────
        do_eval = bool(ppo_cfg.log_every) and ((it + 1) % ppo_cfg.log_every == 0)
        if do_eval:
            eval_metrics = evaluate_current_policy(policy, env_cfg, ppo_cfg)
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

        # ── Print ────────────────────────────────────────────────────
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
                f"train_ret {logs['train_mean_episode_total_reward'][-1]: .3f} | "
                f"eval_ret {logs['eval_mean_episode_total_reward'][-1]: .3f} | "
                f"comp {logs['eval_task_completion_rate'][-1]:.2f} | "
                f"surv {logs['eval_survival_rate'][-1]:.2f} | "
                f"dur {logs['eval_mean_duration'][-1]:.1f} | "
                f"S[π {logs['striker_loss_policy'][-1]:.4f} "
                f"V {logs['striker_loss_value'][-1]:.4f} "
                f"H {logs['striker_entropy'][-1]:.4f} "
                f"ev {logs['striker_explained_variance'][-1]:.4f}] | "
                f"J[π {logs['jammer_loss_policy'][-1]:.4f} "
                f"V {logs['jammer_loss_value'][-1]:.4f} "
                f"H {logs['jammer_entropy'][-1]:.4f} "
                f"ev {logs['jammer_explained_variance'][-1]:.4f}]"
                f"{norm_log}"
            )

        if it + 1 >= ppo_cfg.n_iters:
            break

    # Shutdown
    try:
        collector.shutdown()
    except Exception:
        pass

    return base_env, policy, critic, logs, reward_normalizer
