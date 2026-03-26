from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    import types
    _this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_this_dir.parent))
    _pkg_name = "dual_mappo"
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_this_dir)]
        _pkg.__package__ = _pkg_name
        _pkg.__file__ = str(_this_dir / "__init__.py")
        sys.modules[_pkg_name] = _pkg
    __package__ = _pkg_name

import torch

from .config import EnvConfig, ExperimentConfig, NetworkConfig, PPOConfig
from .models import make_combined_critic, make_combined_policy
from .rewards import RewardConfig
from .trainer import build_env, train_mappo
from .visualization import TestRunner, animate_rollout, plot_training


@dataclass
class CurriculumStage:
    index: int
    n_iters: int
    n_strikers: int
    n_jammers: int
    n_targets: int
    n_radars: int
    radar_kill_probability: float
    label: str


def _parse_range(text: str, cast_type):
    parts = [p.strip() for p in text.split(":")]
    if len(parts) != 2:
        raise ValueError(f"Invalid range '{text}'. Expected format min:max")
    lo = cast_type(parts[0])
    hi = cast_type(parts[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _split_iters(total: int, chunks: int) -> List[int]:
    total = int(total)
    chunks = int(chunks)
    if total <= 0 or chunks <= 0:
        return []
    base = total // chunks
    rem = total % chunks
    out = [base + (1 if i < rem else 0) for i in range(chunks)]
    return [x for x in out if x > 0]


def _lerp_int(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def _lerp_float(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _build_linear_curriculum(args: argparse.Namespace, total_iters: int) -> List[CurriculumStage]:
    stages = max(1, int(args.curriculum_stages))
    stage_iters = _split_iters(total_iters, stages)
    if not stage_iters:
        return []

    out: List[CurriculumStage] = []
    for i, n_iters in enumerate(stage_iters):
        if len(stage_iters) == 1:
            t = 1.0
        else:
            t = i / float(len(stage_iters) - 1)
        out.append(
            CurriculumStage(
                index=len(out) + 1,
                n_iters=n_iters,
                n_strikers=max(1, _lerp_int(args.start_n_strikers, args.n_strikers, t)),
                n_jammers=max(1, _lerp_int(args.start_n_jammers, args.n_jammers, t)),
                n_targets=max(1, _lerp_int(args.start_n_targets, args.n_targets, t)),
                n_radars=max(1, _lerp_int(args.start_n_radars, args.n_radars, t)),
                radar_kill_probability=max(
                    0.0,
                    min(1.0, _lerp_float(args.start_radar_kill_probability, args.end_radar_kill_probability, t)),
                ),
                label=f"curriculum_{i + 1}",
            )
        )
    return out


def _build_domain_randomization(
    args: argparse.Namespace,
    total_iters: int,
    start_index: int,
) -> List[CurriculumStage]:
    if total_iters <= 0:
        return []

    rng = random.Random(int(args.seed) + 1093)

    ns_lo, ns_hi = _parse_range(args.dr_n_strikers_range, int)
    nj_lo, nj_hi = _parse_range(args.dr_n_jammers_range, int)
    nt_lo, nt_hi = _parse_range(args.dr_n_targets_range, int)
    nr_lo, nr_hi = _parse_range(args.dr_n_radars_range, int)
    kp_lo, kp_hi = _parse_range(args.dr_radar_kill_probability_range, float)

    chunks = max(1, int(math.ceil(total_iters / max(1, args.dr_stage_iters))))
    stage_iters = _split_iters(total_iters, chunks)

    out: List[CurriculumStage] = []
    for i, n_iters in enumerate(stage_iters):
        out.append(
            CurriculumStage(
                index=start_index + i,
                n_iters=n_iters,
                n_strikers=max(1, rng.randint(ns_lo, ns_hi)),
                n_jammers=max(1, rng.randint(nj_lo, nj_hi)),
                n_targets=max(1, rng.randint(nt_lo, nt_hi)),
                n_radars=max(1, rng.randint(nr_lo, nr_hi)),
                radar_kill_probability=max(0.0, min(1.0, rng.uniform(kp_lo, kp_hi))),
                label=f"domain_randomized_{i + 1}",
            )
        )
    return out


def _merge_logs(dst: Dict[str, List[float]], src: Dict[str, List[float]]) -> None:
    for key, values in src.items():
        dst.setdefault(key, [])
        dst[key].extend(values)


def _adapt_checkpoint_for_stage(
    checkpoint: Optional[Dict[str, Any]],
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    net_cfg: NetworkConfig,
) -> Optional[Dict[str, Any]]:
    if checkpoint is None:
        return None

    temp_env = build_env(env_cfg, ppo_cfg)
    temp_policy = make_combined_policy(temp_env, hidden=net_cfg.actor_hidden, depth=net_cfg.depth)
    temp_critic = make_combined_critic(temp_env, hidden=net_cfg.critic_hidden, depth=net_cfg.depth)

    src_policy = checkpoint.get("policy_state_dict") if isinstance(checkpoint, dict) else None
    src_critic = checkpoint.get("critic_state_dict") if isinstance(checkpoint, dict) else None

    dst_policy = temp_policy.state_dict()
    dst_critic = temp_critic.state_dict()

    matched_policy = 0
    matched_critic = 0

    if isinstance(src_policy, dict):
        for key, tensor in src_policy.items():
            if key in dst_policy and tuple(dst_policy[key].shape) == tuple(tensor.shape):
                dst_policy[key] = tensor.detach().to(dst_policy[key].device, dtype=dst_policy[key].dtype)
                matched_policy += 1

    if isinstance(src_critic, dict):
        for key, tensor in src_critic.items():
            if key in dst_critic and tuple(dst_critic[key].shape) == tuple(tensor.shape):
                dst_critic[key] = tensor.detach().to(dst_critic[key].device, dtype=dst_critic[key].dtype)
                matched_critic += 1

    print(
        f"Checkpoint adaptation: policy matched {matched_policy}/{len(dst_policy)}, "
        f"critic matched {matched_critic}/{len(dst_critic)}"
    )

    adapted = {
        "policy_state_dict": dst_policy,
        "critic_state_dict": dst_critic,
        "reward_normalizer_state_dict": checkpoint.get("reward_normalizer_state_dict"),
        "from_stage_label": checkpoint.get("stage_label"),
    }
    return adapted


def build_parser() -> argparse.ArgumentParser:
    env_defaults = EnvConfig()
    ppo_defaults = PPOConfig()
    net_defaults = NetworkConfig()
    reward_defaults = RewardConfig()

    p = argparse.ArgumentParser(description="Dual-MAPPO curriculum runner (run.py remains unchanged)")

    p.add_argument("--n_strikers", type=int, default=env_defaults.n_strikers)
    p.add_argument("--n_jammers", type=int, default=env_defaults.n_jammers)
    p.add_argument("--n_targets", type=int, default=env_defaults.n_targets)
    p.add_argument("--n_radars", type=int, default=env_defaults.n_radars)

    p.add_argument("--start_n_strikers", type=int, default=1)
    p.add_argument("--start_n_jammers", type=int, default=1)
    p.add_argument("--start_n_targets", type=int, default=1)
    p.add_argument("--start_n_radars", type=int, default=1)

    p.add_argument("--start_radar_kill_probability", type=float, default=0.25)
    p.add_argument("--end_radar_kill_probability", type=float, default=env_defaults.radar_kill_probability)

    p.add_argument("--num_envs", type=int, default=ppo_defaults.num_envs)
    p.add_argument("--max_steps", type=int, default=env_defaults.max_steps)
    p.add_argument("--n_iters", type=int, default=ppo_defaults.n_iters)
    p.add_argument("--num_epochs", type=int, default=ppo_defaults.num_epochs)
    p.add_argument("--minibatch_size", type=int, default=ppo_defaults.minibatch_size)
    p.add_argument("--actor_lr", type=float, default=ppo_defaults.actor_lr)
    p.add_argument("--critic_lr", type=float, default=ppo_defaults.critic_lr)
    p.add_argument("--clip_eps", type=float, default=ppo_defaults.clip_eps)
    p.add_argument("--entropy_coef", type=float, default=ppo_defaults.entropy_coef)
    p.add_argument("--normalize_rewards", action=argparse.BooleanOptionalAction, default=ppo_defaults.normalize_rewards)
    p.add_argument("--seed", type=int, default=ppo_defaults.seed)
    p.add_argument("--curriculum_stages", type=int, default=4)

    p.add_argument("--enable_domain_randomization", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dr_fraction", type=float, default=0.3)
    p.add_argument("--dr_stage_iters", type=int, default=10)
    p.add_argument("--dr_n_strikers_range", type=str, default="1:4")
    p.add_argument("--dr_n_jammers_range", type=str, default="1:4")
    p.add_argument("--dr_n_targets_range", type=str, default="1:4")
    p.add_argument("--dr_n_radars_range", type=str, default="1:4")
    p.add_argument("--dr_radar_kill_probability_range", type=str, default="0.1:1.0")

    p.add_argument("--actor_hidden", type=int, default=net_defaults.actor_hidden)
    p.add_argument("--critic_hidden", type=int, default=net_defaults.critic_hidden)
    p.add_argument("--depth", type=int, default=net_defaults.depth)

    p.add_argument("--target_destroyed", type=float, default=reward_defaults.target_destroyed)
    p.add_argument("--agent_destroyed", type=float, default=reward_defaults.agent_destroyed)
    p.add_argument("--timestep_penalty", type=float, default=reward_defaults.timestep_penalty)
    p.add_argument("--team_spirit", type=float, default=reward_defaults.team_spirit)
    p.add_argument("--striker_progress_scale", type=float, default=reward_defaults.striker_progress_scale)
    p.add_argument("--jammer_progress_scale", type=float, default=reward_defaults.jammer_progress_scale)
    p.add_argument("--jammer_jam_bonus", type=float, default=reward_defaults.jammer_jam_bonus)

    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--save_name", type=str, default="dual_mappo_curriculum.pt")
    p.add_argument("--save_stage_checkpoints", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--load_checkpoint", type=str, default=None)
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--no_animate", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.n_iters <= 0:
        raise ValueError("n_iters must be > 0")

    reward_cfg = RewardConfig(
        target_destroyed=args.target_destroyed,
        agent_destroyed=args.agent_destroyed,
        timestep_penalty=args.timestep_penalty,
        team_spirit=args.team_spirit,
        striker_progress_scale=args.striker_progress_scale,
        jammer_progress_scale=args.jammer_progress_scale,
        jammer_jam_bonus=args.jammer_jam_bonus,
    )

    ppo_template = PPOConfig(
        num_envs=args.num_envs,
        n_iters=1,
        num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        normalize_rewards=args.normalize_rewards,
        seed=args.seed,
    )
    net_cfg = NetworkConfig(
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        depth=args.depth,
    )

    dr_fraction = max(0.0, min(1.0, float(args.dr_fraction)))
    if args.enable_domain_randomization:
        dr_iters = int(round(args.n_iters * dr_fraction))
        cur_iters = int(args.n_iters - dr_iters)
    else:
        dr_iters = 0
        cur_iters = int(args.n_iters)

    stages: List[CurriculumStage] = []
    stages.extend(_build_linear_curriculum(args, cur_iters))
    stages.extend(_build_domain_randomization(args, dr_iters, start_index=len(stages) + 1))

    if not stages:
        raise RuntimeError("No curriculum stages were generated. Check n_iters and settings.")

    print("\n=== Curriculum Plan ===")
    for st in stages:
        print(
            f"Stage {st.index:02d} [{st.label}] | iters={st.n_iters:3d} | "
            f"S/J/T/R={st.n_strikers}/{st.n_jammers}/{st.n_targets}/{st.n_radars} | "
            f"radar_kill_probability={st.radar_kill_probability:.3f}"
        )

    incoming_checkpoint: Optional[Dict[str, Any]] = None
    if args.load_checkpoint:
        incoming_checkpoint = torch.load(args.load_checkpoint, map_location=ppo_template.device)
        print(f"Loaded initial checkpoint from: {args.load_checkpoint}")

    all_logs: Dict[str, List[float]] = {}
    stage_records: List[Dict[str, Any]] = []
    final_env_cfg: Optional[EnvConfig] = None
    final_policy = None
    final_critic = None
    final_reward_normalizer = None

    for st in stages:
        stage_reward_cfg = copy.deepcopy(reward_cfg)
        env_cfg = EnvConfig(
            n_strikers=st.n_strikers,
            n_jammers=st.n_jammers,
            n_targets=st.n_targets,
            n_radars=st.n_radars,
            max_steps=args.max_steps,
            radar_kill_probability=st.radar_kill_probability,
            reward_config=stage_reward_cfg,
        )
        ppo_cfg = PPOConfig(
            num_envs=ppo_template.num_envs,
            n_iters=st.n_iters,
            num_epochs=ppo_template.num_epochs,
            minibatch_size=ppo_template.minibatch_size,
            actor_lr=ppo_template.actor_lr,
            critic_lr=ppo_template.critic_lr,
            clip_eps=ppo_template.clip_eps,
            entropy_coef=ppo_template.entropy_coef,
            normalize_rewards=ppo_template.normalize_rewards,
            seed=ppo_template.seed + st.index,
        )

        stage_exp_cfg = ExperimentConfig(env=env_cfg, ppo=ppo_cfg, net=net_cfg).finalize()
        adapted_ckpt = _adapt_checkpoint_for_stage(incoming_checkpoint, stage_exp_cfg.env, stage_exp_cfg.ppo, stage_exp_cfg.net)

        print(
            f"\n--- Training stage {st.index}/{len(stages)} ({st.label}) ---\n"
            f"n_iters={st.n_iters}, S/J/T/R={st.n_strikers}/{st.n_jammers}/{st.n_targets}/{st.n_radars}, "
            f"radar_kill_probability={st.radar_kill_probability:.3f}"
        )

        base_env, policy, critic, logs, reward_normalizer = train_mappo(
            stage_exp_cfg.env,
            stage_exp_cfg.ppo,
            stage_exp_cfg.net,
            checkpoint=adapted_ckpt,
        )

        _merge_logs(all_logs, logs)

        stage_end_eval = float("nan")
        if logs.get("eval_mean_episode_total_reward"):
            stage_end_eval = logs["eval_mean_episode_total_reward"][-1]

        stage_record = {
            "index": st.index,
            "label": st.label,
            "n_iters": st.n_iters,
            "n_strikers": st.n_strikers,
            "n_jammers": st.n_jammers,
            "n_targets": st.n_targets,
            "n_radars": st.n_radars,
            "radar_kill_probability": st.radar_kill_probability,
            "last_eval_mean_episode_total_reward": stage_end_eval,
        }
        stage_records.append(stage_record)

        if args.save_stage_checkpoints:
            stage_name = Path(args.save_name).stem
            stage_suffix = Path(args.save_name).suffix or ".pt"
            stage_path = Path(args.save_dir) / f"{stage_name}_stage_{st.index:02d}{stage_suffix}"
            stage_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "env_cfg": stage_exp_cfg.env,
                    "ppo_cfg": stage_exp_cfg.ppo,
                    "net_cfg": stage_exp_cfg.net,
                    "logs": logs,
                    "reward_normalizer_state_dict": (
                        reward_normalizer.state_dict() if reward_normalizer is not None else None
                    ),
                    "stage_record": stage_record,
                    "stage_label": st.label,
                },
                stage_path,
            )
            print(f"Saved stage checkpoint: {stage_path}")

        incoming_checkpoint = {
            "policy_state_dict": policy.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "reward_normalizer_state_dict": (
                reward_normalizer.state_dict() if reward_normalizer is not None else None
            ),
            "stage_label": st.label,
        }
        final_env_cfg = stage_exp_cfg.env
        final_policy = policy
        final_critic = critic
        final_reward_normalizer = reward_normalizer

    if final_env_cfg is None or final_policy is None or final_critic is None:
        raise RuntimeError("Curriculum did not produce a final policy.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / args.save_name
    torch.save(
        {
            "policy_state_dict": final_policy.state_dict(),
            "critic_state_dict": final_critic.state_dict(),
            "env_cfg": final_env_cfg,
            "ppo_cfg": ppo_template,
            "net_cfg": net_cfg,
            "logs": all_logs,
            "reward_normalizer_state_dict": (
                final_reward_normalizer.state_dict() if final_reward_normalizer is not None else None
            ),
            "curriculum_stages": stage_records,
        },
        save_path,
    )
    print(f"\nSaved curriculum checkpoint to: {save_path}")

    if all_logs.get("eval_mean_episode_total_reward"):
        print("Last aggregated metrics:")
        print(f"  train_mean_episode_total_reward = {all_logs['train_mean_episode_total_reward'][-1]:.4f}")
        print(f"  eval_mean_episode_total_reward  = {all_logs['eval_mean_episode_total_reward'][-1]:.4f}")
        print(f"  eval_task_completion_rate       = {all_logs['eval_task_completion_rate'][-1]:.4f}")
        print(f"  eval_survival_rate              = {all_logs['eval_survival_rate'][-1]:.4f}")
        print(f"  eval_mean_duration              = {all_logs['eval_mean_duration'][-1]:.4f}")

    if not args.no_plot:
        try:
            plot_training(all_logs)
        except Exception as exc:
            print(f"plot_training warning (continuing): {type(exc).__name__}: {exc}")

    if not args.no_animate:
        try:
            for _ in range(5):
                tester = TestRunner(final_policy, env_cfg=final_env_cfg, device=ppo_template.device, seed=999)
                frames = tester.rollout()
                animate_rollout(frames, tester.env)
                print(f"Visualized rollout with {len(frames)} frames")
        except Exception as exc:
            print(f"animate_rollout warning (continuing): {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
