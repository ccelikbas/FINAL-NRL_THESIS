from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from .trainer import build_env, evaluate_current_policy, train_mappo
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


def _split_iters(total: int, chunks: int) -> List[int]:
    total = int(total)
    chunks = int(chunks)
    if total <= 0 or chunks <= 0:
        return []
    base = total // chunks
    rem = total % chunks
    out = [base + (1 if i < rem else 0) for i in range(chunks)]
    return [x for x in out if x > 0]


def _build_requested_five_stage_plan(args: argparse.Namespace) -> List[CurriculumStage]:
    rng = random.Random(int(args.seed) + 1093)

    stage2_template = dict(n_strikers=1, n_jammers=1, n_targets=1, n_radars=1, radar_kill_probability=1.0)
    stage3_template = dict(n_strikers=1, n_jammers=1, n_targets=2, n_radars=2, radar_kill_probability=1.0)
    stage4_template = dict(n_strikers=2, n_jammers=2, n_targets=2, n_radars=2, radar_kill_probability=1.0)

    stages: List[CurriculumStage] = [
        CurriculumStage(
            index=1,
            label="stage_1",
            n_iters=args.stage1_iters,
            n_strikers=1,
            n_jammers=1,
            n_targets=1,
            n_radars=1,
            radar_kill_probability=0.5,
        ),
        CurriculumStage(
            index=2,
            label="stage_2",
            n_iters=args.stage2_iters,
            **stage2_template,
        ),
        CurriculumStage(
            index=3,
            label="stage_3",
            n_iters=args.stage3_iters,
            **stage3_template,
        ),
        CurriculumStage(
            index=4,
            label="stage_4",
            n_iters=args.stage4_iters,
            **stage4_template,
        ),
    ]

    random_stage_choices = [
        ("stage2", stage2_template),
        ("stage3", stage3_template),
        ("stage4", stage4_template),
    ]
    # Final phase: domain randomization per ITERATION (not every N iters).
    # Each stage below has n_iters=1 and randomised S/J/R/T sampled within user bounds.
    # This keeps one actor policy continuously optimised while exposing varied configs.
    next_index = 5
    for i in range(int(args.stage5_iters)):
        if bool(args.stage5_use_bounds):
            sampled_cfg = {
                "n_strikers": rng.randint(int(args.stage5_min_strikers), int(args.stage5_max_strikers)),
                "n_jammers": rng.randint(int(args.stage5_min_jammers), int(args.stage5_max_jammers)),
                "n_targets": rng.randint(int(args.stage5_min_targets), int(args.stage5_max_targets)),
                "n_radars": rng.randint(int(args.stage5_min_radars), int(args.stage5_max_radars)),
                "radar_kill_probability": rng.uniform(
                    float(args.stage5_min_radar_kill_probability),
                    float(args.stage5_max_radar_kill_probability),
                ),
            }
            choice_name = "bounds"
            cfg = sampled_cfg
        else:
            choice_name, cfg = random_stage_choices[rng.randint(0, len(random_stage_choices) - 1)]
        stages.append(
            CurriculumStage(
                index=next_index,
                label=f"stage_5_random_{i + 1}_{choice_name}",
                n_iters=1,
                n_strikers=cfg["n_strikers"],
                n_jammers=cfg["n_jammers"],
                n_targets=cfg["n_targets"],
                n_radars=cfg["n_radars"],
                radar_kill_probability=max(0.0, min(1.0, float(cfg["radar_kill_probability"]))),
            )
        )
        next_index += 1

    return stages


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
        temp_policy.load_state_dict(src_policy, strict=True)
        dst_policy = temp_policy.state_dict()
        matched_policy = len([v for v in dst_policy.values() if isinstance(v, torch.Tensor)])

    if isinstance(src_critic, dict):
        for key, tensor in src_critic.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if key in dst_critic and isinstance(dst_critic[key], torch.Tensor) and tuple(dst_critic[key].shape) == tuple(tensor.shape):
                dst_critic[key] = tensor.detach().to(dst_critic[key].device, dtype=dst_critic[key].dtype)
                matched_critic += 1

    print(
        f"Checkpoint adaptation: policy strict_load tensors={matched_policy}, "
        f"critic matched {matched_critic}/{len(dst_critic)}"
    )

    adapted = {
        "policy_state_dict": dst_policy,
        "critic_state_dict": dst_critic,
        "reward_normalizer_state_dict": checkpoint.get("reward_normalizer_state_dict"),
        "from_stage_label": checkpoint.get("stage_label"),
    }
    return adapted


def _evaluate_generalized_policy(
    checkpoint: Dict[str, Any],
    reward_cfg: RewardConfig,
    ppo_template: PPOConfig,
    net_cfg: NetworkConfig,
    max_steps: int,
    n_eval_episodes: int,
) -> List[Dict[str, Any]]:
    eval_setups = _get_generalized_eval_setups()

    print("\n=== Generalized Single-Policy Evaluation ===")
    print(f"Episodes per configuration: {n_eval_episodes}")

    results: List[Dict[str, Any]] = []
    for label, ns, nj, nr, nt in eval_setups:
        eval_env_cfg = EnvConfig(
            n_strikers=ns,
            n_jammers=nj,
            n_targets=nt,
            n_radars=nr,
            max_steps=max_steps,
            radar_kill_probability=1.0,
            reward_config=copy.deepcopy(reward_cfg),
        )
        eval_ppo_cfg = PPOConfig(
            num_envs=ppo_template.num_envs,
            n_iters=1,
            num_epochs=ppo_template.num_epochs,
            minibatch_size=ppo_template.minibatch_size,
            actor_lr=ppo_template.actor_lr,
            critic_lr=ppo_template.critic_lr,
            clip_eps=ppo_template.clip_eps,
            entropy_coef=ppo_template.entropy_coef,
            normalize_rewards=ppo_template.normalize_rewards,
            seed=ppo_template.seed,
            log_every=ppo_template.log_every,
            device=ppo_template.device,
        )

        adapted_ckpt = _adapt_checkpoint_for_stage(checkpoint, eval_env_cfg, eval_ppo_cfg, net_cfg)
        if adapted_ckpt is None:
            raise RuntimeError("Generalized evaluation requires a trained checkpoint.")

        eval_env = build_env(eval_env_cfg, eval_ppo_cfg)
        eval_policy = make_combined_policy(eval_env, hidden=net_cfg.actor_hidden, depth=net_cfg.depth)
        eval_policy.load_state_dict(adapted_ckpt["policy_state_dict"], strict=True)
        eval_policy = eval_policy.to(eval_ppo_cfg.device)

        metrics = evaluate_current_policy(
            eval_policy,
            eval_env_cfg,
            eval_ppo_cfg,
            n_eval_episodes=int(n_eval_episodes),
        )

        results.append(
            {
                "config": label,
                "completion_rate": float(metrics["eval_task_completion_rate"]),
                "survival_rate": float(metrics["eval_survival_rate"]),
                "mean_duration": float(metrics["eval_mean_duration"]),
            }
        )

    headers = ("Config", "Completion", "Survival", "Mean Time")
    print("\n" + f"{headers[0]:<14}{headers[1]:>14}{headers[2]:>12}{headers[3]:>14}")
    print("-" * 54)
    for row in results:
        print(
            f"{row['config']:<14}"
            f"{row['completion_rate']:>14.4f}"
            f"{row['survival_rate']:>12.4f}"
            f"{row['mean_duration']:>14.2f}"
        )

    return results


def _get_generalized_eval_setups() -> List[tuple[str, int, int, int, int]]:
    # (label, n_strikers, n_jammers, n_radars, n_targets)
    return [
        ("1s1j1r1t", 1, 1, 1, 1),
        ("1s1j2r2t", 1, 1, 2, 2),
        ("2s2j2r2t", 2, 2, 2, 2),
        ("2s2j3r3t", 2, 2, 3, 3),
    ]


def _assert_actor_policy_shape_consistency(
    ppo_template: PPOConfig,
    net_cfg: NetworkConfig,
    env_cfgs: List[EnvConfig],
) -> None:
    """Assert actor parameter shapes are identical across all curriculum/eval configs.

    This guards the requirement that one actor policy architecture is used for all
    configurations. If this fails, strict policy transfer would also fail.
    """
    if not env_cfgs:
        return

    base_env = build_env(env_cfgs[0], ppo_template)
    base_policy = make_combined_policy(base_env, hidden=net_cfg.actor_hidden, depth=net_cfg.depth)
    base_shapes = {
        k: tuple(v.shape)
        for k, v in base_policy.state_dict().items()
        if isinstance(v, torch.Tensor)
    }

    for idx, cfg in enumerate(env_cfgs[1:], start=2):
        env_i = build_env(cfg, ppo_template)
        pol_i = make_combined_policy(env_i, hidden=net_cfg.actor_hidden, depth=net_cfg.depth)
        shapes_i = {
            k: tuple(v.shape)
            for k, v in pol_i.state_dict().items()
            if isinstance(v, torch.Tensor)
        }
        if shapes_i != base_shapes:
            raise RuntimeError(
                f"Actor policy shape mismatch at config #{idx}: "
                f"S/J/T/R={cfg.n_strikers}/{cfg.n_jammers}/{cfg.n_targets}/{cfg.n_radars}."
            )


def build_parser() -> argparse.ArgumentParser:
    env_defaults = EnvConfig()
    ppo_defaults = PPOConfig()
    net_defaults = NetworkConfig()
    reward_defaults = RewardConfig()

    p = argparse.ArgumentParser(description="Dual-MAPPO curriculum runner (run.py remains unchanged)")

    p.add_argument("--stage1_iters", type=int, default=50)
    p.add_argument("--stage2_iters", type=int, default=50)
    p.add_argument("--stage3_iters", type=int, default=200)
    p.add_argument("--stage4_iters", type=int, default=100)
    p.add_argument("--stage5_iters", type=int, default=200)
    # Stage-5 randomization mode:
    # --stage5_use_bounds=True => sample S/J/R/T and radar kill prob from bounds every iteration.
    # --stage5_use_bounds=False => sample from predefined stage2/3/4 templates every iteration.
    p.add_argument("--stage5_use_bounds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stage5_min_strikers", type=int, default=1)
    p.add_argument("--stage5_max_strikers", type=int, default=2)
    p.add_argument("--stage5_min_jammers", type=int, default=1)
    p.add_argument("--stage5_max_jammers", type=int, default=2)
    p.add_argument("--stage5_min_radars", type=int, default=1)
    p.add_argument("--stage5_max_radars", type=int, default=3)
    p.add_argument("--stage5_min_targets", type=int, default=1)
    p.add_argument("--stage5_max_targets", type=int, default=3)
    p.add_argument("--stage5_min_radar_kill_probability", type=float, default=0.5)
    p.add_argument("--stage5_max_radar_kill_probability", type=float, default=1.0)
    p.add_argument("--generalized_eval_episodes", type=int, default=30)
    p.add_argument("--generalized_eval", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--num_envs", type=int, default=ppo_defaults.num_envs)
    p.add_argument("--max_steps", type=int, default=env_defaults.max_steps)
    p.add_argument("--num_epochs", type=int, default=ppo_defaults.num_epochs)
    p.add_argument("--minibatch_size", type=int, default=ppo_defaults.minibatch_size)
    p.add_argument("--actor_lr", type=float, default=ppo_defaults.actor_lr)
    p.add_argument("--critic_lr", type=float, default=ppo_defaults.critic_lr)
    p.add_argument("--clip_eps", type=float, default=ppo_defaults.clip_eps)
    p.add_argument("--entropy_coef", type=float, default=ppo_defaults.entropy_coef)
    p.add_argument("--normalize_rewards", action=argparse.BooleanOptionalAction, default=ppo_defaults.normalize_rewards)
    p.add_argument("--seed", type=int, default=ppo_defaults.seed)
    p.add_argument("--log_every", type=int, default=ppo_defaults.log_every)

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

    for key in ("stage1_iters", "stage2_iters", "stage3_iters", "stage4_iters", "stage5_iters"):
        if int(getattr(args, key)) <= 0:
            raise ValueError(f"{key} must be > 0")
    if int(args.generalized_eval_episodes) <= 0:
        raise ValueError("generalized_eval_episodes must be > 0")
    for low_key, high_key in (
        ("stage5_min_strikers", "stage5_max_strikers"),
        ("stage5_min_jammers", "stage5_max_jammers"),
        ("stage5_min_radars", "stage5_max_radars"),
        ("stage5_min_targets", "stage5_max_targets"),
    ):
        if int(getattr(args, low_key)) <= 0:
            raise ValueError(f"{low_key} must be > 0")
        if int(getattr(args, high_key)) < int(getattr(args, low_key)):
            raise ValueError(f"{high_key} must be >= {low_key}")
    if not (0.0 <= float(args.stage5_min_radar_kill_probability) <= 1.0):
        raise ValueError("stage5_min_radar_kill_probability must be in [0,1]")
    if not (0.0 <= float(args.stage5_max_radar_kill_probability) <= 1.0):
        raise ValueError("stage5_max_radar_kill_probability must be in [0,1]")
    if float(args.stage5_max_radar_kill_probability) < float(args.stage5_min_radar_kill_probability):
        raise ValueError("stage5_max_radar_kill_probability must be >= stage5_min_radar_kill_probability")

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
        log_every=args.log_every,
    )
    net_cfg = NetworkConfig(
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        depth=args.depth,
    )

    stages: List[CurriculumStage] = _build_requested_five_stage_plan(args)

    if not stages:
        raise RuntimeError("No curriculum stages were generated. Check n_iters and settings.")

    print("\n=== Curriculum Plan ===")
    print("Actor policy transfer mode: strict=True (same actor weights/biases carried each stage)")
    print("Domain randomization: enabled in stage 5 (new random config every iteration)")
    print(f"Reward normalization: {'enabled' if ppo_template.normalize_rewards else 'disabled'} and state carried across stages")
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

    # Verify once that actor architecture is identical across all planned/eval configurations.
    # This enforces the single-policy-shape requirement before training starts.
    shape_check_cfgs: List[EnvConfig] = []
    for st in stages:
        shape_check_cfgs.append(
            EnvConfig(
                n_strikers=st.n_strikers,
                n_jammers=st.n_jammers,
                n_targets=st.n_targets,
                n_radars=st.n_radars,
                max_steps=args.max_steps,
                radar_kill_probability=st.radar_kill_probability,
                reward_config=copy.deepcopy(reward_cfg),
            )
        )
    shape_check_cfgs.extend(
        [
            EnvConfig(n_strikers=1, n_jammers=1, n_targets=1, n_radars=1, max_steps=args.max_steps, reward_config=copy.deepcopy(reward_cfg)),
            EnvConfig(n_strikers=1, n_jammers=1, n_targets=2, n_radars=2, max_steps=args.max_steps, reward_config=copy.deepcopy(reward_cfg)),
            EnvConfig(n_strikers=2, n_jammers=2, n_targets=2, n_radars=2, max_steps=args.max_steps, reward_config=copy.deepcopy(reward_cfg)),
            EnvConfig(n_strikers=2, n_jammers=2, n_targets=3, n_radars=3, max_steps=args.max_steps, reward_config=copy.deepcopy(reward_cfg)),
        ]
    )
    _assert_actor_policy_shape_consistency(ppo_template, net_cfg, shape_check_cfgs)
    print("Actor shape consistency check: PASSED across curriculum + evaluation configurations")

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

    final_checkpoint_payload = {
        "policy_state_dict": final_policy.state_dict(),
        "critic_state_dict": final_critic.state_dict(),
        "reward_normalizer_state_dict": (
            final_reward_normalizer.state_dict() if final_reward_normalizer is not None else None
        ),
        "stage_label": "final_curriculum_policy",
    }

    generalized_eval_results: List[Dict[str, Any]] = []
    if bool(args.generalized_eval):
        generalized_eval_results = _evaluate_generalized_policy(
            checkpoint=final_checkpoint_payload,
            reward_cfg=reward_cfg,
            ppo_template=ppo_template,
            net_cfg=net_cfg,
            max_steps=args.max_steps,
            n_eval_episodes=args.generalized_eval_episodes,
        )

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
            "generalized_policy_eval": generalized_eval_results,
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
            print("\nAnimating generalized policy: 5 rollouts per evaluation config (20 total)")
            eval_setups = _get_generalized_eval_setups()
            rollout_counter = 0
            for label, ns, nj, nr, nt in eval_setups:
                anim_env_cfg = EnvConfig(
                    n_strikers=ns,
                    n_jammers=nj,
                    n_targets=nt,
                    n_radars=nr,
                    max_steps=args.max_steps,
                    radar_kill_probability=1.0,
                    reward_config=copy.deepcopy(reward_cfg),
                )
                anim_ppo_cfg = PPOConfig(
                    num_envs=ppo_template.num_envs,
                    n_iters=1,
                    num_epochs=ppo_template.num_epochs,
                    minibatch_size=ppo_template.minibatch_size,
                    actor_lr=ppo_template.actor_lr,
                    critic_lr=ppo_template.critic_lr,
                    clip_eps=ppo_template.clip_eps,
                    entropy_coef=ppo_template.entropy_coef,
                    normalize_rewards=ppo_template.normalize_rewards,
                    seed=ppo_template.seed,
                    log_every=ppo_template.log_every,
                    device=ppo_template.device,
                )

                adapted_ckpt = _adapt_checkpoint_for_stage(final_checkpoint_payload, anim_env_cfg, anim_ppo_cfg, net_cfg)
                if adapted_ckpt is None:
                    raise RuntimeError("Animation requires a trained checkpoint payload.")

                anim_env = build_env(anim_env_cfg, anim_ppo_cfg)
                anim_policy = make_combined_policy(anim_env, hidden=net_cfg.actor_hidden, depth=net_cfg.depth)
                anim_policy.load_state_dict(adapted_ckpt["policy_state_dict"], strict=True)
                anim_policy = anim_policy.to(anim_ppo_cfg.device)

                for r in range(5):
                    tester = TestRunner(anim_policy, env_cfg=anim_env_cfg, device=anim_ppo_cfg.device, seed=999 + r)
                    frames = tester.rollout()
                    animate_rollout(frames, tester.env)
                    rollout_counter += 1
                    print(
                        f"Visualized rollout {rollout_counter}/20 "
                        f"for {label} (run {r + 1}/5) with {len(frames)} frames"
                    )
        except Exception as exc:
            print(f"animate_rollout warning (continuing): {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
