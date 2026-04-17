"""Curriculum learning runner for FOFE-MAPPO.

Edit the CURRICULUM and EVAL sections below to configure your curriculum
phases and evaluation scenarios.  The training loop uses a **global
iteration counter** that never resets across phase transitions, so
evaluation runs at fixed global intervals regardless of config changes.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

if __package__ in (None, ""):
    import types
    _this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_this_dir.parent))
    _pkg_name = "fofe_mappo"
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_this_dir)]
        _pkg.__package__ = _pkg_name
        _pkg.__file__ = str(_this_dir / "__init__.py")
        sys.modules[_pkg_name] = _pkg
    __package__ = _pkg_name

import torch
import numpy as np
import matplotlib.pyplot as plt

from .config import EnvConfig, ExperimentConfig, FOFEConfig, NetworkConfig, PPOConfig
from .models import make_combined_critic, make_combined_policy
from .rewards import RewardConfig
from .trainer import build_env, evaluate_current_policy, train_mappo
from .visualization import TestRunner, animate_rollout, plot_training


# =====================================================================
#  CURRICULUM CONFIGURATION  –  edit this section freely
# =====================================================================

@dataclass
class ScenarioConfig:
    """A single environment scenario configuration."""
    n_strikers: int = 1
    n_jammers: int = 1
    n_known_targets: int = 1
    n_unknown_targets: int = 0
    n_known_radars: int = 1
    n_unknown_radars: int = 0
    radar_kill_probability: float = 1.0

    @property
    def label(self) -> str:
        parts = [f"{self.n_strikers}s{self.n_jammers}j"]
        parts.append(f"{self.n_known_targets}kt")
        if self.n_unknown_targets > 0:
            parts.append(f"{self.n_unknown_targets}ut")
        parts.append(f"{self.n_known_radars}kr")
        if self.n_unknown_radars > 0:
            parts.append(f"{self.n_unknown_radars}ur")
        if self.radar_kill_probability < 1.0:
            parts.append(f"p{self.radar_kill_probability:.2f}")
        return "_".join(parts)


@dataclass
class Fixed:
    """Use a single fixed configuration for the entire phase."""
    config: ScenarioConfig


@dataclass
class RandomChoice:
    """Randomly pick one configuration per iteration from a list."""
    choices: List[ScenarioConfig]


@dataclass
class RandomRange:
    """Sample each parameter independently from [min, max] per iteration."""
    n_strikers: Tuple[int, int] = (1, 1)
    n_jammers: Tuple[int, int] = (1, 1)
    n_known_targets: Tuple[int, int] = (1, 1)
    n_unknown_targets: Tuple[int, int] = (0, 0)
    n_known_radars: Tuple[int, int] = (1, 1)
    n_unknown_radars: Tuple[int, int] = (0, 0)
    radar_kill_probability: Tuple[float, float] = (1.0, 1.0)


ConfigType = Union[Fixed, RandomChoice, RandomRange]


@dataclass
class CurriculumPhase:
    """One phase of the curriculum.

    iters : (start, end) — half-open range [start, end) of global iterations.
    config: Fixed, RandomChoice, or RandomRange.
            For RandomChoice / RandomRange a *new* config is sampled every
            iteration so the policy learns generalised behaviour.
    """
    name: str
    iters: Tuple[int, int]
    config: ConfigType


# ──────────────────── CURRICULUM DEFINITION ────────────────────────
# Phases must be contiguous: phase[i].iters[1] == phase[i+1].iters[0].

CURRICULUM: List[CurriculumPhase] = [
    CurriculumPhase(
        name="basic_1v1",
        iters=(0, 25),
        config=Fixed(ScenarioConfig(
            n_strikers=1, n_jammers=1,
            n_known_targets=1, n_known_radars=1,
            radar_kill_probability=1.0,
        )),
    ),
    CurriculumPhase(
        name="scaling_1v2",
        iters=(25, 100),
        config=RandomChoice([
            ScenarioConfig(n_strikers=1, n_jammers=1,
                           n_known_targets=1, n_known_radars=1),
            ScenarioConfig(n_strikers=1, n_jammers=1,
                           n_known_targets=2, n_known_radars=2),
        ]),
    ),
    CurriculumPhase(
        name="scaling_2v2",
        iters=(100, 150),
        config=RandomChoice([
            ScenarioConfig(n_strikers=1, n_jammers=1,
                           n_known_targets=2, n_known_radars=2),
            ScenarioConfig(n_strikers=2, n_jammers=2,
                           n_known_targets=2, n_known_radars=2),
        ]),
    ),
    # CurriculumPhase(
    #     name="domain_randomization",
    #     iters=(200, 400),
    #     config=RandomRange(
    #         n_strikers=(1, 2),
    #         n_jammers=(1, 2),
    #         n_known_targets=(1, 3),
    #         n_known_radars=(1, 3),
    #         radar_kill_probability=(1.0),
    #     ),
    # ),
]


# ──────────────────── EVALUATION SCENARIOS ─────────────────────────
# Tested every EVAL_EVERY global iterations; each runs EVAL_EPISODES.

EVAL_SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig(n_strikers=1, n_jammers=1, n_known_targets=1, n_known_radars=1),
    ScenarioConfig(n_strikers=1, n_jammers=1, n_known_targets=2, n_known_radars=2),
    ScenarioConfig(n_strikers=2, n_jammers=2, n_known_targets=2, n_known_radars=2)
    # ScenarioConfig(n_strikers=2, n_jammers=2, n_known_targets=3, n_known_radars=3),
]

EVAL_EVERY: int = 5        # run multi-scenario eval every N global iterations
EVAL_EPISODES: int = 100     # episodes per eval scenario


# ──────────────────── FOFE CONFIGURATION ───────────────────────────

FOFE_CONFIG = FOFEConfig(use_fofe=True)


# =====================================================================
#  IMPLEMENTATION  –  you usually do not need to edit below this line
# =====================================================================

# ── helpers ──────────────────────────────────────────────────────────

def _get_total_iters() -> int:
    if not CURRICULUM:
        return 0
    return max(p.iters[1] for p in CURRICULUM)


def _get_phase(global_iter: int) -> Optional[CurriculumPhase]:
    for phase in CURRICULUM:
        if phase.iters[0] <= global_iter < phase.iters[1]:
            return phase
    return None


def _sample_config(phase: CurriculumPhase, rng: random.Random) -> ScenarioConfig:
    cfg = phase.config
    if isinstance(cfg, Fixed):
        return cfg.config
    if isinstance(cfg, RandomChoice):
        return rng.choice(cfg.choices)
    if isinstance(cfg, RandomRange):
        return ScenarioConfig(
            n_strikers=rng.randint(*cfg.n_strikers),
            n_jammers=rng.randint(*cfg.n_jammers),
            n_known_targets=rng.randint(*cfg.n_known_targets),
            n_unknown_targets=rng.randint(*cfg.n_unknown_targets),
            n_known_radars=rng.randint(*cfg.n_known_radars),
            n_unknown_radars=rng.randint(*cfg.n_unknown_radars),
            radar_kill_probability=rng.uniform(*cfg.radar_kill_probability),
        )
    raise TypeError(f"Unknown config type: {type(cfg)}")


def _scenario_to_env_cfg(
    scenario: ScenarioConfig,
    max_steps: int,
    reward_cfg: RewardConfig,
    fofe_cfg: Optional[FOFEConfig] = None,
) -> EnvConfig:
    env_cfg = EnvConfig(
        n_strikers=scenario.n_strikers,
        n_jammers=scenario.n_jammers,
        n_known_targets=scenario.n_known_targets,
        n_unknown_targets=scenario.n_unknown_targets,
        n_known_radars=scenario.n_known_radars,
        n_unknown_radars=scenario.n_unknown_radars,
        max_steps=max_steps,
        radar_kill_probability=scenario.radar_kill_probability,
        reward_config=copy.deepcopy(reward_cfg),
    )
    if fofe_cfg is not None:
        env_cfg._use_fofe = fofe_cfg.use_fofe
    return env_cfg


def _next_eval_boundary(global_iter: int, eval_every: int) -> int:
    if eval_every <= 0:
        return 10**9
    return ((global_iter // eval_every) + 1) * eval_every


def _merge_logs(dst: Dict[str, List[float]], src: Dict[str, List[float]]) -> None:
    for key, values in src.items():
        dst.setdefault(key, [])
        dst[key].extend(values)


# ── checkpoint adaptation ────────────────────────────────────────────

def _adapt_checkpoint_for_stage(
    checkpoint: Optional[Dict[str, Any]],
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    net_cfg: NetworkConfig,
    fofe_cfg: Optional[FOFEConfig] = None,
) -> Optional[Dict[str, Any]]:
    """Adapt checkpoint weights for a (possibly changed) env configuration.

    - Actor policy: always strict load (architecture is config-invariant).
    - Critic: strict when FOFE is on (shapes invariant); partial shape-
      matching when FOFE is off (critic input may change with entity count).
    """
    if checkpoint is None:
        return None

    temp_env = build_env(env_cfg, ppo_cfg)
    temp_policy = make_combined_policy(
        temp_env, hidden=net_cfg.actor_hidden, depth=net_cfg.depth, fofe_cfg=fofe_cfg,
    )
    temp_critic = make_combined_critic(
        temp_env, hidden=net_cfg.critic_hidden, depth=net_cfg.depth, fofe_cfg=fofe_cfg,
    )

    src_policy = checkpoint.get("policy_state_dict")
    src_critic = checkpoint.get("critic_state_dict")

    # ── Policy: always strict ────────────────────────────────────────
    dst_policy = temp_policy.state_dict()
    matched_policy = 0
    if isinstance(src_policy, dict):
        temp_policy.load_state_dict(src_policy, strict=True)
        dst_policy = temp_policy.state_dict()
        matched_policy = len([v for v in dst_policy.values() if isinstance(v, torch.Tensor)])

    # ── Critic: try strict first, fall back to partial shape-matching ──
    dst_critic = temp_critic.state_dict()
    matched_critic = 0
    critic_mode = "skip"
    if isinstance(src_critic, dict):
        try:
            temp_critic.load_state_dict(src_critic, strict=True)
            dst_critic = temp_critic.state_dict()
            matched_critic = len(dst_critic)
            critic_mode = "strict"
        except RuntimeError:
            # Shape mismatch (e.g. value_head changes with agent count) —
            # copy only the parameters whose shapes match exactly.
            for key, tensor in src_critic.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                if (key in dst_critic
                        and isinstance(dst_critic[key], torch.Tensor)
                        and tuple(dst_critic[key].shape) == tuple(tensor.shape)):
                    dst_critic[key] = tensor.detach().to(
                        dst_critic[key].device, dtype=dst_critic[key].dtype,
                    )
                    matched_critic += 1
            critic_mode = "partial"

    print(
        f"  Checkpoint adapt: policy={matched_policy} tensors (strict), "
        f"critic={matched_critic}/{len(dst_critic)} ({critic_mode})"
    )

    return {
        "policy_state_dict": dst_policy,
        "critic_state_dict": dst_critic,
        "reward_normalizer_state_dict": checkpoint.get("reward_normalizer_state_dict"),
    }


# ── multi-scenario evaluation ────────────────────────────────────────

def _run_multi_scenario_eval(
    checkpoint: Dict[str, Any],
    eval_scenarios: List[ScenarioConfig],
    reward_cfg: RewardConfig,
    ppo_template: PPOConfig,
    net_cfg: NetworkConfig,
    fofe_cfg: Optional[FOFEConfig],
    max_steps: int,
    n_eval_episodes: int,
) -> Dict[str, Dict[str, float]]:
    """Evaluate current policy on every scenario. Returns {label: metrics}."""
    results: Dict[str, Dict[str, float]] = {}

    for scenario in eval_scenarios:
        env_cfg = _scenario_to_env_cfg(scenario, max_steps, reward_cfg, fofe_cfg)
        adapted = _adapt_checkpoint_for_stage(checkpoint, env_cfg, ppo_template, net_cfg, fofe_cfg)
        if adapted is None:
            continue

        eval_env = build_env(env_cfg, ppo_template)
        eval_policy = make_combined_policy(
            eval_env, hidden=net_cfg.actor_hidden, depth=net_cfg.depth, fofe_cfg=fofe_cfg,
        )
        eval_policy.load_state_dict(adapted["policy_state_dict"], strict=True)
        eval_policy = eval_policy.to(ppo_template.device)

        metrics = evaluate_current_policy(
            eval_policy, env_cfg, ppo_template, n_eval_episodes=n_eval_episodes,
        )

        results[scenario.label] = {
            "completion_rate": float(metrics["eval_task_completion_rate"]),
            "survival_rate": float(metrics["eval_survival_rate"]),
            "mean_duration": float(metrics["eval_mean_duration"]),
            "mean_reward": float(metrics["eval_mean_episode_total_reward"]),
        }

    return results


def _print_eval_results(global_iter: int, results: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{'=' * 70}")
    print(f"  Multi-scenario evaluation @ global iteration {global_iter}")
    print(f"{'=' * 70}")
    hdr = ("Scenario", "Completion", "Survival", "Duration", "Reward")
    print(f"  {hdr[0]:<24} {hdr[1]:>10} {hdr[2]:>10} {hdr[3]:>10} {hdr[4]:>10}")
    print(f"  {'-' * 68}")

    comp_v, surv_v, dur_v, rew_v = [], [], [], []
    for label, m in results.items():
        print(
            f"  {label:<24} "
            f"{m['completion_rate']:>10.4f} "
            f"{m['survival_rate']:>10.4f} "
            f"{m['mean_duration']:>10.2f} "
            f"{m['mean_reward']:>10.3f}"
        )
        comp_v.append(m["completion_rate"])
        surv_v.append(m["survival_rate"])
        dur_v.append(m["mean_duration"])
        rew_v.append(m["mean_reward"])

    if comp_v:
        n = len(comp_v)
        print(f"  {'-' * 68}")
        print(
            f"  {'AVERAGE':<24} "
            f"{sum(comp_v)/n:>10.4f} "
            f"{sum(surv_v)/n:>10.4f} "
            f"{sum(dur_v)/n:>10.2f} "
            f"{sum(rew_v)/n:>10.3f}"
        )
    print()


# ── shape-consistency guard ──────────────────────────────────────────

def _assert_actor_policy_shape_consistency(
    ppo_template: PPOConfig,
    net_cfg: NetworkConfig,
    fofe_cfg: Optional[FOFEConfig],
    env_cfgs: List[EnvConfig],
) -> None:
    if not env_cfgs:
        return
    base_env = build_env(env_cfgs[0], ppo_template)
    base_policy = make_combined_policy(
        base_env, hidden=net_cfg.actor_hidden, depth=net_cfg.depth, fofe_cfg=fofe_cfg,
    )
    base_shapes = {
        k: tuple(v.shape)
        for k, v in base_policy.state_dict().items()
        if isinstance(v, torch.Tensor)
    }
    for idx, cfg in enumerate(env_cfgs[1:], start=2):
        env_i = build_env(cfg, ppo_template)
        pol_i = make_combined_policy(
            env_i, hidden=net_cfg.actor_hidden, depth=net_cfg.depth, fofe_cfg=fofe_cfg,
        )
        shapes_i = {
            k: tuple(v.shape)
            for k, v in pol_i.state_dict().items()
            if isinstance(v, torch.Tensor)
        }
        if shapes_i != base_shapes:
            raise RuntimeError(
                f"Actor policy shape mismatch at config #{idx}: "
                f"S={cfg.n_strikers} J={cfg.n_jammers} "
                f"T={cfg.n_targets} R={cfg.n_radars}."
            )


# ── dashboard plotting ───────────────────────────────────────────────

def plot_curriculum_dashboard(
    training_logs: Dict[str, List[float]],
    eval_history: Dict[int, Dict[str, Dict[str, float]]],
    curriculum: List[CurriculumPhase],
) -> None:
    """Plot curriculum training dashboard with per-scenario eval metrics."""

    def _plot_valid(ax, x, y, label, **kwargs):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(y)
        if not np.any(valid):
            return
        ax.plot(x[valid], y[valid], marker="o", markersize=2, label=label, **kwargs)

    def _add_phase_shading(ax, curriculum_phases):
        bg = ["#e8f0fe", "#fef7e0", "#e8fce8", "#fce8e8", "#f0e8fc",
              "#fce8f0", "#e0f7fe", "#fefce0"]
        for i, phase in enumerate(curriculum_phases):
            ax.axvspan(
                phase.iters[0], phase.iters[1],
                alpha=0.18, color=bg[i % len(bg)], zorder=0,
            )

    eval_iters = sorted(eval_history.keys())
    if not eval_iters:
        plot_training(training_logs)
        return

    first_eval = next(iter(eval_history.values()))
    scenario_labels = list(first_eval.keys())

    fig, axes = plt.subplots(3, 3, figsize=(26, 17))

    # ── Row 0 col 0: Training reward ─────────────────────────────────
    ax = axes[0, 0]
    if "train_mean_episode_total_reward" in training_logs:
        iters = np.arange(1, len(training_logs["train_mean_episode_total_reward"]) + 1)
        _plot_valid(ax, iters, training_logs["train_mean_episode_total_reward"], "train_reward")
    _add_phase_shading(ax, curriculum)
    ax.set_title("Training Episode Reward")
    ax.set_xlabel("Global Iteration")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Row 0 col 1: Policy + Value loss ─────────────────────────────
    ax = axes[0, 1]
    for key, lbl, clr in [
        ("striker_loss_policy", "striker_pi", "tab:blue"),
        ("striker_loss_value", "striker_V", "tab:orange"),
        ("jammer_loss_policy", "jammer_pi", "tab:green"),
        ("jammer_loss_value", "jammer_V", "tab:red"),
    ]:
        if key in training_logs:
            iters = np.arange(1, len(training_logs[key]) + 1)
            _plot_valid(ax, iters, training_logs[key], lbl, color=clr)
    _add_phase_shading(ax, curriculum)
    ax.set_title("Policy & Value Loss")
    ax.set_xlabel("Global Iteration")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Row 0 col 2: Entropy / KL ────────────────────────────────────
    ax = axes[0, 2]
    for key, lbl, clr in [
        ("striker_entropy", "striker_H", "tab:green"),
        ("jammer_entropy", "jammer_H", "tab:olive"),
        ("striker_approx_kl", "striker_KL", "tab:red"),
        ("jammer_approx_kl", "jammer_KL", "tab:pink"),
    ]:
        if key in training_logs:
            iters = np.arange(1, len(training_logs[key]) + 1)
            _plot_valid(ax, iters, training_logs[key], lbl, color=clr)
    _add_phase_shading(ax, curriculum)
    ax.set_title("Entropy & KL Divergence")
    ax.set_xlabel("Global Iteration")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Row 1-2: Multi-scenario eval metrics ─────────────────────────
    metric_axes = [
        (axes[1, 0], "completion_rate", "Task Completion Rate"),
        (axes[1, 1], "survival_rate", "Survival Rate"),
        (axes[1, 2], "mean_duration", "Mean Duration"),
        (axes[2, 0], "mean_reward", "Mean Reward"),
    ]

    for ax, metric_key, title in metric_axes:
        for lbl in scenario_labels:
            vals = [eval_history[ei][lbl][metric_key] for ei in eval_iters]
            _plot_valid(ax, eval_iters, vals, lbl)

        avg = []
        for ei in eval_iters:
            sv = [eval_history[ei][lbl][metric_key] for lbl in scenario_labels]
            finite = [v for v in sv if math.isfinite(v)]
            avg.append(sum(finite) / len(finite) if finite else float("nan"))
        _plot_valid(
            ax, eval_iters, avg, "AVERAGE",
            linewidth=2.5, color="black", linestyle="--",
        )
        _add_phase_shading(ax, curriculum)
        ax.set_title(f"Eval: {title}")
        ax.set_xlabel("Global Iteration")
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)

    # ── Row 2 col 1: Curriculum phase timeline ────────────────────────
    ax = axes[2, 1]
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
               "tab:brown", "tab:cyan", "tab:olive"]
    for i, phase in enumerate(curriculum):
        clr = palette[i % len(palette)]
        ax.barh(
            0, phase.iters[1] - phase.iters[0], left=phase.iters[0],
            height=0.5, color=clr, alpha=0.7, edgecolor="black",
        )
        mid = (phase.iters[0] + phase.iters[1]) / 2
        cfg = phase.config
        if isinstance(cfg, Fixed):
            desc = f"{phase.name}\n{cfg.config.label}"
        elif isinstance(cfg, RandomChoice):
            desc = f"{phase.name}\nRndChoice({len(cfg.choices)})"
        elif isinstance(cfg, RandomRange):
            desc = f"{phase.name}\nRndRange"
        else:
            desc = phase.name
        ax.text(mid, 0, desc, ha="center", va="center", fontsize=6, fontweight="bold")
    ax.set_title("Curriculum Phases")
    ax.set_xlabel("Global Iteration")
    ax.set_yticks([])

    # ── Row 2 col 2: Time per iteration ───────────────────────────────
    ax = axes[2, 2]
    if "iter_time_s" in training_logs:
        iters = np.arange(1, len(training_logs["iter_time_s"]) + 1)
        _plot_valid(ax, iters, training_logs["iter_time_s"], "total", color="tab:blue")
    if "iter_time_excl_eval_s" in training_logs:
        iters = np.arange(1, len(training_logs["iter_time_excl_eval_s"]) + 1)
        _plot_valid(ax, iters, training_logs["iter_time_excl_eval_s"], "train only", color="tab:orange")
    _add_phase_shading(ax, curriculum)
    ax.set_title("Time per Iteration (s)")
    ax.set_xlabel("Global Iteration")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Curriculum MAPPO Training Dashboard", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# =====================================================================
#  CLI  (training hyper-parameters; curriculum is defined above)
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    env_d = EnvConfig()
    ppo_d = PPOConfig()
    net_d = NetworkConfig()
    rew_d = RewardConfig()

    p = argparse.ArgumentParser(
        description="Curriculum MAPPO runner.  "
        "Edit CURRICULUM / EVAL_SCENARIOS at the top of this file to configure.",
    )

    # PPO
    p.add_argument("--num_envs", type=int, default=ppo_d.num_envs)
    p.add_argument("--max_steps", type=int, default=env_d.max_steps)
    p.add_argument("--num_epochs", type=int, default=ppo_d.num_epochs)
    p.add_argument("--minibatch_size", type=int, default=ppo_d.minibatch_size)
    p.add_argument("--actor_lr", type=float, default=ppo_d.actor_lr)
    p.add_argument("--critic_lr", type=float, default=ppo_d.critic_lr)
    p.add_argument("--clip_eps", type=float, default=ppo_d.clip_eps)
    p.add_argument("--entropy_coef", type=float, default=ppo_d.entropy_coef)
    p.add_argument("--normalize_rewards", action=argparse.BooleanOptionalAction, default=ppo_d.normalize_rewards)
    p.add_argument("--seed", type=int, default=ppo_d.seed)
    p.add_argument("--log_every", type=int, default=ppo_d.log_every)

    # Network
    p.add_argument("--actor_hidden", type=int, default=net_d.actor_hidden)
    p.add_argument("--critic_hidden", type=int, default=net_d.critic_hidden)
    p.add_argument("--depth", type=int, default=net_d.depth)

    # Rewards
    p.add_argument("--target_destroyed", type=float, default=rew_d.target_destroyed)
    p.add_argument("--agent_destroyed", type=float, default=rew_d.agent_destroyed)
    p.add_argument("--timestep_penalty", type=float, default=rew_d.timestep_penalty)
    p.add_argument("--team_spirit", type=float, default=rew_d.team_spirit)
    p.add_argument("--striker_progress_scale", type=float, default=rew_d.striker_progress_scale)
    p.add_argument("--jammer_progress_scale", type=float, default=rew_d.jammer_progress_scale)
    p.add_argument("--jammer_jam_bonus", type=float, default=rew_d.jammer_jam_bonus)

    # Save / load
    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--save_name", type=str, default="curriculum_mappo.pt")
    p.add_argument("--load_checkpoint", type=str, default=None)
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--no_animate", action="store_true")

    # CLI overrides for the top-of-file eval settings
    p.add_argument("--eval_every", type=int, default=None,
                   help="Override EVAL_EVERY (default: use value at top of file)")
    p.add_argument("--eval_episodes", type=int, default=None,
                   help="Override EVAL_EPISODES (default: use value at top of file)")

    return p


# =====================================================================
#  MAIN
# =====================================================================

def main() -> None:
    args = build_parser().parse_args()

    eval_every = args.eval_every if args.eval_every is not None else EVAL_EVERY
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else EVAL_EPISODES
    fofe_cfg = FOFE_CONFIG

    # ── validate curriculum ──────────────────────────────────────────
    total_iters = _get_total_iters()
    if total_iters <= 0:
        raise RuntimeError("CURRICULUM is empty or has no iterations.")
    for i, phase in enumerate(CURRICULUM):
        if phase.iters[0] >= phase.iters[1]:
            raise ValueError(f"Phase '{phase.name}' has empty range {phase.iters}")
        if i > 0 and phase.iters[0] != CURRICULUM[i - 1].iters[1]:
            raise ValueError(
                f"Gap/overlap between '{CURRICULUM[i-1].name}' (ends {CURRICULUM[i-1].iters[1]}) "
                f"and '{phase.name}' (starts {phase.iters[0]})"
            )
    if not EVAL_SCENARIOS:
        raise ValueError("EVAL_SCENARIOS must not be empty.")

    # ── build shared configs ─────────────────────────────────────────
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

    # ── print plan ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CURRICULUM PLAN")
    print("=" * 70)
    print(f"  Total global iterations : {total_iters}")
    print(f"  Multi-scenario eval     : every {eval_every} iters, "
          f"{eval_episodes} episodes/scenario, {len(EVAL_SCENARIOS)} scenarios")
    print()
    for phase in CURRICULUM:
        cfg = phase.config
        if isinstance(cfg, Fixed):
            desc = f"Fixed({cfg.config.label})"
        elif isinstance(cfg, RandomChoice):
            desc = f"RandomChoice([{', '.join(c.label for c in cfg.choices)}])"
        elif isinstance(cfg, RandomRange):
            desc = (
                f"RandomRange(S={cfg.n_strikers}, J={cfg.n_jammers}, "
                f"kT={cfg.n_known_targets}, uT={cfg.n_unknown_targets}, "
                f"kR={cfg.n_known_radars}, uR={cfg.n_unknown_radars}, "
                f"p={cfg.radar_kill_probability})"
            )
        else:
            desc = str(cfg)
        print(f"  [{phase.iters[0]:4d}, {phase.iters[1]:4d})  {phase.name:25s} {desc}")
    print()
    print("  Eval scenarios:")
    for sc in EVAL_SCENARIOS:
        print(f"    - {sc.label}")
    print("=" * 70)

    # ── shape-consistency check ──────────────────────────────────────
    shape_check_cfgs: List[EnvConfig] = []
    # eval scenarios
    for sc in EVAL_SCENARIOS:
        shape_check_cfgs.append(_scenario_to_env_cfg(sc, args.max_steps, reward_cfg, fofe_cfg))
    # curriculum corners
    for phase in CURRICULUM:
        cfg = phase.config
        if isinstance(cfg, Fixed):
            shape_check_cfgs.append(
                _scenario_to_env_cfg(cfg.config, args.max_steps, reward_cfg, fofe_cfg),
            )
        elif isinstance(cfg, RandomChoice):
            for c in cfg.choices:
                shape_check_cfgs.append(
                    _scenario_to_env_cfg(c, args.max_steps, reward_cfg, fofe_cfg),
                )
        elif isinstance(cfg, RandomRange):
            for s in cfg.n_strikers:
                for j in cfg.n_jammers:
                    for kt in cfg.n_known_targets:
                        for kr in cfg.n_known_radars:
                            shape_check_cfgs.append(_scenario_to_env_cfg(
                                ScenarioConfig(
                                    n_strikers=s, n_jammers=j,
                                    n_known_targets=kt, n_known_radars=kr,
                                ),
                                args.max_steps, reward_cfg, fofe_cfg,
                            ))
    _assert_actor_policy_shape_consistency(ppo_template, net_cfg, fofe_cfg, shape_check_cfgs)
    print("Actor shape consistency check: PASSED")

    # ── optional initial checkpoint ──────────────────────────────────
    incoming_checkpoint: Optional[Dict[str, Any]] = None
    if args.load_checkpoint:
        incoming_checkpoint = torch.load(args.load_checkpoint, map_location=ppo_template.device)
        print(f"Loaded initial checkpoint: {args.load_checkpoint}")

    # ── training state ───────────────────────────────────────────────
    all_training_logs: Dict[str, List[float]] = {}
    eval_history: Dict[int, Dict[str, Dict[str, float]]] = {}
    iter_configs: List[Dict[str, Any]] = []
    rng = random.Random(args.seed + 7777)

    # ==================================================================
    #  MAIN CURRICULUM LOOP
    # ==================================================================
    global_iter = 0
    while global_iter < total_iters:
        phase = _get_phase(global_iter)
        if phase is None:
            raise RuntimeError(f"No curriculum phase for global iteration {global_iter}")

        # ── determine batch size and scenario config ─────────────────
        if isinstance(phase.config, Fixed):
            # batch consecutive Fixed iterations, split at eval boundaries
            phase_end = phase.iters[1]
            next_eval = _next_eval_boundary(global_iter, eval_every)
            batch_end = min(phase_end, next_eval, total_iters)
            n_batch = batch_end - global_iter
            scenario = phase.config.config
        else:
            # RandomChoice / RandomRange: sample fresh each iteration
            n_batch = 1
            scenario = _sample_config(phase, rng)

        # record which config each global iteration used
        for _ in range(n_batch):
            iter_configs.append({
                "phase": phase.name,
                "config": scenario.label,
                "n_strikers": scenario.n_strikers,
                "n_jammers": scenario.n_jammers,
                "n_known_targets": scenario.n_known_targets,
                "n_unknown_targets": scenario.n_unknown_targets,
                "n_known_radars": scenario.n_known_radars,
                "n_unknown_radars": scenario.n_unknown_radars,
                "radar_kill_probability": scenario.radar_kill_probability,
            })

        # ── build env / ppo configs ──────────────────────────────────
        env_cfg = _scenario_to_env_cfg(scenario, args.max_steps, reward_cfg, fofe_cfg)
        ppo_cfg = PPOConfig(
            num_envs=ppo_template.num_envs,
            n_iters=n_batch,
            num_epochs=ppo_template.num_epochs,
            minibatch_size=ppo_template.minibatch_size,
            actor_lr=ppo_template.actor_lr,
            critic_lr=ppo_template.critic_lr,
            clip_eps=ppo_template.clip_eps,
            entropy_coef=ppo_template.entropy_coef,
            normalize_rewards=ppo_template.normalize_rewards,
            seed=ppo_template.seed + global_iter,
            log_every=ppo_template.log_every,
        )
        # propagate global offset so trainer.py prints correct iter numbers
        ppo_cfg.iteration_offset = global_iter

        exp_cfg = ExperimentConfig(
            env=env_cfg, ppo=ppo_cfg, net=net_cfg, fofe=fofe_cfg,
        ).finalize()

        # ── adapt checkpoint ─────────────────────────────────────────
        adapted = _adapt_checkpoint_for_stage(
            incoming_checkpoint, exp_cfg.env, exp_cfg.ppo, exp_cfg.net, fofe_cfg,
        )

        print(
            f"\n--- Global iter [{global_iter}, {global_iter + n_batch}) | "
            f"Phase: {phase.name} | Config: {scenario.label} | "
            f"n_iters={n_batch} ---"
        )

        # ── train ────────────────────────────────────────────────────
        _base_env, policy, critic, logs, reward_normalizer = train_mappo(
            exp_cfg.env, exp_cfg.ppo, exp_cfg.net, fofe_cfg, adapted,
        )

        _merge_logs(all_training_logs, logs)

        # ── update checkpoint ────────────────────────────────────────
        incoming_checkpoint = {
            "policy_state_dict": policy.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "reward_normalizer_state_dict": (
                reward_normalizer.state_dict() if reward_normalizer is not None else None
            ),
        }

        global_iter += n_batch

        # ── multi-scenario evaluation ────────────────────────────────
        if eval_every > 0 and global_iter % eval_every == 0:
            eval_results = _run_multi_scenario_eval(
                checkpoint=incoming_checkpoint,
                eval_scenarios=EVAL_SCENARIOS,
                reward_cfg=reward_cfg,
                ppo_template=ppo_template,
                net_cfg=net_cfg,
                fofe_cfg=fofe_cfg,
                max_steps=args.max_steps,
                n_eval_episodes=eval_episodes,
            )
            eval_history[global_iter] = eval_results
            _print_eval_results(global_iter, eval_results)

    # ── final evaluation (if not already on a boundary) ──────────────
    if total_iters not in eval_history:
        eval_results = _run_multi_scenario_eval(
            checkpoint=incoming_checkpoint,
            eval_scenarios=EVAL_SCENARIOS,
            reward_cfg=reward_cfg,
            ppo_template=ppo_template,
            net_cfg=net_cfg,
            fofe_cfg=fofe_cfg,
            max_steps=args.max_steps,
            n_eval_episodes=eval_episodes,
        )
        eval_history[total_iters] = eval_results
        _print_eval_results(total_iters, eval_results)

    # ── save ─────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / args.save_name
    torch.save(
        {
            "policy_state_dict": incoming_checkpoint["policy_state_dict"],
            "critic_state_dict": incoming_checkpoint["critic_state_dict"],
            "reward_normalizer_state_dict": incoming_checkpoint["reward_normalizer_state_dict"],
            "net_cfg": net_cfg,
            "ppo_template": ppo_template,
            "fofe_cfg": fofe_cfg,
            "reward_cfg": reward_cfg,
            "training_logs": all_training_logs,
            "eval_history": eval_history,
            "iter_configs": iter_configs,
            "curriculum_phases": [
                {"name": p.name, "iters": p.iters, "config_type": type(p.config).__name__}
                for p in CURRICULUM
            ],
            "eval_scenario_labels": [sc.label for sc in EVAL_SCENARIOS],
        },
        save_path,
    )
    print(f"\nSaved curriculum checkpoint to: {save_path}")

    # ── dashboard ────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            plot_curriculum_dashboard(all_training_logs, eval_history, CURRICULUM)
        except Exception as exc:
            print(f"plot warning (continuing): {type(exc).__name__}: {exc}")

    # ── animation ────────────────────────────────────────────────────
    if not args.no_animate:
        try:
            n_rollouts = 5
            total_vis = n_rollouts * len(EVAL_SCENARIOS)
            print(f"\nAnimating: {n_rollouts} rollouts x {len(EVAL_SCENARIOS)} eval scenarios "
                  f"= {total_vis} total")
            counter = 0
            for scenario in EVAL_SCENARIOS:
                anim_env_cfg = _scenario_to_env_cfg(
                    scenario, args.max_steps, reward_cfg, fofe_cfg,
                )
                adapted = _adapt_checkpoint_for_stage(
                    incoming_checkpoint, anim_env_cfg, ppo_template, net_cfg, fofe_cfg,
                )
                if adapted is None:
                    continue
                anim_env = build_env(anim_env_cfg, ppo_template)
                anim_policy = make_combined_policy(
                    anim_env, hidden=net_cfg.actor_hidden,
                    depth=net_cfg.depth, fofe_cfg=fofe_cfg,
                )
                anim_policy.load_state_dict(adapted["policy_state_dict"], strict=True)
                anim_policy = anim_policy.to(ppo_template.device)

                for r in range(n_rollouts):
                    tester = TestRunner(
                        anim_policy, env_cfg=anim_env_cfg,
                        device=ppo_template.device, seed=999 + r,
                    )
                    frames = tester.rollout()
                    animate_rollout(frames, tester.env)
                    counter += 1
                    print(
                        f"  Rollout {counter}/{total_vis} | "
                        f"{scenario.label} run {r + 1}/{n_rollouts} | "
                        f"{len(frames)} frames"
                    )
        except Exception as exc:
            print(f"animate warning (continuing): {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
