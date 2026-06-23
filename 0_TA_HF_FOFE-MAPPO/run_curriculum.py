"""Curriculum runner for FOFE-MAPPO with per-environment domain randomization.

To run from prev checkpoint: 
cd "0_TA_HF_FOFE-MAPPO"
python run_curriculum.py --load_checkpoint runs/curriculum_mappo.pt
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Force Inductor to compile Triton kernels in the main process instead of a
# subprocess pool. Works around a Triton bug where worker subprocesses return
# an asm dict missing the 'cubin' key on some Linux + CUDA toolchains, which
# raises `KeyError: 'cubin'` during the first PPO update. setdefault lets the
# user override with `TORCHINDUCTOR_COMPILE_THREADS=N python ...` if desired.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

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

import numpy as np
import torch
import matplotlib.pyplot as plt

from .config import (
    DomainRandomization, EnvConfig, EnvExtensionsConfig, ExperimentConfig,
    FOFEConfig, NetworkConfig, PPOConfig,
)
from .nlr_style import (
    apply_nlr_style,
    NLR_PRIMARY, NLR_SECONDARY, NLR_ACCENT, NLR_GRAY, NLR_DARKGRAY,
    NLR_LIGHTBLUE_50, NLR_LIGHTBLUE_20, NLR_TERRA_50,
)
from .rewards import RewardConfig
from .trainer import build_env, train_mappo
from .visualization import TestRunner, animate_rollout
from .HF_visualization import HFTestRunner, hf_animate_rollout

apply_nlr_style()


# =====================================================================
#  CURRICULUM DEFINITION  —  edit this section freely
# =====================================================================
# Field rule (per section):
#   None        → inherit from config.py
#   scalar      → fixed for the section
#   (lo, hi)    → per-environment domain randomization, inclusive
# `scenario`: None → config; "S1" / "S2" → fixed for the section.

# Convenience type alias for the field rule above.
IntField   = Union[None, int, Tuple[int, int]]
FloatField = Union[None, float, Tuple[float, float]]


@dataclass
class CurriculumSection:
    """One contiguous stage of the curriculum.

    Sections run back-to-back; a single global iteration counter spans the
    whole curriculum (it never resets at section boundaries).
    """
    name: str
    n_iters: int

    # Team composition
    n_strikers:        IntField = None
    n_jammers:         IntField = None
    n_known_targets:   IntField = None
    n_unknown_targets: IntField = None
    n_known_radars:    IntField = None
    n_unknown_radars:  IntField = None

    # Episode / scenario / threat
    max_steps:              IntField        = None
    scenario:               Optional[str]   = None     # None | "S1" | "S2"  (fixed, not DR)
    radar_kill_probability: FloatField      = None

    # Inter-agent communication: None → inherit from config.py, True/False → fixed.
    communicate:            Optional[bool]  = None

    # Evaluation only (ignored during training): which checkpoint to evaluate
    # this scenario with. None → use the run's --checkpoint default. A bare name
    # (e.g. "BaselineV2.pt") resolves under runs/; an absolute path is used as-is.
    policy_file:            Optional[str]   = None


CURRICULUM: List[CurriculumSection] = [
    # ── Stage 1: the simplest possible case, fixed, to bootstrap behaviour ──
    CurriculumSection(
        name="Baseline 2sx4j 0.01k",
        n_iters=50,
        n_strikers=1, n_jammers=2,
        n_known_targets=2, n_unknown_targets=0,
        n_known_radars=6, n_unknown_radars=0,
        radar_kill_probability=0.01,
        scenario="S2"
    )
    # ── Stage 2: randomize the threat field per-env (targets + radars vary) ──
    # CurriculumSection(
    #     name="Baseline 2sx4j 0.025k",
    #     n_iters=0,
    #     n_strikers=1, n_jammers=2,
    #     n_known_targets=2, n_unknown_targets=0,
    #     n_known_radars=6, n_unknown_radars=0,
    #     radar_kill_probability=0.1,
    #     scenario="S2"
    # )
    # ── Stage 3: full per-env DR including team size + softer kills ──────────
#     CurriculumSection(
#         name="Baseline 2sx4j 0.05k",
#         n_iters=500,
#         n_strikers=2, n_jammers=4,
#         n_known_targets=1, n_unknown_targets=0,
#         n_known_radars=1, n_unknown_radars=0,
#         radar_kill_probability=0.05,
#         scenario="S2"
#     ), 
#     CurriculumSection(
#         name="Baseline 2sx4j 0.1k",
#         n_iters=500,
#         n_strikers=2, n_jammers=4,
#         n_known_targets=1, n_unknown_targets=0,
#         n_known_radars=1, n_unknown_radars=0,
#         radar_kill_probability=0.1,
#         scenario="S2"
#     ), 
#     CurriculumSection(
#         name="Baseline 2sx4j 0.25k",
#         n_iters=1000,
#         n_strikers=2, n_jammers=4,
#         n_known_targets=1, n_unknown_targets=0,
#         n_known_radars=1, n_unknown_radars=0,
#         radar_kill_probability=0.25,
#         scenario="S2"
#     )
]

# check if this is configured correctly. 
FOFE_CONFIG = FOFEConfig(use_fofe=False)


# =====================================================================
#  IMPLEMENTATION  —  you usually do not need to edit below this line
# =====================================================================

# ── field resolution: (value_for_EnvConfig, dr_range_or_None) ─────────

def _resolve_int(value: IntField, config_default: int) -> Tuple[int, Optional[Tuple[int, int]]]:
    if value is None:
        return int(config_default), None
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"DR range must be (lo, hi), got {value!r}")
        lo, hi = int(value[0]), int(value[1])
        if lo > hi:
            raise ValueError(f"DR range lo>hi: {value!r}")
        # EnvConfig count is the range MAX (tensors are allocated at this size).
        return hi, (lo, hi)
    return int(value), None


def _resolve_float(value: FloatField, config_default: float) -> Tuple[float, Optional[Tuple[float, float]]]:
    if value is None:
        return float(config_default), None
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"DR range must be (lo, hi), got {value!r}")
        lo, hi = float(value[0]), float(value[1])
        if lo > hi:
            raise ValueError(f"DR range lo>hi: {value!r}")
        return hi, (lo, hi)
    return float(value), None


def _section_to_env_cfg(
    section: CurriculumSection,
    defaults: EnvConfig,
    reward_cfg: RewardConfig,
    fofe_cfg: FOFEConfig,
) -> EnvConfig:
    """Build an EnvConfig (+ DomainRandomization) for a section.

    Count fields are set to the range maximum so the env allocates tensors big
    enough; the DomainRandomization carries the (lo, hi) ranges that drive the
    per-env activation masks at reset.
    """
    ns,  dr_ns  = _resolve_int(section.n_strikers,        defaults.n_strikers)
    nj,  dr_nj  = _resolve_int(section.n_jammers,         defaults.n_jammers)
    nkt, dr_nkt = _resolve_int(section.n_known_targets,   defaults.n_known_targets)
    nut, dr_nut = _resolve_int(section.n_unknown_targets, defaults.n_unknown_targets)
    nkr, dr_nkr = _resolve_int(section.n_known_radars,    defaults.n_known_radars)
    nur, dr_nur = _resolve_int(section.n_unknown_radars,  defaults.n_unknown_radars)
    ms,  dr_ms  = _resolve_int(section.max_steps,         defaults.max_steps)
    kp,  dr_kp  = _resolve_float(section.radar_kill_probability, defaults.radar_kill_probability)

    scenario = section.scenario if section.scenario is not None else defaults.scenario
    communicate = (
        section.communicate if section.communicate is not None
        else getattr(defaults, "communicate", True)
    )

    dr = DomainRandomization(
        n_strikers=dr_ns, n_jammers=dr_nj,
        n_known_targets=dr_nkt, n_unknown_targets=dr_nut,
        n_known_radars=dr_nkr, n_unknown_radars=dr_nur,
        radar_kill_probability=dr_kp, max_steps=dr_ms,
    )
    if not dr.active():
        dr = None

    env_cfg = EnvConfig(
        n_strikers=ns, n_jammers=nj,
        n_known_targets=nkt, n_unknown_targets=nut,
        n_known_radars=nkr, n_unknown_radars=nur,
        max_steps=ms, scenario=scenario,
        radar_kill_probability=kp,
        communicate=communicate,
        # Flat-MLP obs slots are constant across sections (obs_dim must match
        # for the carried policy/critic to transfer), so carry them from the
        # leading config defaults rather than per-section overrides.
        n_other_agent_obs_slots=getattr(defaults, "n_other_agent_obs_slots", 3),
        n_radar_obs_slots=getattr(defaults, "n_radar_obs_slots", 2),
        n_target_obs_slots=getattr(defaults, "n_target_obs_slots", 2),
        reward_config=copy.deepcopy(reward_cfg),
        dr=dr,
    )
    env_cfg._use_fofe = fofe_cfg.use_fofe
    return env_cfg


def _section_label(section: CurriculumSection, env_cfg: EnvConfig) -> str:
    """One-line human description of a section's resolved config."""
    def fmt(value, fixed):
        if isinstance(value, (tuple, list)):
            return f"[{value[0]}-{value[1]}]"
        return str(fixed)
    parts = [
        f"S{fmt(section.n_strikers, env_cfg.n_strikers)}",
        f"J{fmt(section.n_jammers, env_cfg.n_jammers)}",
        f"kT{fmt(section.n_known_targets, env_cfg.n_known_targets)}",
        f"kR{fmt(section.n_known_radars, env_cfg.n_known_radars)}",
    ]
    if section.n_unknown_targets is not None and (
        isinstance(section.n_unknown_targets, (tuple, list)) or section.n_unknown_targets > 0
    ):
        parts.append(f"uT{fmt(section.n_unknown_targets, env_cfg.n_unknown_targets)}")
    if section.n_unknown_radars is not None and (
        isinstance(section.n_unknown_radars, (tuple, list)) or section.n_unknown_radars > 0
    ):
        parts.append(f"uR{fmt(section.n_unknown_radars, env_cfg.n_unknown_radars)}")
    parts.append(f"kill{fmt(section.radar_kill_probability, env_cfg.radar_kill_probability)}")
    parts.append(f"steps{fmt(section.max_steps, env_cfg.max_steps)}")
    parts.append(env_cfg.scenario)
    parts.append(f"comm{'ON' if env_cfg.communicate else 'OFF'}")
    return " ".join(parts)


def _viz_env_cfg(env_cfg: EnvConfig) -> EnvConfig:
    """Concrete (non-DR) copy of a section's config for visualization.

    Drops the DR ranges so the rollout shows a clean scenario at the section's
    maximum entity counts (no masked-out / parked entities). The trained policy
    is count-invariant, and its role split matches these max counts."""
    viz = copy.deepcopy(env_cfg)
    viz.dr = None
    return viz


# ── log merging ──────────────────────────────────────────────────────

def _merge_logs(dst: Dict[str, List[float]], src: Dict[str, List[float]]) -> None:
    for key, values in src.items():
        dst.setdefault(key, [])
        dst[key].extend(values)


def _state_dict_to_cpu(sd: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Detach a state_dict to CPU and strip torch.compile's `_orig_mod.` prefix.

    Moving to CPU prevents the carried checkpoint from pinning VRAM while the
    next section rebuilds its env/policy/critic. Stripping `_orig_mod.` keeps
    checkpoints portable: on Linux+GPU the trainer compiles the actor/critic
    nets, which inserts `_orig_mod.` into every parameter key — that breaks
    strict loads into a fresh uncompiled policy (e.g. `_rebuild_policy` for
    rollouts, or resuming on Windows where Triton is unavailable)."""
    if sd is None:
        return None
    return {
        k.replace("_orig_mod.", ""): (v.detach().to("cpu") if torch.is_tensor(v) else v)
        for k, v in sd.items()
    }


# ── dashboard ─────────────────────────────────────────────────────────

def plot_curriculum_dashboard(
    logs: Dict[str, List[float]],
    section_bounds: List[Tuple[str, int, int]],
    save_path: Optional[str] = None,
) -> None:
    """Dashboard spanning the entire curriculum.

    section_bounds: list of (name, start_iter, end_iter) half-open ranges.
    """
    palette = [
        NLR_PRIMARY, NLR_ACCENT, NLR_SECONDARY, NLR_DARKGRAY,
        NLR_LIGHTBLUE_50, NLR_TERRA_50, NLR_GRAY, NLR_LIGHTBLUE_20,
    ]

    def _plot_valid(ax, x, y, label, **kwargs):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(y)
        if not np.any(valid):
            return
        ax.plot(x[valid], y[valid], marker="o", markersize=2, label=label, **kwargs)

    def _shade(ax):
        for i, (_, s, e) in enumerate(section_bounds):
            ax.axvspan(s, e, alpha=0.16, color=palette[i % len(palette)], zorder=0)

    fig, axes = plt.subplots(3, 3, figsize=(26, 17))

    # Row 0 col 0: training reward
    ax = axes[0, 0]
    key = "train_mean_episode_total_reward"
    if logs.get(key):
        x = np.arange(1, len(logs[key]) + 1)
        _plot_valid(ax, x, logs[key], "train_reward", color=NLR_PRIMARY)
    _shade(ax)
    ax.set_title("Training Episode Reward"); ax.set_xlabel("Global Iteration")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Row 0 col 1: losses
    ax = axes[0, 1]
    for k, lbl, clr in [
        ("striker_loss_policy", "striker_pi", NLR_PRIMARY),
        ("striker_loss_value", "striker_V", NLR_ACCENT),
        ("jammer_loss_policy", "jammer_pi", NLR_SECONDARY),
        ("jammer_loss_value", "jammer_V", NLR_TERRA_50),
    ]:
        if logs.get(k):
            _plot_valid(ax, np.arange(1, len(logs[k]) + 1), logs[k], lbl, color=clr)
    _shade(ax)
    ax.set_title("Policy & Value Loss"); ax.set_xlabel("Global Iteration")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Row 0 col 2: entropy / KL
    ax = axes[0, 2]
    for k, lbl, clr in [
        ("striker_entropy", "striker_H", NLR_PRIMARY),
        ("jammer_entropy", "jammer_H", NLR_SECONDARY),
        ("striker_approx_kl", "striker_KL", NLR_ACCENT),
        ("jammer_approx_kl", "jammer_KL", NLR_TERRA_50),
    ]:
        if logs.get(k):
            _plot_valid(ax, np.arange(1, len(logs[k]) + 1), logs[k], lbl, color=clr)
    _shade(ax)
    ax.set_title("Entropy & KL Divergence"); ax.set_xlabel("Global Iteration")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Rows 1-2: evaluation metrics (section-mirrored DR)
    eval_axes = [
        (axes[1, 0], "eval_task_completion_rate", "Eval: Task Completion Rate"),
        (axes[1, 1], "eval_survival_rate", "Eval: Survival Rate"),
        (axes[1, 2], "eval_mean_duration", "Eval: Mean Duration"),
        (axes[2, 0], "eval_mean_episode_total_reward", "Eval: Mean Reward"),
    ]
    for ax, key, title in eval_axes:
        if logs.get(key):
            _plot_valid(ax, np.arange(1, len(logs[key]) + 1), logs[key], key, color=NLR_PRIMARY)
        _shade(ax)
        ax.set_title(title); ax.set_xlabel("Global Iteration")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Row 2 col 1: section timeline
    ax = axes[2, 1]
    for i, (name, s, e) in enumerate(section_bounds):
        ax.barh(0, e - s, left=s, height=0.5, color=palette[i % len(palette)],
                alpha=0.8, edgecolor=NLR_DARKGRAY)
        ax.text((s + e) / 2, 0, name, ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_title("Curriculum Sections"); ax.set_xlabel("Global Iteration"); ax.set_yticks([])

    # Row 2 col 2: time per iter
    ax = axes[2, 2]
    for k, lbl, clr in [
        ("iter_time_s", "total", NLR_PRIMARY),
        ("iter_time_excl_eval_s", "train only", NLR_ACCENT),
    ]:
        if logs.get(k):
            _plot_valid(ax, np.arange(1, len(logs[k]) + 1), logs[k], lbl, color=clr)
    _shade(ax)
    ax.set_title("Time per Iteration (s)"); ax.set_xlabel("Global Iteration")
    ax.set_ylabel("seconds"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("Curriculum MAPPO Training Dashboard", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=120)
            print(f"Saved dashboard to: {save_path}")
        except Exception as exc:
            print(f"dashboard save warning (continuing): {type(exc).__name__}: {exc}")
    plt.close(fig)


# ── standalone plots ─────────────────────────────────────────────────

def _plot_valid_xy(ax, x, y, label, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    if not np.any(valid):
        return
    ax.plot(x[valid], y[valid], marker="o", markersize=2, label=label, **kwargs)


def _shade_sections(ax, section_bounds: List[Tuple[str, int, int]]) -> None:
    palette = [
        NLR_PRIMARY, NLR_ACCENT, NLR_SECONDARY, NLR_DARKGRAY,
        NLR_LIGHTBLUE_50, NLR_TERRA_50, NLR_GRAY, NLR_LIGHTBLUE_20,
    ]
    for i, (_, s, e) in enumerate(section_bounds):
        ax.axvspan(s, e, alpha=0.16, color=palette[i % len(palette)], zorder=0)


def _save_fig(fig, save_path: Optional[str], label: str) -> None:
    if not save_path:
        plt.close(fig); return
    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"Saved {label} to: {save_path}")
    except Exception as exc:
        print(f"{label} save warning (continuing): {type(exc).__name__}: {exc}")
    plt.close(fig)


def plot_curriculum_reward(
    logs: Dict[str, List[float]],
    section_bounds: List[Tuple[str, int, int]],
    save_path: Optional[str] = None,
) -> None:
    """Training episode reward over the full curriculum."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    key = "train_mean_episode_total_reward"
    if logs.get(key):
        x = np.arange(1, len(logs[key]) + 1)
        _plot_valid_xy(ax, x, logs[key], "train_reward", color=NLR_PRIMARY)
    _shade_sections(ax, section_bounds)
    ax.set_title("Training Episode Reward", fontsize=13, fontweight="bold")
    ax.set_xlabel("Global Iteration"); ax.set_ylabel("Mean Episode Reward")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save_path, "reward plot")


def plot_curriculum_eval_rates(
    logs: Dict[str, List[float]],
    section_bounds: List[Tuple[str, int, int]],
    save_path: Optional[str] = None,
) -> None:
    """Eval rates over the full curriculum: survival, task completion,
    targets destroyed — all bounded in [0, 1] so they share one axis."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    series = [
        ("eval_survival_rate",        "survival rate",         NLR_PRIMARY),
        ("eval_task_completion_rate", "task completion rate",  NLR_ACCENT),
        ("eval_targets_destroyed_rate", "targets destroyed rate", NLR_SECONDARY),
    ]
    for key, label, color in series:
        if logs.get(key):
            x = np.arange(1, len(logs[key]) + 1)
            _plot_valid_xy(ax, x, logs[key], label, color=color)
    _shade_sections(ax, section_bounds)
    ax.set_title("Eval: Survival / Completion / Targets-Destroyed Rates",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Global Iteration"); ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save_path, "eval-rates plot")


# =====================================================================
#  CLI
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    ppo_d = PPOConfig()
    net_d = NetworkConfig()
    rew_d = RewardConfig()

    p = argparse.ArgumentParser(
        description="Curriculum MAPPO runner with per-environment domain "
        "randomization. Edit the CURRICULUM list at the top of this file.",
    )
    # PPO / training
    p.add_argument("--num_envs", type=int, default=ppo_d.num_envs)
    p.add_argument("--num_epochs", type=int, default=ppo_d.num_epochs)
    p.add_argument("--minibatch_size", type=int, default=ppo_d.minibatch_size)
    p.add_argument("--actor_lr", type=float, default=ppo_d.actor_lr)
    p.add_argument("--critic_lr", type=float, default=ppo_d.critic_lr)
    p.add_argument("--clip_eps", type=float, default=ppo_d.clip_eps)
    p.add_argument("--entropy_coef", type=float, default=ppo_d.entropy_coef)
    p.add_argument("--normalize_rewards", action=argparse.BooleanOptionalAction, default=ppo_d.normalize_rewards)
    p.add_argument("--seed", type=int, default=ppo_d.seed)
    p.add_argument("--log_every", type=int, default=ppo_d.log_every,
                   help="Evaluation cadence (global iters) — eval mirrors the active section's DR.")
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
    # Save / load / output
    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--save_name", type=str, default="curriculum_mappo.pt")
    p.add_argument("--load_checkpoint", type=str, default=None)
    p.add_argument("--n_rollouts", type=int, default=15, help="Final-stage rollouts to animate.")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--no_animate", action="store_true")
    return p


# =====================================================================
#  MAIN
# =====================================================================

def main() -> None:
    args = build_parser().parse_args()

    if not CURRICULUM:
        raise RuntimeError("CURRICULUM is empty — define at least one section.")
    for sec in CURRICULUM:
        if sec.n_iters <= 0:
            raise ValueError(f"Section '{sec.name}' has n_iters={sec.n_iters} (must be > 0).")

    fofe_cfg = FOFE_CONFIG
    ext_cfg = EnvExtensionsConfig()                       # defaults from config.py
    hf_radar_cfg = ext_cfg.hf_radar if ext_cfg.use_hf_radar else None
    env_defaults = EnvConfig()                            # config.py leading values

    reward_cfg = RewardConfig(
        target_destroyed=args.target_destroyed,
        agent_destroyed=args.agent_destroyed,
        timestep_penalty=args.timestep_penalty,
        team_spirit=args.team_spirit,
        striker_progress_scale=args.striker_progress_scale,
        jammer_progress_scale=args.jammer_progress_scale,
        jammer_jam_bonus=args.jammer_jam_bonus,
    )
    net_cfg = NetworkConfig(
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        depth=args.depth,
    )

    # Resolve every section up front (also validates DR ranges via the env).
    resolved: List[Tuple[CurriculumSection, EnvConfig]] = [
        (sec, _section_to_env_cfg(sec, env_defaults, reward_cfg, fofe_cfg))
        for sec in CURRICULUM
    ]
    total_iters = sum(sec.n_iters for sec in CURRICULUM)

    # ── plan print-out ───────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  CURRICULUM PLAN")
    print("=" * 78)
    print(f"  FOFE encoding   : {'ENABLED' if fofe_cfg.use_fofe else 'DISABLED'}")
    print(f"  HF radar model  : {'ENABLED' if hf_radar_cfg is not None else 'DISABLED'}")
    print(f"  num_envs        : {args.num_envs}")
    print(f"  eval cadence    : every {args.log_every} iters (mirrors section DR)")
    print(f"  total iterations: {total_iters}")
    print("-" * 78)
    cursor = 0
    for sec, env_cfg in resolved:
        dr_tag = "DR" if env_cfg.dr is not None else "fixed"
        print(f"  [{cursor:4d},{cursor + sec.n_iters:4d})  {sec.name:20s} "
              f"({dr_tag})  {_section_label(sec, env_cfg)}")
        cursor += sec.n_iters
    print("=" * 78 + "\n")

    # ── optional initial checkpoint ──────────────────────────────────
    checkpoint: Optional[Dict[str, Any]] = None
    if args.load_checkpoint:
        device = PPOConfig().device
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        print(f"Loaded initial checkpoint: {args.load_checkpoint}")

    # ── run sections ─────────────────────────────────────────────────
    all_logs: Dict[str, List[float]] = {}
    section_bounds: List[Tuple[str, int, int]] = []
    global_iter = 0
    last_env_cfg: Optional[EnvConfig] = None

    for sec, env_cfg in resolved:
        ppo_cfg = PPOConfig(
            num_envs=args.num_envs,
            n_iters=sec.n_iters,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef,
            normalize_rewards=args.normalize_rewards,
            seed=args.seed + global_iter,
            log_every=args.log_every,
        )
        # Align the internal eval cadence + printed iter numbers to global iters.
        ppo_cfg.iteration_offset = global_iter

        exp_cfg = ExperimentConfig(
            env=env_cfg, ppo=ppo_cfg, net=net_cfg, fofe=fofe_cfg, ext=ext_cfg,
        ).finalize()

        print("\n" + "-" * 78)
        print(f"  SECTION '{sec.name}'  iters [{global_iter}, {global_iter + sec.n_iters})  "
              f"{_section_label(sec, env_cfg)}")
        print("-" * 78)

        base_env, policy, critic, logs, reward_normalizer = train_mappo(
            exp_cfg.env, exp_cfg.ppo, exp_cfg.net, fofe_cfg, checkpoint,
            hf_radar_cfg=hf_radar_cfg,
        )

        _merge_logs(all_logs, logs)
        section_bounds.append((sec.name, global_iter, global_iter + sec.n_iters))
        global_iter += sec.n_iters
        last_env_cfg = env_cfg

        # Carry weights forward on CPU (strict load works — FOFE nets are
        # count-invariant). Offloading to CPU lets us release this section's
        # GPU env/policy/critic/collector BEFORE the next section allocates,
        # otherwise both sections' models coexist in VRAM and OOM on small GPUs.
        checkpoint = {
            "policy_state_dict": _state_dict_to_cpu(policy.state_dict()),
            "critic_state_dict": _state_dict_to_cpu(critic.state_dict()),
            "reward_normalizer_state_dict": _state_dict_to_cpu(
                reward_normalizer.state_dict() if reward_normalizer is not None else None
            ),
        }
        del base_env, policy, critic, logs, reward_normalizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── save ─────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / args.save_name
    torch.save(
        {
            "policy_state_dict": checkpoint["policy_state_dict"],
            "critic_state_dict": checkpoint["critic_state_dict"],
            "reward_normalizer_state_dict": checkpoint["reward_normalizer_state_dict"],
            "net_cfg": net_cfg,
            "fofe_cfg": fofe_cfg,
            "ext_cfg": ext_cfg,
            "reward_cfg": reward_cfg,
            "training_logs": all_logs,
            "curriculum": [
                {"name": s.name, "n_iters": s.n_iters, "label": _section_label(s, c)}
                for s, c in resolved
            ],
            "section_bounds": section_bounds,
        },
        save_path,
    )
    print(f"\nSaved curriculum checkpoint to: {save_path}")

    # ── dashboard (spans the whole curriculum) ───────────────────────
    project_dir = Path(__file__).resolve().parent
    if not args.no_plot:
        try:
            plot_curriculum_dashboard(
                all_logs, section_bounds,
                save_path=str(project_dir / "plots" / "curriculum_dashboard.png"),
            )
        except Exception as exc:
            print(f"plot warning (continuing): {type(exc).__name__}: {exc}")
        try:
            plot_curriculum_reward(
                all_logs, section_bounds,
                save_path=str(project_dir / "plots" / "curriculum_reward.png"),
            )
        except Exception as exc:
            print(f"reward plot warning (continuing): {type(exc).__name__}: {exc}")
        try:
            plot_curriculum_eval_rates(
                all_logs, section_bounds,
                save_path=str(project_dir / "plots" / "curriculum_eval_rates.png"),
            )
        except Exception as exc:
            print(f"eval-rates plot warning (continuing): {type(exc).__name__}: {exc}")

    # ── final-stage rollouts ─────────────────────────────────────────
    if not args.no_animate and last_env_cfg is not None:
        try:
            vis_dir = project_dir / "visualisations"
            vis_dir.mkdir(parents=True, exist_ok=True)
            viz_cfg = _viz_env_cfg(last_env_cfg)          # concrete (non-DR) final config
            device = PPOConfig().device
            policy = _rebuild_policy(viz_cfg, net_cfg, fofe_cfg, hf_radar_cfg,
                                     checkpoint["policy_state_dict"], device)
            print(f"\nAnimating {args.n_rollouts} final-stage rollouts "
                  f"({_section_label(resolved[-1][0], last_env_cfg)})")
            for i in range(args.n_rollouts):
                gif_path = vis_dir / f"curriculum_final_{i + 1:02d}.gif"
                if hf_radar_cfg is not None:
                    tester = HFTestRunner(policy, env_cfg=viz_cfg, hf_cfg=hf_radar_cfg,
                                          device=device, seed=999 + i)
                    frames = tester.rollout()
                    hf_animate_rollout(frames, tester.env, save_path=str(gif_path))
                else:
                    tester = TestRunner(policy, env_cfg=viz_cfg, device=device, seed=999 + i)
                    frames = tester.rollout()
                    animate_rollout(frames, tester.env, save_path=str(gif_path))
                print(f"  rollout {i + 1}/{args.n_rollouts} ({len(frames)} frames) → {gif_path.name}")
        except Exception as exc:
            print(f"animate warning (continuing): {type(exc).__name__}: {exc}")


def _rebuild_policy(env_cfg, net_cfg, fofe_cfg, hf_radar_cfg, policy_state_dict, device):
    """Build a fresh policy sized to env_cfg and strict-load the trained weights.

    FOFE parameter shapes are count-invariant, so the load is exact; only the
    wrapper's role split (n_strikers / n_jammers) is taken from env_cfg."""
    from .models import make_combined_policy
    probe_ppo = PPOConfig(num_envs=1, device=device)
    env = build_env(env_cfg, probe_ppo, hf_radar_cfg=hf_radar_cfg)
    policy = make_combined_policy(env, hidden=net_cfg.actor_hidden,
                                  depth=net_cfg.depth, fofe_cfg=fofe_cfg)
    policy.load_state_dict(policy_state_dict, strict=True)
    return policy.to(device)


if __name__ == "__main__":
    main()
