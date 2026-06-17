from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon

from .config import EnvConfig
from .environment import StrikeEA2DEnv
from .nlr_style import (
    apply_nlr_style,
    NLR_PRIMARY,
    NLR_SECONDARY,
    NLR_ACCENT,
    NLR_GRAY,
    NLR_DARKGRAY,
    NLR_LIGHTBLUE_50,
    NLR_LIGHTBLUE_20,
    NLR_TERRA_50,
)

apply_nlr_style()

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


def _plot_valid(ax, series: List[float], label: str, **kwargs) -> None:
    y = np.asarray(series, dtype=float)
    if y.size == 0:
        return
    x = np.arange(1, y.size + 1)
    valid = np.isfinite(y)
    if not np.any(valid):
        return
    ax.plot(x[valid], y[valid], marker="o", markersize=2, label=label, **kwargs)


def _draw_episode_reward(ax, logs: Dict[str, List[float]]) -> None:
    if "train_mean_episode_total_reward" in logs:
        _plot_valid(ax, logs["train_mean_episode_total_reward"], "train_ep_return_total")
    for key in sorted(logs.keys()):
        if not key.startswith("train_component_"):
            continue
        y = np.asarray(logs[key], dtype=float)
        if y.size == 0:
            continue
        valid = np.isfinite(y)
        if not np.any(valid) or np.nanmax(np.abs(y[valid])) <= 1e-12:
            continue
        label = key.replace("train_component_", "")
        _plot_valid(ax, logs[key], label)
    ax.set_title("Training Episode Reward")
    ax.set_xlabel("Iteration")
    ax.legend(fontsize=6)
    ax.grid(True)


def _draw_policy_value_loss(ax, logs: Dict[str, List[float]]) -> None:
    if "striker_loss_policy" in logs:
        _plot_valid(ax, logs["striker_loss_policy"], "striker_policy_loss", color=NLR_PRIMARY)
    if "striker_loss_value" in logs:
        _plot_valid(ax, logs["striker_loss_value"], "striker_value_loss", color=NLR_ACCENT)
    if "jammer_loss_policy" in logs:
        _plot_valid(ax, logs["jammer_loss_policy"], "jammer_policy_loss", color=NLR_SECONDARY)
    if "jammer_loss_value" in logs:
        _plot_valid(ax, logs["jammer_loss_value"], "jammer_value_loss", color=NLR_TERRA_50)
    ax.set_title("Policy & Value Loss (Striker + Jammer)")
    ax.set_xlabel("Iteration")
    ax.legend()
    ax.grid(True)


def _draw_eval_kpi(ax, logs: Dict[str, List[float]]) -> None:
    if "eval_survival_rate" in logs:
        _plot_valid(ax, logs["eval_survival_rate"], "eval_survival_ratio")
    if "eval_task_completion_rate" in logs:
        _plot_valid(ax, logs["eval_task_completion_rate"], "eval_completion_ratio")
    if "eval_targets_destroyed_rate" in logs:
        _plot_valid(ax, logs["eval_targets_destroyed_rate"], "eval_targets_destroyed_ratio")
    ax.set_title("Eval Survival, Completion & Targets Destroyed")
    ax.set_xlabel("Iteration")
    ax.legend()
    ax.grid(True)


def _save_standalone_panels(
    logs: Dict[str, List[float]],
    save_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """Save three real, standalone figures (not crops): episode reward,
    policy/value loss, and eval KPI ratios.
    """
    specs = [
        ("episode_reward.png",     _draw_episode_reward),
        ("policy_value_loss.png",  _draw_policy_value_loss),
        ("eval_kpi.png",           _draw_eval_kpi),
    ]
    for fname, draw_fn in specs:
        fig_i, ax_i = plt.subplots(figsize=(8, 5))
        draw_fn(ax_i, logs)
        fig_i.tight_layout()
        out = save_path / fname
        fig_i.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig_i)
        print(f"Saved plot: {out}")


def plot_training(
    logs: Dict[str, List[float]],
    save_dir: Optional[str] = None,
    show: Optional[bool] = None,
) -> None:
    """Plot training curves with combined striker/jammer diagnostics.

    Layout (2×3):
        Row 0: Training reward | Combined policy+value loss | Combined entropy/KL/clip/EV
        Row 1: Eval return | Eval survival+completion | Eval duration

    Parameters
    ----------
    save_dir : str | None
        If provided, the directory is created (if missing) and the figure is
        saved as ``training_dashboard.png``. FOFE diagnostics (if available)
        are saved as ``fofe_diagnostics.png`` in the same directory.
    show : bool | None
        Whether to display the figure interactively. Defaults to True when
        ``save_dir`` is None and False when ``save_dir`` is set, so headless
        runs don't block. Pass an explicit bool to override.
    """
    if show is None:
        show = save_dir is None
    save_path: Optional[Path] = None
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(22, 10))

    # --- Row 0, Col 0: Training Episode Reward ---
    _draw_episode_reward(axes[0, 0], logs)

    # --- Row 0, Col 1: Combined striker+jammer losses ---
    _draw_policy_value_loss(axes[0, 1], logs)

    # --- Row 0, Col 2: Combined striker+jammer diagnostics ---
    ax = axes[0, 2]
    if "striker_entropy" in logs:
        _plot_valid(ax, logs["striker_entropy"], "striker_entropy", color=NLR_PRIMARY)
    if "jammer_entropy" in logs:
        _plot_valid(ax, logs["jammer_entropy"], "jammer_entropy", color=NLR_LIGHTBLUE_50)
    if "striker_approx_kl" in logs:
        _plot_valid(ax, logs["striker_approx_kl"], "striker_kl_approx", color=NLR_ACCENT)
    if "jammer_approx_kl" in logs:
        _plot_valid(ax, logs["jammer_approx_kl"], "jammer_kl_approx", color=NLR_TERRA_50)
    if "striker_clip_ratio" in logs:
        _plot_valid(ax, logs["striker_clip_ratio"], "striker_clip_ratio", color=NLR_SECONDARY)
    if "jammer_clip_ratio" in logs:
        _plot_valid(ax, logs["jammer_clip_ratio"], "jammer_clip_ratio", color=NLR_LIGHTBLUE_20)
    if "striker_explained_variance" in logs:
        _plot_valid(ax, logs["striker_explained_variance"], "striker_explained_var", color=NLR_DARKGRAY)
    if "jammer_explained_variance" in logs:
        _plot_valid(ax, logs["jammer_explained_variance"], "jammer_explained_var", color=NLR_GRAY)
    ax.set_title("Entropy / KL / Clip / EV (Striker + Jammer)")
    ax.set_xlabel("Iteration")
    ax.legend(fontsize=7)
    ax.grid(True)

    # --- Row 1, Col 0: Time per Iteration ---
    ax = axes[1, 0]
    if "iter_time_s" in logs:
        _plot_valid(ax, logs["iter_time_s"], "total (incl. eval)", color=NLR_PRIMARY)
    if "iter_time_excl_eval_s" in logs:
        y_excl = np.asarray(logs["iter_time_excl_eval_s"], dtype=float)
        if y_excl.size > 0 and np.any(np.isfinite(y_excl)):
            _plot_valid(ax, logs["iter_time_excl_eval_s"], "training only", color=NLR_ACCENT)
    if "eval_time_s" in logs:
        y_eval = np.asarray(logs["eval_time_s"], dtype=float)
        if y_eval.size > 0 and np.nanmax(np.where(np.isfinite(y_eval), y_eval, 0.0)) > 1e-6:
            _plot_valid(ax, logs["eval_time_s"], "eval only", color=NLR_SECONDARY, linestyle="--")
    ax.set_title("Time per Iteration (s)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=7)
    ax.grid(True)

    # --- Row 1, Col 1: Eval Survival, Completion & Targets Destroyed ---
    _draw_eval_kpi(axes[1, 1], logs)

    # --- Row 1, Col 2: Eval Mission Duration ---
    ax = axes[1, 2]
    if "eval_mean_duration" in logs:
        _plot_valid(ax, logs["eval_mean_duration"], "eval_mission_duration")
    ax.set_title("Eval Mission Duration")
    ax.set_xlabel("Iteration")
    ax.legend()
    ax.grid(True)

    fig.suptitle("Dual-MAPPO Training Dashboard (Striker + Jammer)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        out_file = save_path / "training_dashboard.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {out_file}")
        _save_standalone_panels(logs, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    # Show / save FOFE diagnostics if FOFE log keys have finite data
    _plot_fofe_diagnostics(logs, save_dir=save_dir, show=show)


def _plot_fofe_diagnostics(
    logs: Dict[str, List[float]],
    save_dir: Optional[str] = None,
    show: Optional[bool] = None,
) -> None:
    """Optional FOFE KPI diagnostics dashboard (2x3 grid).

    Only displays if the logs contain FOFE keys with at least some finite data.
    Backward-compatible: silently returns if keys are missing.

    Layout:
        Row 0: Channel Dominance (Striker) | Channel Dominance (Jammer) | Collapse Fraction
        Row 1: Visible Entity Counts       | All-Masked Fraction        | SEE Gradient Norms
    """
    # Check if any FOFE data exists
    test_key = "fofe_striker_dominance_agents"
    if test_key not in logs:
        return
    y = np.asarray(logs[test_key], dtype=float)
    if y.size == 0 or not np.any(np.isfinite(y)):
        return

    if show is None:
        show = save_dir is None
    save_path: Optional[Path] = None
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    def _plot_valid(ax, series, label, **kwargs):
        y = np.asarray(series, dtype=float)
        if y.size == 0:
            return
        x = np.arange(1, y.size + 1)
        valid = np.isfinite(y)
        if not np.any(valid):
            return
        ax.plot(x[valid], y[valid], marker="o", markersize=2, label=label, **kwargs)

    def _safe(key):
        return logs.get(key, [])

    fig, axes = plt.subplots(2, 3, figsize=(22, 10))

    # ── Row 0, Col 0: Channel Dominance (Striker) ────────────────
    ax = axes[0, 0]
    _plot_valid(ax, _safe("fofe_striker_dominance_agents"),  "agents",  color=NLR_PRIMARY)
    _plot_valid(ax, _safe("fofe_striker_dominance_targets"), "targets", color=NLR_ACCENT)
    _plot_valid(ax, _safe("fofe_striker_dominance_radars"),  "radars",  color=NLR_SECONDARY)
    ax.set_title("Channel Dominance (Striker)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Norm share")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True)

    # ── Row 0, Col 1: Channel Dominance (Jammer) ─────────────────
    ax = axes[0, 1]
    _plot_valid(ax, _safe("fofe_jammer_dominance_agents"),  "agents",  color=NLR_PRIMARY)
    _plot_valid(ax, _safe("fofe_jammer_dominance_targets"), "targets", color=NLR_ACCENT)
    _plot_valid(ax, _safe("fofe_jammer_dominance_radars"),  "radars",  color=NLR_SECONDARY)
    ax.set_title("Channel Dominance (Jammer)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Norm share")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True)

    # ── Row 0, Col 2: Collapse Fraction ──────────────────────────
    ax = axes[0, 2]
    for role, ls in [("striker", "-"), ("jammer", "--")]:
        for ch, col in [("agents", NLR_PRIMARY), ("targets", NLR_ACCENT), ("radars", NLR_SECONDARY)]:
            _plot_valid(ax, _safe(f"fofe_{role}_collapse_{ch}"),
                        f"{role[0].upper()}_{ch}", color=col, linestyle=ls)
    ax.set_title("Collapse Fraction (norm < eps)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True)

    # ── Row 1, Col 0: Visible Entity Counts ──────────────────────
    ax = axes[1, 0]
    for role, ls in [("striker", "-"), ("jammer", "--")]:
        for ch, col in [("agents", NLR_PRIMARY), ("targets", NLR_ACCENT), ("radars", NLR_SECONDARY)]:
            _plot_valid(ax, _safe(f"fofe_{role}_visible_{ch}_mean"),
                        f"{role[0].upper()}_{ch}", color=col, linestyle=ls)
    ax.set_title("Visible Entity Counts (mean)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Count")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True)

    # ── Row 1, Col 1: All-Masked Fraction ────────────────────────
    ax = axes[1, 1]
    for role, ls in [("striker", "-"), ("jammer", "--")]:
        for ch, col in [("agents", NLR_PRIMARY), ("targets", NLR_ACCENT), ("radars", NLR_SECONDARY)]:
            _plot_valid(ax, _safe(f"fofe_{role}_all_masked_{ch}"),
                        f"{role[0].upper()}_{ch}", color=col, linestyle=ls)
    ax.set_title("All-Masked Fraction")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True)

    # ── Row 1, Col 2: SEE Gradient Norms ─────────────────────────
    ax = axes[1, 2]
    _plot_valid(ax, _safe("fofe_striker_actor_see_grad_norm"),  "S_actor",  color=NLR_PRIMARY)
    _plot_valid(ax, _safe("fofe_striker_critic_see_grad_norm"), "S_critic", color=NLR_ACCENT)
    _plot_valid(ax, _safe("fofe_jammer_actor_see_grad_norm"),  "J_actor",  color=NLR_SECONDARY)
    _plot_valid(ax, _safe("fofe_jammer_critic_see_grad_norm"), "J_critic", color=NLR_TERRA_50)
    ax.set_title("SEE Gradient Norms (post-clip)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 norm")
    ax.legend(fontsize=8)
    ax.grid(True)

    fig.suptitle("FOFE Diagnostics Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        out_file = save_path / "fofe_diagnostics.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {out_file}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------------------------------------------------
# Test Runner & Animation (same as original, adapted for CombinedPolicy)
# ------------------------------------------------------------------

class TestRunner:
    def __init__(self, policy, *, env_cfg: EnvConfig, device: torch.device, seed: int = 999):
        self.policy = policy
        self.policy.eval()
        self.policy.deterministic = True
        self.env = StrikeEA2DEnv(
            num_envs=1,
            max_steps=env_cfg.max_steps,
            device=device,
            seed=seed,
            n_strikers=env_cfg.n_strikers,
            n_jammers=env_cfg.n_jammers,
            n_targets=env_cfg.n_targets,
            n_radars=env_cfg.n_radars,
            n_known_targets=env_cfg.n_known_targets,
            n_unknown_targets=env_cfg.n_unknown_targets,
            n_known_radars=env_cfg.n_known_radars,
            n_unknown_radars=env_cfg.n_unknown_radars,
            dt=env_cfg.dt,
            world_bounds=env_cfg.world_bounds,
            v_max=env_cfg.v_max,
            accel_magnitude=env_cfg.accel_magnitude,
            dpsi_max=env_cfg.dpsi_max,
            h_accel_magnitude_fraction=env_cfg.h_accel_magnitude_fraction,
            min_turn_radius=env_cfg.min_turn_radius,
            R_obs=env_cfg.R_obs,
            R_comm=env_cfg.R_comm,
            communicate=getattr(env_cfg, "communicate", True),
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
            radar_min_sep=getattr(env_cfg, 'radar_min_sep', 0.5),
            scenario=getattr(env_cfg, 'scenario', 'S1'),
            s2_radar_min_sep=getattr(env_cfg, 's2_radar_min_sep', 0.2),
            s2_target_min_sep=getattr(env_cfg, 's2_target_min_sep', 0.2),
            use_fofe=getattr(env_cfg, 'use_fofe', False),
        )

    @torch.no_grad()
    def rollout(self) -> List[Dict[str, torch.Tensor]]:
        td = self.env.reset()
        frames = [self._snapshot()]
        for _ in range(self.env.max_steps):
            td = self.policy(td)
            td = self.env.step(td)
            frames.append(self._snapshot())
            if bool(td.get(("next", "done")).item()):
                break
            td = td.get("next")
        return frames

    def _snapshot(self) -> Dict[str, torch.Tensor]:
        env = self.env
        snap = {
            "agent_pos": env.agent_pos[0].detach().cpu(),
            "agent_alive": env.agent_alive[0].detach().cpu(),
            "agent_heading": env.agent_heading[0].detach().cpu(),
            "target_pos": env.target_pos[0].detach().cpu(),
            "target_alive": env.target_alive[0].detach().cpu(),
            "target_known": env.target_known[0].detach().cpu(),
            "radar_pos": env.radar_pos[0].detach().cpu(),
            "radar_known": env.radar_known[0].detach().cpu(),
            "radar_eff_range": env.radar_eff_range[0].detach().cpu(),
        }
        if hasattr(env, "last_reward_components") and env.last_reward_components:
            snap["reward_components"] = {
                k: v[0].detach().cpu().clone() for k, v in env.last_reward_components.items()
            }
        return snap


def animate_rollout(
    frames: List[Dict[str, torch.Tensor]],
    env: StrikeEA2DEnv,
    interval_ms: int = 70,
    save_path: Optional[str] = None,
    show: Optional[bool] = None,
):
    # --- Pre-compute reward time-series from frames ---
    reward_ts: Dict[str, List[float]] = {}
    total_ts: List[float] = []
    for fr in frames:
        rc = fr.get("reward_components")
        if rc is None:
            continue
        step_total = 0.0
        for comp_name, comp_tensor in rc.items():
            val = float(comp_tensor.sum().item())
            reward_ts.setdefault(comp_name, []).append(val)
            step_total += val
        total_ts.append(step_total)

    active_components = {k: v for k, v in reward_ts.items() if any(abs(x) > 1e-9 for x in v)}

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_rew = fig.add_subplot(gs[0, 1])

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title(f"Dual-MAPPO Rollout | R_obs={env.R_obs:.2f}, R_comm={env.R_comm:.2f}")

    striker_color = NLR_PRIMARY
    jammer_color = NLR_ACCENT

    striker_sc = ax.scatter([], [], s=60, marker="^", color=striker_color, label="Strikers")
    jammer_sc = ax.scatter([], [], s=60, marker="s", color=jammer_color, label="Jammers")
    target_known_sc = ax.scatter([], [], s=80, marker="*", label="Targets (known)", color="#a65e00")
    target_unknown_sc = ax.scatter([], [], s=80, marker="*", label="Targets (unknown)", facecolors="none", edgecolors="#a65e00", linewidths=1.8)
    radar_known_sc = ax.scatter([], [], s=80, marker="X", label="Radars (known)", color="#243c9b")
    radar_unknown_sc = ax.scatter([], [], s=80, marker="X", label="Radars (unknown)", facecolors="none", edgecolors="#243c9b", linewidths=1.8)

    # Trailing path behind agents: per-agent fading LineCollection.
    trail_len = 40
    trail_max_alpha = 0.45
    agent_trail_colors = (
        [mcolors.to_rgba(striker_color)] * env.n_strikers
        + [mcolors.to_rgba(jammer_color)] * env.n_jammers
    )
    agent_trail_history: List[List[tuple]] = [[] for _ in range(env.n_agents)]
    agent_trail_lcs = [
        ax.add_collection(LineCollection([], linewidths=2.0, zorder=1.5, capstyle="round"))
        for _ in range(env.n_agents)
    ]

    radar_circles = [ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="C3", alpha=0.6, lw=2)) for _ in range(env.n_radars)]
    jammer_circles = [ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="C4", alpha=0.5, lw=1.5, ls="--")) for _ in range(env.n_jammers)]
    obs_circles = [ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="C0", alpha=0.35, lw=1.0, ls=":")) for _ in range(env.n_agents)]
    comm_circles = [ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="black", alpha=0.18, lw=0.9, ls="-.")) for _ in range(env.n_agents)]
    striker_arcs = [ax.add_patch(Polygon(np.empty((0, 2)), closed=True, fc="C2", alpha=0.18, ec="C2")) for _ in range(env.n_strikers)]
    heading_lines = [ax.plot([], [])[0] for _ in range(env.n_agents)]
    comm_lines = [ax.plot([], [], color="black", lw=1.4, alpha=0.7)[0] for _ in range(max(env.n_agents - 1, 1))]
    ax.plot([], [], color="black", lw=1.4, alpha=0.7, label="Comm MST (<= R_comm)")
    ax.plot([], [], color="black", lw=0.9, ls="-.", alpha=0.35, label="R_comm")
    ax.plot([], [], color="C0", lw=1.0, ls=":", alpha=0.45, label="R_obs")
    ax.plot([], [], color=NLR_DARKGRAY, lw=2.0, alpha=trail_max_alpha, label=f"Agent trail (last {trail_len} steps)")
    ax.legend(loc="lower right", fontsize=5, markerscale=0.6, handlelength=1.5, borderpad=0.3, labelspacing=0.3)

    ax_rew.set_xlabel("Timestep")
    ax_rew.set_ylabel("Reward (team sum)")
    ax_rew.set_ylim(-0.25, 0.25)
    ax_rew.set_title("Per-Timestep Reward Components")

    n_reward_steps = len(total_ts)
    if n_reward_steps > 0:
        timesteps = np.arange(1, n_reward_steps + 1)
        for comp_name, values in sorted(active_components.items()):
            ax_rew.plot(timesteps, values, lw=1.5, alpha=0.8, label=comp_name)
        ax_rew.plot(timesteps, total_ts, lw=2.5, color="black", label="total", zorder=10)
        ax_rew.axhline(0, color="gray", lw=0.5)
        ax_rew.set_xlim(1, max(n_reward_steps, 2))
        ax_rew.legend(loc="best", fontsize=7, ncol=2)
    ax_rew.grid(True, alpha=0.3)

    time_vline = ax_rew.axvline(x=1, color="red", lw=1.5, ls="--", alpha=0.8)

    empty_xy = np.empty((0, 2), dtype=float)

    def init():
        striker_sc.set_offsets(empty_xy)
        jammer_sc.set_offsets(empty_xy)
        target_known_sc.set_offsets(empty_xy)
        target_unknown_sc.set_offsets(empty_xy)
        radar_known_sc.set_offsets(empty_xy)
        radar_unknown_sc.set_offsets(empty_xy)
        for c in [*radar_circles, *jammer_circles, *obs_circles, *comm_circles]:
            c.set_visible(False)
        for a in striker_arcs:
            a.set_visible(False)
        for ln in heading_lines:
            ln.set_data([], [])
        for cl in comm_lines:
            cl.set_data([], [])
        for hist in agent_trail_history:
            hist.clear()
        for lc in agent_trail_lcs:
            lc.set_segments([])
        time_vline.set_xdata([1])
        return [
            striker_sc,
            jammer_sc,
            target_known_sc,
            target_unknown_sc,
            radar_known_sc,
            radar_unknown_sc,
            *radar_circles,
            *jammer_circles,
            *obs_circles,
            *comm_circles,
            *striker_arcs,
            *heading_lines,
            *comm_lines,
            *agent_trail_lcs,
            time_vline,
        ]

    def update(i: int):
        fr = frames[i]
        ap = fr["agent_pos"]
        aa = fr["agent_alive"]
        ah = fr["agent_heading"]
        tp = fr["target_pos"]
        ta = fr["target_alive"]
        tk = fr["target_known"]
        rp = fr["radar_pos"]
        rk = fr["radar_known"]
        rr = fr["radar_eff_range"]

        ap_km = ap * 1000
        tp_km = tp * 1000
        rp_km = rp * 1000

        # Reset trail history when the animation loops back to frame 0.
        if i == 0:
            for hist in agent_trail_history:
                hist.clear()

        # Update per-agent fading trails (only while the agent is alive).
        for k in range(env.n_agents):
            if aa[k].item():
                agent_trail_history[k].append((float(ap_km[k, 0]), float(ap_km[k, 1])))
                if len(agent_trail_history[k]) > trail_len:
                    del agent_trail_history[k][: len(agent_trail_history[k]) - trail_len]
            pts = agent_trail_history[k]
            if len(pts) >= 2:
                pts_arr = np.asarray(pts, dtype=float)
                segs = np.stack([pts_arr[:-1], pts_arr[1:]], axis=1)
                n = segs.shape[0]
                alphas = np.linspace(0.05, trail_max_alpha, n)
                colors = np.tile(np.asarray(agent_trail_colors[k], dtype=float), (n, 1))
                colors[:, 3] = alphas
                agent_trail_lcs[k].set_segments(segs)
                agent_trail_lcs[k].set_color(colors)
            else:
                agent_trail_lcs[k].set_segments([])

        striker_xy = ap_km[: env.n_strikers][aa[: env.n_strikers]]
        jammer_xy = ap_km[env.n_strikers:][aa[env.n_strikers:]]
        striker_sc.set_offsets(striker_xy.numpy() if striker_xy.numel() else empty_xy)
        jammer_sc.set_offsets(jammer_xy.numpy() if jammer_xy.numel() else empty_xy)

        alive_known_targets = ta & tk
        alive_unknown_targets = ta & (~tk)

        if alive_known_targets.any():
            target_known_sc.set_offsets(tp_km[alive_known_targets].numpy())
        else:
            target_known_sc.set_offsets(empty_xy)

        if alive_unknown_targets.any():
            target_unknown_sc.set_offsets(tp_km[alive_unknown_targets].numpy())
        else:
            target_unknown_sc.set_offsets(empty_xy)

        known_radars = rk
        unknown_radars = ~rk

        if known_radars.any():
            radar_known_sc.set_offsets(rp_km[known_radars].numpy())
        else:
            radar_known_sc.set_offsets(empty_xy)

        if unknown_radars.any():
            radar_unknown_sc.set_offsets(rp_km[unknown_radars].numpy())
        else:
            radar_unknown_sc.set_offsets(empty_xy)

        for j, c in enumerate(radar_circles):
            c.set_visible(True)
            c.set_center((float(rp_km[j, 0]), float(rp_km[j, 1])))
            c.set_radius(float(rr[j]) * 1000)

        for j, jc in enumerate(jammer_circles):
            idx = env.n_strikers + j
            if aa[idx].item():
                jc.set_visible(True)
                jc.set_center((float(ap_km[idx, 0]), float(ap_km[idx, 1])))
                jc.set_radius(env.jammer.jam_radius * 1000)
            else:
                jc.set_visible(False)

        for k, oc in enumerate(obs_circles):
            if aa[k].item():
                oc.set_visible(True)
                oc.set_center((float(ap_km[k, 0]), float(ap_km[k, 1])))
                oc.set_radius(float(env.R_obs) * 1000)
            else:
                oc.set_visible(False)

        for k, cc in enumerate(comm_circles):
            if aa[k].item():
                cc.set_visible(True)
                cc.set_center((float(ap_km[k, 0]), float(ap_km[k, 1])))
                cc.set_radius(float(env.R_comm) * 1000)
            else:
                cc.set_visible(False)

        half_fov = 0.5 * env.striker.engage_fov_deg
        r_str = env.striker.engage_range * 1000
        for s, sa in enumerate(striker_arcs):
            if aa[s].item():
                cx, cy = float(ap_km[s, 0]), float(ap_km[s, 1])
                th1 = math.radians(math.degrees(float(ah[s])) - half_fov)
                th2 = th1 + math.radians(2 * half_fov)
                angles = np.linspace(th1, th2, 24)
                verts = np.vstack(([cx, cy], np.column_stack((cx + r_str * np.cos(angles), cy + r_str * np.sin(angles)))))
                sa.set_visible(True)
                sa.set_xy(verts)
            else:
                sa.set_visible(False)

        for k, ln in enumerate(heading_lines):
            if aa[k].item():
                x, y = float(ap_km[k, 0]), float(ap_km[k, 1])
                ln.set_data([x, x + 30 * math.cos(float(ah[k]))], [y, y + 30 * math.sin(float(ah[k]))])
            else:
                ln.set_data([], [])

        alive_idx = torch.where(aa)[0]
        edges_for_plot = []
        if alive_idx.numel() > 1:
            alive_pos_world = ap[alive_idx]  # [Na,2]
            alive_pos_km = ap_km[alive_idx]  # [Na,2]
            na = alive_idx.numel()
            dmat = torch.cdist(alive_pos_world, alive_pos_world)
            adj = dmat <= env.R_comm
            if not getattr(env, "communicate", True):
                adj = torch.zeros_like(adj)  # no comm links when comm disabled

            visited = torch.zeros(na, dtype=torch.bool)
            components = []
            for start in range(na):
                if visited[start]:
                    continue
                queue = [start]
                visited[start] = True
                comp = [start]
                while queue:
                    u = queue.pop(0)
                    neigh = torch.where(adj[u])[0].tolist()
                    for v in neigh:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                            comp.append(v)
                components.append(comp)

            for comp in components:
                if len(comp) <= 1:
                    continue

                parent = {u: u for u in comp}

                def _find(u):
                    while parent[u] != u:
                        parent[u] = parent[parent[u]]
                        u = parent[u]
                    return u

                def _union(u, v):
                    ru, rv = _find(u), _find(v)
                    if ru == rv:
                        return False
                    parent[rv] = ru
                    return True

                cand = []
                for i in range(len(comp)):
                    for j in range(i + 1, len(comp)):
                        u, v = comp[i], comp[j]
                        if bool(adj[u, v].item()):
                            cand.append((float(dmat[u, v].item()), u, v))
                cand.sort(key=lambda x: x[0])

                added = 0
                for _w, u, v in cand:
                    if _union(u, v):
                        p1 = alive_pos_km[u]
                        p2 = alive_pos_km[v]
                        edges_for_plot.append((float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])))
                        added += 1
                        if added == len(comp) - 1:
                            break

        for li, line in enumerate(comm_lines):
            if li < len(edges_for_plot):
                x1, y1, x2, y2 = edges_for_plot[li]
                line.set_data([x1, x2], [y1, y2])
            else:
                line.set_data([], [])

        ax.set_xlabel(f"t={i} | Agents: {int(aa.sum().item())}/{env.n_agents} | Targets: {int(ta.sum().item())}/{env.n_targets}")

        reward_step = min(i, n_reward_steps) if n_reward_steps > 0 else 1
        time_vline.set_xdata([reward_step])

        return [
            striker_sc,
            jammer_sc,
            target_known_sc,
            target_unknown_sc,
            radar_known_sc,
            radar_unknown_sc,
            *radar_circles,
            *jammer_circles,
            *obs_circles,
            *striker_arcs,
            *heading_lines,
            *comm_lines,
            *agent_trail_lcs,
            time_vline,
        ]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), init_func=init, interval=interval_ms, blit=False, repeat=False)
    # Save to GIF first (if requested), then optionally display. Saving after
    # plt.show() doesn't work because the figure may be closed by then.
    if show is None:
        show = save_path is None
    if save_path is not None:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        writer = animation.PillowWriter(fps=max(1, int(round(1000 / max(1, interval_ms)))))
        ani.save(str(out_file), writer=writer)
        print(f"Saved rollout GIF: {out_file}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return ani


# ------------------------------------------------------------------
# Comparison Dashboard: MAPPO (Legacy) vs FOFE-MAPPO
# ------------------------------------------------------------------

_LEGACY_LABEL = "MAPPO (Legacy)"
_FOFE_LABEL   = "FOFE-MAPPO"

# Two coordinated NLR palettes per metric family:
#   Legacy (baseline) = lighter tint  (50 % tone or light grey)
#   FOFE              = primary tone   (full NLR colour)
_METRIC_COLORS = {
    "blue":   {_LEGACY_LABEL: NLR_LIGHTBLUE_50, _FOFE_LABEL: NLR_PRIMARY},
    "orange": {_LEGACY_LABEL: NLR_TERRA_50,     _FOFE_LABEL: NLR_ACCENT},
    "green":  {_LEGACY_LABEL: NLR_LIGHTBLUE_20, _FOFE_LABEL: NLR_SECONDARY},
    "purple": {_LEGACY_LABEL: NLR_GRAY,         _FOFE_LABEL: NLR_DARKGRAY},
}

# Method styling: combine line pattern + marker shape + width for readability.
_METHOD_STYLES = {
    _LEGACY_LABEL: {
        "linestyle": (0, (6, 2, 1.2, 2)),
        "linewidth": 1.2,
        "alpha": 0.95,
        "marker": "X",
        "markersize": 3.5,
        "markevery": 10,
        "markeredgewidth": 0.8,
    },
    _FOFE_LABEL: {
        "linestyle": "-",
        "linewidth": 1.4,
        "alpha": 1.0,
        "marker": "o",
        "markersize": 3.0,
        "markevery": 10,
        "markeredgewidth": 0.8,
        "markerfacecolor": "white",
    },
}


def plot_comparison(
    legacy_logs: Dict[str, List[float]],
    fofe_logs: Dict[str, List[float]],
) -> None:
    """Comparison dashboard: MAPPO (Legacy) vs FOFE-MAPPO.

    Layout (2×3) — same grid as the single-method dashboard:
        Row 0: Training total reward | Combined losses | Entropy/KL/clip/EV
        Row 1: Eval total return     | Survival+completion | Duration

    Reward plots show ONLY total reward for both methods.
    All other plots show every metric for both methods, with dashed lines
    for Legacy and solid lines for FOFE-MAPPO.
    """

    def _plot_method(ax, series, label, color, style_dict, **extra):
        y = np.asarray(series, dtype=float)
        if y.size == 0:
            return
        x = np.arange(1, y.size + 1)
        valid = np.isfinite(y)
        if not np.any(valid):
            return
        ax.plot(
            x[valid], y[valid],
            label=label, color=color,
            linestyle=style_dict.get("linestyle", "-"),
            linewidth=style_dict.get("linewidth", 2.0),
            alpha=style_dict.get("alpha", 1.0),
            marker=style_dict.get("marker", "o"),
            markersize=style_dict.get("markersize", 3),
            markevery=style_dict.get("markevery", None),
            markeredgewidth=style_dict.get("markeredgewidth", 0.8),
            markerfacecolor=style_dict.get("markerfacecolor", None),
            **extra,
        )

    def _dual(ax, key, metric_label, metric_family, legacy, fofe):
        legacy_color = _METRIC_COLORS[metric_family][_LEGACY_LABEL]
        fofe_color = _METRIC_COLORS[metric_family][_FOFE_LABEL]
        if key in legacy:
            _plot_method(ax, legacy[key],
                         f"{metric_label} [{_LEGACY_LABEL}]", legacy_color,
                         _METHOD_STYLES[_LEGACY_LABEL])
        if key in fofe:
            _plot_method(ax, fofe[key],
                         f"{metric_label} [{_FOFE_LABEL}]", fofe_color,
                         _METHOD_STYLES[_FOFE_LABEL])

    fig, axes = plt.subplots(2, 3, figsize=(24, 11))

    # ── Row 0, Col 0: Training Total Reward (only total) ─────────
    ax = axes[0, 0]
    _dual(ax, "train_mean_episode_total_reward", "Total Reward",
            "blue", legacy_logs, fofe_logs)
    ax.set_title("Training Episode Reward")
    ax.set_xlabel("Iteration")
    ax.legend(fontsize=8)
    ax.grid(True)

    # ── Row 0, Col 1: Combined losses ────────────────────────────
    ax = axes[0, 1]
    loss_items = [
        ("striker_loss_policy", "striker_policy_loss", "blue"),
        ("striker_loss_value",  "striker_value_loss",  "orange"),
        ("jammer_loss_policy",  "jammer_policy_loss",  "green"),
        ("jammer_loss_value",   "jammer_value_loss",   "purple"),
    ]
    for key, lbl, col in loss_items:
        _dual(ax, key, lbl, col, legacy_logs, fofe_logs)
    ax.set_title("Policy & Value Loss (Striker + Jammer)")
    ax.set_xlabel("Iteration")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True)

    # ── Row 0, Col 2: Entropy / KL / Clip / EV ──────────────────
    ax = axes[0, 2]
    diag_items = [
        ("striker_entropy",            "striker_entropy",       "green"),
        ("jammer_entropy",             "jammer_entropy",        "green"),
        ("striker_approx_kl",          "striker_kl_approx",     "orange"),
        ("jammer_approx_kl",           "jammer_kl_approx",      "orange"),
        ("striker_clip_ratio",         "striker_clip_ratio",    "blue"),
        ("jammer_clip_ratio",          "jammer_clip_ratio",     "blue"),
        ("striker_explained_variance", "striker_explained_var", "purple"),
        ("jammer_explained_variance",  "jammer_explained_var",  "purple"),
    ]
    for key, lbl, col in diag_items:
        _dual(ax, key, lbl, col, legacy_logs, fofe_logs)
    ax.set_title("Entropy / KL / Clip / EV (Striker + Jammer)")
    ax.set_xlabel("Iteration")
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True)

    # ── Row 1, Col 0: Time per Iteration ─────────────────────────
    ax = axes[1, 0]
    _dual(ax, "iter_time_s", "iter_time_s", "blue", legacy_logs, fofe_logs)
    ax.set_title("Time per Iteration (s)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=8)
    ax.grid(True)

    # ── Row 1, Col 1: Eval Survival, Completion & Targets Destroyed ──
    ax = axes[1, 1]
    _dual(ax, "eval_survival_rate",          "survival_rate",          "blue",   legacy_logs, fofe_logs)
    _dual(ax, "eval_task_completion_rate",   "completion_rate",        "orange", legacy_logs, fofe_logs)
    _dual(ax, "eval_targets_destroyed_rate", "targets_destroyed_rate", "green",  legacy_logs, fofe_logs)
    ax.set_title("Eval Survival, Completion & Targets Destroyed")
    ax.set_xlabel("Iteration")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)

    # ── Row 1, Col 2: Eval Mission Duration ──────────────────────
    ax = axes[1, 2]
    _dual(ax, "eval_mean_duration", "mission_duration", "blue",
          legacy_logs, fofe_logs)
    ax.set_title("Eval Mission Duration")
    ax.set_xlabel("Iteration")
    ax.legend(fontsize=8)
    ax.grid(True)

    fig.suptitle(
        f"Comparison: {_LEGACY_LABEL}  vs  {_FOFE_LABEL}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ------------------------------------------------------------------
# Side-by-side rollout animation: Legacy vs FOFE
# ------------------------------------------------------------------

def _draw_world_panel(ax, env, frames, title):
    """Set up one world-view panel and return the drawable artists + update fn."""
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title(f"{title} | R_obs={env.R_obs:.2f}, R_comm={env.R_comm:.2f}")

    empty_xy = np.empty((0, 2), dtype=float)

    striker_sc = ax.scatter([], [], s=60, marker="^", label="Strikers")
    jammer_sc = ax.scatter([], [], s=60, marker="s", label="Jammers")
    target_known_sc = ax.scatter([], [], s=80, marker="*", label="Targets (known)", color="#a65e00")
    target_unknown_sc = ax.scatter([], [], s=80, marker="*", label="Targets (unknown)",
                                   facecolors="none", edgecolors="#a65e00", linewidths=1.8)
    radar_known_sc = ax.scatter([], [], s=80, marker="X", label="Radars (known)", color="#243c9b")
    radar_unknown_sc = ax.scatter([], [], s=80, marker="X", label="Radars (unknown)",
                                  facecolors="none", edgecolors="#243c9b", linewidths=1.8)

    radar_circles = [ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="C3", alpha=0.6, lw=2))
                     for _ in range(env.n_radars)]
    jammer_circles = [ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="C4", alpha=0.5, lw=1.5, ls="--"))
                      for _ in range(env.n_jammers)]
    obs_circles = [ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="C0", alpha=0.35, lw=1.0, ls=":"))
                   for _ in range(env.n_agents)]
    comm_circles = [ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="black", alpha=0.18, lw=0.9, ls="-."))
                    for _ in range(env.n_agents)]
    striker_arcs = [ax.add_patch(Polygon(np.empty((0, 2)), closed=True, fc="C2", alpha=0.18, ec="C2"))
                    for _ in range(env.n_strikers)]
    heading_lines = [ax.plot([], [])[0] for _ in range(env.n_agents)]
    comm_lines = [ax.plot([], [], color="black", lw=1.4, alpha=0.7)[0]
                  for _ in range(max(env.n_agents - 1, 1))]
    ax.plot([], [], color="black", lw=1.4, alpha=0.7, label="Comm MST (<= R_comm)")
    ax.plot([], [], color="black", lw=0.9, ls="-.", alpha=0.35, label="R_comm")
    ax.plot([], [], color="C0", lw=1.0, ls=":", alpha=0.45, label="R_obs")
    ax.legend(loc="upper right", fontsize=6)

    artists = {
        "striker_sc": striker_sc, "jammer_sc": jammer_sc,
        "target_known_sc": target_known_sc, "target_unknown_sc": target_unknown_sc,
        "radar_known_sc": radar_known_sc, "radar_unknown_sc": radar_unknown_sc,
        "radar_circles": radar_circles, "jammer_circles": jammer_circles,
        "obs_circles": obs_circles, "comm_circles": comm_circles,
        "striker_arcs": striker_arcs,
        "heading_lines": heading_lines, "comm_lines": comm_lines,
    }

    def _update_panel(i):
        fr = frames[min(i, len(frames) - 1)]
        ap = fr["agent_pos"]; aa = fr["agent_alive"]; ah = fr["agent_heading"]
        tp = fr["target_pos"]; ta = fr["target_alive"]; tk = fr["target_known"]
        rp = fr["radar_pos"]; rk = fr["radar_known"]; rr = fr["radar_eff_range"]

        ap_km, tp_km, rp_km = ap * 1000, tp * 1000, rp * 1000

        s_xy = ap_km[:env.n_strikers][aa[:env.n_strikers]]
        j_xy = ap_km[env.n_strikers:][aa[env.n_strikers:]]
        striker_sc.set_offsets(s_xy.numpy() if s_xy.numel() else empty_xy)
        jammer_sc.set_offsets(j_xy.numpy() if j_xy.numel() else empty_xy)

        ak_t = ta & tk; au_t = ta & (~tk)
        target_known_sc.set_offsets(tp_km[ak_t].numpy() if ak_t.any() else empty_xy)
        target_unknown_sc.set_offsets(tp_km[au_t].numpy() if au_t.any() else empty_xy)
        radar_known_sc.set_offsets(rp_km[rk].numpy() if rk.any() else empty_xy)
        radar_unknown_sc.set_offsets(rp_km[~rk].numpy() if (~rk).any() else empty_xy)

        for j, c in enumerate(radar_circles):
            c.set_visible(True)
            c.set_center((float(rp_km[j, 0]), float(rp_km[j, 1])))
            c.set_radius(float(rr[j]) * 1000)
        for j, jc in enumerate(jammer_circles):
            idx = env.n_strikers + j
            if aa[idx].item():
                jc.set_visible(True)
                jc.set_center((float(ap_km[idx, 0]), float(ap_km[idx, 1])))
                jc.set_radius(env.jammer.jam_radius * 1000)
            else:
                jc.set_visible(False)
        for k, oc in enumerate(obs_circles):
            if aa[k].item():
                oc.set_visible(True)
                oc.set_center((float(ap_km[k, 0]), float(ap_km[k, 1])))
                oc.set_radius(float(env.R_obs) * 1000)
            else:
                oc.set_visible(False)

        for k, cc in enumerate(comm_circles):
            if aa[k].item():
                cc.set_visible(True)
                cc.set_center((float(ap_km[k, 0]), float(ap_km[k, 1])))
                cc.set_radius(float(env.R_comm) * 1000)
            else:
                cc.set_visible(False)

        half_fov = 0.5 * env.striker.engage_fov_deg
        r_str = env.striker.engage_range * 1000
        for s, sa in enumerate(striker_arcs):
            if aa[s].item():
                cx, cy = float(ap_km[s, 0]), float(ap_km[s, 1])
                th1 = math.radians(math.degrees(float(ah[s])) - half_fov)
                th2 = th1 + math.radians(2 * half_fov)
                angles = np.linspace(th1, th2, 24)
                verts = np.vstack(([cx, cy], np.column_stack((cx + r_str * np.cos(angles),
                                                               cy + r_str * np.sin(angles)))))
                sa.set_visible(True); sa.set_xy(verts)
            else:
                sa.set_visible(False)

        for k, ln in enumerate(heading_lines):
            if aa[k].item():
                x, y = float(ap_km[k, 0]), float(ap_km[k, 1])
                ln.set_data([x, x + 30 * math.cos(float(ah[k]))],
                            [y, y + 30 * math.sin(float(ah[k]))])
            else:
                ln.set_data([], [])

        # Communication MST
        alive_idx = torch.where(aa)[0]
        edges = []
        if alive_idx.numel() > 1:
            alive_pos_w = ap[alive_idx]; alive_pos_km = ap_km[alive_idx]
            na = alive_idx.numel()
            dmat = torch.cdist(alive_pos_w, alive_pos_w)
            adj = dmat <= env.R_comm
            if not getattr(env, "communicate", True):
                adj = torch.zeros_like(adj)  # no comm links when comm disabled
            visited = torch.zeros(na, dtype=torch.bool)
            components = []
            for start in range(na):
                if visited[start]: continue
                queue = [start]; visited[start] = True; comp = [start]
                while queue:
                    u = queue.pop(0)
                    for v in torch.where(adj[u])[0].tolist():
                        if not visited[v]:
                            visited[v] = True; queue.append(v); comp.append(v)
                components.append(comp)
            for comp in components:
                if len(comp) <= 1: continue
                parent = {u: u for u in comp}
                def _find(u, _p=parent):
                    while _p[u] != u: _p[u] = _p[_p[u]]; u = _p[u]
                    return u
                def _union(u, v, _p=parent):
                    ru, rv = _find(u), _find(v)
                    if ru == rv: return False
                    _p[rv] = ru; return True
                cand = sorted(((float(dmat[comp[i], comp[j]].item()), comp[i], comp[j])
                                for i in range(len(comp)) for j in range(i+1, len(comp))
                                if bool(adj[comp[i], comp[j]].item())),
                               key=lambda x: x[0])
                added = 0
                for _w, u, v in cand:
                    if _union(u, v):
                        p1, p2 = alive_pos_km[u], alive_pos_km[v]
                        edges.append((float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])))
                        added += 1
                        if added == len(comp) - 1: break

        for li, line in enumerate(comm_lines):
            if li < len(edges):
                x1, y1, x2, y2 = edges[li]
                line.set_data([x1, x2], [y1, y2])
            else:
                line.set_data([], [])

        ax.set_xlabel(f"t={i} | Agents: {int(aa.sum().item())}/{env.n_agents} "
                       f"| Targets: {int(ta.sum().item())}/{env.n_targets}")

    return artists, _update_panel


def animate_comparison_rollout(
    legacy_frames: List[Dict[str, torch.Tensor]],
    fofe_frames: List[Dict[str, torch.Tensor]],
    legacy_env: StrikeEA2DEnv,
    fofe_env: StrikeEA2DEnv,
    interval_ms: int = 70,
):
    """Side-by-side rollout animation: MAPPO (Legacy) vs FOFE-MAPPO."""
    fig, (ax_l, ax_f) = plt.subplots(1, 2, figsize=(22, 9))

    _, update_legacy = _draw_world_panel(ax_l, legacy_env, legacy_frames, _LEGACY_LABEL)
    _, update_fofe   = _draw_world_panel(ax_f, fofe_env,   fofe_frames,   _FOFE_LABEL)

    n_frames = max(len(legacy_frames), len(fofe_frames))

    def _init():
        return []

    def _update(i):
        update_legacy(i)
        update_fofe(i)
        return []

    fig.suptitle(f"Rollout Comparison: {_LEGACY_LABEL}  vs  {_FOFE_LABEL}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    ani = animation.FuncAnimation(fig, _update, frames=n_frames,
                                  init_func=_init, interval=interval_ms,
                                  blit=False, repeat=False)
    plt.show()
    return ani
