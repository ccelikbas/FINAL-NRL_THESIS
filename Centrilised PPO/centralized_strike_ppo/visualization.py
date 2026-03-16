from __future__ import annotations

import contextlib
import math
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.patches import Circle, Polygon

from .config import EnvConfig
from .environment import StrikeEA2DEnv

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


def plot_training(logs: Dict[str, List[float]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if "eval_mean_episode_total_reward" in logs:
        axes[0].plot(logs["eval_mean_episode_total_reward"], label="eval_ep_return_total")
    axes[0].set_title("Mission Evaluation")
    axes[0].set_xlabel("Iteration")
    axes[0].legend()
    axes[0].grid(True)

    if "loss_policy" in logs:
        axes[1].plot(logs["loss_policy"], label="policy")
    if "loss_value" in logs:
        axes[1].plot(logs["loss_value"], label="value")
    axes[1].set_title("Losses")
    axes[1].set_xlabel("Iteration")
    axes[1].legend()
    axes[1].grid(True)

    if "eval_survival_rate" in logs:
        axes[2].plot(logs["eval_survival_rate"], label="eval_survival")
    if "eval_task_completion_rate" in logs:
        axes[2].plot(logs["eval_task_completion_rate"], label="eval_completion")
    if "eval_mean_duration" in logs:
        axes[2].plot(logs["eval_mean_duration"], label="eval_duration")
    if "clip_ratio" in logs:
        axes[2].plot(logs["clip_ratio"], label="clip_ratio")
    axes[2].set_title("Eval + Diagnostics")
    axes[2].set_xlabel("Iteration")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    plt.show()


class TestRunner:
    def __init__(self, policy, *, env_cfg: EnvConfig, device: torch.device, seed: int = 999):
        self.policy = policy.eval()
        self.env = StrikeEA2DEnv(
            num_envs=1,
            max_steps=env_cfg.max_steps,
            device=device,
            seed=seed,
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

    @torch.no_grad()
    def rollout(self) -> List[Dict[str, torch.Tensor]]:
        td = self.env.reset()
        frames = [self._snapshot()]
        with _deterministic_context():
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
        return {
            "agent_pos": env.agent_pos[0].detach().cpu(),
            "agent_alive": env.agent_alive[0].detach().cpu(),
            "agent_heading": env.agent_heading[0].detach().cpu(),
            "target_pos": env.target_pos[0].detach().cpu(),
            "target_alive": env.target_alive[0].detach().cpu(),
            "radar_pos": env.radar_pos[0].detach().cpu(),
            "radar_eff_range": env.radar_eff_range[0].detach().cpu(),
        }


def animate_rollout(frames: List[Dict[str, torch.Tensor]], env: StrikeEA2DEnv, interval_ms: int = 70):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title("Centralized PPO Rollout")

    striker_sc = ax.scatter([], [], s=60, marker="^", label="Strikers")
    jammer_sc = ax.scatter([], [], s=60, marker="s", label="Jammers")
    target_sc = ax.scatter([], [], s=80, marker="*", label="Targets")
    radar_sc = ax.scatter([], [], s=80, marker="X", label="Radars")

    radar_circles = [ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="C3", alpha=0.6, lw=2)) for _ in range(env.n_radars)]
    jammer_circles = [ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="C4", alpha=0.5, lw=1.5, ls="--")) for _ in range(env.n_jammers)]
    striker_arcs = [ax.add_patch(Polygon(np.empty((0, 2)), closed=True, fc="C2", alpha=0.18, ec="C2")) for _ in range(env.n_strikers)]
    heading_lines = [ax.plot([], [])[0] for _ in range(env.n_agents)]
    ax.legend(loc="upper right")

    empty_xy = np.empty((0, 2), dtype=float)

    def init():
        striker_sc.set_offsets(empty_xy)
        jammer_sc.set_offsets(empty_xy)
        target_sc.set_offsets(empty_xy)
        radar_sc.set_offsets(empty_xy)
        for c in [*radar_circles, *jammer_circles]:
            c.set_visible(False)
        for a in striker_arcs:
            a.set_visible(False)
        for ln in heading_lines:
            ln.set_data([], [])
        return [striker_sc, jammer_sc, target_sc, radar_sc, *radar_circles, *jammer_circles, *striker_arcs, *heading_lines]

    def update(i: int):
        fr = frames[i]
        ap = fr["agent_pos"]
        aa = fr["agent_alive"]
        ah = fr["agent_heading"]
        tp = fr["target_pos"]
        ta = fr["target_alive"]
        rp = fr["radar_pos"]
        rr = fr["radar_eff_range"]

        ap_km = ap * 1000
        tp_km = tp * 1000
        rp_km = rp * 1000

        striker_xy = ap_km[: env.n_strikers][aa[: env.n_strikers]]
        jammer_xy = ap_km[env.n_strikers:][aa[env.n_strikers:]]
        striker_sc.set_offsets(striker_xy.numpy() if striker_xy.numel() else empty_xy)
        jammer_sc.set_offsets(jammer_xy.numpy() if jammer_xy.numel() else empty_xy)
        target_sc.set_offsets(tp_km[ta].numpy() if ta.any() else empty_xy)
        radar_sc.set_offsets(rp_km.numpy() if rp_km.numel() else empty_xy)

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

        ax.set_xlabel(f"t={i} | Agents: {int(aa.sum().item())}/{env.n_agents} | Targets: {int(ta.sum().item())}/{env.n_targets}")
        return [striker_sc, jammer_sc, target_sc, radar_sc, *radar_circles, *jammer_circles, *striker_arcs, *heading_lines]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), init_func=init, interval=interval_ms, blit=True, repeat=False)
    plt.show()
    return ani