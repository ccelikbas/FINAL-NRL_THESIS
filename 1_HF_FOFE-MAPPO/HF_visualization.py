"""
Visualization for the high-fidelity angular radar model.

Extends the base visualization with angular coverage rendering:
  - Dashed circle at R_unconstrained (full range, no jamming)
  - Filled polar polygon showing per-angle effective detection range
  - Main-lobe and side-lobe notches cut by each active jammer

Re-exports unchanged functions (plot_training, plot_fofe_diagnostics,
plot_comparison) from visualization.py for convenience.
"""

from __future__ import annotations

import contextlib
import math
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.patches import Circle, Polygon

from .config import EnvConfig, HFRadarConfig
from .HF_environment import HFStrikeEA2DEnv

# Re-export unchanged utilities from base visualization
from .visualization import (
    plot_training,
    _plot_fofe_diagnostics as plot_fofe_diagnostics,
    plot_comparison,
)

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
# Angular coverage helpers
# ------------------------------------------------------------------

_N_ANGLE_POINTS = 360  # discretisation resolution for polar boundary


def _compute_radar_boundary_km(
    radar_x: float,
    radar_y: float,
    jammer_pos: np.ndarray,
    jammer_alive: np.ndarray,
    env: HFStrikeEA2DEnv,
) -> np.ndarray:
    """Compute polar boundary vertices for one radar's effective range.

    Parameters
    ----------
    radar_x, radar_y : float
        Radar position in world coordinates (NOT km).
    jammer_pos : np.ndarray [J, 2]
        Jammer positions in world coordinates.
    jammer_alive : np.ndarray [J]  bool
        Which jammers are alive.
    env : HFStrikeEA2DEnv
        Environment (for hf_cfg and radar_range_unconstrained).

    Returns
    -------
    np.ndarray [N, 2]
        Polygon vertices in **km** coordinates (world * 1000).
    """
    R_unc = env.radar_range_unconstrained
    hf_cfg = env.hf_cfg
    theta_main_half = math.radians(hf_cfg.theta_main_deg / 2)
    theta_side_half = math.radians(hf_cfg.theta_side_deg / 2)
    gt_over_gs = hf_cfg.G_t / hf_cfg.G_S

    angles = np.linspace(0, 2 * np.pi, _N_ANGLE_POINTS, endpoint=False)
    r_eff = np.full(_N_ANGLE_POINTS, R_unc)

    J = jammer_pos.shape[0]
    for j in range(J):
        if not jammer_alive[j]:
            continue
        jx, jy = jammer_pos[j]
        R_J = math.hypot(jx - radar_x, jy - radar_y)
        if R_J < 1e-12:
            R_J = 1e-12
        theta_j = math.atan2(jy - radar_y, jx - radar_x)

        # BT ranges (stand-off formulas, clamped to R_unc)
        R_main = min(math.sqrt(R_unc * R_J), R_unc)
        R_side = min((R_unc * R_unc * gt_over_gs * R_J * R_J) ** 0.25, R_unc)

        # Angular delta for each discrete angle
        delta = np.abs(angles - theta_j) % (2 * np.pi)
        delta = np.minimum(delta, 2 * np.pi - delta)

        # Per-angle effective range from this jammer
        jammer_r = np.where(
            delta <= theta_main_half,
            R_main,
            np.where(delta <= theta_side_half, R_side, R_unc),
        )

        # Deepest cut wins
        r_eff = np.minimum(r_eff, jammer_r)

    # Convert to cartesian km
    x_km = (radar_x + r_eff * np.cos(angles)) * 1000.0
    y_km = (radar_y + r_eff * np.sin(angles)) * 1000.0
    return np.column_stack([x_km, y_km])


# ------------------------------------------------------------------
# HF Test Runner
# ------------------------------------------------------------------

class HFTestRunner:
    """Rollout runner that uses HFStrikeEA2DEnv for the HF radar model."""

    def __init__(
        self,
        policy,
        *,
        env_cfg: EnvConfig,
        hf_cfg: HFRadarConfig,
        device: torch.device,
        seed: int = 999,
    ):
        self.policy = policy
        self.policy.eval()
        self.policy.deterministic = True
        self.env = HFStrikeEA2DEnv(
            hf_cfg=hf_cfg,
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


# ------------------------------------------------------------------
# HF Rollout Animation
# ------------------------------------------------------------------

def hf_animate_rollout(
    frames: List[Dict[str, torch.Tensor]],
    env: HFStrikeEA2DEnv,
    interval_ms: int = 70,
):
    """Animate a rollout with angular radar coverage rendering.

    For each radar the visualisation shows:
      - Thin red dashed circle at R_unconstrained (full range)
      - Filled light-red polygon showing the actual effective detection boundary
        with notches cut by active jammers (white = safe zones)
    """
    # --- Pre-compute reward time-series ---
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

    active_components: Dict[str, List[float]] = {}
    for name, vals in reward_ts.items():
        arr = np.asarray(vals, dtype=float)
        if np.nanmax(np.abs(arr)) > 1e-12:
            active_components[name] = vals

    # --- Figure layout ---
    fig, (ax, ax_rew) = plt.subplots(
        1, 2, figsize=(22, 9), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title(f"HF Radar | R_unc={env.radar_range_unconstrained:.3f} | "
                 f"R_obs={env.R_obs:.2f}, R_comm={env.R_comm:.2f}")

    empty_xy = np.empty((0, 2), dtype=float)

    # --- Static artists ---
    striker_sc = ax.scatter([], [], s=60, marker="^", label="Strikers")
    jammer_sc = ax.scatter([], [], s=60, marker="s", label="Jammers")
    target_known_sc = ax.scatter([], [], s=80, marker="*", label="Targets (known)", color="#a65e00")
    target_unknown_sc = ax.scatter(
        [], [], s=80, marker="*", label="Targets (unknown)",
        facecolors="none", edgecolors="#a65e00", linewidths=1.8,
    )
    radar_known_sc = ax.scatter([], [], s=80, marker="X", label="Radars (known)", color="#243c9b")
    radar_unknown_sc = ax.scatter(
        [], [], s=80, marker="X", label="Radars (unknown)",
        facecolors="none", edgecolors="#243c9b", linewidths=1.8,
    )

    # Dashed circles at R_unconstrained (always shown)
    R_unc_km = env.radar_range_unconstrained * 1000.0
    radar_unc_circles = [
        ax.add_patch(Circle((0, 0), R_unc_km, fill=False, edgecolor="red",
                            alpha=0.4, lw=1.0, ls="--"))
        for _ in range(env.n_radars)
    ]

    # Filled polygons for effective range (angular coverage)
    radar_coverage_polys = [
        ax.add_patch(Polygon(
            np.empty((0, 2)), closed=True,
            fc=(1.0, 0.6, 0.6, 0.25), ec="red", lw=1.2, alpha=0.7,
        ))
        for _ in range(env.n_radars)
    ]

    # Jammer, obs, comm circles
    jammer_circles = [
        ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="C4",
                            alpha=0.5, lw=1.5, ls="--"))
        for _ in range(env.n_jammers)
    ]
    obs_circles = [
        ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="C0",
                            alpha=0.35, lw=1.0, ls=":"))
        for _ in range(env.n_agents)
    ]
    comm_circles = [
        ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="black",
                            alpha=0.18, lw=0.9, ls="-."))
        for _ in range(env.n_agents)
    ]
    striker_arcs = [
        ax.add_patch(Polygon(np.empty((0, 2)), closed=True,
                             fc="C2", alpha=0.18, ec="C2"))
        for _ in range(env.n_strikers)
    ]
    heading_lines = [ax.plot([], [])[0] for _ in range(env.n_agents)]
    comm_lines = [
        ax.plot([], [], color="black", lw=1.4, alpha=0.7)[0]
        for _ in range(max(env.n_agents - 1, 1))
    ]
    ax.plot([], [], color="black", lw=1.4, alpha=0.7, label="Comm MST (<= R_comm)")
    ax.plot([], [], color="black", lw=0.9, ls="-.", alpha=0.35, label="R_comm")
    ax.plot([], [], color="C0", lw=1.0, ls=":", alpha=0.45, label="R_obs")
    ax.legend(loc="upper right")

    # --- Reward subplot ---
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

    # ---- init / update ----

    def init():
        striker_sc.set_offsets(empty_xy)
        jammer_sc.set_offsets(empty_xy)
        target_known_sc.set_offsets(empty_xy)
        target_unknown_sc.set_offsets(empty_xy)
        radar_known_sc.set_offsets(empty_xy)
        radar_unknown_sc.set_offsets(empty_xy)
        for c in [*radar_unc_circles, *jammer_circles, *obs_circles, *comm_circles]:
            c.set_visible(False)
        for p in radar_coverage_polys:
            p.set_visible(False)
        for a in striker_arcs:
            a.set_visible(False)
        for ln in heading_lines:
            ln.set_data([], [])
        for cl in comm_lines:
            cl.set_data([], [])
        time_vline.set_xdata([1])
        return []

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

        ap_km = ap * 1000
        tp_km = tp * 1000
        rp_km = rp * 1000

        # Scatter agents
        striker_xy = ap_km[:env.n_strikers][aa[:env.n_strikers]]
        jammer_xy = ap_km[env.n_strikers:][aa[env.n_strikers:]]
        striker_sc.set_offsets(striker_xy.numpy() if striker_xy.numel() else empty_xy)
        jammer_sc.set_offsets(jammer_xy.numpy() if jammer_xy.numel() else empty_xy)

        # Targets
        alive_known_targets = ta & tk
        alive_unknown_targets = ta & (~tk)
        target_known_sc.set_offsets(
            tp_km[alive_known_targets].numpy() if alive_known_targets.any() else empty_xy
        )
        target_unknown_sc.set_offsets(
            tp_km[alive_unknown_targets].numpy() if alive_unknown_targets.any() else empty_xy
        )

        # Radars scatter
        radar_known_sc.set_offsets(rp_km[rk].numpy() if rk.any() else empty_xy)
        radar_unknown_sc.set_offsets(rp_km[~rk].numpy() if (~rk).any() else empty_xy)

        # ---- HF Radar coverage polygons ----
        jammer_pos_np = ap[env.n_strikers:].numpy()        # [J, 2] world
        jammer_alive_np = aa[env.n_strikers:].numpy()      # [J] bool

        for j in range(env.n_radars):
            rx, ry = float(rp[j, 0]), float(rp[j, 1])

            # Dashed circle at R_unconstrained
            radar_unc_circles[j].set_visible(True)
            radar_unc_circles[j].set_center((rx * 1000, ry * 1000))
            radar_unc_circles[j].set_radius(R_unc_km)

            # Angular coverage polygon
            verts = _compute_radar_boundary_km(
                rx, ry, jammer_pos_np, jammer_alive_np, env
            )
            radar_coverage_polys[j].set_visible(True)
            radar_coverage_polys[j].set_xy(verts)

        # Jammer circles (show jam_radius for reference, though not used by HF model)
        for j, jc in enumerate(jammer_circles):
            idx = env.n_strikers + j
            if aa[idx].item():
                jc.set_visible(True)
                jc.set_center((float(ap_km[idx, 0]), float(ap_km[idx, 1])))
                jc.set_radius(env.jammer.jam_radius * 1000)
            else:
                jc.set_visible(False)

        # Obs / comm circles
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

        # Striker engagement arcs
        half_fov = 0.5 * env.striker.engage_fov_deg
        r_str = env.striker.engage_range * 1000
        for s, sa in enumerate(striker_arcs):
            if aa[s].item():
                cx, cy = float(ap_km[s, 0]), float(ap_km[s, 1])
                th1 = math.radians(math.degrees(float(ah[s])) - half_fov)
                th2 = th1 + math.radians(2 * half_fov)
                angs = np.linspace(th1, th2, 24)
                verts = np.vstack((
                    [cx, cy],
                    np.column_stack((cx + r_str * np.cos(angs),
                                     cy + r_str * np.sin(angs))),
                ))
                sa.set_visible(True)
                sa.set_xy(verts)
            else:
                sa.set_visible(False)

        # Heading lines
        for k, ln in enumerate(heading_lines):
            if aa[k].item():
                x, y = float(ap_km[k, 0]), float(ap_km[k, 1])
                ln.set_data(
                    [x, x + 30 * math.cos(float(ah[k]))],
                    [y, y + 30 * math.sin(float(ah[k]))],
                )
            else:
                ln.set_data([], [])

        # Communication MST
        alive_idx = torch.where(aa)[0]
        edges_for_plot = []
        if alive_idx.numel() > 1:
            alive_pos_world = ap[alive_idx]
            alive_pos_km_sel = ap_km[alive_idx]
            na = alive_idx.numel()
            dmat = torch.cdist(alive_pos_world, alive_pos_world)
            adj = dmat <= env.R_comm

            visited = torch.zeros(na, dtype=torch.bool)
            components: List[List[int]] = []
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
                for ci in range(len(comp)):
                    for cj in range(ci + 1, len(comp)):
                        u, v = comp[ci], comp[cj]
                        if bool(adj[u, v].item()):
                            cand.append((float(dmat[u, v].item()), u, v))
                cand.sort(key=lambda x: x[0])

                added = 0
                for _w, u, v in cand:
                    if _union(u, v):
                        p1 = alive_pos_km_sel[u]
                        p2 = alive_pos_km_sel[v]
                        edges_for_plot.append(
                            (float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))
                        )
                        added += 1
                        if added == len(comp) - 1:
                            break

        for li, line in enumerate(comm_lines):
            if li < len(edges_for_plot):
                x1, y1, x2, y2 = edges_for_plot[li]
                line.set_data([x1, x2], [y1, y2])
            else:
                line.set_data([], [])

        ax.set_xlabel(
            f"t={i} | Agents: {int(aa.sum().item())}/{env.n_agents} "
            f"| Targets: {int(ta.sum().item())}/{env.n_targets}"
        )

        reward_step = min(i, n_reward_steps) if n_reward_steps > 0 else 1
        time_vline.set_xdata([reward_step])

        return []

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), init_func=init,
        interval=interval_ms, blit=False, repeat=False,
    )
    plt.show()
    return ani
