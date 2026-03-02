"""Visualisation utilities: training curves and rollout animation."""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Polygon


# ------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------

def plot_training(logs: Dict[str, List[float]], save_dir: Optional[str] = None):
    """Plot (and optionally save) reward and loss curves from a training run."""

    fig1, ax1 = plt.subplots()
    ax1.plot(logs["episode_reward_mean"])
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Mean episode reward")
    ax1.set_title("Training: Episode reward mean"); ax1.grid(True)

    fig2, ax2 = plt.subplots()
    for k in ("loss_total", "loss_policy", "loss_value", "loss_entropy"):
        ax2.plot(logs[k], label=k.replace("loss_", ""))
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Loss")
    ax2.set_title("Training: Loss curves"); ax2.legend(); ax2.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(os.path.join(save_dir, "episode_reward_mean.png"), dpi=160, bbox_inches="tight")
        fig2.savefig(os.path.join(save_dir, "loss_curves.png"),         dpi=160, bbox_inches="tight")

    plt.show()


# ------------------------------------------------------------------
# Rollout animation
# ------------------------------------------------------------------

def animate_rollout(
    frames:      List[Dict],
    env,                          # StrikeEA2DEnv (for geometry constants)
    interval_ms: int = 70,
    title:       str = "Strike–EA Rollout",
) -> animation.FuncAnimation:
    """Produce a matplotlib animation from rollout snapshots.
    
    Coordinate scaling: 1 unit = 1000 km, so axes show 0-1000 km.
    """

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
    ax.set_aspect("equal", adjustable="box"); ax.set_title(title)
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")

    striker_sc = ax.scatter([], [], s=60, marker="^", label="Strikers")
    jammer_sc  = ax.scatter([], [], s=60, marker="s", label="Jammers")
    target_sc  = ax.scatter([], [], s=80, marker="*", label="Targets")
    radar_sc   = ax.scatter([], [], s=80, marker="X", label="Radars")

    radar_circles   = [ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="C3", alpha=0.6, lw=2))            for _ in range(env.n_radars)]
    jammer_circles  = [ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="C4", alpha=0.5, lw=1.5, ls="--")) for _ in range(env.n_jammers)]
    striker_arcs    = [ax.add_patch(Polygon(np.empty((0, 2)), closed=True, fc="C2", alpha=0.18, ec="C2"))       for _ in range(env.n_strikers)]
    heading_lines   = [ax.plot([], [])[0] for _ in range(env.n_agents)]
    ax.legend(loc="upper right")

    empty_xy = np.empty((0, 2), dtype=float)

    def init():
        striker_sc.set_offsets(empty_xy); jammer_sc.set_offsets(empty_xy)
        target_sc.set_offsets(empty_xy);  radar_sc.set_offsets(empty_xy)
        for c in [*radar_circles, *jammer_circles]: c.set_visible(False)
        for a in striker_arcs:   a.set_visible(False)
        for ln in heading_lines: ln.set_data([], [])
        return [striker_sc, jammer_sc, target_sc, radar_sc, *radar_circles, *heading_lines]

    def update(i):
        fr  = frames[i]
        ap  = fr["agent_pos"]
        aa  = fr["agent_alive"]
        ah  = fr["agent_heading"]
        tp  = fr["target_pos"]
        ta  = fr["target_alive"]
        rp  = fr["radar_pos"]
        rr  = fr.get("radar_eff_range")

        # Scale positions from normalized (0-1) to km (0-1000)
        ap_km = ap * 1000
        tp_km = tp * 1000
        rp_km = rp * 1000
        
        striker_xy = ap_km[: env.n_strikers][aa[: env.n_strikers]]
        jammer_xy  = ap_km[env.n_strikers: ][aa[env.n_strikers: ]]
        striker_sc.set_offsets(striker_xy.numpy() if striker_xy.numel() else empty_xy)
        jammer_sc.set_offsets( jammer_xy.numpy()  if jammer_xy.numel()  else empty_xy)
        target_sc.set_offsets( tp_km[ta].numpy()  if ta.any()           else empty_xy)
        radar_sc.set_offsets(  rp_km.numpy()      if rp_km.numel()      else empty_xy)

        # radar effective-range circles (scaled to km)
        if rp_km.numel() and rr is not None:
            for j, c in enumerate(radar_circles):
                c.set_visible(True)
                c.set_center((float(rp_km[j, 0]), float(rp_km[j, 1])))
                c.set_radius(float(rr[j]) * 1000)

        # jammer jam-range circles (scaled to km)
        for j, jc in enumerate(jammer_circles):
            idx = env.n_strikers + j
            if aa[idx].item():
                jc.set_visible(True)
                jc.set_center((float(ap_km[idx, 0]), float(ap_km[idx, 1])))
                jc.set_radius(env.jammer.jam_radius * 1000)
            else:
                jc.set_visible(False)

        # striker FOV arcs (scaled to km)
        half_fov = 0.5 * env.striker.engage_fov_deg
        r_str    = env.striker.engage_range * 1000
        for s, sa in enumerate(striker_arcs):
            if aa[s].item():
                cx, cy = float(ap_km[s, 0]), float(ap_km[s, 1])
                th1    = math.radians(math.degrees(float(ah[s])) - half_fov)
                th2    = th1 + math.radians(2 * half_fov)
                angles = np.linspace(th1, th2, 24)
                verts  = np.vstack(([cx, cy], np.column_stack((cx + r_str * np.cos(angles), cy + r_str * np.sin(angles)))))
                sa.set_visible(True); sa.set_xy(verts)
            else:
                sa.set_visible(False)

        # heading indicator lines (scaled to km)
        heading_scale = 30
        for k, ln in enumerate(heading_lines):
            if aa[k].item():
                x, y = float(ap_km[k, 0]), float(ap_km[k, 1])
                ln.set_data([x, x + heading_scale * math.cos(float(ah[k]))],
                            [y, y + heading_scale * math.sin(float(ah[k]))])
            else:
                ln.set_data([], [])

        # Show alive counts in xlabel
        alive_agents = int(aa.sum().item())
        alive_targets = int(ta.sum().item())
        ax.set_xlabel(f"t={i} | Agents: {alive_agents}/{env.n_agents} | Targets: {alive_targets}/{env.n_targets}")
        return [striker_sc, jammer_sc, target_sc, radar_sc,
                *radar_circles, *jammer_circles, *striker_arcs, *heading_lines]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        init_func=init, interval=interval_ms, blit=True, repeat=False,
    )
    plt.show()
    return ani
