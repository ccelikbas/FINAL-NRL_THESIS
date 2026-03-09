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
    """Plot (and optionally save) reward, loss, and diagnostic curves from a training run."""

    figs = []

    # --- 1. Episode reward ---
    fig1, ax1 = plt.subplots()
    ax1.plot(logs["episode_reward_mean"])
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Mean episode reward")
    ax1.set_title("Training: Episode reward mean"); ax1.grid(True)
    figs.append((fig1, "episode_reward_mean.png"))

    # --- 2. Policy / value loss ---
    fig2, ax2 = plt.subplots()
    for k in ("loss_policy", "loss_value"):
        if k in logs:
            ax2.plot(logs[k], label=k.replace("loss_", ""))
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Loss")
    ax2.set_title("Training: Loss curves"); ax2.legend(); ax2.grid(True)
    figs.append((fig2, "loss_curves.png"))

    # --- 3. PPO diagnostic panel (entropy, approx_kl, clip_fraction, adv_std) ---
    diag_keys = [
        ("entropy",        "Entropy",        "Policy entropy"),
        ("approx_kl",      "Approx KL",      "Approximate KL divergence"),
        ("clip_fraction",  "Clip fraction",   "PPO clip fraction"),
        ("advantage_std",  "Advantage Std",   "Advantage standard deviation"),
    ]
    present = [(k, ylabel, title) for k, ylabel, title in diag_keys if k in logs and len(logs[k]) > 0]
    if present:
        n = len(present)
        fig3, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        for idx, (k, ylabel, title) in enumerate(present):
            ax = axes[0, idx]
            ax.plot(logs[k], color=f"C{idx + 2}")
            ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.grid(True)
        fig3.suptitle("PPO Diagnostics", fontsize=14, y=1.02)
        fig3.tight_layout()
        figs.append((fig3, "ppo_diagnostics.png"))

    # --- 4. Per-agent reward (if tracked) ---
    agent_rew_keys = [k for k in logs if k.startswith("reward_agent_")]
    if agent_rew_keys:
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        for k in sorted(agent_rew_keys):
            ax4.plot(logs[k], label=k.replace("reward_", ""))
        ax4.set_xlabel("Iteration"); ax4.set_ylabel("Mean per-agent reward")
        ax4.set_title("Per-Agent Reward"); ax4.legend(); ax4.grid(True)
        figs.append((fig4, "per_agent_reward.png"))

    # --- 5. Per-agent entropy (if tracked) ---
    agent_ent_keys = [k for k in logs if k.startswith("entropy_agent_")]
    if agent_ent_keys:
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        for k in sorted(agent_ent_keys):
            ax5.plot(logs[k], label=k.replace("entropy_", ""))
        ax5.set_xlabel("Iteration"); ax5.set_ylabel("Policy entropy")
        ax5.set_title("Per-Agent Policy Entropy"); ax5.legend(); ax5.grid(True)
        figs.append((fig5, "per_agent_entropy.png"))

    # --- Save all figures ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for fig, fname in figs:
            fig.savefig(os.path.join(save_dir, fname), dpi=160, bbox_inches="tight")

    plt.show()


# ------------------------------------------------------------------
# Evaluation reward-component plots
# ------------------------------------------------------------------

def plot_evaluation_rewards(
    results: Dict,
    save_dir: Optional[str] = None,
    max_step: Optional[int] = None,
):
    """Plot per-step reward components from a PolicyEvaluator run.

    Creates two figures:
    1. **Stacked area** – reward components over timesteps (shows relative
       magnitude and sign of each component, plus total reward line).
    2. **Individual lines** – each component as a separate line so the scale
       of small components is visible.

    Parameters
    ----------
    results : dict returned by ``PolicyEvaluator.evaluate()``.
    save_dir : optional directory to save PNGs.
    max_step : trim x-axis to this step (default: auto from episode counts).
    """
    rps = results.get("reward_components_per_step")
    if rps is None:
        return

    ep_counts = np.array(rps["_episode_count"])
    # Auto-determine max relevant step (where at least 5 % of episodes were still running)
    if max_step is None:
        threshold = max(1, int(0.05 * ep_counts[0]))
        valid = np.where(ep_counts >= threshold)[0]
        max_step = int(valid[-1]) + 1 if len(valid) else len(ep_counts)

    x = np.arange(max_step)

    component_names = [
        "target_destroyed", "striker_approach", "jammer_approach",
        "timestep_penalty", "border_penalty", "radar_avoidance",
        "agent_destroyed",
    ]
    colors = {
        "target_destroyed": "#2ca02c",   # green  – positive sparse
        "striker_approach":  "#1f77b4",   # blue   – positive dense
        "jammer_approach":   "#17becf",   # cyan   – positive dense
        "timestep_penalty":  "#ff7f0e",   # orange – negative dense
        "border_penalty":    "#d62728",   # red    – negative dense
        "radar_avoidance":   "#9467bd",   # purple – negative dense
        "agent_destroyed":   "#e377c2",   # pink   – negative sparse
    }
    labels = {
        "target_destroyed": "Target destroyed",
        "striker_approach":  "Striker approach",
        "jammer_approach":   "Jammer approach",
        "timestep_penalty":  "Timestep penalty",
        "border_penalty":    "Border penalty",
        "radar_avoidance":   "Radar avoidance",
        "agent_destroyed":   "Agent destroyed",
    }

    # Gather arrays
    comp_arrays = {}
    for name in component_names:
        arr = np.array(rps.get(name, [0.0] * max_step))[:max_step]
        comp_arrays[name] = arr

    total = np.array(rps.get("total", [0.0] * max_step))[:max_step]

    # --- Figure 1: Stacked area with total line ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Separate positive and negative components for stacked fill
    pos_names = [n for n in component_names if comp_arrays[n].max() > 1e-8]
    neg_names = [n for n in component_names if comp_arrays[n].min() < -1e-8]

    # Stack positives from 0 upward
    pos_cum = np.zeros(max_step)
    for name in pos_names:
        arr = np.maximum(comp_arrays[name], 0)
        ax1.fill_between(x, pos_cum, pos_cum + arr, alpha=0.55,
                         color=colors[name], label=labels[name])
        pos_cum += arr

    # Stack negatives from 0 downward
    neg_cum = np.zeros(max_step)
    for name in neg_names:
        arr = np.minimum(comp_arrays[name], 0)
        ax1.fill_between(x, neg_cum, neg_cum + arr, alpha=0.55,
                         color=colors[name],
                         label=labels[name] if name not in pos_names else None)
        neg_cum += arr

    ax1.plot(x, total, color="black", lw=2, label="Total reward")
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Mean reward (summed over agents)")
    n_ep = int(results.get("n_episodes", 0))
    ax1.set_title(f"Per-Step Reward Components (avg over {n_ep} episodes)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Figure 2: Individual component lines ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for name in component_names:
        arr = comp_arrays[name]
        if np.abs(arr).max() > 1e-10:
            ax2.plot(x, arr, label=labels[name], color=colors[name], lw=1.5)
    ax2.plot(x, total, color="black", lw=2, ls="--", label="Total reward")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Mean reward (summed over agents)")
    ax2.set_title(f"Per-Step Reward Components – Line View (avg over {n_ep} episodes)")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Figure 3: Cumulative reward over time ---
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    cum_total = np.cumsum(total)
    for name in component_names:
        arr = comp_arrays[name]
        if np.abs(arr).max() > 1e-10:
            ax3.plot(x, np.cumsum(arr), label=labels[name], color=colors[name], lw=1.5)
    ax3.plot(x, cum_total, color="black", lw=2, label="Total (cumulative)")
    ax3.axhline(0, color="gray", lw=0.5)
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Cumulative reward (summed over agents)")
    ax3.set_title(f"Cumulative Reward Components (avg over {n_ep} episodes)")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(os.path.join(save_dir, "eval_reward_components_stacked.png"), dpi=160, bbox_inches="tight")
        fig2.savefig(os.path.join(save_dir, "eval_reward_components_lines.png"),   dpi=160, bbox_inches="tight")
        fig3.savefig(os.path.join(save_dir, "eval_reward_cumulative.png"),         dpi=160, bbox_inches="tight")

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
