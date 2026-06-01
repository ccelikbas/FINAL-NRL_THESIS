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
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon

from .config import EnvConfig, HFRadarConfig
from .HF_environment import HFStrikeEA2DEnv
from .nlr_style import apply_nlr_style, NLR_PRIMARY, NLR_ACCENT, NLR_DARKGRAY

# Re-export unchanged utilities from base visualization
from .visualization import (
    plot_training,
    _plot_fofe_diagnostics as plot_fofe_diagnostics,
    plot_comparison,
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


# ------------------------------------------------------------------
# Angular coverage helpers
# ------------------------------------------------------------------

_N_ANGLE_POINTS = 360  # discretisation resolution for polar boundary


def _active_jammer_radar_mask_np(
    jammer_pos: np.ndarray,        # [J, 2] world
    jammer_alive: np.ndarray,      # [J] bool
    jammer_pointing: np.ndarray,   # [J] radians, world frame
    radar_pos: np.ndarray,         # [R, 2] world
    jammer_lobe_half: float,       # cone half-width (radians)
    max_jammed_radars: Optional[int],
) -> np.ndarray:
    """[J, R] bool — radar is actively jammed by jammer this frame.

    Mirrors ``HFStrikeEA2DEnv._restrict_to_closest_in_cone_radars`` (and
    the directional gate in ``_compute_hf_radar_eff_range``): a pair is
    active iff the jammer is alive, the radar lies inside the jammer's
    cone, and (when ``max_jammed_radars`` is set) the radar is among the
    K closest in-cone radars for that jammer.
    """
    J = jammer_pos.shape[0]
    R = radar_pos.shape[0]
    if J == 0 or R == 0:
        return np.zeros((J, R), dtype=bool)

    rel = radar_pos[None, :, :] - jammer_pos[:, None, :]        # [J, R, 2]
    angle_j_to_r = np.arctan2(rel[..., 1], rel[..., 0])         # [J, R]
    delta = angle_j_to_r - jammer_pointing[:, None]
    delta = np.arctan2(np.sin(delta), np.cos(delta))            # wrap to [-pi, pi]
    in_cone = (np.abs(delta) <= jammer_lobe_half) & jammer_alive[:, None]

    K = max_jammed_radars
    if K is not None and 0 < K < R:
        dist = np.linalg.norm(rel, axis=-1)                     # [J, R]
        dist_masked = np.where(in_cone, dist, np.inf)
        # Indices of the K smallest distances per jammer.
        top_idx = np.argpartition(dist_masked, K - 1, axis=-1)[:, :K]
        keep = np.zeros_like(in_cone)
        np.put_along_axis(keep, top_idx, True, axis=-1)
        in_cone = in_cone & keep

    return in_cone


def _compute_radar_boundary_km(
    radar_x: float,
    radar_y: float,
    jammer_pos: np.ndarray,
    jammer_alive: np.ndarray,
    env: HFStrikeEA2DEnv,
    jammer_pointing: Optional[np.ndarray] = None,
    active_jammers: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute polar boundary vertices for one radar's effective range.

    The underlying radar physics (main-/side-lobe BT ranges) is unchanged.
    With directional jammers we additionally gate each jammer's
    contribution by whether the radar lies inside that jammer's cone —
    matching ``HFStrikeEA2DEnv._compute_hf_radar_eff_range``.

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
    jammer_pointing : np.ndarray [J] or None
        World-frame pointing direction (radians) for each jammer
        (= heading + chosen bearing). If None, the boundary falls back
        to the legacy omnidirectional behaviour for that frame.
    active_jammers : np.ndarray [J] bool or None
        Per-jammer mask of which jammers are *currently* jamming this
        specific radar (i.e. radar passes the cone test AND is one of
        the jammer's K closest in-cone radars). Mirrors the K=cap
        applied in ``_compute_hf_radar_eff_range``. If None, the cap is
        ignored (legacy behaviour: every alive in-cone jammer cuts).

    Returns
    -------
    np.ndarray [N, 2]
        Polygon vertices in **km** coordinates (world * 1000).
    """
    R_unc = env.radar_range_unconstrained
    hf_cfg = env.hf_cfg
    theta_main_half = math.radians(hf_cfg.theta_main_deg / 2)
    theta_side_half = math.radians(hf_cfg.theta_side_deg / 2)
    jammer_lobe_half = getattr(env, "_jammer_main_lobe_half", None)
    meters_per_world_unit = env._meters_per_world_unit

    angles = np.linspace(0, 2 * np.pi, _N_ANGLE_POINTS, endpoint=False)
    r_eff = np.full(_N_ANGLE_POINTS, R_unc)

    J = jammer_pos.shape[0]
    for j in range(J):
        if not jammer_alive[j]:
            continue
        # K-closest capacity gate (mirrors env._restrict_to_closest_in_cone_radars).
        # When supplied, this already encodes both the cone test and the cap,
        # so the explicit cone branch below is skipped.
        if active_jammers is not None and not bool(active_jammers[j]):
            continue
        jx, jy = jammer_pos[j]
        R_J_world = math.hypot(jx - radar_x, jy - radar_y)
        R_J_m = max(R_J_world * meters_per_world_unit, 1e-12)
        theta_j = math.atan2(jy - radar_y, jx - radar_x)

        # Directional gate (only when no active_jammers mask was passed in).
        if (
            active_jammers is None
            and jammer_pointing is not None
            and jammer_lobe_half is not None
        ):
            angle_j_to_r = math.atan2(radar_y - jy, radar_x - jx)
            d_point = math.atan2(
                math.sin(angle_j_to_r - float(jammer_pointing[j])),
                math.cos(angle_j_to_r - float(jammer_pointing[j])),
            )
            if abs(d_point) > jammer_lobe_half:
                continue

        # BT ranges (direct JSR=1 formulas, clamped to R_unc)
        R_main_m = (max(env._r_main_bt_coeff * R_J_m * R_J_m, 0.0)) ** 0.25
        R_main = min(R_main_m / meters_per_world_unit, R_unc)
        R_side_m = (max(env._r_side_bt_coeff * R_J_m * R_J_m, 0.0)) ** 0.25
        R_side = min(R_side_m / meters_per_world_unit, R_unc)

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


def _jammer_cone_polygon_km(
    jx_world: float,
    jy_world: float,
    pointing_rad: float,
    half_width_rad: float,
    radius_km: float,
    n_arc: int = 32,
) -> np.ndarray:
    """Triangular wedge in km coordinates centered at the jammer.

    The wedge fan radius (``radius_km``) is purely a display choice —
    the directional jammer is treated as range-unlimited in the physics.
    """
    cx_km = jx_world * 1000.0
    cy_km = jy_world * 1000.0
    ang = np.linspace(
        pointing_rad - half_width_rad,
        pointing_rad + half_width_rad,
        n_arc,
    )
    arc_x = cx_km + radius_km * np.cos(ang)
    arc_y = cy_km + radius_km * np.sin(ang)
    return np.vstack((
        [cx_km, cy_km],
        np.column_stack([arc_x, arc_y]),
    ))


def _jammer_cone_ring_polygon_km(
    jx_world: float,
    jy_world: float,
    pointing_rad: float,
    half_width_rad: float,
    r_inner_km: float,
    r_outer_km: float,
    n_arc: int = 32,
) -> np.ndarray:
    """Annular wedge segment between r_inner_km and r_outer_km. When
    r_inner_km <= 0 this collapses to an apex triangle, matching the
    legacy `_jammer_cone_polygon_km` shape for the innermost ring.
    """
    cx_km = jx_world * 1000.0
    cy_km = jy_world * 1000.0
    ang = np.linspace(
        pointing_rad - half_width_rad,
        pointing_rad + half_width_rad,
        n_arc,
    )
    outer_x = cx_km + r_outer_km * np.cos(ang)
    outer_y = cy_km + r_outer_km * np.sin(ang)
    if r_inner_km <= 0.0:
        return np.vstack((
            [cx_km, cy_km],
            np.column_stack([outer_x, outer_y]),
        ))
    inner_x = cx_km + r_inner_km * np.cos(ang)
    inner_y = cy_km + r_inner_km * np.sin(ang)
    return np.vstack((
        np.column_stack([outer_x, outer_y]),
        np.column_stack([inner_x[::-1], inner_y[::-1]]),
    ))


# Cone gradient configuration: alpha tapers linearly from `_CONE_ALPHA_NEAR`
# at the jammer to 0 at the cone tip across `_CONE_N_RINGS` ring segments.
_CONE_N_RINGS = 16
_CONE_BASE_RGB = (1.0, 0.85, 0.2)
_CONE_ALPHA_NEAR = 0.35


def _make_cone_gradient_artists(ax) -> list:
    """Create the stack of ring-segment Polygons for one jammer cone."""
    alphas = np.linspace(_CONE_ALPHA_NEAR, 0.0, _CONE_N_RINGS + 1)[:-1]
    return [
        ax.add_patch(Polygon(
            np.empty((0, 2)), closed=True,
            fc=(*_CONE_BASE_RGB, float(a)), ec="none", zorder=1,
        ))
        for a in alphas
    ]


def _update_cone_gradient(
    patches: list,
    jx_world: float,
    jy_world: float,
    pointing_rad: float,
    half_width_rad: float,
    radius_km: float,
    n_arc: int = 32,
) -> None:
    r_edges = np.linspace(0.0, radius_km, len(patches) + 1)
    for k, patch in enumerate(patches):
        verts = _jammer_cone_ring_polygon_km(
            jx_world, jy_world,
            pointing_rad, half_width_rad,
            r_inner_km=float(r_edges[k]),
            r_outer_km=float(r_edges[k + 1]),
            n_arc=n_arc,
        )
        patch.set_visible(True)
        patch.set_xy(verts)


def _hide_cone_gradient(patches: list) -> None:
    for p in patches:
        p.set_visible(False)


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
            radar_min_sep=getattr(env_cfg, 'radar_min_sep', 0.5),
            scenario=getattr(env_cfg, 'scenario', 'S1'),
            s2_radar_min_sep=getattr(env_cfg, 's2_radar_min_sep', 0.2),
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
        # Directional jammer bearings (radians, relative to jammer heading).
        # Present when the HF env exposes them — pre-first-step they are
        # zero, which the animation treats as "pointing along own heading".
        if hasattr(env, "jammer_bearing"):
            snap["jammer_bearing"] = env.jammer_bearing[0].detach().cpu()
        if hasattr(env, "last_reward_components") and env.last_reward_components:
            snap["reward_components"] = {
                k: v[0].detach().cpu().clone() for k, v in env.last_reward_components.items()
            }
        # Per-(agent, radar) actor margin features used by the animation's
        # margin subplot. Pulled from _build_radar_extra so the values are
        # the geometric truth (un-masked by visibility). Columns:
        #   1 → range margin to R_unc      ∈ [-1, +1]
        #   2 → angular margin to cone     ∈ [-1, +1]
        if hasattr(env, "_build_radar_extra"):
            radar_extra = env._build_radar_extra(env.num_envs, env.n_agents, env.n_radars)
            snap["m_range"] = radar_extra[0, ..., 1].detach().cpu().clone()  # [A, R]
            snap["m_angle"] = radar_extra[0, ..., 2].detach().cpu().clone()  # [A, R]
        return snap


# ------------------------------------------------------------------
# HF Rollout Animation
# ------------------------------------------------------------------

def hf_animate_rollout(
    frames: List[Dict[str, torch.Tensor]],
    env: HFStrikeEA2DEnv,
    interval_ms: int = 70,
    save_path: Optional[str] = None,
    show: Optional[bool] = None,
):
    """Animate a rollout with angular radar coverage rendering.

    For each radar the visualisation shows:
      - Thin red dashed circle at R_unconstrained (full range)
      - Filled light-red polygon showing the actual effective detection boundary
        with notches cut by active jammers (white = safe zones)

    Parameters
    ----------
    save_path : str | None
        If provided, the rollout is saved as a GIF at this path (parents
        created if needed). If None, the animation is shown interactively.
    show : bool | None
        Display the figure window. Defaults to True when ``save_path`` is None
        and False otherwise so headless runs don't block.
    """
    # --- Pre-compute reward time-series (kept commented; replaced by the
    #     per-agent margin subplot below for verification of the new
    #     actor-obs features) ---
    # reward_ts: Dict[str, List[float]] = {}
    # total_ts: List[float] = []
    # for fr in frames:
    #     rc = fr.get("reward_components")
    #     if rc is None:
    #         continue
    #     step_total = 0.0
    #     for comp_name, comp_tensor in rc.items():
    #         val = float(comp_tensor.sum().item())
    #         reward_ts.setdefault(comp_name, []).append(val)
    #         step_total += val
    #     total_ts.append(step_total)
    #
    # active_components: Dict[str, List[float]] = {}
    # for name, vals in reward_ts.items():
    #     arr = np.asarray(vals, dtype=float)
    #     if np.nanmax(np.abs(arr)) > 1e-12:
    #         active_components[name] = vals

    # --- Pre-compute per-agent margin time-series ---
    # For each agent and timestep, take the MIN over radars of each margin
    # — i.e. the most-threatening radar per metric. Dead agents become NaN
    # so their lines break at the death step. Shape per series: [n_steps].
    n_agents_total = env.n_agents
    m_range_ts = [[] for _ in range(n_agents_total)]
    m_angle_ts = [[] for _ in range(n_agents_total)]
    for fr in frames:
        alive = fr.get("agent_alive")
        mr = fr.get("m_range")
        ma = fr.get("m_angle")
        if mr is None or ma is None or alive is None:
            continue
        # mr / ma : [A, R]   → reduce over R
        mr_per_agent = mr.min(dim=-1).values if mr.numel() > 0 else mr.new_zeros(n_agents_total)
        ma_per_agent = ma.min(dim=-1).values if ma.numel() > 0 else ma.new_zeros(n_agents_total)
        for k in range(n_agents_total):
            if bool(alive[k].item()):
                m_range_ts[k].append(float(mr_per_agent[k].item()))
                m_angle_ts[k].append(float(ma_per_agent[k].item()))
            else:
                m_range_ts[k].append(float("nan"))
                m_angle_ts[k].append(float("nan"))
    n_margin_steps = len(m_range_ts[0]) if n_agents_total > 0 else 0

    # --- Figure layout ---
    fig, (ax, ax_marg) = plt.subplots(
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
    striker_color = NLR_PRIMARY
    jammer_color = NLR_ACCENT
    striker_sc = ax.scatter([], [], s=60, marker="^", color=striker_color, label="Strikers")
    jammer_sc = ax.scatter([], [], s=60, marker="s", color=jammer_color, label="Jammers")

    # Trailing path behind each agent: per-agent fading LineCollection.
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

    # Obs / comm circles
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
    # Directional-jammer cone wedges (one per jammer), rendered as a
    # radial alpha gradient: opaque near the jammer, fully transparent
    # at the cone tip. Drawn behind other artists so markers stay
    # visible on top.
    jammer_lobe_half = getattr(env, "_jammer_main_lobe_half", 0.0)
    jammer_cone_polys = [
        _make_cone_gradient_artists(ax) for _ in range(env.n_jammers)
    ]
    heading_lines = [ax.plot([], [])[0] for _ in range(env.n_agents)]
    comm_lines = [
        ax.plot([], [], color="black", lw=1.4, alpha=0.7)[0]
        for _ in range(max(env.n_agents - 1, 1))
    ]
    ax.plot([], [], color="black", lw=1.4, alpha=0.7, label="Comm MST (<= R_comm)")
    ax.plot([], [], color="black", lw=0.9, ls="-.", alpha=0.35, label="R_comm")
    ax.plot([], [], color="C0", lw=1.0, ls=":", alpha=0.45, label="R_obs")
    if env.n_jammers > 0:
        ax.plot([], [], color=_CONE_BASE_RGB, lw=6, alpha=_CONE_ALPHA_NEAR,
                label=f"Jammer cone ({math.degrees(2*jammer_lobe_half):.0f} deg)")
    ax.plot([], [], color=NLR_DARKGRAY, lw=2.0, alpha=trail_max_alpha,
            label=f"Agent trail (last {trail_len} steps)")
    ax.legend(loc="lower right", fontsize=8, markerscale=0.6,
              handlelength=1.5, borderpad=0.3, labelspacing=0.3)

    # --- Reward subplot (replaced — kept commented for reference) ---
    # ax_rew.set_xlabel("Timestep")
    # ax_rew.set_ylabel("Reward (team sum)")
    # ax_rew.set_ylim(-0.25, 0.25)
    # ax_rew.set_title("Per-Timestep Reward Components")
    #
    # n_reward_steps = len(total_ts)
    # if n_reward_steps > 0:
    #     timesteps = np.arange(1, n_reward_steps + 1)
    #     for comp_name, values in sorted(active_components.items()):
    #         ax_rew.plot(timesteps, values, lw=1.5, alpha=0.8, label=comp_name)
    #     ax_rew.plot(timesteps, total_ts, lw=2.5, color="black", label="total", zorder=10)
    #     ax_rew.axhline(0, color="gray", lw=0.5)
    #     ax_rew.set_xlim(1, max(n_reward_steps, 2))
    #     ax_rew.legend(loc="best", fontsize=7, ncol=2)
    # ax_rew.grid(True, alpha=0.3)
    #
    # time_vline = ax_rew.axvline(x=1, color="red", lw=1.5, ls="--", alpha=0.8)

    # --- Margin subplot (per agent, min over radars) ---
    # Two lines per agent: solid = range margin (vs R_unc), dashed =
    # angular margin (vs ±θ_side/2 jammed-cone boundary). y-axis fixed
    # to [-1, +1] to match the actor obs scale. Positive = safe.
    ax_marg.set_xlabel("Timestep")
    ax_marg.set_ylabel("Margin (positive = safe)")
    ax_marg.set_ylim(-1.05, 1.05)
    ax_marg.set_title("Per-Agent Margins (min over radars)")
    ax_marg.axhline(0, color="gray", lw=0.5)

    if n_margin_steps > 0:
        timesteps = np.arange(1, n_margin_steps + 1)
        for k in range(n_agents_total):
            role = "striker" if k < env.n_strikers else "jammer"
            color = f"C{k}"
            ax_marg.plot(
                timesteps, m_range_ts[k],
                lw=1.6, ls="-", color=color, alpha=0.9,
                label=f"agent {k} ({role}) range",
            )
            ax_marg.plot(
                timesteps, m_angle_ts[k],
                lw=1.4, ls="--", color=color, alpha=0.9,
                label=f"agent {k} ({role}) angle",
            )
        ax_marg.set_xlim(1, max(n_margin_steps, 2))
        ax_marg.legend(loc="best", fontsize=7, ncol=2)
    ax_marg.grid(True, alpha=0.3)

    time_vline = ax_marg.axvline(x=1, color="red", lw=1.5, ls="--", alpha=0.8)
    # n_reward_steps left in scope for the update() callback's vline math.
    n_reward_steps = n_margin_steps

    # ---- init / update ----

    def init():
        striker_sc.set_offsets(empty_xy)
        jammer_sc.set_offsets(empty_xy)
        target_known_sc.set_offsets(empty_xy)
        target_unknown_sc.set_offsets(empty_xy)
        radar_known_sc.set_offsets(empty_xy)
        radar_unknown_sc.set_offsets(empty_xy)
        for c in [*radar_unc_circles, *obs_circles, *comm_circles]:
            c.set_visible(False)
        for p in radar_coverage_polys:
            p.set_visible(False)
        for a in striker_arcs:
            a.set_visible(False)
        for w in jammer_cone_polys:
            _hide_cone_gradient(w)
        for ln in heading_lines:
            ln.set_data([], [])
        for cl in comm_lines:
            cl.set_data([], [])
        for hist in agent_trail_history:
            hist.clear()
        for lc in agent_trail_lcs:
            lc.set_segments([])
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
        jammer_heading_np = ah[env.n_strikers:].numpy()    # [J] rad
        jammer_bearing_np = (
            fr["jammer_bearing"].numpy()
            if "jammer_bearing" in fr
            else np.zeros(env.n_jammers)
        )
        # World-frame pointing per jammer (heading + chosen bearing).
        jammer_pointing_np = jammer_heading_np + jammer_bearing_np

        # Per-frame mask of which jammer→radar pairs are actively
        # jamming after the K-closest cap (mirrors the physics in
        # HFStrikeEA2DEnv._compute_hf_radar_eff_range). [J, R] bool.
        active_jr = _active_jammer_radar_mask_np(
            jammer_pos_np,
            jammer_alive_np,
            jammer_pointing_np,
            rp.numpy(),
            float(jammer_lobe_half),
            getattr(env, "_jammer_max_jammed_radars", None),
        )

        for j in range(env.n_radars):
            rx, ry = float(rp[j, 0]), float(rp[j, 1])

            # Dashed circle at R_unconstrained
            radar_unc_circles[j].set_visible(True)
            radar_unc_circles[j].set_center((rx * 1000, ry * 1000))
            radar_unc_circles[j].set_radius(R_unc_km)

            # Angular coverage polygon. Only the (up to K) closest
            # in-cone jammers per radar contribute, matching the physics.
            verts = _compute_radar_boundary_km(
                rx, ry, jammer_pos_np, jammer_alive_np, env,
                jammer_pointing=jammer_pointing_np,
                active_jammers=active_jr[:, j],
            )
            radar_coverage_polys[j].set_visible(True)
            radar_coverage_polys[j].set_xy(verts)

        # Directional jammer cones (drawn from each jammer in its
        # currently-selected bearing). Length is a display-only choice;
        # the jammer is treated as range-unlimited in the physics.
        cone_radius_km = 1.5 * 1000.0 * (env.high - env.low)
        for j in range(env.n_jammers):
            if not bool(jammer_alive_np[j]):
                _hide_cone_gradient(jammer_cone_polys[j])
                continue
            jx, jy = float(jammer_pos_np[j, 0]), float(jammer_pos_np[j, 1])
            _update_cone_gradient(
                jammer_cone_polys[j],
                jx, jy,
                pointing_rad=float(jammer_pointing_np[j]),
                half_width_rad=jammer_lobe_half,
                radius_km=cone_radius_km,
            )

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
    # Save first (if requested), then optionally display. Saving after
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


_HF_LEGACY_LABEL = "MAPPO (Legacy)"
_HF_FOFE_LABEL = "FOFE-MAPPO"


def _draw_hf_world_panel(ax, env: HFStrikeEA2DEnv, frames: List[Dict[str, torch.Tensor]], title: str):
    """Set up one HF world panel and return an update callable."""
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title(
        f"{title} | HF Radar | R_unc={env.radar_range_unconstrained:.3f} | "
        f"R_obs={env.R_obs:.2f}, R_comm={env.R_comm:.2f}"
    )

    empty_xy = np.empty((0, 2), dtype=float)

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

    # Dashed circles at unconstrained range
    r_unc_km = env.radar_range_unconstrained * 1000.0
    radar_unc_circles = [
        ax.add_patch(Circle((0, 0), r_unc_km, fill=False, edgecolor="red", alpha=0.4, lw=1.0, ls="--"))
        for _ in range(env.n_radars)
    ]

    # Per-angle effective coverage polygons
    radar_coverage_polys = [
        ax.add_patch(Polygon(
            np.empty((0, 2)), closed=True,
            fc=(1.0, 0.6, 0.6, 0.25), ec="red", lw=1.2, alpha=0.7,
        ))
        for _ in range(env.n_radars)
    ]

    obs_circles = [
        ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="C0", alpha=0.35, lw=1.0, ls=":"))
        for _ in range(env.n_agents)
    ]
    comm_circles = [
        ax.add_patch(Circle((0, 0), 0, fill=False, edgecolor="black", alpha=0.18, lw=0.9, ls="-."))
        for _ in range(env.n_agents)
    ]
    striker_arcs = [
        ax.add_patch(Polygon(np.empty((0, 2)), closed=True, fc="C2", alpha=0.18, ec="C2"))
        for _ in range(env.n_strikers)
    ]
    jammer_lobe_half = getattr(env, "_jammer_main_lobe_half", 0.0)
    jammer_cone_polys = [
        _make_cone_gradient_artists(ax) for _ in range(env.n_jammers)
    ]
    heading_lines = [ax.plot([], [])[0] for _ in range(env.n_agents)]
    comm_lines = [
        ax.plot([], [], color="black", lw=1.4, alpha=0.7)[0]
        for _ in range(max(env.n_agents - 1, 1))
    ]
    ax.plot([], [], color="black", lw=1.4, alpha=0.7, label="Comm MST (<= R_comm)")
    ax.plot([], [], color="black", lw=0.9, ls="-.", alpha=0.35, label="R_comm")
    ax.plot([], [], color="C0", lw=1.0, ls=":", alpha=0.45, label="R_obs")
    if env.n_jammers > 0:
        ax.plot([], [], color=_CONE_BASE_RGB, lw=6, alpha=_CONE_ALPHA_NEAR,
                label=f"Jammer cone ({math.degrees(2*jammer_lobe_half):.0f} deg)")
    ax.legend(loc="upper right", fontsize=6)

    def _update_panel(i: int):
        fr = frames[min(i, len(frames) - 1)]
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

        # Agents
        striker_xy = ap_km[:env.n_strikers][aa[:env.n_strikers]]
        jammer_xy = ap_km[env.n_strikers:][aa[env.n_strikers:]]
        striker_sc.set_offsets(striker_xy.numpy() if striker_xy.numel() else empty_xy)
        jammer_sc.set_offsets(jammer_xy.numpy() if jammer_xy.numel() else empty_xy)

        # Targets
        alive_known_targets = ta & tk
        alive_unknown_targets = ta & (~tk)
        target_known_sc.set_offsets(tp_km[alive_known_targets].numpy() if alive_known_targets.any() else empty_xy)
        target_unknown_sc.set_offsets(tp_km[alive_unknown_targets].numpy() if alive_unknown_targets.any() else empty_xy)

        # Radars
        radar_known_sc.set_offsets(rp_km[rk].numpy() if rk.any() else empty_xy)
        radar_unknown_sc.set_offsets(rp_km[~rk].numpy() if (~rk).any() else empty_xy)

        # HF angular coverage
        jammer_pos_np = ap[env.n_strikers:].numpy()
        jammer_alive_np = aa[env.n_strikers:].numpy()
        jammer_heading_np = ah[env.n_strikers:].numpy()
        jammer_bearing_np = (
            fr["jammer_bearing"].numpy()
            if "jammer_bearing" in fr
            else np.zeros(env.n_jammers)
        )
        jammer_pointing_np = jammer_heading_np + jammer_bearing_np

        active_jr = _active_jammer_radar_mask_np(
            jammer_pos_np,
            jammer_alive_np,
            jammer_pointing_np,
            rp.numpy(),
            float(jammer_lobe_half),
            getattr(env, "_jammer_max_jammed_radars", None),
        )

        for j in range(env.n_radars):
            rx, ry = float(rp[j, 0]), float(rp[j, 1])

            radar_unc_circles[j].set_visible(True)
            radar_unc_circles[j].set_center((rx * 1000, ry * 1000))
            radar_unc_circles[j].set_radius(r_unc_km)

            verts = _compute_radar_boundary_km(
                rx, ry, jammer_pos_np, jammer_alive_np, env,
                jammer_pointing=jammer_pointing_np,
                active_jammers=active_jr[:, j],
            )
            radar_coverage_polys[j].set_visible(True)
            radar_coverage_polys[j].set_xy(verts)

        # Directional jammer cones (display-only, range-unlimited).
        cone_radius_km = 1.5 * 1000.0 * (env.high - env.low)
        for j in range(env.n_jammers):
            if not bool(jammer_alive_np[j]):
                _hide_cone_gradient(jammer_cone_polys[j])
                continue
            jx, jy = float(jammer_pos_np[j, 0]), float(jammer_pos_np[j, 1])
            _update_cone_gradient(
                jammer_cone_polys[j],
                jx, jy,
                pointing_rad=float(jammer_pointing_np[j]),
                half_width_rad=jammer_lobe_half,
                radius_km=cone_radius_km,
            )

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
                verts = np.vstack(([cx, cy], np.column_stack((cx + r_str * np.cos(angs), cy + r_str * np.sin(angs)))))
                sa.set_visible(True)
                sa.set_xy(verts)
            else:
                sa.set_visible(False)

        # Heading lines
        for k, ln in enumerate(heading_lines):
            if aa[k].item():
                x, y = float(ap_km[k, 0]), float(ap_km[k, 1])
                ln.set_data([x, x + 30 * math.cos(float(ah[k]))], [y, y + 30 * math.sin(float(ah[k]))])
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

        ax.set_xlabel(
            f"t={i} | Agents: {int(aa.sum().item())}/{env.n_agents} "
            f"| Targets: {int(ta.sum().item())}/{env.n_targets}"
        )

    return {}, _update_panel


def animate_hf_comparison_rollout(
    legacy_frames: List[Dict[str, torch.Tensor]],
    fofe_frames: List[Dict[str, torch.Tensor]],
    legacy_env: HFStrikeEA2DEnv,
    fofe_env: HFStrikeEA2DEnv,
    interval_ms: int = 70,
):
    """Side-by-side rollout animation with HF radar fidelity on both panels."""
    fig, (ax_l, ax_f) = plt.subplots(1, 2, figsize=(22, 9))

    _, update_legacy = _draw_hf_world_panel(ax_l, legacy_env, legacy_frames, _HF_LEGACY_LABEL)
    _, update_fofe = _draw_hf_world_panel(ax_f, fofe_env, fofe_frames, _HF_FOFE_LABEL)

    n_frames = max(len(legacy_frames), len(fofe_frames))

    def _init():
        return []

    def _update(i):
        update_legacy(i)
        update_fofe(i)
        return []

    fig.suptitle(
        f"HF Rollout Comparison: {_HF_LEGACY_LABEL}  vs  {_HF_FOFE_LABEL}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        init_func=_init,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    plt.show()
    return ani
