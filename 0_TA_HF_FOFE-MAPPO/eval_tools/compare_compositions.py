"""compare_compositions.py — analyse ONE policy across several team compositions.

For each composition (n_strikers, n_jammers) the policy is rolled out for ``A``
different seeds. Every rollout is rendered as a single "tactical picture"
subplot showing:

  * agent trajectories (strikers = solid, jammers = dashed), start = hollow o,
    end = filled marker,
  * targets (survived vs destroyed) and radars,
  * each radar's un-jammed range (dashed) and its jammer-notched effective
    coverage at the end of the episode.

The subplots are laid out as an ``A x N`` dashboard (A rows = seeds/runs,
N cols = compositions) styled for use in a research paper.

Read-only: rolls out a checkpoint, writes a PNG. Nothing in the environment,
trainer, or config is modified.

Configure the run by editing the CONFIG block at the top of the file
(POLICY_PATH, CONFIGS, N_RUNS_PER_CONFIG, BASE_SEED). CLI flags override them.

Example (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\compare_compositions.py
"""
from __future__ import annotations
import argparse, copy, sys, types
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PKG_NAME = "fofe_mappo"
if __package__ in (None, ""):
    sys.path.insert(0, str(_THIS_DIR.parent))
    if _PKG_NAME not in sys.modules:
        _pkg = types.ModuleType(_PKG_NAME)
        _pkg.__path__ = [str(_THIS_DIR.parent), str(_THIS_DIR)]  # parent = sim modules, this dir = eval_tools
        _pkg.__package__ = _PKG_NAME
        _pkg.__file__ = str(_THIS_DIR / "__init__.py")
        sys.modules[_PKG_NAME] = _pkg
    __package__ = _PKG_NAME

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon, Circle

from .environment import coalition_fragmentation
from .HF_visualization import (
    HFTestRunner,
    _active_jammer_radar_mask_np,
    _compute_radar_boundary_km,
)
from .evaluate_policy import _LoadedCheckpoint, _build_policy_for_scenario
from .nlr_style import NLR_PRIMARY, NLR_ACCENT, NLR_DARKGRAY, NLR_GRAY

# ===================================================================
# CONFIG — edit these to control the analysis
# ===================================================================

# Policy to analyse. Relative paths are resolved against this file's folder
# so it works regardless of the current working directory.
POLICY_PATH = "runs/FINALV1/complete_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt"

# Compositions to test, as (n_strikers, n_jammers). One dashboard column each.
CONFIGS = [(2, 2), (2, 3), (2, 4)]

# Number of rollouts (different seeds) per composition. One dashboard row each.
N_RUNS_PER_CONFIG = 5

# Seeds are BASE_SEED, BASE_SEED+1, ... one per run/row.
BASE_SEED = 1002

# Coalition-fragmentation neighbour radius (world units), for the frag metric.
FRAG_RADIUS = 0.2

# Output dashboard PNG (relative paths resolved against this file's folder).
OUT_PATH = "eval_results/composition_dashboard.png"

# ===================================================================
# ENVIRONMENT CONFIG — the scenario the policy is evaluated in.
# These override the checkpoint's stored env config so the analysis runs
# in exactly the setup you want (should match how the policy was trained).
# ===================================================================

# Spawn scenario: "S1" or "S2".
SCENARIO = "S2"

# Target / radar counts. "known" = revealed to the agents at spawn,
# "unknown" = hidden until sensed. Totals are known + unknown.
N_KNOWN_TARGETS = 4
N_UNKNOWN_TARGETS = 0
N_KNOWN_RADARS = 6
N_UNKNOWN_RADARS = 0

# Match the training setup: FOFE observation encoding + inter-agent comms.
USE_FOFE = True
COMMUNICATE = True

# ===================================================================


# --- Marker / colour conventions (shared across every subplot) ---
_TARGET_ALIVE_C = "#a65e00"   # survived target
_TARGET_DEAD_C = NLR_GRAY     # destroyed target
_RADAR_C = "#243c9b"


def _resolve(path_str: str) -> Path:
    """Resolve a config path relative to this file's folder if not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else (_THIS_DIR.parent / p)  # runs/ , eval_results/ live in the parent (0_TA_...)


def _striker_colors(n: int):
    cmap = plt.get_cmap("winter")
    return [cmap(0.15 + 0.7 * (i / max(n - 1, 1))) for i in range(n)]


def _jammer_colors(n: int):
    cmap = plt.get_cmap("autumn")
    return [cmap(0.0 + 0.6 * (i / max(n - 1, 1))) for i in range(n)]


def rollout_config(ckpt, ns: int, nj: int, seed: int, device):
    """Roll the policy out once for one composition/seed. Returns (frames, env)."""
    env_cfg = copy.deepcopy(ckpt.base_env_cfg)
    env_cfg.n_strikers, env_cfg.n_jammers = ns, nj
    if hasattr(env_cfg, "dr"):
        env_cfg.dr = None  # deterministic layout for a clean picture

    # --- Apply the ENVIRONMENT CONFIG block ---
    # HFTestRunner reads these fields off env_cfg directly to build the env,
    # so set both the known/unknown counts and the resolved totals here
    # (normally EnvConfig.__post_init__ derives the totals; we bypass it).
    env_cfg.scenario = SCENARIO
    env_cfg.n_known_targets, env_cfg.n_unknown_targets = N_KNOWN_TARGETS, N_UNKNOWN_TARGETS
    env_cfg.n_targets = N_KNOWN_TARGETS + N_UNKNOWN_TARGETS
    env_cfg.n_known_radars, env_cfg.n_unknown_radars = N_KNOWN_RADARS, N_UNKNOWN_RADARS
    env_cfg.n_radars = N_KNOWN_RADARS + N_UNKNOWN_RADARS
    env_cfg.communicate = COMMUNICATE
    # FOFE observation encoding. The read-only use_fofe property reflects the
    # private _use_fofe field (normally set by finalize()); set it directly so
    # the env emits FOFE obs, matching the FOFE-trained policy.
    env_cfg._use_fofe = USE_FOFE

    policy = _build_policy_for_scenario(ckpt, env_cfg, device)
    runner = HFTestRunner(policy, env_cfg=env_cfg, hf_cfg=ckpt.hf_radar_cfg,
                          device=device, seed=seed)
    frames = runner.rollout()
    return frames, runner.env


def _draw_radar_coverage(ax, fr, env):
    """Draw each radar's R_unc circle + jammer-notched effective coverage (km)."""
    ap = fr["agent_pos"]
    aa = fr["agent_alive"]
    ah = fr["agent_heading"]
    rp = fr["radar_pos"]

    R_unc_km = env.radar_range_unconstrained * 1000.0
    lobe_half = getattr(env, "_jammer_main_lobe_half", 0.0)

    jammer_pos = ap[env.n_strikers:].numpy()
    jammer_alive = aa[env.n_strikers:].numpy()
    jammer_pointing = (
        ah[env.n_strikers:].numpy()
        + (fr["jammer_bearing"].numpy() if "jammer_bearing" in fr
           else np.zeros(env.n_jammers))
    )
    active_jr = _active_jammer_radar_mask_np(
        jammer_pos, jammer_alive, jammer_pointing, rp.numpy(),
        float(lobe_half), getattr(env, "_jammer_max_jammed_radars", None),
    )

    for j in range(env.n_radars):
        rx, ry = float(rp[j, 0]), float(rp[j, 1])
        ax.add_patch(Circle((rx * 1000, ry * 1000), R_unc_km, fill=False,
                            edgecolor="#c23b3b", alpha=0.35, lw=0.9, ls="--",
                            zorder=1))
        verts = _compute_radar_boundary_km(
            rx, ry, jammer_pos, jammer_alive, env,
            jammer_pointing=jammer_pointing, active_jammers=active_jr[:, j],
        )
        ax.add_patch(MplPolygon(verts, closed=True, fc=(0.76, 0.23, 0.23, 0.11),
                                ec=(0.76, 0.23, 0.23, 0.55), lw=0.9, zorder=1.2))


def draw_run(ax, frames, env, ns, nj):
    """Render one rollout as a tactical-picture subplot.

    Returns a dict of end-of-episode KPIs for this rollout:
        frag              seed-mean coalition fragmentation
        survival          fraction of agents still alive at the end
        targets_destroyed fraction of targets destroyed at the end
    """
    s_colors = _striker_colors(ns)
    j_colors = _jammer_colors(nj)

    # --- Trajectories + per-step fragmentation ---
    fr_frag = []
    s_traj = [[] for _ in range(ns)]
    j_traj = [[] for _ in range(nj)]
    for fr in frames:
        ap_ = fr["agent_pos"]
        al_ = fr["agent_alive"].bool()
        f, _ = coalition_fragmentation(ap_, FRAG_RADIUS, alive_mask=al_)
        fr_frag.append(float(f))
        for k in range(ns):
            s_traj[k].append(ap_[k].numpy())
        for k in range(nj):
            j_traj[k].append(ap_[ns + k].numpy())

    # --- Radar coverage (end-of-episode snapshot) behind everything ---
    _draw_radar_coverage(ax, frames[-1], env)

    # --- Targets / radars (initial layout; colour by survival at end) ---
    tp = frames[0]["target_pos"].numpy() * 1000
    ta_end = frames[-1]["target_alive"].bool().numpy()
    if (~ta_end).any():
        ax.scatter(tp[~ta_end, 0], tp[~ta_end, 1], s=90, marker="*",
                   facecolors="none", edgecolors=_TARGET_DEAD_C, linewidths=1.4,
                   zorder=5)
    if ta_end.any():
        ax.scatter(tp[ta_end, 0], tp[ta_end, 1], s=100, marker="*",
                   color=_TARGET_ALIVE_C, zorder=6)
    rp = frames[0]["radar_pos"].numpy() * 1000
    ax.scatter(rp[:, 0], rp[:, 1], s=80, marker="X", color=_RADAR_C, zorder=4)

    # --- Striker paths (solid) ---
    for k in range(ns):
        a = np.array(s_traj[k]) * 1000
        ax.plot(a[:, 0], a[:, 1], "-", color=s_colors[k], lw=1.8, zorder=3)
        ax.plot(a[0, 0], a[0, 1], "o", color=s_colors[k], ms=6, mfc="none",
                mew=1.4, zorder=3.5)
        ax.plot(a[-1, 0], a[-1, 1], "^", color=s_colors[k], ms=8, zorder=3.5)

    # --- Jammer paths (dashed) ---
    for k in range(nj):
        a = np.array(j_traj[k]) * 1000
        ax.plot(a[:, 0], a[:, 1], "--", color=j_colors[k], lw=1.5, zorder=3)
        ax.plot(a[0, 0], a[0, 1], "o", color=j_colors[k], ms=6, mfc="none",
                mew=1.4, zorder=3.5)
        ax.plot(a[-1, 0], a[-1, 1], "s", color=j_colors[k], ms=7, zorder=3.5)

    lo = float(getattr(env, "low", 0.0)) * 1000
    hi = float(getattr(env, "high", 1.0)) * 1000
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25)

    # --- End-of-episode KPIs (ta_end computed above) ---
    survival = float(frames[-1]["agent_alive"].bool().numpy().mean())
    targets_destroyed = float((~ta_end).mean())
    return {
        "frag": float(np.mean(fr_frag)),
        "survival": survival,
        "targets_destroyed": targets_destroyed,
    }


def _legend_handles():
    return [
        Line2D([0], [0], color=NLR_PRIMARY, lw=1.8, label="Striker path"),
        Line2D([0], [0], color=NLR_ACCENT, lw=1.5, ls="--", label="Jammer path"),
        Line2D([0], [0], marker="o", color=NLR_DARKGRAY, ls="none", mfc="none",
               mew=1.4, ms=7, label="Start"),
        Line2D([0], [0], marker="^", color=NLR_PRIMARY, ls="none", ms=8,
               label="Striker end"),
        Line2D([0], [0], marker="s", color=NLR_ACCENT, ls="none", ms=7,
               label="Jammer end"),
        Line2D([0], [0], marker="*", color=_TARGET_ALIVE_C, ls="none", ms=11,
               label="Target (survived)"),
        Line2D([0], [0], marker="*", color=_TARGET_DEAD_C, ls="none", ms=11,
               mfc="none", mew=1.4, label="Target (destroyed)"),
        Line2D([0], [0], marker="X", color=_RADAR_C, ls="none", ms=9,
               label="Radar"),
        Line2D([0], [0], color="#c23b3b", lw=0.9, ls="--", label="Radar range"),
        Patch(fc=(0.76, 0.23, 0.23, 0.11), ec=(0.76, 0.23, 0.23, 0.55),
              label="Effective coverage"),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=POLICY_PATH,
                    help="policy .pt to analyse (default: POLICY_PATH)")
    ap.add_argument("--runs", type=int, default=N_RUNS_PER_CONFIG,
                    help="rollouts (rows) per composition")
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    ckpt_path = _resolve(args.checkpoint)
    out_path = _resolve(args.out)
    A = args.runs
    N = len(CONFIGS)
    seeds = [args.base_seed + r for r in range(A)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {ckpt_path.name} on {device} ...")
    ckpt = _LoadedCheckpoint(ckpt_path, device)
    print(f"env: scenario={SCENARIO}  targets={N_KNOWN_TARGETS}k+{N_UNKNOWN_TARGETS}u  "
          f"radars={N_KNOWN_RADARS}k+{N_UNKNOWN_RADARS}u  "
          f"fofe={USE_FOFE}  communicate={COMMUNICATE}")

    fig, axes = plt.subplots(A, N, figsize=(4.6 * N, 4.6 * A), squeeze=False)

    for ci, (ns, nj) in enumerate(CONFIGS):
        cfg_metrics = []
        for ri in range(A):
            ax = axes[ri, ci]
            frames, env = rollout_config(ckpt, ns, nj, seeds[ri], device)
            m = draw_run(ax, frames, env, ns, nj)
            cfg_metrics.append(m)
            ax.set_title(
                f"seed {seeds[ri]}   ·   tgt={m['targets_destroyed']:.2f}   ·   "
                f"surv={m['survival']:.2f}   ·   frag={m['frag']:.2f}",
                fontsize=8)
            if ci == 0:
                ax.set_ylabel("Y (km)", fontsize=8)
            if ri == A - 1:
                ax.set_xlabel("X (km)", fontsize=8)
            print(f"  {ns}s{nj}j seed {seeds[ri]}: "
                  f"tgt={m['targets_destroyed']:.3f}  surv={m['survival']:.3f}  "
                  f"frag={m['frag']:.3f}")

        # Column header spanning the top subplot of each composition.
        mean_td = np.mean([m["targets_destroyed"] for m in cfg_metrics])
        mean_su = np.mean([m["survival"] for m in cfg_metrics])
        mean_fr = np.mean([m["frag"] for m in cfg_metrics])
        axes[0, ci].annotate(
            f"{ns} strikers · {nj} jammers\n"
            f"mean: tgt={mean_td:.2f} · surv={mean_su:.2f} · frag={mean_fr:.2f}",
            xy=(0.5, 1.16), xycoords="axes fraction", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=NLR_PRIMARY,
        )

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=5,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Composition-agnostic policy analysis — {ckpt_path.name}\n"
        f"{A} rollouts per composition (top = agent paths & radar coverage)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
