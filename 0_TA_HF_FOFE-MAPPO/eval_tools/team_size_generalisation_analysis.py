"""team_size_generalisation_analysis.py — sweep ONE trained policy over a GRID of
team compositions (n_strikers × n_jammers) and render a dashboard of KPI heatmaps,
all computed from the SAME rollouts. Read-only (never modifies a checkpoint).

For every (n_strikers, n_jammers) cell the policy is rolled out for N_RUNS parallel
episodes (× N_SEEDS seeds) in the fixed evaluation world (scenario / radars / targets
/ kill below). Each cell yields the mean of every KPI; the KPIs are drawn as a grid
of heatmaps (rows = strikers, cols = jammers, colour = KPI value):

    survival · task completion · targets destroyed · fragmentation (frag)
    · mean episode reward · mean duration

The FOFE (or fixed-slot legacy) policy can be evaluated at any composition because
its per-role obs encoding is composition-agnostic; cells far from the training
composition are extrapolation and are expected to degrade. The training region is
outlined on each heatmap for reference.

Run (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\team_size_generalisation_analysis.py \\
      --checkpoint runs\\2s2-4jV6.pt
(--checkpoint accepts an absolute path, a path relative to the repo root, or one
relative to this file's folder; the default POLICY_PATH below is file-relative.)
"""
from __future__ import annotations
import argparse, copy, sys, traceback, types
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
from matplotlib.patches import Rectangle

from .config import PPOConfig
from .trainer import build_env
from .environment import coalition_fragmentation
from .evaluate_policy import _LoadedCheckpoint, _build_policy_for_scenario

# ===================================================================
# CONFIG — edit these (CLI flags override)
# ===================================================================
POLICY_PATH = "runs/2s2-4jV7.pt"
STRIKERS = [1, 2, 3, 4]              # y-axis of the grid
JAMMERS = [1, 2, 3, 4, 5, 6, 7, 8]  # x-axis of the grid
N_RUNS = 128                         # parallel episodes per cell (per seed)
N_SEEDS = 2                          # repeats per cell (averaged); raise for smoother maps
BASE_SEED = 500

# Fixed evaluation world (match training). Kept constant across the whole sweep.
SCENARIO = "S2"
N_KNOWN_RADARS, N_UNKNOWN_RADARS = 6, 0
N_KNOWN_TARGETS, N_UNKNOWN_TARGETS = 2, 0
KILL = 0.5
FRAG_RADIUS = 0.2
USE_FOFE = True
COMMUNICATE = True

# Training region to outline on each heatmap: strikers in TRAIN_S, jammers in TRAIN_J
TRAIN_S = [2]
TRAIN_J = [2, 3, 4]

OUT_PATH = "escort_analysis/team_size_generalisation.png"

KPIS = [  # (key, title, colormap, value range or None)
    ("survival",          "Survival rate",        "viridis", (0, 1)),
    ("completion",        "Task completion rate", "viridis", (0, 1)),
    ("targets_destroyed", "Targets-destroyed rate", "viridis", (0, 1)),
    ("frag",              "Fragmentation (frag)", "magma",   (0, None)),
    ("mean_reward",       "Mean episode reward",  "cividis", (None, None)),
    ("mean_duration",     "Mean duration (steps)", "cividis", (None, None)),
]


def _resolve(p):
    p = Path(p)
    if p.is_absolute():
        return p
    if p.exists():              # relative to CWD (e.g. repo root)
        return p.resolve()
    return _THIS_DIR.parent / p  # else relative to the parent (0_TA_...); runs/ , escort_analysis/ live there


def _alive_of(env):
    for n in ("agent_alive", "alive", "_alive"):
        a = getattr(env, n, None)
        if a is not None:
            return a.bool()


def _make_cfg(base, ns, nj):
    ec = copy.deepcopy(base)
    ec.n_strikers, ec.n_jammers = ns, nj
    ec.scenario = SCENARIO
    ec.n_known_targets, ec.n_unknown_targets = N_KNOWN_TARGETS, N_UNKNOWN_TARGETS
    ec.n_targets = N_KNOWN_TARGETS + N_UNKNOWN_TARGETS
    ec.n_known_radars, ec.n_unknown_radars = N_KNOWN_RADARS, N_UNKNOWN_RADARS
    ec.n_radars = N_KNOWN_RADARS + N_UNKNOWN_RADARS
    ec.radar_kill_probability = KILL
    ec.communicate = COMMUNICATE
    ec._use_fofe = USE_FOFE
    ec.dr = None
    return ec


def run_cell(ckpt, base, ns, nj, n_runs, seed, device):
    """One composition, n_runs parallel episodes. Returns dict of mean KPIs."""
    ec = _make_cfg(base, ns, nj)
    policy = _build_policy_for_scenario(ckpt, ec, device)
    policy.eval(); policy.deterministic = True
    env = build_env(ec, PPOConfig(num_envs=n_runs, device=str(device)), hf_radar_cfg=ckpt.hf_radar_cfg)
    B = n_runs
    frag_sum = np.zeros(B); frag_cnt = np.zeros(B)
    rew_sum = np.zeros(B); ep_len = np.zeros(B)
    fin_alive = None; fin_talive = None
    with torch.no_grad():
        td = env.reset()
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(env.max_steps):
            td = policy(td); td = env.step(td)
            al = _alive_of(env); pos = env.agent_pos
            active = ~done
            am = active.detach().cpu().numpy()
            f, _ = coalition_fragmentation(pos, FRAG_RADIUS, alive_mask=al)
            frag_sum += np.where(am, f.detach().cpu().numpy(), 0.0); frag_cnt += am
            step_r = sum(v.sum(-1) for v in env.last_reward_components.values())  # [B] team reward
            rew_sum += np.where(am, step_r.detach().cpu().numpy(), 0.0)
            ep_len += am
            ta = env.target_alive.bool().detach().cpu().numpy()
            alc = al.detach().cpu().numpy()
            if fin_alive is None:
                fin_alive = alc.copy(); fin_talive = ta.copy()
            fin_alive[am] = alc[am]; fin_talive[am] = ta[am]
            done = done | td.get(("next", "done")).reshape(B).bool()
            if bool(done.all()):
                break
            td = td.get("next")
    return dict(
        survival=float(fin_alive.mean()),
        completion=float(np.mean(~fin_talive.any(axis=1))),
        targets_destroyed=float(1.0 - fin_talive.mean()),
        frag=float(np.mean(frag_sum / np.clip(frag_cnt, 1, None))),
        mean_reward=float(rew_sum.mean()),
        mean_duration=float(ep_len.mean()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=POLICY_PATH)
    ap.add_argument("--n_runs", type=int, default=N_RUNS)
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    ap.add_argument("--strikers", default=None, help="comma list override, e.g. '2' or '1,2,3'")
    ap.add_argument("--jammers", default=None, help="comma list override, e.g. '2,4'")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    strikers = [int(x) for x in args.strikers.split(",")] if args.strikers else STRIKERS
    jammers = [int(x) for x in args.jammers.split(",")] if args.jammers else JAMMERS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _LoadedCheckpoint(_resolve(args.checkpoint), device)
    base = ckpt.base_env_cfg
    name = Path(args.checkpoint).stem
    print(f"policy={name}  grid: strikers={strikers} x jammers={jammers}  "
          f"n_runs={args.n_runs} x n_seeds={args.n_seeds}  world: S2 {N_KNOWN_RADARS}kr {N_KNOWN_TARGETS}kt kill={KILL}\n")

    nS, nJ = len(strikers), len(jammers)
    grids = {k: np.full((nS, nJ), np.nan) for k, *_ in KPIS}
    for si, ns in enumerate(strikers):
        for ji, nj in enumerate(jammers):
            try:
                acc = []
                for s in range(args.n_seeds):
                    acc.append(run_cell(ckpt, base, ns, nj, args.n_runs, BASE_SEED + s, device))
                cell = {k: float(np.mean([a[k] for a in acc])) for k in grids}
                for k in grids:
                    grids[k][si, ji] = cell[k]
                print(f"  {ns}s{nj}j: surv={cell['survival']:.2f} comp={cell['completion']:.2f} "
                      f"tgt={cell['targets_destroyed']:.2f} frag={cell['frag']:.3f} "
                      f"rew={cell['mean_reward']:.1f} dur={cell['mean_duration']:.0f}")
            except Exception as exc:  # noqa: BLE001 — mark failed cells, keep sweeping
                print(f"  {ns}s{nj}j: FAILED ({type(exc).__name__}: {exc})")
                traceback.print_exc()

    # ---- dashboard of heatmaps ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (key, title, cmap, vr) in zip(axes.ravel(), KPIS):
        g = grids[key]
        vmin = vr[0] if vr and vr[0] is not None else np.nanmin(g)
        vmax = vr[1] if vr and vr[1] is not None else np.nanmax(g)
        im = ax.imshow(g, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(nJ)); ax.set_xticklabels(jammers)
        ax.set_yticks(range(nS)); ax.set_yticklabels(strikers)
        ax.set_xlabel("n_jammers"); ax.set_ylabel("n_strikers")
        ax.set_title(title, fontsize=11)
        for si in range(nS):
            for ji in range(nJ):
                v = g[si, ji]
                if not np.isnan(v):
                    ax.text(ji, si, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="white")
        # outline training region
        for si, ns in enumerate(strikers):
            for ji, nj in enumerate(jammers):
                if ns in TRAIN_S and nj in TRAIN_J:
                    ax.add_patch(Rectangle((ji - 0.5, si - 0.5), 1, 1, fill=False,
                                           edgecolor="red", lw=1.6))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Team-size generalisation — {name}  (red = training region; "
                 f"{args.n_runs}×{args.n_seeds} episodes/cell, S2 {N_KNOWN_RADARS} radars kill={KILL})",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = _resolve(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    # CSV dump
    csv = out.with_suffix(".csv")
    with open(csv, "w") as f:
        f.write("kpi,n_strikers,n_jammers,value\n")
        for k in grids:
            for si, ns in enumerate(strikers):
                for ji, nj in enumerate(jammers):
                    f.write(f"{k},{ns},{nj},{grids[k][si,ji]:.4f}\n")
    print(f"\nsaved dashboard -> {out}")
    print(f"saved CSV       -> {csv}")


if __name__ == "__main__":
    main()
