"""team_size_generalisation_analysis.py — sweep a trained policy (or COMPARE two
policies) over a GRID of team compositions (n_strikers × n_jammers) and render KPI
heatmaps, all from the SAME rollouts. Read-only (never modifies a checkpoint).

For every (n_strikers, n_jammers) cell a policy is rolled out for N_RUNS parallel
episodes (× N_SEEDS seeds) in the fixed evaluation world (scenario / radars / targets
/ kill below). Each cell yields per-episode KPIs; the cell means are drawn as heatmaps
(rows = strikers, cols = jammers, colour = KPI value).

TWO MODES
---------
* SINGLE policy (only --checkpoint set): the original 6-KPI dashboard
      survival · task completion · targets destroyed · fragmentation
      · mean episode reward · mean duration

* COMPARE two policies (--checkpoint = complete model, --baseline = baseline model):
  a 3×3 dashboard —
      rows  = Targets destroyed · Survival rate · Duration
      cols  = complete · baseline · difference (complete − baseline)
  The complete & baseline columns share one colour scale per row for easy comparison;
  the difference column uses a diverging map centred at 0. Every square prints the
  cell mean with its 95% confidence interval underneath (mean ± CI_Z·SE over the
  cell's per-episode samples; the Δ column uses the PAIRED per-episode difference,
  so a Δ interval excluding 0 marks a cell where the two policies differ).
  A separate table + CSV
  reports, per cell, the PAIRED one-sided Wilcoxon signed-rank p-value between the two
  policies (targets/survival: higher-is-better; duration: lower-is-better). Pairing is
  by shared per-episode seeds, so both policies see identical worlds (common random
  numbers) and the signed-rank test is valid.

The radar / target COUNTS are set in ONE place per channel (KNOWN_RADARS, KNOWN_TARGETS,
…): give an int for a FIXED count, or an inclusive (lo, hi) tuple to DOMAIN-RANDOMISE it
per episode (each parallel env samples its own count in [lo, hi] at reset). Override per
run with --known_radars / --known_targets etc. (pass 'n' or 'lo,hi'). Under a range the
per-episode rates are normalised to the PRESENT entities, so a cell mixes world
complexities. See DomainRandomization in config.py; hi becomes the allocated count.

The FOFE (or fixed-slot legacy) policy can be evaluated at any composition because its
per-role obs encoding is composition-agnostic; cells far from the training composition
are extrapolation and are expected to degrade. The training region is outlined in red.

Run (repo root, project venv):
  # single policy
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\team_size_generalisation_analysis.py \\
      --checkpoint runs\\2s2-4jV7.pt
  # compare two policies
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\team_size_generalisation_analysis.py \\
      --checkpoint runs\\complete.pt --baseline runs\\baseline.pt
(--checkpoint / --baseline accept an absolute path, a path relative to the repo root,
or one relative to this file's folder; defaults below are file-relative.)
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import wilcoxon

from .config import PPOConfig, DomainRandomization
from .trainer import build_env
from .environment import coalition_fragmentation
from .evaluate_policy import _LoadedCheckpoint, _build_policy_for_scenario
from .nlr_style import (
    NLR_DARKBLUE, NLR_LIGHTBLUE, NLR_LIGHTBLUE_20, NLR_LIGHTBLUE_50,
    NLR_TERRA, NLR_TERRA_20, NLR_TERRA_50, NLR_DARKGRAY,
)
# Shared data drop: this sweep's grids are also written to
# eval_results/analysis_data/, so the presentation figures can be (re)built from
# them without re-running the sweep. See analysis_data.py.
from .analysis_data import save_analysis

# NLR house-style colormaps.
#   sequential  — white → light blue → dark blue  (KPI magnitude)
#   diverging   — terra ← white → dark blue        (difference, centred at 0)
NLR_SEQ = LinearSegmentedColormap.from_list(
    "nlr_seq", ["#ffffff", NLR_LIGHTBLUE_20, NLR_LIGHTBLUE, NLR_DARKBLUE])
NLR_DIV = LinearSegmentedColormap.from_list(
    "nlr_div", [NLR_TERRA, NLR_TERRA_50, NLR_TERRA_20, "#ffffff",
                NLR_LIGHTBLUE_20, NLR_LIGHTBLUE_50, NLR_DARKBLUE])
NLR_SEQ.set_bad("#eeeeee"); NLR_DIV.set_bad("#eeeeee")

# ===================================================================
# CONFIG — edit these (CLI flags override)
# ===================================================================
POLICY_PATH = "runs/FINALV2/complete_stage7of8_DR_j2-4_k0_25.pt"    # "complete" model (--checkpoint)
BASELINE_PATH = "runs/FINALV2/Final_Baseline_Cont_4.pt"  # "baseline" model (--baseline); None → single-policy mode
STRIKERS = [1, 2, 3]             # y-axis of the grid
JAMMERS = [1, 2, 3, 4, 5, 6]  # x-axis of the grid
N_RUNS = 1000                         # parallel episodes per cell (per seed)
N_SEEDS = 1                         # repeats per cell (concatenated); raise for stronger tests
BASE_SEED = 42

# Evaluation world (match training). Radar / target COUNTS: give an int for a
# FIXED count, or an inclusive (lo, hi) tuple to DOMAIN-RANDOMISE it per episode
# (each parallel env in a cell samples its own count in [lo, hi] at reset). Under
# a range the env allocates tensors at hi and masks each episode down, and the
# per-episode rates are normalised to the PRESENT entities, so a cell mixes world
# complexities. Override per run with --known_radars / --known_targets etc.,
# passing either 'n' (fixed) or 'lo,hi' (randomised).
SCENARIO = "S2"
KNOWN_RADARS    = (4,6)        # e.g. (2, 6) to randomise known radars in [2, 6]
UNKNOWN_RADARS  = 0
KNOWN_TARGETS   = (2,4)        # e.g. (1, 3) to randomise known targets in [1, 3]
UNKNOWN_TARGETS = 0
KILL = 0.25
FRAG_RADIUS = 0.2
USE_FOFE = True
COMMUNICATE = True

# Training region to outline on each heatmap: strikers in TRAIN_S, jammers in TRAIN_J
TRAIN_S = [2]
TRAIN_J = [2, 3, 4]

OUT_PATH = "escort_analysis/team_size_generalisation.png"

# Name of this analysis' data dump (eval_results/analysis_data/<name>.json), the
# input of the presentation figures: every heatmap grid (per policy, per KPI),
# its CI half-widths and the per-cell paired p-values.       [CLI: --data-name]
DATA_NAME = "3.3.5_team_size_generalisation"

# Figure resolution (dots per inch). Higher = sharper output (larger files).
DPI = 600

# Per-cell CONFIDENCE INTERVAL printed under the mean in every heatmap square of the
# COMPARISON dashboard (the single-policy dashboard keeps the bare mean). Normal/CLT
# interval on the cell's per-episode samples: mean +- CI_Z * std(ddof=1)/sqrt(n),
# with n = N_RUNS * N_SEEDS episodes. The complete/baseline columns use their own
# samples; the difference column uses the PAIRED per-episode difference
# (complete - baseline), which is valid because both policies are rolled out on the
# same seeds (common random numbers) — so its CI answers "is Δ distinguishable from
# 0?" and lines up with the Wilcoxon table.
SHOW_CI = True
CI_Z = 1.96            # 1.96 = 95%, 1.645 = 90%, 2.576 = 99%
CI_LABEL = "95% CI"    # only used in the printed/CSV headers
CI_FONTSIZE = 6.5      # the mean stays at fontsize 9

# Per-episode KPI keys produced by run_cell (all length-B arrays).
KPI_KEYS = ["survival", "completion", "targets_destroyed", "frag", "reward", "duration"]

# Single-policy dashboard: (key, title, colormap, value range or None)
KPIS = [
    ("survival",          "Survival rate",          "viridis", (0, 1)),
    ("completion",        "Task completion rate",   "viridis", (0, 1)),
    ("targets_destroyed", "Targets-destroyed rate", "viridis", (0, 1)),
    ("frag",              "Fragmentation (frag)",   "magma",   (0, None)),
    ("reward",            "Mean episode reward",    "cividis", (None, None)),
    ("duration",          "Mean duration (steps)",  "cividis", (None, None)),
]

# Comparison dashboard rows: (key, title, colormap, direction). "higher"/"lower" = which
# way is BETTER for the policy, driving the one-sided Wilcoxon alternative.
COMPARE_KPIS = [
    ("targets_destroyed", "Targets destroyed",     "viridis", "higher"),
    ("survival",          "Survival rate",         "viridis", "higher"),
    ("duration",          "Mean duration (steps)", "cividis", "lower"),
]


def _count_spec(v):
    """Normalise a count spec to (alloc_size, dr_range).

    ``v`` is an int (FIXED count) or an inclusive (lo, hi) tuple (per-episode DR).
    ``alloc_size`` is what the env builds tensors at (hi for a range); ``dr_range``
    is the (lo, hi) passed to DomainRandomization, or None when the count is fixed
    (a scalar, or a degenerate lo==hi range).
    """
    if isinstance(v, (tuple, list)):
        lo, hi = int(v[0]), int(v[1])
        return hi, (None if lo == hi else (lo, hi))
    return int(v), None


def _parse_count(s):
    """Parse a CLI count override: 'n' → fixed int; 'lo,hi' → (lo, hi) DR range."""
    parts = [int(x) for x in s.split(",")]
    return parts[0] if len(parts) == 1 else (parts[0], parts[1])


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

    # Radar / target counts. Each spec is an int (fixed) or a (lo, hi) range;
    # _count_spec gives (alloc_size, dr_range). The env allocates tensors at
    # alloc_size (hi for a range) and masks each episode down to a per-env sample.
    kt, dr_kt = _count_spec(KNOWN_TARGETS)
    ut, dr_ut = _count_spec(UNKNOWN_TARGETS)
    kr, dr_kr = _count_spec(KNOWN_RADARS)
    ur, dr_ur = _count_spec(UNKNOWN_RADARS)
    ec.n_known_targets, ec.n_unknown_targets = kt, ut
    ec.n_targets = kt + ut
    ec.n_known_radars, ec.n_unknown_radars = kr, ur
    ec.n_radars = kr + ur

    ec.radar_kill_probability = KILL
    ec.communicate = COMMUNICATE
    ec._use_fofe = USE_FOFE

    dr = DomainRandomization(
        n_known_radars=dr_kr,   n_unknown_radars=dr_ur,
        n_known_targets=dr_kt,  n_unknown_targets=dr_ut,
    )
    ec.dr = dr if dr.active() else None
    return ec


def run_cell(ckpt, base, ns, nj, n_runs, seed, device):
    """One composition, n_runs parallel episodes at a fixed env seed.

    Returns a dict of PER-EPISODE KPI arrays (each length n_runs). Seeding the env
    (via PPOConfig.seed → StrikeEA2DEnv seed) makes the sampled world reproducible, so
    two policies evaluated at the same seed share identical per-episode layouts — the
    common-random-numbers pairing the signed-rank test relies on.
    """
    ec = _make_cfg(base, ns, nj)
    policy = _build_policy_for_scenario(ckpt, ec, device)
    policy.eval(); policy.deterministic = True
    env = build_env(ec, PPOConfig(num_envs=n_runs, device=str(device), seed=int(seed)),
                    hf_radar_cfg=ckpt.hf_radar_cfg)
    B = n_runs
    frag_sum = np.zeros(B); frag_cnt = np.zeros(B)
    rew_sum = np.zeros(B); ep_len = np.zeros(B)
    fin_alive = None; fin_talive = None
    with torch.no_grad():
        td = env.reset()
        # Per-episode PRESENT masks (fixed for the episode; all-True when DR off).
        # Under radar/target DR the absent slots must be excluded from the rates.
        tgt_present = env.target_present.bool().detach().cpu().numpy()   # [B, T]
        agt_present = env.agent_present.bool().detach().cpu().numpy()    # [B, A]
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
    n_agt = np.clip(agt_present.sum(axis=1), 1, None)                   # present agents / episode
    n_tgt = np.clip(tgt_present.sum(axis=1), 1, None)                   # present targets / episode
    tgt_destroyed = (tgt_present & ~fin_talive).sum(axis=1)            # present & down
    return dict(
        survival=((agt_present & fin_alive).sum(axis=1) / n_agt).astype(float),   # frac PRESENT agents alive
        completion=(~(tgt_present & fin_talive).any(axis=1)).astype(float),       # 1 if all present targets down
        targets_destroyed=(tgt_destroyed / n_tgt).astype(float),                  # frac PRESENT targets down
        frag=(frag_sum / np.clip(frag_cnt, 1, None)).astype(float),
        reward=rew_sum.astype(float),
        duration=ep_len.astype(float),
    )


def sweep_policy(ckpt, strikers, jammers, n_runs, n_seeds, base_seed, device, label):
    """Sweep one policy over the grid.

    Returns (grids, per_ep):
      grids[k]      → (nS, nJ) array of cell means for KPI k.
      per_ep[(si,ji)] → dict of concatenated per-episode arrays (or None on failure).
    """
    base = ckpt.base_env_cfg
    nS, nJ = len(strikers), len(jammers)
    grids = {k: np.full((nS, nJ), np.nan) for k in KPI_KEYS}
    per_ep = {}
    print(f"[{label}] sweeping strikers={strikers} x jammers={jammers}  "
          f"n_runs={n_runs} x n_seeds={n_seeds}")
    for si, ns in enumerate(strikers):
        for ji, nj in enumerate(jammers):
            try:
                acc = {k: [] for k in KPI_KEYS}
                for s in range(n_seeds):
                    ep = run_cell(ckpt, base, ns, nj, n_runs, base_seed + s, device)
                    for k in KPI_KEYS:
                        acc[k].append(ep[k])
                cat = {k: np.concatenate(acc[k]) for k in KPI_KEYS}
                per_ep[(si, ji)] = cat
                for k in KPI_KEYS:
                    grids[k][si, ji] = float(np.mean(cat[k]))
                print(f"  [{label}] {ns}s{nj}j: surv={grids['survival'][si,ji]:.2f} "
                      f"comp={grids['completion'][si,ji]:.2f} "
                      f"tgt={grids['targets_destroyed'][si,ji]:.2f} "
                      f"frag={grids['frag'][si,ji]:.3f} "
                      f"rew={grids['reward'][si,ji]:.1f} dur={grids['duration'][si,ji]:.0f}")
            except Exception as exc:  # noqa: BLE001 — mark failed cells, keep sweeping
                per_ep[(si, ji)] = None
                print(f"  [{label}] {ns}s{nj}j: FAILED ({type(exc).__name__}: {exc})")
                traceback.print_exc()
    return grids, per_ep


# ===================================================================
# Plotting helpers
# ===================================================================

def _ci_half(x):
    """Half-width of the normal/CLT confidence interval of the MEAN of `x`.

    CI_Z * std(ddof=1) / sqrt(n) over the per-episode samples; NaN when fewer than
    two finite samples exist (the interval is undefined), 0.0 for a constant cell."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(CI_Z * np.std(x, ddof=1) / np.sqrt(x.size))


def _fmt_ci(h):
    """"±<half-width>" with enough decimals to stay informative.

    At n = N_RUNS * N_SEEDS episodes the CI of a rate is only ~0.005 wide, which
    two decimals would flatten to "±0.00"; the step-count KPIs are ~1.0 wide and
    want fewer. So the precision follows the magnitude."""
    if h >= 10:
        return f"±{h:.1f}"
    if h >= 0.1:
        return f"±{h:.2f}"
    return f"±{h:.3f}"


def _ci_grids(per_ep, strikers, jammers, keys=KPI_KEYS):
    """{kpi: (nS, nJ) grid of CI half-widths} from one policy's per-episode samples."""
    grids = {k: np.full((len(strikers), len(jammers)), np.nan) for k in keys}
    for (si, ji), ep in per_ep.items():
        if ep is None:
            continue                                    # failed cell stays NaN
        for k in keys:
            grids[k][si, ji] = _ci_half(ep[k])
    return grids


def _paired_ci_grids(per_ep_c, per_ep_b, strikers, jammers, keys=KPI_KEYS):
    """{kpi: grid of CI half-widths of the PAIRED difference complete - baseline}.

    Episodes are paired by position, which pairs them by seed: both policies are
    swept with the same BASE_SEED, so episode i of a cell is the same sampled world
    for both. The paired CI is therefore much tighter than combining the two
    independent column CIs, and is the one that says whether Δ differs from 0."""
    grids = {k: np.full((len(strikers), len(jammers)), np.nan) for k in keys}
    for key in per_ep_c:
        c, b = per_ep_c.get(key), per_ep_b.get(key)
        if c is None or b is None:
            continue
        si, ji = key
        for k in keys:
            n = min(len(c[k]), len(b[k]))
            grids[k][si, ji] = _ci_half(np.asarray(c[k][:n]) - np.asarray(b[k][:n]))
    return grids


def _cell_text_color(cmap, norm, value):
    """Pick black or white text for legibility on the mapped cell colour."""
    r, g, b, _ = cmap(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b   # perceived brightness
    return "#ffffff" if luminance < 0.55 else "#14202b"


def _draw_heat(fig, ax, g, cmap, vmin, vmax, strikers, jammers, title,
               cbar_label="", signed=False, ci=None):
    """Heatmap of `g`. When `ci` is given (same-shaped grid of CI half-widths) each
    square prints the mean on top and "+-<half-width>" under it in a smaller font."""
    nS, nJ = len(strikers), len(jammers)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad("#eeeeee")
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(np.ma.masked_invalid(g), origin="lower", cmap=cmap, norm=norm,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(nJ)); ax.set_xticklabels(jammers)
    ax.set_yticks(range(nS)); ax.set_yticklabels(strikers)
    ax.set_xlabel("Jammers"); ax.set_ylabel("Strikers")
    ax.set_title(title, fontsize=11, pad=6)
    fmt = "{:+.2f}" if signed else "{:.2f}"
    show_ci = SHOW_CI and ci is not None
    for si in range(nS):
        for ji in range(nJ):
            v = g[si, ji]
            if np.isnan(v):
                continue
            tcol = _cell_text_color(cmap, norm, v)
            h = ci[si, ji] if show_ci else np.nan
            if not np.isfinite(h):                      # no CI → mean centred as before
                ax.text(ji, si, fmt.format(v), ha="center", va="center",
                        fontsize=9, color=tcol)
                continue
            # mean above centre, CI half-width below it
            ax.text(ji, si + 0.10, fmt.format(v), ha="center", va="center",
                    fontsize=9, color=tcol)
            ax.text(ji, si - 0.17, _fmt_ci(h), ha="center", va="center",
                    fontsize=CI_FONTSIZE, color=tcol, alpha=0.85)
    for si, ns in enumerate(strikers):
        for ji, nj in enumerate(jammers):
            if ns in TRAIN_S and nj in TRAIN_J:
                ax.add_patch(Rectangle((ji - 0.5, si - 0.5), 1, 1, fill=False,
                                       edgecolor=NLR_TERRA, lw=1.8, zorder=5))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=9)


def plot_single_dashboard(grids, strikers, jammers, name, out, n_runs, n_seeds):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for ax, (key, title, _cmap, vr) in zip(axes.ravel(), KPIS):
        g = grids[key]
        vmin = vr[0] if vr and vr[0] is not None else np.nanmin(g)
        vmax = vr[1] if vr and vr[1] is not None else np.nanmax(g)
        _draw_heat(fig, ax, g, NLR_SEQ, vmin, vmax, strikers, jammers, title,
                   cbar_label=title)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved dashboard -> {out}")


def plot_comparison(grids_c, grids_b, strikers, jammers, name_c, name_b, out, n_runs, n_seeds,
                    ci_c=None, ci_b=None, ci_d=None):
    """3 rows (KPIs) × 3 cols (complete / baseline / difference).

    `ci_c` / `ci_b` / `ci_d` are optional {kpi: grid of CI half-widths} for the three
    columns (the difference column's being the PAIRED-difference CI); when given, each
    square prints mean ± half-width."""
    nrow = len(COMPARE_KPIS)
    fig, axes = plt.subplots(nrow, 3, figsize=(15, 4.6 * nrow),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, (key, title, _cmap, direction) in enumerate(COMPARE_KPIS):
        gc, gb = grids_c[key], grids_b[key]
        # shared scale for the complete & baseline columns
        if key in ("targets_destroyed", "survival"):
            vmin, vmax = 0.0, 1.0
        else:
            allv = np.concatenate([gc[~np.isnan(gc)], gb[~np.isnan(gb)]])
            vmin, vmax = (float(allv.min()), float(allv.max())) if allv.size else (0.0, 1.0)
        better = "higher = better" if direction == "higher" else "lower = better"
        _draw_heat(fig, axes[row, 0], gc, NLR_SEQ, vmin, vmax, strikers, jammers,
                   f"{title}\nComm-FOFE-MAPPO", cbar_label=title,
                   ci=(ci_c or {}).get(key))
        _draw_heat(fig, axes[row, 1], gb, NLR_SEQ, vmin, vmax, strikers, jammers,
                   f"{title}\nMAPPO Baseline", cbar_label=title,
                   ci=(ci_b or {}).get(key))
        # difference: Comm-FOFE-MAPPO − MAPPO Baseline, diverging, centred at 0
        diff = gc - gb
        vabs = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
        vabs = vabs if vabs > 0 else 1.0
        _draw_heat(fig, axes[row, 2], diff, NLR_DIV, -vabs, vabs, strikers, jammers,
                   f"{title}\nΔ = Comm-FOFE-MAPPO − MAPPO Baseline",
                   cbar_label=f"Δ  ({better})", signed=True,
                   ci=(ci_d or {}).get(key))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved comparison dashboard -> {out}")


# ===================================================================
# Statistics — paired one-sided Wilcoxon signed-rank per cell
# ===================================================================

def _stars(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def paired_pvalue(comp, base_arr, direction):
    """One-sided paired Wilcoxon signed-rank on d = complete − baseline.

    direction "higher" (targets/survival: complete better when larger) → alternative
    "greater"; "lower" (duration: complete better when smaller) → "less". Returns
    (p_value, n_pairs); p is NaN when all paired differences are zero (test undefined).
    """
    n = min(len(comp), len(base_arr))
    comp, base_arr = comp[:n], base_arr[:n]
    if n == 0 or not np.any(comp - base_arr != 0):
        return np.nan, n
    alt = "greater" if direction == "higher" else "less"
    try:
        _, p = wilcoxon(comp, base_arr, alternative=alt, zero_method="wilcox")
    except ValueError:
        return np.nan, n
    return float(p), n


def compute_stats(per_ep_c, per_ep_b, strikers, jammers):
    rows = []
    for si, ns in enumerate(strikers):
        for ji, nj in enumerate(jammers):
            c, b = per_ep_c.get((si, ji)), per_ep_b.get((si, ji))
            row = {"composition": f"{ns}s{nj}j", "n_strikers": ns, "n_jammers": nj}
            for key, _title, _cmap, direction in COMPARE_KPIS:
                if c is None or b is None:
                    row[key] = (np.nan, 0, np.nan, np.nan)
                else:
                    p, n = paired_pvalue(c[key], b[key], direction)
                    row[key] = (p, n, float(np.median(c[key])), float(np.median(b[key])))
            rows.append(row)
    return rows


def plot_pvalue_table(rows, out):
    kpi_keys = [k for k, *_ in COMPARE_KPIS]
    arrows = {"higher": "↑", "lower": "↓"}
    col_labels = ["composition"] + [
        f"{title}\n({arrows[d]} better)  p" for _k, title, _c, d in COMPARE_KPIS
    ]
    cell_text, cell_colors = [], []
    for r in rows:
        line = [r["composition"]]
        colors = ["white"]
        for key in kpi_keys:
            p, _n, _mc, _mb = r[key]
            line.append("n/a" if np.isnan(p) else f"{p:.3f}{_stars(p)}")
            colors.append("#c8f0c8" if (not np.isnan(p) and p < 0.05) else "white")
        cell_text.append(line)
        cell_colors.append(colors)
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(rows) + 1.8))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.35)
    ax.set_title("Paired one-sided Wilcoxon signed-rank p-values  (H1: complete BETTER "
                 "than baseline)\ngreen = p<0.05 · * p<.05  ** p<.01  *** p<.001 · "
                 "n/a = no non-zero paired differences", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"saved p-value table -> {out}")


def write_pvalue_csv(rows, out):
    with open(out, "w") as f:
        f.write("composition,n_strikers,n_jammers,kpi,direction,n_pairs,"
                "median_complete,median_baseline,p_value,significant\n")
        for r in rows:
            for key, _title, _cmap, direction in COMPARE_KPIS:
                p, n, mc, mb = r[key]
                sig = 1 if (not np.isnan(p) and p < 0.05) else 0
                f.write(f"{r['composition']},{r['n_strikers']},{r['n_jammers']},{key},"
                        f"{direction},{n},{mc:.4f},{mb:.4f},{p:.6f},{sig}\n")
    print(f"saved p-value CSV   -> {out}")


def _world_meta(strikers, jammers, n_runs, n_seeds):
    """The swept world + axes, so a figure script can label and caption itself."""
    return {"scenario": SCENARIO, "strikers": list(strikers), "jammers": list(jammers),
            "n_runs": int(n_runs), "n_seeds": int(n_seeds),
            "n_episodes_per_cell": int(n_runs) * int(n_seeds),
            "base_seed": int(BASE_SEED),
            "known_radars": KNOWN_RADARS, "unknown_radars": UNKNOWN_RADARS,
            "known_targets": KNOWN_TARGETS, "unknown_targets": UNKNOWN_TARGETS,
            "radar_kill_probability": KILL, "frag_radius": FRAG_RADIUS,
            "use_fofe": USE_FOFE, "communicate": COMMUNICATE,
            "train_strikers": list(TRAIN_S), "train_jammers": list(TRAIN_J),
            "ci_z": CI_Z, "ci_label": CI_LABEL}


def export_data(data_name, strikers, jammers, n_runs, n_seeds, name_c, grids_c,
                ci_c=None, name_b=None, grids_b=None, ci_b=None, ci_d=None,
                rows=None) -> None:
    """Write the swept grids to eval_results/analysis_data/.

    Every grid is a (len(strikers), len(jammers)) matrix — rows = striker counts,
    columns = jammer counts, in the order given by "strikers"/"jammers" — holding
    the cell MEAN of that KPI; "*_ci" holds the matching CI half-widths (paired
    for the difference), and "pvalues" the per-cell one-sided Wilcoxon table. In
    single-policy mode only the complete policy's block is written."""
    def _grids(g):
        return {k: np.asarray(v, dtype=float) for k, v in (g or {}).items()}

    payload = {
        "kind": "team_size_generalisation",
        "axes": {"strikers": list(strikers), "jammers": list(jammers)},
        "kpi_keys": list(KPI_KEYS),
        "compare_kpis": [{"key": k, "title": t, "direction": d}
                         for k, t, _c, d in COMPARE_KPIS],
        "policies": {"complete": name_c, "baseline": name_b},
        "grids": {"complete": _grids(grids_c),
                  **({"baseline": _grids(grids_b)} if grids_b else {})},
        "ci": {**({"complete": _grids(ci_c)} if ci_c else {}),
               **({"baseline": _grids(ci_b)} if ci_b else {}),
               **({"difference": _grids(ci_d)} if ci_d else {})},
        # per composition: p-value, n pairs and the two medians, per compared KPI
        "pvalues": [{"composition": r["composition"],
                     "n_strikers": r["n_strikers"], "n_jammers": r["n_jammers"],
                     **{k: {"pvalue": r[k][0], "n_pairs": r[k][1],
                            "median_complete": r[k][2], "median_baseline": r[k][3]}
                        for k, *_ in COMPARE_KPIS}}
                    for r in (rows or [])],
    }
    save_analysis(data_name, payload, meta=_world_meta(strikers, jammers, n_runs, n_seeds),
                  source=Path(__file__).name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=POLICY_PATH,
                    help="COMPLETE model policy (the primary / candidate model).")
    ap.add_argument("--baseline", default=BASELINE_PATH,
                    help="BASELINE model policy to compare against. Omit for single-policy mode.")
    ap.add_argument("--n_runs", type=int, default=N_RUNS)
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    ap.add_argument("--strikers", default=None, help="comma list override, e.g. '2' or '1,2,3'")
    ap.add_argument("--jammers", default=None, help="comma list override, e.g. '2,4'")
    ap.add_argument("--known_radars", default=None,
                    help="count override: 'n' fixed, or 'lo,hi' to randomise KNOWN radars per episode")
    ap.add_argument("--unknown_radars", default=None,
                    help="count override: 'n' fixed, or 'lo,hi' to randomise UNKNOWN radars per episode")
    ap.add_argument("--known_targets", default=None,
                    help="count override: 'n' fixed, or 'lo,hi' to randomise KNOWN targets per episode")
    ap.add_argument("--unknown_targets", default=None,
                    help="count override: 'n' fixed, or 'lo,hi' to randomise UNKNOWN targets per episode")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--data-name", default=DATA_NAME,
                    help="name of the analysis-data dump written to "
                         "eval_results/analysis_data/<name>.json — the input of the "
                         "presentation figures")
    ap.add_argument("--no-data", action="store_true",
                    help="skip writing the analysis-data dump")
    args = ap.parse_args()
    strikers = [int(x) for x in args.strikers.split(",")] if args.strikers else STRIKERS
    jammers = [int(x) for x in args.jammers.split(",")] if args.jammers else JAMMERS

    # CLI count overrides replace the module constants (only when the flag is passed).
    global KNOWN_RADARS, UNKNOWN_RADARS, KNOWN_TARGETS, UNKNOWN_TARGETS
    if args.known_radars    is not None: KNOWN_RADARS    = _parse_count(args.known_radars)
    if args.unknown_radars  is not None: UNKNOWN_RADARS  = _parse_count(args.unknown_radars)
    if args.known_targets   is not None: KNOWN_TARGETS   = _parse_count(args.known_targets)
    if args.unknown_targets is not None: UNKNOWN_TARGETS = _parse_count(args.unknown_targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fmt(v):
        return f"[{v[0]}..{v[1]}]" if isinstance(v, (tuple, list)) else str(v)

    ckpt_c = _LoadedCheckpoint(_resolve(args.checkpoint), device)
    name_c = Path(args.checkpoint).stem
    print(f"complete policy = {name_c}   world: {SCENARIO} "
          f"radars(k={_fmt(KNOWN_RADARS)},u={_fmt(UNKNOWN_RADARS)}) "
          f"targets(k={_fmt(KNOWN_TARGETS)},u={_fmt(UNKNOWN_TARGETS)}) kill={KILL}")
    grids_c, per_ep_c = sweep_policy(ckpt_c, strikers, jammers, args.n_runs, args.n_seeds,
                                     BASE_SEED, device, label="complete")

    out = _resolve(args.out)

    # ---------- single-policy mode ----------
    if not args.baseline:
        plot_single_dashboard(grids_c, strikers, jammers, name_c, out, args.n_runs, args.n_seeds)
        if not args.no_data:
            export_data(args.data_name, strikers, jammers, args.n_runs, args.n_seeds,
                        name_c, grids_c, ci_c=_ci_grids(per_ep_c, strikers, jammers)
                        if SHOW_CI else None)
        csv = out.with_suffix(".csv")
        with open(csv, "w") as f:
            f.write("kpi,n_strikers,n_jammers,value\n")
            for k in KPI_KEYS:
                for si, ns in enumerate(strikers):
                    for ji, nj in enumerate(jammers):
                        f.write(f"{k},{ns},{nj},{grids_c[k][si,ji]:.4f}\n")
        print(f"saved CSV       -> {csv}")
        return

    # ---------- comparison mode ----------
    ckpt_b = _LoadedCheckpoint(_resolve(args.baseline), device)
    name_b = Path(args.baseline).stem
    print(f"baseline policy = {name_b}")
    grids_b, per_ep_b = sweep_policy(ckpt_b, strikers, jammers, args.n_runs, args.n_seeds,
                                     BASE_SEED, device, label="baseline")

    # Per-cell CI half-widths: own samples for each policy column, PAIRED per-episode
    # difference for the Δ column (both policies swept on the same seeds).
    ci_c = _ci_grids(per_ep_c, strikers, jammers) if SHOW_CI else None
    ci_b = _ci_grids(per_ep_b, strikers, jammers) if SHOW_CI else None
    ci_d = _paired_ci_grids(per_ep_c, per_ep_b, strikers, jammers) if SHOW_CI else None

    plot_comparison(grids_c, grids_b, strikers, jammers, name_c, name_b, out,
                    args.n_runs, args.n_seeds, ci_c, ci_b, ci_d)

    rows = compute_stats(per_ep_c, per_ep_b, strikers, jammers)
    plot_pvalue_table(rows, out.with_name(out.stem + "_pvalues.png"))
    write_pvalue_csv(rows, out.with_name(out.stem + "_pvalues.csv"))

    # Data dump: every grid + CI + p-value, so the presentation figures can be
    # rebuilt without re-running the sweep.
    if not args.no_data:
        export_data(args.data_name, strikers, jammers, args.n_runs, args.n_seeds,
                    name_c, grids_c, ci_c, name_b, grids_b, ci_b, ci_d, rows)

    # KPI-value CSV (complete, baseline, diff) for the three compared KPIs
    # (the *_ci columns are the same half-widths printed in the heatmap squares)
    csv = out.with_name(out.stem + "_values.csv")
    with open(csv, "w") as f:
        f.write("kpi,n_strikers,n_jammers,complete,baseline,difference,"
                "complete_ci,baseline_ci,difference_ci\n")
        for key, *_ in COMPARE_KPIS:
            for si, ns in enumerate(strikers):
                for ji, nj in enumerate(jammers):
                    c, b = grids_c[key][si, ji], grids_b[key][si, ji]
                    hc = ci_c[key][si, ji] if ci_c else np.nan
                    hb = ci_b[key][si, ji] if ci_b else np.nan
                    hd = ci_d[key][si, ji] if ci_d else np.nan
                    f.write(f"{key},{ns},{nj},{c:.4f},{b:.4f},{c - b:.4f},"
                            f"{hc:.4f},{hb:.4f},{hd:.4f}\n")
    print(f"saved values CSV -> {csv}")

    # stdout summary of significant cells
    print("\nSignificant cells (paired one-sided Wilcoxon, p<0.05, complete better):")
    any_sig = False
    for r in rows:
        hits = [f"{title} p={r[key][0]:.3f}{_stars(r[key][0])}"
                for key, title, _c, _d in COMPARE_KPIS
                if not np.isnan(r[key][0]) and r[key][0] < 0.05]
        if hits:
            any_sig = True
            print(f"  {r['composition']:>6}: " + " | ".join(hits))
    if not any_sig:
        print("  (none)")


if __name__ == "__main__":
    main()
