"""plot_reward_components.py — plot how each REWARD COMPONENT evolves over the
training iterations for ONE checkpoint, so you can see what the agents were
actually optimising during training.

WHAT IT PLOTS
─────────────
The trainer logs, per iteration, the mean episode reward broken down into named
components (target_destroyed, terminal_bonus, escort, striker_approach, …). This
script reads those stored series straight from the checkpoint's `training_logs`
and draws one line per component over the global iteration axis, with the
curriculum stages shaded in the background.

NOTE ON ROLES
─────────────
The components are logged at the TEAM level (summed over agents) — they are NOT
stored separately per striker / per jammer. So this is a single, honest team
plot of the stored data; it does not split or re-attribute rewards by role.
(Component *names* like `striker_approach` / `jammer_progress` still reveal which
role-specific shaping term was active.)

Deactivated components (all-zero across training) are hidden by default; pass
--all to include them.

OUTPUT
──────
eval_results/reward_components.png  (stable name — re-runs overwrite it).

Run (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\plot_reward_components.py
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\plot_reward_components.py --source eval
"""
from __future__ import annotations
import argparse, sys, types
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

# Importing evaluate_policy registers the fofe_mappo.* config/reward modules that
# the pickled checkpoint references, and gives us the shared path resolver.
from .evaluate_policy import _resolve_policy_path
from .nlr_style import (  # importing applies NLR style
    NLR_GRAY, NLR_DARKGRAY, NLR_PRIMARY, NLR_SECONDARY, NLR_ACCENT,
)

# ===================================================================
# CONFIG — edit these (CLI flags override)
# ===================================================================

# Checkpoint to analyse. Bare name → runs/, "runs/…/x.pt" → project dir, abs → as-is.
POLICY_PATH = "runs/FINALV1/complete_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt"

# Which logged series: "train" (the optimisation signal) or "eval" (clean estimate).
SOURCE = "train"

# Include components that are all-zero across training (deactivated terms)?
SHOW_ZERO = False

# Y-axis limits for the reward-components plot. Clamps the view so the steady-state
# differences are readable (early-training values fall outside and are clipped from
# view, not from the data). Set to None for matplotlib autoscale.
COMPONENT_YLIM = (-10.0, 10.0)

# Running-average window (in datapoints) applied to every line to smooth the noise
# and make trends clearer. 1 = no smoothing (raw). Centred, edge-normalised, so it
# does not shift or shorten the curves.               [CLI: --smooth]
SMOOTH_WINDOW = 25

# Output PNGs (relative paths resolved against the project dir 0_TA_...).
OUT_PATH = "eval_results/reward_components.png"
RATES_OUT_PATH = "eval_results/eval_rates.png"

# Eval-rate series to plot (key, label, colour). Only those present are drawn;
# all are bounded in [0, 1] so they share one axis.
EVAL_RATES = [
    ("eval_survival_rate",          "survival rate",                 NLR_PRIMARY),
    ("eval_task_completion_rate",   "task completion rate",          NLR_ACCENT),
    ("eval_targets_destroyed_rate", "targets destroyed rate",        NLR_SECONDARY),
    ("eval_coalition_fragmentation", "coalition fragmentation (frag)", NLR_DARKGRAY),
]

# ===================================================================


def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_THIS_DIR.parent / p)


def _finite_xy(v):
    """(iteration, value) for the finite samples only.

    Eval series are logged sparsely (NaN between eval iterations); dropping the
    NaNs connects the real samples into a curve instead of an invisible line of
    isolated points. Dense (train) series are unaffected.
    """
    v = np.asarray(v, dtype=float)
    x = np.arange(1, len(v) + 1)
    m = np.isfinite(v)
    return x[m], v[m]


def _smooth(y, w):
    """Centred, edge-normalised running average (length- and position-preserving)."""
    y = np.asarray(y, dtype=float)
    if w <= 1 or y.size < 2:
        return y
    w = int(min(w, y.size))
    kernel = np.ones(w)
    num = np.convolve(y, kernel, mode="same")
    den = np.convolve(np.ones_like(y), kernel, mode="same")   # shrinks window at edges
    return num / den


def _smooth_sectioned(xs, ys, section_bounds, w):
    """Running-average smoothing applied WITHIN each curriculum section only.

    Boundaries between curriculum stages are NOT smoothed across — the average
    resets at every section, so the sharp step at each transition is preserved.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys, dtype=float)
    if w <= 1 or ys.size < 2:
        return ys
    if not section_bounds:
        return _smooth(ys, w)
    out = ys.copy()
    covered = np.zeros(xs.shape, dtype=bool)
    for _name, start, end in section_bounds:
        m = (xs > start) & (xs <= end)          # samples belonging to this stage
        if m.any():
            out[m] = _smooth(ys[m], w)
            covered |= m
    if not covered.all():                       # samples outside any declared stage
        out[~covered] = _smooth(ys[~covered], w)
    return out


def _shade_sections(ax, section_bounds):
    """Shade alternating curriculum stages and label them along the top."""
    if not section_bounds:
        return
    for i, (name, start, end) in enumerate(section_bounds):
        if i % 2 == 1:
            ax.axvspan(start, end, color=NLR_GRAY, alpha=0.08, zorder=0)
        if i > 0:
            ax.axvline(start, color=NLR_GRAY, lw=0.8, ls="--", alpha=0.6, zorder=0.5)
        ax.annotate(str(name), xy=((start + end) / 2, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7, color=NLR_DARKGRAY, rotation=0,
                    xytext=(0, 2), textcoords="offset points")


def plot_eval_rates(logs, section_bounds, ckpt_name, out, smooth=1):
    """Separate figure: every eval rate over the training iterations.

    Survival / completion / targets-destroyed / fragmentation are all in [0, 1]
    so they share one axis. Series are logged only on eval iterations (NaN
    elsewhere) → masked so the lines show gaps rather than dropping to zero.
    Returns the list of plotted labels (empty if none of the keys were present).
    """
    present = [(k, lbl, c) for (k, lbl, c) in EVAL_RATES if logs.get(k) is not None
               and np.any(np.isfinite(np.asarray(logs[k], dtype=float)))]
    if not present:
        return []

    fig, ax = plt.subplots(figsize=(14, 6))
    _shade_sections(ax, section_bounds)
    n_iter = 0
    for key, label, color in present:
        n_iter = max(n_iter, len(logs[key]))
        xs, ys = _finite_xy(logs[key])
        ys = _smooth_sectioned(xs, ys, section_bounds, smooth)
        ax.plot(xs, ys, lw=1.6, color=color, label=label)

    ax.set_xlabel("Global training iteration", fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title(f"Eval rates over training — {ckpt_name}", fontsize=12,
                 fontweight="bold", pad=26)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(1, n_iter)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
              frameon=True, title="eval metric")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [lbl for _, lbl, _ in present]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=POLICY_PATH,
                    help="checkpoint .pt to read training_logs from")
    ap.add_argument("--source", choices=["train", "eval"], default=SOURCE,
                    help="plot the 'train' (optimised) or 'eval' component series")
    ap.add_argument("--all", action="store_true",
                    help="also plot deactivated (all-zero) components")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--rates-out", default=RATES_OUT_PATH,
                    help="output PNG for the eval-rates plot")
    ap.add_argument("--smooth", type=int, default=SMOOTH_WINDOW,
                    help="running-average window in datapoints (1 = raw); smoothing "
                         "resets at each curriculum section so transitions stay sharp")
    args = ap.parse_args()

    ckpt_path = _resolve_policy_path(args.checkpoint, None)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    print(f"reading {ckpt_path.name} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    logs = ckpt.get("training_logs")
    if not logs:
        raise KeyError(f"{ckpt_path.name} has no 'training_logs' — cannot plot reward components.")
    section_bounds = ckpt.get("section_bounds") or []

    prefix = f"{args.source}_component_"
    keys = sorted(k for k in logs if k.startswith(prefix))
    if not keys:
        raise KeyError(f"no '{prefix}*' series in training_logs "
                       f"(available example keys: {list(logs)[:8]})")

    # Build (label, series) pairs, dropping all-zero components unless --all.
    series = []
    skipped = []
    for k in keys:
        v = np.asarray(logs[k], dtype=float)
        finite = v[np.isfinite(v)]
        active = finite.size > 0 and bool(np.any(np.abs(finite) > 1e-9))
        label = k[len(prefix):]
        if active or args.all:
            series.append((label, v))
        else:
            skipped.append(label)

    if not series:
        raise RuntimeError("no non-zero reward components to plot (use --all to force).")

    n_iter = max(len(v) for _, v in series)
    x = np.arange(1, n_iter + 1)

    # Distinct colours for many lines (tab20 gives up to 20 well-separated hues).
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(14, 7))
    _shade_sections(ax, section_bounds)
    for idx, (label, v) in enumerate(series):
        xs, ys = _finite_xy(v)
        ys = _smooth_sectioned(xs, ys, section_bounds, args.smooth)
        ax.plot(xs, ys, lw=1.4, color=cmap(idx % 20), label=label)

    # Total reward — the logged mean episode total (the true total as recorded,
    # not a re-sum of the plotted components), drawn in black on top.
    total_key = f"{args.source}_mean_episode_total_reward"
    if logs.get(total_key) is not None:
        xs, ys = _finite_xy(logs[total_key])
        ys = _smooth_sectioned(xs, ys, section_bounds, args.smooth)
        ax.plot(xs, ys, lw=1.4, color="black", zorder=6, label="TOTAL")

    ax.axhline(0.0, color=NLR_DARKGRAY, lw=0.8, alpha=0.5)
    ax.set_xlabel("Global training iteration", fontsize=11)
    ax.set_ylabel(f"Mean episode reward contribution ({args.source})", fontsize=11)
    ax.set_title(f"Reward components over training — {ckpt_path.name}  "
                 f"[{args.source}, team-level]", fontsize=12, fontweight="bold", pad=26)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_iter)
    if COMPONENT_YLIM is not None:
        ax.set_ylim(*COMPONENT_YLIM)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
              frameon=True, title="component")
    fig.tight_layout()

    out = _resolve_out(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"plotted {len(series)} active components ({args.source}); "
          f"hidden {len(skipped)} zero components"
          + (f": {', '.join(skipped)}" if skipped and not args.all else ""))
    print(f"saved -> {out}")

    # ── second figure: all eval rates over the iterations ──
    rates_out = _resolve_out(args.rates_out)
    rate_labels = plot_eval_rates(logs, section_bounds, ckpt_path.name, rates_out, args.smooth)
    if rate_labels:
        print(f"plotted {len(rate_labels)} eval rates: {', '.join(rate_labels)}")
        print(f"saved -> {rates_out}")
    else:
        print("no eval-rate series found in training_logs — skipped eval-rates plot.")


if __name__ == "__main__":
    main()
