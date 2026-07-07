"""compare_two_policies_logs.py — overlay the TRAINING LOGS of TWO checkpoints.

Pass two policy .pt files; the script reads the logs stored in each checkpoint and
draws two comparison figures, with BOTH policies on the same axes:

  1. Reward curve      — the (non-eval) training reward `train_mean_episode_total_reward`,
                         one curve per policy.
  2. Eval-rates plot   — the eval KPIs (survival / task completion / targets-destroyed /
                         coalition fragmentation), each shown for both policies.

In the eval-rates plot the COLOUR encodes the metric and the LINE STYLE encodes the
policy (solid = policy A, dashed = policy B), so the two policies sit on one axis
without eight indistinguishable colours. NLR house colours throughout.

Lines are smoothed with a per-section running average (curriculum transitions stay
sharp); see --smooth. Outputs use stable names (overwritten each run).

Run (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\compare_two_policies_logs.py \\
      runs/FINALV1/complete_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt \\
      runs/FINALV1/baseline_S1_20260704/stage5of5_...pt  --label-a Complete --label-b Baseline
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
from matplotlib.lines import Line2D

# Reuse the checkpoint path resolver and the finite / per-section smoothing helpers.
from .evaluate_policy import _resolve_policy_path
from .plot_reward_components import _finite_xy, _smooth_sectioned
from .nlr_style import NLR_PRIMARY, NLR_SECONDARY, NLR_ACCENT, NLR_DARKGRAY

# ===================================================================
# CONFIG — edit these (CLI flags / positional args override)
# ===================================================================

# The two policies to compare (bare name → runs/, "runs/…/x.pt" → project dir, abs → as-is).
POLICY_A = "runs/FINALV1/complete_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt"
POLICY_B = "runs/FINALV1/baseline_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt"
LABEL_A = "Complete"
LABEL_B = "Baseline"

# Running-average window (datapoints); resets at each curriculum section so
# transitions stay sharp. 1 = raw.                        [CLI: --smooth]
SMOOTH_WINDOW = 25

# Output PNGs (relative paths resolved against the project dir 0_TA_...).
REWARD_OUT = "eval_results/compare_reward_curves.png"
RATES_OUT = "eval_results/compare_eval_rates.png"

# ===================================================================

# The (non-eval) training reward series.
TRAIN_REWARD_KEY = "train_mean_episode_total_reward"

# Eval-rate series (key, label, NLR colour). All bounded in [0, 1] → one axis.
EVAL_RATES = [
    ("eval_survival_rate",           "survival rate",                  NLR_PRIMARY),
    ("eval_task_completion_rate",    "task completion rate",           NLR_ACCENT),
    ("eval_targets_destroyed_rate",  "targets destroyed rate",         NLR_SECONDARY),
    ("eval_coalition_fragmentation", "coalition fragmentation (frag)", NLR_DARKGRAY),
]

# Per-policy encodings.
POLICY_COLORS = [NLR_PRIMARY, NLR_ACCENT]   # reward plot: one colour per policy
POLICY_STYLES = ["-", "--"]                 # eval plot: line style per policy


def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_THIS_DIR.parent / p)


def _load_logs(path: Path):
    """Return (training_logs, section_bounds) from a checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    logs = ck.get("training_logs")
    if not logs:
        raise KeyError(f"{path.name} has no 'training_logs' — cannot plot its curves.")
    return logs, (ck.get("section_bounds") or [])


def plot_reward(policies, out, smooth):
    """One (train) reward curve per policy on a shared axis."""
    fig, ax = plt.subplots(figsize=(13, 6))
    n_iter = 0
    for label, logs, bounds, color in policies:
        v = logs.get(TRAIN_REWARD_KEY)
        if v is None:
            print(f"  ! {label}: no '{TRAIN_REWARD_KEY}' in logs — skipped.")
            continue
        n_iter = max(n_iter, len(v))
        xs, ys = _finite_xy(v)
        ys = _smooth_sectioned(xs, ys, bounds, smooth)
        ax.plot(xs, ys, lw=1.9, color=color, label=label)

    ax.axhline(0.0, color=NLR_DARKGRAY, lw=0.8, alpha=0.5)
    ax.set_xlabel("Global training iteration", fontsize=11)
    ax.set_ylabel("Mean episode reward (train)", fontsize=11)
    # ax.set_title("Training reward — policy comparison", fontsize=12, fontweight="bold")
    if n_iter:
        ax.set_xlim(1, n_iter)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, frameon=True, title="policy")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved reward comparison -> {out}")


def plot_eval_rates(policies, out, smooth):
    """Every eval KPI for both policies: colour = metric, line style = policy."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    n_iter = 0
    plotted_metrics = set()
    for p_idx, (label, logs, bounds, _color) in enumerate(policies):
        style = POLICY_STYLES[p_idx % len(POLICY_STYLES)]
        for key, klabel, kcolor in EVAL_RATES:
            v = logs.get(key)
            if v is None or not np.any(np.isfinite(np.asarray(v, dtype=float))):
                continue
            n_iter = max(n_iter, len(v))
            xs, ys = _finite_xy(v)
            ys = _smooth_sectioned(xs, ys, bounds, smooth)
            ax.plot(xs, ys, lw=1.6, color=kcolor, ls=style)
            plotted_metrics.add(klabel)

    ax.set_xlabel("Global training iteration", fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("Eval rates — policy comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    if n_iter:
        ax.set_xlim(1, n_iter)
    ax.grid(True, alpha=0.3)

    # Two legends: colour → metric, line style → policy.
    kpi_handles = [Line2D([0], [0], color=c, lw=1.8, label=l)
                   for _, l, c in EVAL_RATES if l in plotted_metrics]
    pol_handles = [Line2D([0], [0], color=NLR_DARKGRAY, lw=1.8,
                          ls=POLICY_STYLES[i % len(POLICY_STYLES)], label=pol[0])
                   for i, pol in enumerate(policies)]
    leg1 = ax.legend(handles=kpi_handles, title="eval metric", loc="center left",
                     bbox_to_anchor=(1.01, 0.68), fontsize=9, frameon=True)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=pol_handles, title="policy", loc="center left",
                     bbox_to_anchor=(1.01, 0.30), fontsize=9, frameon=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    # bbox_extra_artists ensures BOTH outside legends are inside the saved crop.
    fig.savefig(out, dpi=200, bbox_inches="tight", bbox_extra_artists=[leg1, leg2])
    plt.close(fig)
    print(f"saved eval-rates comparison -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy_a", nargs="?", default=POLICY_A, help="first checkpoint .pt")
    ap.add_argument("policy_b", nargs="?", default=POLICY_B, help="second checkpoint .pt")
    ap.add_argument("--label-a", default=None, help="legend label for policy A")
    ap.add_argument("--label-b", default=None, help="legend label for policy B")
    ap.add_argument("--smooth", type=int, default=SMOOTH_WINDOW,
                    help="running-average window (1 = raw); resets at each curriculum section")
    ap.add_argument("--reward-out", default=REWARD_OUT)
    ap.add_argument("--rates-out", default=RATES_OUT)
    args = ap.parse_args()

    path_a = _resolve_policy_path(args.policy_a, None)
    path_b = _resolve_policy_path(args.policy_b, None)
    for p in (path_a, path_b):
        if p is None or not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {p}")

    # Labels: explicit flag wins; else the config default when the config path was
    # used, otherwise the checkpoint stem. Disambiguate if the two collide.
    label_a = args.label_a if args.label_a is not None else (
        LABEL_A if args.policy_a == POLICY_A else path_a.stem)
    label_b = args.label_b if args.label_b is not None else (
        LABEL_B if args.policy_b == POLICY_B else path_b.stem)
    if label_a == label_b:
        label_a, label_b = f"{label_a} (A)", f"{label_b} (B)"

    print(f"A: {label_a}  [{path_a.name}]")
    print(f"B: {label_b}  [{path_b.name}]")
    logs_a, bounds_a = _load_logs(path_a)
    logs_b, bounds_b = _load_logs(path_b)

    policies = [
        (label_a, logs_a, bounds_a, POLICY_COLORS[0]),
        (label_b, logs_b, bounds_b, POLICY_COLORS[1]),
    ]

    plot_reward(policies, _resolve_out(args.reward_out), args.smooth)
    plot_eval_rates(policies, _resolve_out(args.rates_out), args.smooth)


if __name__ == "__main__":
    main()
