"""compare_two_policies_logs.py — overlay the TRAINING LOGS of TWO OR MORE checkpoints.

List any number of policies (A, B, C, D, …) in the POLICIES config below; the
script reads the logs stored in each checkpoint and draws two comparison figures,
with ALL policies on the same axes:

  1. Reward curve      — the (non-eval) training reward `train_mean_episode_total_reward`,
                         one curve per policy (colour = policy).
  2. Eval-rates plot   — the eval KPIs (survival / task completion / targets-destroyed /
                         coalition fragmentation), shown for every policy.

In the eval-rates plot the COLOUR encodes the metric and the LINE STYLE encodes the
policy (solid = A, dashed = B, dash-dot = C, dotted = D), so the policies sit on one
axis without a wall of indistinguishable colours. NLR house colours throughout.
(Line styles cycle after four policies, so ≤4 stay cleanly distinguishable.)

Each policy can optionally STITCH one or more CONTINUATION checkpoints (for resumed
training) via its `cont` list: the base logs, then each continuation's logs appended
in turn, on one continuous x-axis (a hard section cut marks every join). Order matters.

Two INDEPENDENT policy groups are configured below: POLICIES_S1 (→ the *_S1 PNGs)
and POLICIES_S2 (→ the *_S2 PNGs). Each group draws its OWN pair of figures with
identical styling; leave POLICIES_S2 empty to skip the S2 plots. The groups do not
interact — the S1 figures are never affected by what you put in the S2 group.

Lines are smoothed with a per-section running average (curriculum transitions stay
sharp); see --smooth. Outputs use stable names (overwritten each run).

Run (repo root, project venv) — no args uses the POLICIES_S1/POLICIES_S2 config; or pass paths:
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\compare_two_policies_logs.py \\
      runs/.../complete_FINAL.pt runs/.../baseline_FINAL.pt runs/.../policy_c_FINAL.pt \\
      --labels Complete Baseline "Policy C"
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
from .nlr_style import (NLR_PRIMARY, NLR_SECONDARY, NLR_ACCENT, NLR_DARKGRAY,
                        NLR_GRAY, NLR_CYCLE)

# ===================================================================
# CONFIG — edit these (CLI flags / positional args override)
# ===================================================================

# The policies to compare — add as many as you like (C, D, …); one curve each.
# Each entry is a dict:
#   path  : base checkpoint  (bare name → runs/, "runs/…/x.pt" → project dir, abs → as-is)
#   label : legend label     (defaults to the checkpoint stem if omitted)
#   cont  : ORDERED list of CONTINUATION checkpoints for RESUMED training — each is
#           stitched on AFTER the previous, joining into one continuous x-axis (a
#           hard section cut marks every join). Order matters; [] = no continuation,
#           a single "path.pt" string is also accepted.
#   start_iter : (optional) drop the first N iterations and re-base the x-axis to 0.
#           Use it when a policy's early iterations belong to a different phase that
#           should not show here — e.g. iters 0→10k were trained on S1 before the S2
#           run: set start_iter=10_000 so the curve starts at the S2 beginning. It is
#           applied AFTER stitching, and may differ per policy — each still starts at
#           0 on the graph. Omit or 0 = keep the whole history.
#
# ── S1 GROUP ──  (draws the *_S1 PNGs; leave as-is — these are the good S1 policies)
POLICIES_S1: list[dict] = [
    # dict(path="runs/FINALV1/complete_s1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt",
    #      label="Complete V1",
    #      cont=[]),
    dict(path="runs/FINALV2/complete_stage7of8_DR_j2-4_k0_25.pt",
         label="Complete V2",
         cont=[]),
    dict(path="runs/FINALV2/baseline_stage11of11_DR_j2-4_k0_25_FINAL.pt",
         label="Baseline V2",
         cont=["runs/FINALV2/FINAL_baseline_s1_cont.pt",
               "runs/FINALV2/Final_Baseline_Cont_2.pt",
               "runs/FINALV2/Final_Baseline_Cont_3.pt",
               "runs/FINALV2/Final_Baseline_Cont_4.pt",
               "runs/FINALV2/Final_Baseline_Cont_5.pt"]),
    # dict(path="runs/.../policy_c.pt", label="Policy C", cont=[]),
    # dict(path="runs/.../policy_d.pt", label="Policy D", cont=[]),
]

# ── S2 GROUP ──  (draws the *_S2 PNGs; SAME dict format as POLICIES_S1 above).
# Fill in your S2 checkpoints here. Leave the list empty to skip the S2 plots
# entirely — the S1 figures are unaffected either way.
POLICIES_S2: list[dict] = [
    dict(path="runs/FINALV1/complete_S2_20260704/stage3of3_S2_DR_j2-4_k0_25_FINAL.pt",
         label="OLD - Complete S2",
         start_iter=5000,          # set to e.g. 10_000 to drop the S1 phase (0→10k)
         cont=[]),
    dict(path="runs/FINALV2/S2_Baseline_stage9of9_S2_DR_j2-4_k0_25_FINAL.pt",
         label="Baseline S2",
         start_iter=10000,          # may differ per policy; each still starts at 0 on the graph
         cont=[]),
]

# Running-average window (datapoints); resets at each curriculum section so
# transitions stay sharp. 1 = raw.                        [CLI: --smooth]
SMOOTH_WINDOW = 25

# Output PNGs (relative paths resolved against the project dir 0_TA_...). One pair
# per group; the S1 and S2 figures are written to distinct files so neither group
# overwrites the other.
REWARD_OUT_S1 = "eval_results/compare_reward_curves_S1.png"
RATES_OUT_S1 = "eval_results/compare_eval_rates_S1.png"
REWARD_OUT_S2 = "eval_results/compare_reward_curves_S2.png"
RATES_OUT_S2 = "eval_results/compare_eval_rates_S2.png"

# Figure resolution (dots per inch). Higher = sharper output (larger files). [CLI: --dpi]
DPI = 600

# Curriculum radar-kill-probability (P_kill) stages: (start_iter, p_kill). Each
# stage runs until the next stage's start (the last runs to the end of training).
# Annotated as small, slightly-gray dashed markers with a tiny value label low on
# the axis — deliberately subtle so they don't overpower the curves.
SHOW_PKILL_STAGES = False
PKILL_STAGES = [
    (0,    0.025),
    (1000, 0.05),
    (2000, 0.10),
    (3000, 0.15),
    (4000, 0.20),
    (5000, 0.25),
]

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

# Per-policy encodings (cycle when there are more policies than entries).
POLICY_COLORS = NLR_CYCLE                    # reward plot: one colour per policy
POLICY_STYLES = ["-", "--", "-.", ":"]      # eval plot: line style per policy


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


def _series_len(logs) -> int:
    """Number of logged iterations (max over all series)."""
    return max((len(v) for v in logs.values()), default=0)


def _stitch_logs(base_logs, base_bounds, cont_logs, cont_bounds):
    """Concatenate a continuation run's logs AFTER the base run's, so the plot
    continues from the base's last iteration. Every series is padded to its run's
    length first (keeping 1:1 iteration alignment), then base+continuation are
    concatenated. Section bounds are merged with the continuation shifted by the
    base length, so per-section smoothing keeps a hard cut at the join too."""
    base_len, cont_len = _series_len(base_logs), _series_len(cont_logs)

    def _pad(seq, n):
        seq = list(seq)
        return seq + [float("nan")] * (n - len(seq))

    stitched = {
        k: _pad(base_logs.get(k, []), base_len) + _pad(cont_logs.get(k, []), cont_len)
        for k in (set(base_logs) | set(cont_logs))
    }
    merged_bounds = list(base_bounds) + [
        (name, int(s) + base_len, int(e) + base_len) for (name, s, e) in cont_bounds
    ]
    return stitched, merged_bounds


def _as_cont_list(spec) -> list[str]:
    """Normalise a continuation spec (None / str / list[str]) to an ordered list."""
    if spec is None:
        return []
    if isinstance(spec, str):
        return [spec]
    return [s for s in spec if s]


def _maybe_stitch(base_logs, base_bounds, cont_spec, who):
    """Stitch an ORDERED chain of continuation checkpoints after the base run.

    Each continuation is appended onto the running (already-stitched) result, so
    a sequence of resumed runs joins into one continuous x-axis in the order given.
    Accepts None, a single path string, or a list of path strings.
    """
    cont_paths = _as_cont_list(cont_spec)
    if not cont_paths:
        return base_logs, base_bounds
    logs, bounds = base_logs, base_bounds
    n = len(cont_paths)
    for i, cont_path_str in enumerate(cont_paths, 1):
        cont_path = _resolve_policy_path(cont_path_str, None)
        if cont_path is None or not cont_path.exists():
            raise FileNotFoundError(f"continuation checkpoint not found: {cont_path}")
        cont_logs, cont_bounds = _load_logs(cont_path)
        prev_len = _series_len(logs)
        logs, bounds = _stitch_logs(logs, bounds, cont_logs, cont_bounds)
        print(f"  {who}: stitched +{cont_path.name}  "
              f"(step {i}/{n}: {prev_len} iters → +{_series_len(cont_logs)} continuation)")
    if n > 1:
        print(f"  {who}: total after {n} continuations = {_series_len(logs)} iters")
    return logs, bounds


def _trim_start(logs, bounds, start_iter, who=""):
    """Drop the first `start_iter` iterations of a (stitched) policy and RE-BASE its
    x-axis to 0. Use it when a policy's early iterations belong to a DIFFERENT phase
    (e.g. 0→10k trained on S1) that should not appear on this plot: after trimming,
    iteration `start_iter` becomes the first point, so two policies with DIFFERENT
    start_iter values both begin at the left edge (0) of the graph.

    Every log series is sliced from `start_iter` onward (index i == iteration i, so
    the slice is iteration-aligned), and section bounds are shifted by -start_iter;
    sections lying entirely in the trimmed-away region are dropped."""
    t = int(start_iter or 0)
    if t <= 0:
        return logs, bounds
    trimmed = {k: list(v)[t:] for k, v in logs.items()}
    shifted = [(name, int(s) - t, int(e) - t) for (name, s, e) in bounds if int(e) > t]
    if who:
        print(f"  {who}: start_iter={t} → dropped first {t} iters, re-based to 0 "
              f"({_series_len(trimmed)} iters remain)")
    return trimmed, shifted


def _annotate_pkill(ax, n_iter):
    """Mark the P_kill curriculum stages subtly, low on the axis.

    Draws a short slightly-gray dashed vertical at each stage boundary and a tiny
    value label centred under each stage, so the markers stay unobtrusive and do
    not overpower the curves. Uses a blended transform (x in data coordinates, y
    in axes fraction) so it also works on the reward axis (arbitrary y range)."""
    if not (SHOW_PKILL_STAGES and n_iter):
        return
    trans = ax.get_xaxis_transform()        # x = data, y = axes fraction (0=bottom)
    n_stages = len(PKILL_STAGES)
    for i, (start, pk) in enumerate(PKILL_STAGES):
        if start > n_iter:
            break
        end = min(PKILL_STAGES[i + 1][0] if i + 1 < n_stages else n_iter, n_iter)
        if start > 0:                       # short boundary marker (skip the x=0 edge)
            ax.plot([start, start], [0.0, 0.05], transform=trans, color=NLR_GRAY,
                    lw=0.7, ls="--", alpha=0.6, clip_on=False, zorder=1)
        ax.text((start + end) / 2.0, 0.012, f"{pk:g}", transform=trans, ha="center",
                va="bottom", fontsize=6.0, color=NLR_GRAY, alpha=0.9, zorder=1)
    ax.text(0.004, 0.052, "P_kill", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.0, color=NLR_GRAY, alpha=0.9, style="italic")


def plot_reward(policies, out, smooth, dpi=DPI):
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
    _annotate_pkill(ax, n_iter)
    ax.legend(loc="lower right", fontsize=10, frameon=True, title="policy")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved reward comparison -> {out}")


def plot_eval_rates(policies, out, smooth, dpi=DPI, title_suffix=""):
    """Every eval KPI for every policy: colour = metric, line style = policy."""
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
    ax.set_title(f"Eval rates — policy comparison{title_suffix}", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    if n_iter:
        ax.set_xlim(1, n_iter)
    ax.grid(True, alpha=0.3)
    _annotate_pkill(ax, n_iter)

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
    fig.savefig(out, dpi=dpi, bbox_inches="tight", bbox_extra_artists=[leg1, leg2])
    plt.close(fig)
    print(f"saved eval-rates comparison -> {out}")


def _specs_from_cli(args) -> list[dict]:
    """Ordered list of policy specs (path/label/cont) from CLI positional paths
    (no stitching via the CLI — configure continuations in the config groups)."""
    labels = args.labels or []
    return [dict(path=p, label=(labels[i] if i < len(labels) else None), cont=[])
            for i, p in enumerate(args.policies)]


def _run_group(specs, reward_out, rates_out, smooth, dpi, tag, title_suffix="") -> None:
    """Load + stitch + plot ONE policy group into its own pair of figures. An
    empty group is skipped (nothing to plot). `tag` names the group in the log;
    `title_suffix` is appended to the eval-rates title (e.g. ' (S2)')."""
    if not specs:
        print(f"[{tag}] no policies configured — skipping (no figures written).")
        return

    print(f"[{tag}] policies:")
    # Load + stitch each policy in order.
    loaded = []   # [label, logs, bounds, ckpt_name]
    for i, spec in enumerate(specs):
        path = _resolve_policy_path(spec["path"], None)
        if path is None or not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        label = spec.get("label") or path.stem
        logs, bounds = _load_logs(path)
        logs, bounds = _maybe_stitch(logs, bounds, spec.get("cont"), f"{chr(65 + i)}/{label}")
        logs, bounds = _trim_start(logs, bounds, spec.get("start_iter"), f"{chr(65 + i)}/{label}")
        loaded.append([label, logs, bounds, path.name])

    # Disambiguate any duplicate labels with the policy letter.
    labels = [x[0] for x in loaded]
    if len(set(labels)) < len(labels):
        for i, x in enumerate(loaded):
            x[0] = f"{x[0]} ({chr(65 + i)})"

    policies = []
    for i, (label, logs, bounds, name) in enumerate(loaded):
        policies.append((label, logs, bounds, POLICY_COLORS[i % len(POLICY_COLORS)]))
        print(f"  {chr(65 + i)}: {label}  [{name}]")

    if len(policies) > len(POLICY_STYLES):
        print(f"  ! {len(policies)} policies but only {len(POLICY_STYLES)} line "
              f"styles — eval-rates styles will repeat (curves may be hard to tell "
              f"apart). Reduce policies or add styles to POLICY_STYLES.")

    plot_reward(policies, _resolve_out(reward_out), smooth, dpi)
    plot_eval_rates(policies, _resolve_out(rates_out), smooth, dpi, title_suffix)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policies", nargs="*", default=None, metavar="CKPT",
                    help="base checkpoint(s) to compare, in order; when given, these "
                         "override the config groups and are drawn to the *_S1 PNGs "
                         "(only). For continuations / the S2 group, edit the config.")
    ap.add_argument("--labels", nargs="+", default=None, metavar="LABEL",
                    help="legend labels for the positional policies, in order "
                         "(default: the checkpoint stem)")
    ap.add_argument("--smooth", type=int, default=SMOOTH_WINDOW,
                    help="running-average window (1 = raw); resets at each curriculum section")
    ap.add_argument("--dpi", type=int, default=DPI, help="output resolution (dots per inch)")
    ap.add_argument("--reward-out-s1", default=REWARD_OUT_S1)
    ap.add_argument("--rates-out-s1", default=RATES_OUT_S1)
    ap.add_argument("--reward-out-s2", default=REWARD_OUT_S2)
    ap.add_argument("--rates-out-s2", default=RATES_OUT_S2)
    args = ap.parse_args()

    # CLI positional paths → an ad-hoc single group, drawn to the S1 outputs
    # (exactly the old behaviour). No S2 group in this mode.
    if args.policies:
        _run_group(_specs_from_cli(args), args.reward_out_s1, args.rates_out_s1,
                   args.smooth, args.dpi, tag="CLI", title_suffix=" (S1)")
        return

    # Default: run BOTH config groups, each to its own pair of figures.
    if not POLICIES_S1 and not POLICIES_S2:
        raise SystemExit("No policies to plot — populate POLICIES_S1 / POLICIES_S2 "
                         "or pass paths on the CLI.")
    _run_group([dict(s) for s in POLICIES_S1], args.reward_out_s1, args.rates_out_s1,
               args.smooth, args.dpi, tag="S1", title_suffix=" (S1)")
    _run_group([dict(s) for s in POLICIES_S2], args.reward_out_s2, args.rates_out_s2,
               args.smooth, args.dpi, tag="S2", title_suffix=" (S2)")


if __name__ == "__main__":
    main()
