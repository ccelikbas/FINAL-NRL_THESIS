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

    # S2 scenarios
    dict(path="runs/FINALV3/complete_S2_FINAL.pt",
         label="V3 - Complete S2",
         cont=[]),
    dict(path="runs/FINALV3/baseline_S2_FINAL.pt",
         label="V3 - Baseline S2",
         cont=[]),
    dict(path="runs/FINALV2/S2_Baseline_stage9of9_S2_DR_j2-4_k0_25_FINAL.pt",
         label="V2 - Baseline S2",
         cont=[]),
    # S1 scenarios
    # dict(path="runs/FINALV2/complete_stage7of8_DR_j2-4_k0_25.pt",
    #      label="Complete V2",
    #      cont=[]),
    # dict(path="runs/FINALV2/baseline_stage11of11_DR_j2-4_k0_25_FINAL.pt",
    #      label="Baseline V2",
    #      cont=["runs/FINALV2/FINAL_baseline_s1_cont.pt",
    #            "runs/FINALV2/Final_Baseline_Cont_2.pt",
    #            "runs/FINALV2/Final_Baseline_Cont_3.pt",
    #            "runs/FINALV2/Final_Baseline_Cont_4.pt",
    #            "runs/FINALV2/Final_Baseline_Cont_5.pt"]),
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

# Convergence-comparison LaTeX table (Complete vs Baseline, rows S1 & S2), written
# in ADDITION to the figures. For each (scenario, model), computed on the SAME
# stitched + start_iter-rebased series as the reward plot:
#   Iterations (#) : the iteration at which the MAX training reward is reached —
#                    counted from 0 in S1, and from the re-based S2 start in S2.
#   Time (h)       : SUM of the per-iteration TRAINING time (iter_time_excl_eval_s,
#                    i.e. eval time excluded — already stored per-iter in the .pt)
#                    over iterations 0..peak, reported in HOURS.
#   Reward         : the maximum training reward after the final-stage start
#                    (S1 >= 5000, S2 >= 4000 on the plotted/re-based x-axis).
# Complete vs Baseline are matched per group by an optional `model="complete"` /
# "baseline" key on the policy dict, else inferred from a label containing
# "complete"/"baseline".
TABLE_OUT = "eval_results/convergence_comparison.tex"
# Window for locating the reward PEAK: 1 = RAW reward (the literal "maximum
# achieved reward"); set to SMOOTH_WINDOW to find the peak on the SMOOTHED curve
# drawn in the plots (steadier, ignores single-iteration spikes). [CLI: --table-smooth]
TABLE_SMOOTH_WINDOW = 1

# The final curriculum stage starts here on the plotted/re-based x-axis. The
# convergence table searches for the peak reward only from this point onward, and
# the figures mark the same point with a subtle vertical guide.
FINAL_STATE_START_ITER = {"S1": 5000, "S2": 4000}

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

# The (non-eval) training reward series, and the per-iteration TRAINING-only wall
# time (eval already subtracted) — both logged every iteration and stored in the .pt.
TRAIN_REWARD_KEY = "train_mean_episode_total_reward"
TRAIN_TIME_KEY = "iter_time_excl_eval_s"

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


def _annotate_final_state_start(ax, n_iter, start_iter, label=None):
    """Mark the first iteration included in the convergence-table peak search."""
    if start_iter is None:
        return
    start = int(start_iter)
    if start <= 0 or not n_iter or start > n_iter:
        return
    trans = ax.get_xaxis_transform()
    ax.axvline(start, color=NLR_DARKGRAY, lw=1.0, ls="--", alpha=0.65, zorder=1.5)
    ax.text(start, 0.985, label or f"final stage start ({start:,})", transform=trans,
            ha="right", va="top", rotation=90, fontsize=8.0,
            color=NLR_DARKGRAY, alpha=0.9, zorder=2)


def plot_reward(policies, out, smooth, dpi=DPI, final_start_iter=None, final_start_label=None):
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
    _annotate_final_state_start(ax, n_iter, final_start_iter, final_start_label)
    ax.legend(loc="lower right", fontsize=10, frameon=True, title="policy")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved reward comparison -> {out}")


def plot_eval_rates(policies, out, smooth, dpi=DPI, title_suffix="", final_start_iter=None, final_start_label=None):
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
    _annotate_final_state_start(ax, n_iter, final_start_iter, final_start_label)

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


def _policy_model(spec) -> "str | None":
    """Classify a policy as 'complete' / 'baseline' for the convergence table.
    Prefers an explicit `model` key on the dict; else infers from the label."""
    m = spec.get("model")
    if m:
        return str(m).strip().lower()
    lbl = (spec.get("label") or "").lower()
    if "baseline" in lbl:
        return "baseline"
    if "complete" in lbl:
        return "complete"
    return None


def _run_group(specs, reward_out, rates_out, smooth, dpi, tag, title_suffix="", final_start_iter=None) -> list[dict]:
    """Load + stitch + plot ONE policy group into its own pair of figures, and
    RETURN the per-policy data (label / model / stitched+rebased logs / bounds) so
    the convergence table can be built. An empty group is skipped. `tag` names the
    group in the log; `title_suffix` is appended to the eval-rates title."""
    if not specs:
        print(f"[{tag}] no policies configured — skipping (no figures written).")
        return []

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
    entries: list[dict] = []
    for i, (label, logs, bounds, name) in enumerate(loaded):
        policies.append((label, logs, bounds, POLICY_COLORS[i % len(POLICY_COLORS)]))
        entries.append({"label": label, "model": _policy_model(specs[i]),
                        "logs": logs, "bounds": bounds,
                        "final_start_iter": final_start_iter})
        print(f"  {chr(65 + i)}: {label}  [{name}]")

    if len(policies) > len(POLICY_STYLES):
        print(f"  ! {len(policies)} policies but only {len(POLICY_STYLES)} line "
              f"styles — eval-rates styles will repeat (curves may be hard to tell "
              f"apart). Reduce policies or add styles to POLICY_STYLES.")

    final_label = None
    if final_start_iter:
        final_label = f"final stage start ({int(final_start_iter / 1000)}k)"
    plot_reward(policies, _resolve_out(reward_out), smooth, dpi, final_start_iter, final_label)
    plot_eval_rates(policies, _resolve_out(rates_out), smooth, dpi, title_suffix,
                    final_start_iter, final_label)
    return entries


# =====================================================================
# Convergence-comparison table (Complete vs Baseline; rows S1 & S2)
# =====================================================================

def _convergence_stats(entry: dict, smooth_window: int, final_start_iter: int = 1) -> "dict | None":
    """For one policy's (already stitched + start_iter-rebased) logs, return
    {iters, hours, reward}:
      iters  = 1-based iteration at which the peak training reward occurs (from 0
               in S1, from the re-based S2 start in S2);
      hours  = summed TRAINING time (eval excluded) over iterations 0..peak / 3600;
      reward = the reward at that peak.
    The peak search ignores curriculum ramp-up by considering only samples with
    plotted_iteration >= final_start_iter.
    `smooth_window` > 1 locates the peak on the section-smoothed curve; 1 = raw."""
    logs, bounds = entry["logs"], entry["bounds"]
    rew = np.asarray(logs.get(TRAIN_REWARD_KEY, []), dtype=float)
    if rew.size == 0 or not np.any(np.isfinite(rew)):
        return None
    if smooth_window and smooth_window > 1:
        xs = np.arange(1, rew.size + 1)
        series = np.asarray(_smooth_sectioned(xs, rew, bounds, smooth_window), dtype=float)
    else:
        series = rew
    min_iter = max(1, int(final_start_iter or 1))
    finite = np.where(np.isfinite(series))[0]
    eligible = finite[(finite + 1) >= min_iter]
    if eligible.size == 0:
        print(f"  ! convergence table: {entry.get('label', 'policy')} has no "
              f"finite reward samples at/after iteration {min_iter}; skipped.")
        return None
    peak = int(eligible[np.argmax(series[eligible])])       # 0-based peak index
    t = np.asarray(logs.get(TRAIN_TIME_KEY, []), dtype=float)
    hours = None
    if t.size:
        hours = float(np.nansum(t[:min(peak + 1, t.size)])) / 3600.0
    return {"iters": peak + 1, "hours": hours, "reward": float(series[peak]),
            "search_start_iter": min_iter}


def _pick_model(entries: list, model: str) -> "dict | None":
    for e in entries or []:
        if e.get("model") == model:
            return e
    return None


def _fmt_iters(s): return "--" if s is None else str(int(s["iters"]))
def _fmt_time(s):  return "--" if (s is None or s["hours"] is None) else f"{s['hours']:.2f}"
def _fmt_reward(s): return "--" if s is None else f"{s['reward']:.2f}"


def build_convergence_table(rows: list) -> str:
    """rows: list of (scenario_str, complete_stats|None, baseline_stats|None)."""
    body = []
    for scen, comp, base in rows:
        body.append(
            f"        {scen} & {_fmt_iters(comp)} & {_fmt_time(comp)} & {_fmt_reward(comp)} "
            f"& {_fmt_iters(base)} & {_fmt_time(base)} & {_fmt_reward(base)} " + r"\\")
    return "\n".join([
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Convergence comparison between the complete and baseline models in Scenarios~S1 and~S2. The peak reward is searched only after the final curriculum stage starts (S1: iteration~5000; S2: iteration~4000).}",
        r"    \label{tab:convergence_comparison}",
        r"    \small",
        r"    \begin{tabular}{lcccccc}",
        r"        \toprule",
        r"        & \multicolumn{3}{c}{\textbf{Complete}}",
        r"        & \multicolumn{3}{c}{\textbf{Baseline}} \\",
        r"        \cmidrule(lr){2-4}",
        r"        \cmidrule(lr){5-7}",
        r"        \textbf{Scenario}",
        r"        & \textbf{Iterations (\#)}",
        r"        & \textbf{Time (h)}",
        r"        & \textbf{Reward}",
        r"        & \textbf{Iterations (\#)}",
        r"        & \textbf{Time (h)}",
        r"        & \textbf{Reward} \\",
        r"        \midrule",
        *body,
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ])


def write_convergence_table(s1_entries: list, s2_entries: list,
                            smooth_window: int, out_path: Path) -> None:
    """Build + print + save the S1/S2 Complete-vs-Baseline convergence table."""
    rows = []
    for scen, entries in (("S1", s1_entries), ("S2", s2_entries)):
        comp, base = _pick_model(entries, "complete"), _pick_model(entries, "baseline")
        if entries and comp is None:
            print(f"  ! convergence table: no 'complete' policy identified in {scen} "
                  f"(add model='complete' or use a label containing 'complete').")
        if entries and base is None:
            print(f"  ! convergence table: no 'baseline' policy identified in {scen} "
                  f"(add model='baseline' or use a label containing 'baseline').")
        start_iter = FINAL_STATE_START_ITER.get(scen, 1)
        rows.append((scen, _convergence_stats(comp, smooth_window, start_iter) if comp else None,
                     _convergence_stats(base, smooth_window, start_iter) if base else None))

    mode = "raw" if not (smooth_window and smooth_window > 1) else f"smoothed(w={smooth_window})"
    print(f"\nConvergence comparison (peak reward located on {mode} curve "
          f"after final-stage start: S1>={FINAL_STATE_START_ITER['S1']}, "
          f"S2>={FINAL_STATE_START_ITER['S2']}; time = training-only hours to peak):")
    print(f"  {'':4s}  {'Complete (iters / h / reward)':32s}  Baseline (iters / h / reward)")
    for scen, comp, base in rows:
        def _cell(s):
            return "--" if s is None else f"{_fmt_iters(s):>6s} / {_fmt_time(s):>6s} / {_fmt_reward(s):>7s}"
        print(f"  {scen:4s}  {_cell(comp):32s}  {_cell(base)}")

    tex = build_convergence_table(rows)
    print("\n" + tex + "\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex, encoding="utf-8")
    print(f"saved convergence table -> {out_path}")


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
    ap.add_argument("--table-out", default=TABLE_OUT)
    ap.add_argument("--table-smooth", type=int, default=TABLE_SMOOTH_WINDOW,
                    help="window for the convergence-table peak (1 = raw max reward; "
                         "SMOOTH_WINDOW = locate the peak on the smoothed curve)")
    args = ap.parse_args()

    # CLI positional paths → an ad-hoc single group, drawn to the S1 outputs
    # (exactly the old behaviour). No S2 group / convergence table in this mode.
    if args.policies:
        _run_group(_specs_from_cli(args), args.reward_out_s1, args.rates_out_s1,
                   args.smooth, args.dpi, tag="CLI", title_suffix=" (S1)")
        return

    # Default: run BOTH config groups, each to its own pair of figures.
    if not POLICIES_S1 and not POLICIES_S2:
        raise SystemExit("No policies to plot — populate POLICIES_S1 / POLICIES_S2 "
                         "or pass paths on the CLI.")
    s1_entries = _run_group([dict(s) for s in POLICIES_S1], args.reward_out_s1,
                            args.rates_out_s1, args.smooth, args.dpi, tag="S1",
                            title_suffix=" (S1)",
                            final_start_iter=FINAL_STATE_START_ITER["S1"])
    s2_entries = _run_group([dict(s) for s in POLICIES_S2], args.reward_out_s2,
                            args.rates_out_s2, args.smooth, args.dpi, tag="S2",
                            title_suffix=" (S2)",
                            final_start_iter=FINAL_STATE_START_ITER["S2"])

    # Convergence-comparison table (Complete vs Baseline, rows S1 & S2).
    write_convergence_table(s1_entries, s2_entries, args.table_smooth,
                            _resolve_out(args.table_out))


if __name__ == "__main__":
    main()
