"""Compare policy/value losses from two or more FOFE-MAPPO checkpoints.

The script reads the `training_logs` stored inside each checkpoint and saves two
stable PNGs that are overwritten on every run:

  1. Value loss comparison  - striker and jammer shown as separate subplots.
  2. Policy loss comparison - striker and jammer shown as separate subplots.

Run with no arguments to use the POLICIES config below, or pass checkpoints in
the same style as 3.3.1_compare_two_policies_logs.py:

  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\3.3.6_compare_policy_losses.py \\
      runs/FINALV1/complete_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt \\
      runs/FINALV1/baseline_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt \\
      --labels Complete Baseline
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PKG_NAME = "fofe_mappo"
if __package__ in (None, ""):
    sys.path.insert(0, str(_THIS_DIR.parent))
    if _PKG_NAME not in sys.modules:
        _pkg = types.ModuleType(_PKG_NAME)
        _pkg.__path__ = [str(_THIS_DIR.parent), str(_THIS_DIR)]
        _pkg.__package__ = _PKG_NAME
        _pkg.__file__ = str(_THIS_DIR / "__init__.py")
        sys.modules[_PKG_NAME] = _pkg
    __package__ = _PKG_NAME

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .evaluate_policy import _resolve_policy_path
from .plot_reward_components import _finite_xy, _smooth_sectioned
from .nlr_style import (
    NLR_ACCENT,
    NLR_CYCLE,
    NLR_DARKGRAY,
    NLR_GRAY,
    NLR_PRIMARY,
    NLR_SECONDARY,
)


# ===================================================================
# CONFIG - edit these defaults, or override with CLI positional args.
# ===================================================================

# Each entry:
#   path  : base checkpoint. Bare names resolve under runs/; paths with folders
#           resolve relative to 0_TA_HF_FOFE-MAPPO; absolute paths are used as-is.
#   label : legend label. Defaults to the checkpoint stem if omitted.
#   cont  : optional continuation checkpoints to stitch after `path`.
POLICIES: list[dict] = [
    dict(
        path="runs/FINALV2/complete_stage7of8_DR_j2-4_k0_25.pt",
        label="Complete",
        cont=[],
    ),
    dict(
        path="runs/FINALV2/baseline_stage11of11_DR_j2-4_k0_25_FINAL.pt",
        label="Baseline",
        cont=[],
    ),
]

SMOOTH_WINDOW = 25
DPI = 600

VALUE_OUT = "eval_results/compare_value_loss.png"
POLICY_OUT = "eval_results/compare_policy_loss.png"


LOSS_GROUPS = {
    "value": [
        ("striker_loss_value", "Striker value loss"),
        ("jammer_loss_value", "Jammer value loss"),
    ],
    "policy": [
        ("striker_loss_policy", "Striker policy loss"),
        ("jammer_loss_policy", "Jammer policy loss"),
    ],
}

ROLE_COLORS = {
    "striker": NLR_PRIMARY,
    "jammer": NLR_SECONDARY,
}


def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_THIS_DIR.parent / p)


def _load_logs(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    logs = ckpt.get("training_logs")
    if not logs:
        raise KeyError(f"{path.name} has no 'training_logs'; cannot plot losses.")
    return logs, (ckpt.get("section_bounds") or [])


def _series_len(logs) -> int:
    return max((len(v) for v in logs.values()), default=0)


def _pad(seq, n: int) -> list:
    seq = list(seq)
    return seq + [float("nan")] * (n - len(seq))


def _stitch_logs(base_logs, base_bounds, cont_logs, cont_bounds):
    base_len, cont_len = _series_len(base_logs), _series_len(cont_logs)
    stitched = {
        key: _pad(base_logs.get(key, []), base_len)
        + _pad(cont_logs.get(key, []), cont_len)
        for key in (set(base_logs) | set(cont_logs))
    }
    merged_bounds = list(base_bounds) + [
        (name, int(start) + base_len, int(end) + base_len)
        for (name, start, end) in cont_bounds
    ]
    return stitched, merged_bounds


def _as_cont_list(spec) -> list[str]:
    if spec is None:
        return []
    if isinstance(spec, str):
        return [spec]
    return [s for s in spec if s]


def _maybe_stitch(base_logs, base_bounds, cont_spec, label: str):
    logs, bounds = base_logs, base_bounds
    cont_paths = _as_cont_list(cont_spec)
    for idx, cont_path_str in enumerate(cont_paths, 1):
        cont_path = _resolve_policy_path(cont_path_str, None)
        if cont_path is None or not cont_path.exists():
            raise FileNotFoundError(f"continuation checkpoint not found: {cont_path}")
        cont_logs, cont_bounds = _load_logs(cont_path)
        before = _series_len(logs)
        logs, bounds = _stitch_logs(logs, bounds, cont_logs, cont_bounds)
        print(
            f"  {label}: stitched +{cont_path.name} "
            f"({idx}/{len(cont_paths)}: {before} + {_series_len(cont_logs)} iters)"
        )
    return logs, bounds


def _policy_specs_from_args(args) -> list[dict]:
    if args.policies:
        labels = args.labels or []
        return [
            dict(path=p, label=(labels[i] if i < len(labels) else None), cont=[])
            for i, p in enumerate(args.policies)
        ]
    return [dict(spec) for spec in POLICIES]


def _load_policies(specs: list[dict]) -> list[tuple[str, dict, list, str]]:
    loaded = []
    for i, spec in enumerate(specs):
        path = _resolve_policy_path(spec["path"], None)
        if path is None or not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        label = spec.get("label") or path.stem
        logs, bounds = _load_logs(path)
        logs, bounds = _maybe_stitch(logs, bounds, spec.get("cont"), label)
        loaded.append([label, logs, bounds, path.name])

    labels = [item[0] for item in loaded]
    if len(set(labels)) < len(labels):
        for i, item in enumerate(loaded):
            item[0] = f"{item[0]} ({chr(65 + i)})"
    return loaded


def _policy_color(policy_idx: int) -> str:
    return NLR_CYCLE[policy_idx % len(NLR_CYCLE)]


def _plot_loss_group(
    policies: list[tuple[str, dict, list, str]],
    group_name: str,
    out: Path,
    smooth: int,
    dpi: int,
) -> None:
    loss_specs = LOSS_GROUPS[group_name]
    fig, axes = plt.subplots(
        len(loss_specs), 1, figsize=(13, 8), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    max_iter = 0
    plotted_any = False

    for ax, (key, title) in zip(axes, loss_specs):
        role = "striker" if key.startswith("striker_") else "jammer"
        ax.axhline(0.0, color=NLR_DARKGRAY, lw=0.8, alpha=0.45)

        for p_idx, (label, logs, bounds, _name) in enumerate(policies):
            values = logs.get(key)
            if values is None:
                print(f"  ! {label}: no '{key}' in training_logs; skipped.")
                continue
            if not np.any(np.isfinite(np.asarray(values, dtype=float))):
                print(f"  ! {label}: '{key}' has no finite values; skipped.")
                continue

            xs, ys = _finite_xy(values)
            ys = _smooth_sectioned(xs, ys, bounds, smooth)
            max_iter = max(max_iter, len(values))
            plotted_any = True
            ax.plot(
                xs,
                ys,
                lw=1.8,
                color=_policy_color(p_idx),
                label=label,
            )

        ax.set_ylabel(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        if group_name == "value":
            ax.set_ylim(0.0, 50.0)
        ax.text(
            0.01,
            0.92,
            role.capitalize(),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color=ROLE_COLORS[role],
            fontweight="bold",
        )

    if not plotted_any:
        raise RuntimeError(f"no finite {group_name} loss series found to plot.")

    axes[-1].set_xlabel("Global training iteration", fontsize=11)
    if max_iter:
        for ax in axes:
            ax.set_xlim(1, max_iter)

    title = "Value loss comparison" if group_name == "value" else "Policy loss comparison"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="policy",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=min(len(labels), 4),
            frameon=True,
            fontsize=9,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {group_name} loss comparison -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "policies",
        nargs="*",
        default=None,
        metavar="CKPT",
        help=(
            "base checkpoint(s) to compare, in order; overrides POLICIES. "
            "For stitched continuations, edit POLICIES."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="legend labels for positional policies, in order",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=SMOOTH_WINDOW,
        help="running-average window (1 = raw); resets at curriculum sections",
    )
    parser.add_argument("--dpi", type=int, default=DPI, help="output resolution")
    parser.add_argument("--value-out", default=VALUE_OUT)
    parser.add_argument("--policy-out", default=POLICY_OUT)
    args = parser.parse_args()

    specs = _policy_specs_from_args(args)
    if not specs:
        raise SystemExit("No policies to plot; populate POLICIES or pass paths.")

    policies = _load_policies(specs)
    for i, (label, _logs, _bounds, name) in enumerate(policies):
        print(f"{chr(65 + i)}: {label} [{name}]")

    _plot_loss_group(
        policies,
        "value",
        _resolve_out(args.value_out),
        max(1, int(args.smooth)),
        args.dpi,
    )
    _plot_loss_group(
        policies,
        "policy",
        _resolve_out(args.policy_out),
        max(1, int(args.smooth)),
        args.dpi,
    )


if __name__ == "__main__":
    main()
