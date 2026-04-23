from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


PHASE_COLORS = [
    "#e8f0fe",
    "#fef7e0",
    "#e8fce8",
    "#fce8e8",
    "#f0e8fc",
    "#fce8f0",
    "#e0f7fe",
    "#fefce0",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a curriculum checkpoint (HF_cur.pt style) and recreate each dashboard "
            "panel as a separate plot."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("2_policies_runs/saved_runs/HF_cur.pt"),
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--mode",
        choices=["show", "save", "both"],
        default="show",
        help="show: interactive windows, save: png files, both: save and show",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("2_policies_runs/hf_cur_individual_plots"),
        help="Directory for saved plots (used in save/both mode)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG dpi when saving",
    )
    parser.add_argument(
        "--non_blocking_show",
        action="store_true",
        help="Open all figures at once instead of one-by-one blocking windows",
    )
    return parser


def _ensure_fofe_package_alias(repo_root: Path) -> None:
    pkg_name = "fofe_mappo"
    if pkg_name in sys.modules:
        return

    hf_dir = repo_root / "1_HF_FOFE-MAPPO"
    if not hf_dir.exists():
        return

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(hf_dir)]
    pkg.__package__ = pkg_name
    pkg.__file__ = str(hf_dir / "__init__.py")
    sys.modules[pkg_name] = pkg


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def _load_checkpoint(checkpoint_path: Path, repo_root: Path) -> Dict[str, Any]:
    _ensure_fofe_package_alias(repo_root)
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint must be a dict, got: {type(obj)}")
    return obj


def _as_training_logs(checkpoint: Dict[str, Any]) -> Dict[str, List[float]]:
    logs = checkpoint.get("training_logs")
    if logs is None:
        logs = checkpoint.get("logs")
    if not isinstance(logs, dict):
        return {}
    return logs


def _as_eval_history(
    checkpoint: Dict[str, Any],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    raw = checkpoint.get("eval_history")
    if not isinstance(raw, dict):
        return {}

    cleaned: Dict[int, Dict[str, Dict[str, float]]] = {}
    for key, value in raw.items():
        try:
            iter_idx = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            cleaned[iter_idx] = value
    return cleaned


def _as_curriculum_spans(checkpoint: Dict[str, Any]) -> List[Tuple[int, int, str, str]]:
    raw = checkpoint.get("curriculum_phases")
    if not isinstance(raw, list):
        return []

    spans: List[Tuple[int, int, str, str]] = []
    for idx, phase in enumerate(raw):
        if not isinstance(phase, dict):
            continue
        iters = phase.get("iters")
        if not isinstance(iters, (list, tuple)) or len(iters) != 2:
            continue
        try:
            start = int(iters[0])
            end = int(iters[1])
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue

        name = str(phase.get("name", f"phase_{idx}"))
        config_type = str(phase.get("config_type", ""))
        spans.append((start, end, name, config_type))

    return spans


def _plot_valid(ax: plt.Axes, x: List[float], y: List[float], label: str, **kwargs: Any) -> None:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size == 0 or y_arr.size == 0:
        return

    n = min(len(x_arr), len(y_arr))
    if n == 0:
        return

    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    valid = np.isfinite(y_arr)
    if not np.any(valid):
        return

    ax.plot(x_arr[valid], y_arr[valid], label=label, **kwargs)


def _add_phase_shading(ax: plt.Axes, phases: List[Tuple[int, int, str, str]]) -> None:
    for i, (start, end, _, _) in enumerate(phases):
        ax.axvspan(start, end, alpha=0.18, color=PHASE_COLORS[i % len(PHASE_COLORS)], zorder=0)


def _style_axis(ax: plt.Axes, title: str, xlabel: str = "Global Iteration") -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _scenario_labels(eval_history: Dict[int, Dict[str, Dict[str, float]]]) -> List[str]:
    if not eval_history:
        return []
    first_iter = sorted(eval_history.keys())[0]
    first = eval_history.get(first_iter, {})
    if not isinstance(first, dict):
        return []
    return [str(label) for label in first.keys()]


def _eval_values(
    eval_history: Dict[int, Dict[str, Dict[str, float]]],
    eval_iters: List[int],
    scenario: str,
    metric_key: str,
) -> List[float]:
    values: List[float] = []
    for iter_idx in eval_iters:
        per_iter = eval_history.get(iter_idx, {})
        scenario_metrics = per_iter.get(scenario, {}) if isinstance(per_iter, dict) else {}
        metric = scenario_metrics.get(metric_key) if isinstance(scenario_metrics, dict) else None
        values.append(_safe_float(metric))
    return values


def _plot_eval_metric(
    ax: plt.Axes,
    eval_history: Dict[int, Dict[str, Dict[str, float]]],
    phases: List[Tuple[int, int, str, str]],
    metric_key: str,
    title: str,
) -> None:
    eval_iters = sorted(eval_history.keys())
    labels = _scenario_labels(eval_history)

    if not eval_iters or not labels:
        _style_axis(ax, title)
        ax.text(0.5, 0.5, "No eval_history in checkpoint", ha="center", va="center", transform=ax.transAxes)
        return

    for label in labels:
        vals = _eval_values(eval_history, eval_iters, label, metric_key)
        _plot_valid(ax, eval_iters, vals, label, marker="o", markersize=2)

    avg_vals: List[float] = []
    for iter_idx in eval_iters:
        finite: List[float] = []
        for label in labels:
            per_iter = eval_history.get(iter_idx, {})
            scenario_metrics = per_iter.get(label, {}) if isinstance(per_iter, dict) else {}
            v = _safe_float(scenario_metrics.get(metric_key) if isinstance(scenario_metrics, dict) else None)
            if math.isfinite(v):
                finite.append(v)
        avg_vals.append(sum(finite) / len(finite) if finite else float("nan"))

    _plot_valid(
        ax,
        eval_iters,
        avg_vals,
        "AVERAGE",
        linewidth=2.5,
        color="black",
        linestyle="--",
        marker="o",
        markersize=2,
    )

    _add_phase_shading(ax, phases)
    _style_axis(ax, title)
    ax.legend(fontsize=7, loc="best")


def _make_fig_ax() -> Tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(11, 6))


def create_individual_plots(
    checkpoint: Dict[str, Any],
) -> List[Tuple[str, plt.Figure]]:
    training_logs = _as_training_logs(checkpoint)
    eval_history = _as_eval_history(checkpoint)
    phases = _as_curriculum_spans(checkpoint)

    figures: List[Tuple[str, plt.Figure]] = []

    # 1. Training episode reward
    fig, ax = _make_fig_ax()
    key = "train_mean_episode_total_reward"
    if key in training_logs:
        x = list(range(1, len(training_logs[key]) + 1))
        y = [_safe_float(v) for v in training_logs[key]]
        _plot_valid(ax, x, y, "train_reward", marker="o", markersize=2)
    _add_phase_shading(ax, phases)
    _style_axis(ax, "Training Episode Reward")
    ax.legend(fontsize=8)
    figures.append(("01_training_reward.png", fig))

    # 2. Policy and value loss
    fig, ax = _make_fig_ax()
    for k, label, color in [
        ("striker_loss_policy", "striker_pi", "tab:blue"),
        ("striker_loss_value", "striker_V", "tab:orange"),
        ("jammer_loss_policy", "jammer_pi", "tab:green"),
        ("jammer_loss_value", "jammer_V", "tab:red"),
    ]:
        if k in training_logs:
            x = list(range(1, len(training_logs[k]) + 1))
            y = [_safe_float(v) for v in training_logs[k]]
            _plot_valid(ax, x, y, label, color=color)
    _add_phase_shading(ax, phases)
    _style_axis(ax, "Policy and Value Loss")
    ax.legend(fontsize=8)
    figures.append(("02_policy_value_loss.png", fig))

    # 3. Entropy and KL divergence
    fig, ax = _make_fig_ax()
    for k, label, color in [
        ("striker_entropy", "striker_H", "tab:green"),
        ("jammer_entropy", "jammer_H", "tab:olive"),
        ("striker_approx_kl", "striker_KL", "tab:red"),
        ("jammer_approx_kl", "jammer_KL", "tab:pink"),
    ]:
        if k in training_logs:
            x = list(range(1, len(training_logs[k]) + 1))
            y = [_safe_float(v) for v in training_logs[k]]
            _plot_valid(ax, x, y, label, color=color)
    _add_phase_shading(ax, phases)
    _style_axis(ax, "Entropy and KL Divergence")
    ax.legend(fontsize=8)
    figures.append(("03_entropy_kl.png", fig))

    # 4-7. Eval metrics
    for filename, metric_key, title in [
        ("04_eval_completion_rate.png", "completion_rate", "Eval Task Completion Rate"),
        ("05_eval_survival_rate.png", "survival_rate", "Eval Survival Rate"),
        ("06_eval_mean_duration.png", "mean_duration", "Eval Mean Duration"),
        ("07_eval_mean_reward.png", "mean_reward", "Eval Mean Reward"),
    ]:
        fig, ax = _make_fig_ax()
        _plot_eval_metric(ax, eval_history, phases, metric_key, title)
        figures.append((filename, fig))

    # 8. Curriculum phase timeline
    fig, ax = _make_fig_ax()
    palette = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:cyan",
        "tab:olive",
    ]
    if phases:
        for i, (start, end, name, config_type) in enumerate(phases):
            color = palette[i % len(palette)]
            width = end - start
            ax.barh(0, width, left=start, height=0.5, color=color, alpha=0.75, edgecolor="black")
            mid = start + (width / 2.0)
            text = f"{name}\\n{config_type}" if config_type else name
            ax.text(mid, 0, text, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.set_yticks([])
    else:
        ax.text(0.5, 0.5, "No curriculum_phases in checkpoint", ha="center", va="center", transform=ax.transAxes)
    _style_axis(ax, "Curriculum Phases")
    figures.append(("08_curriculum_phases.png", fig))

    # 9. Time per iteration
    fig, ax = _make_fig_ax()
    if "iter_time_s" in training_logs:
        x = list(range(1, len(training_logs["iter_time_s"]) + 1))
        y = [_safe_float(v) for v in training_logs["iter_time_s"]]
        _plot_valid(ax, x, y, "total", color="tab:blue")
    if "iter_time_excl_eval_s" in training_logs:
        x = list(range(1, len(training_logs["iter_time_excl_eval_s"]) + 1))
        y = [_safe_float(v) for v in training_logs["iter_time_excl_eval_s"]]
        _plot_valid(ax, x, y, "train only", color="tab:orange")
    _add_phase_shading(ax, phases)
    _style_axis(ax, "Time per Iteration (s)")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=8)
    figures.append(("09_iteration_time.png", fig))

    return figures


def main() -> None:
    args = build_parser().parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = _resolve_path(args.checkpoint, repo_root)
    output_dir = _resolve_path(args.output_dir, repo_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = _load_checkpoint(checkpoint_path, repo_root)
    figures = create_individual_plots(checkpoint)

    should_save = args.mode in {"save", "both"}
    should_show = args.mode in {"show", "both"}

    if should_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, fig in figures:
            fig.tight_layout()
            fig.savefig(output_dir / filename, dpi=args.dpi)
        print(f"Saved {len(figures)} plots to: {output_dir}")

    if should_show:
        if args.non_blocking_show:
            for _, fig in figures:
                fig.tight_layout()
                fig.show()
            plt.show()
        else:
            for idx, (filename, fig) in enumerate(figures, start=1):
                fig.tight_layout()
                print(f"Showing {idx}/{len(figures)}: {filename}")
                plt.figure(fig.number)
                plt.show()

    for _, fig in figures:
        plt.close(fig)


if __name__ == "__main__":
    main()
