from __future__ import annotations

"""
run_generalisation.py
─────────────────────
Zero-shot generalisation test for a saved FOFE-MAPPO policy.

Loads a trained checkpoint and evaluates the frozen policy on arbitrary
team compositions without any fine-tuning.  The FOFE permutation-invariant
architecture allows the same weight matrix to handle variable S/J/T/R counts.

Usage
-----
    .\\.venv\\Scripts\\python.exe -m FOFE-MAPPO.run_generalisation --checkpoint runs/saved_runs/1x2_FOFE_14_04.pt
    .\\.venv\\Scripts\\python.exe -m FOFE-MAPPO.run_generalisation --checkpoint runs/fofe_mappo.pt --n_eval_episodes 100

If the virtual environment is already activated, you can use:
    python.exe -m FOFE-MAPPO.run_generalisation --checkpoint runs/fofe_mappo.pt

Editing test configurations
----------------------------
Change the TEST_CONFIGS list near the top of this file.
Each entry is a dict with keys:

    label             (str)  shown in the results table — keep ≤ 16 chars
    n_strikers        (int)  number of striker agents
    n_jammers         (int)  number of jammer agents
    n_known_targets   (int)  targets visible from the start
    n_unknown_targets (int)  targets that start hidden  (default 0)
    n_known_radars    (int)  radars visible from the start
    n_unknown_radars  (int)  radars that start hidden   (default 0)

The trained composition from the checkpoint is automatically prepended as a
baseline row.  Pass --no_baseline to suppress it.
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
_CHECKPOINT_PKG_ALIAS = "fofe_mappo"

if __package__ in (None, ""):
    import types
    sys.path.insert(0, str(_THIS_DIR.parent))
    if _CHECKPOINT_PKG_ALIAS not in sys.modules:
        _pkg = types.ModuleType(_CHECKPOINT_PKG_ALIAS)
        _pkg.__path__ = [str(_THIS_DIR)]
        _pkg.__package__ = _CHECKPOINT_PKG_ALIAS
        _pkg.__file__ = str(_THIS_DIR / "__init__.py")
        sys.modules[_CHECKPOINT_PKG_ALIAS] = _pkg
    __package__ = _CHECKPOINT_PKG_ALIAS

import torch
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None

from .config import EnvConfig, ExperimentConfig, FOFEConfig, NetworkConfig, PPOConfig
from .models import make_combined_policy
from .trainer import build_env, evaluate_current_policy


def _ensure_checkpoint_import_aliases() -> None:
    """Expose stable module aliases expected by legacy checkpoints."""
    import types

    if _CHECKPOINT_PKG_ALIAS not in sys.modules:
        alias_pkg = types.ModuleType(_CHECKPOINT_PKG_ALIAS)
        alias_pkg.__path__ = [str(_THIS_DIR)]
        alias_pkg.__package__ = _CHECKPOINT_PKG_ALIAS
        alias_pkg.__file__ = str(_THIS_DIR / "__init__.py")
        sys.modules[_CHECKPOINT_PKG_ALIAS] = alias_pkg

    config_module = sys.modules.get(EnvConfig.__module__)
    if config_module is not None:
        sys.modules.setdefault(f"{_CHECKPOINT_PKG_ALIAS}.config", config_module)


# ======================================================================
#  >>>  EDIT TEST CONFIGURATIONS HERE  <<<
#
#  Format: sNjNtNrN  →  n_strikers x n_jammers x n_known_targets x n_known_radars
#
#  Current examples cover:
#    scale-down  : 1 × 1 × 1 × 1
#    trained     : 2 × 2 × 2 × 2   (auto-added from checkpoint as baseline)
#    scale-up    : 3 × 3 × 3 × 3
#    asymmetric  : 2 × 2 × 4 × 4   (more threats than agents)
#
#  Add / remove / reorder rows freely.
# ======================================================================

TEST_CONFIGS: List[Dict[str, Any]] = [
    {
        "label":             "1s_1j_1t_1r",
        "n_strikers":        1,
        "n_jammers":         1,
        "n_known_targets":   1,
        "n_unknown_targets": 0,
        "n_known_radars":    1,
        "n_unknown_radars":  0,
    },
        {
        "label":             "1s_1j_2t_2r",
        "n_strikers":        1,
        "n_jammers":         1,
        "n_known_targets":   2,
        "n_unknown_targets": 0,
        "n_known_radars":    2,
        "n_unknown_radars":  0,
    },
    {
        "label":             "2s_2j_2t_2r",
        "n_strikers":        2,
        "n_jammers":         2,
        "n_known_targets":   2,
        "n_unknown_targets": 0,
        "n_known_radars":    2,
        "n_unknown_radars":  0,
    },
    {
        "label":             "4s_4j_4t_4r",
        "n_strikers":        2,
        "n_jammers":         2,
        "n_known_targets":   3,
        "n_unknown_targets": 0,
        "n_known_radars":    3,
        "n_unknown_radars":  0,
    },
        {
        "label":             "5s_5j_5t_5r",
        "n_strikers":        3,
        "n_jammers":         3,
        "n_known_targets":   3,
        "n_unknown_targets": 0,
        "n_known_radars":    3,
        "n_unknown_radars":  0,
    },
]

# ======================================================================


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _build_eval_env_cfg(base: EnvConfig, cfg: Dict[str, Any]) -> EnvConfig:
    """
    Clone the checkpoint's EnvConfig, replacing only the team composition.
    All physics, sensor ranges, and reward weights come from the trained run.
    """
    return EnvConfig(
        # ── Team composition from the test entry ──────────────────────
        n_strikers=cfg["n_strikers"],
        n_jammers=cfg["n_jammers"],
        n_known_targets=cfg["n_known_targets"],
        n_unknown_targets=cfg.get("n_unknown_targets", 0),
        n_known_radars=cfg["n_known_radars"],
        n_unknown_radars=cfg.get("n_unknown_radars", 0),
        # ── World / episode (from checkpoint) ────────────────────────
        world_bounds=base.world_bounds,
        dt=base.dt,
        max_steps=base.max_steps,
        n_env_layouts=base.n_env_layouts,
        radar_min_sep=getattr(base, "radar_min_sep", 0.5),
        target_spawn_angle_range=base.target_spawn_angle_range,
        # ── Kinematics (from checkpoint) ─────────────────────────────
        v_max=base.v_max,
        accel_magnitude=base.accel_magnitude,
        dpsi_max=base.dpsi_max,
        h_accel_magnitude_fraction=base.h_accel_magnitude_fraction,
        min_turn_radius=base.min_turn_radius,
        # ── Sensors (from checkpoint) ─────────────────────────────────
        R_obs=base.R_obs,
        R_comm=base.R_comm,
        # ── Agent capabilities (from checkpoint) ──────────────────────
        striker_engage_range=base.striker_engage_range,
        striker_engage_fov=base.striker_engage_fov,
        striker_v_min=base.striker_v_min,
        jammer_jam_radius=base.jammer_jam_radius,
        jammer_jam_effect=base.jammer_jam_effect,
        jammer_v_min=base.jammer_v_min,
        # ── Threats (from checkpoint) ─────────────────────────────────
        radar_range=base.radar_range,
        radar_kill_probability=base.radar_kill_probability,
        border_thresh=base.border_thresh,
        # ── Rewards (deep-copied to avoid shared state) ───────────────
        reward_config=copy.deepcopy(base.reward_config),
    )


def _build_eval_ppo_cfg(template: PPOConfig, device: torch.device, seed: int) -> PPOConfig:
    """Minimal PPO config for evaluation (no training needed)."""
    return PPOConfig(
        num_envs=template.num_envs,
        n_iters=1,
        num_epochs=template.num_epochs,
        minibatch_size=template.minibatch_size,
        actor_lr=template.actor_lr,
        critic_lr=template.critic_lr,
        clip_eps=template.clip_eps,
        entropy_coef=template.entropy_coef,
        normalize_rewards=template.normalize_rewards,
        seed=seed,
        device=device,
    )


def _print_results_table(rows: List[Dict[str, Any]], n_eval_episodes: int) -> None:
    """Print a results table with column widths computed from the actual data."""
    BASELINE_SUFFIX = "  ← trained baseline"

    # ── Dynamic column widths ─────────────────────────────────────────
    # Label column: wide enough for the header and the longest label
    W_LBL = max(len("Config"), max(len(r["label"]) for r in rows))

    # Count columns (S/J/T/R): wide enough for the header char and any value
    max_count = max(
        max(r["n_strikers"], r["n_jammers"], r["n_targets"], r["n_radars"])
        for r in rows
    )
    W_CNT = max(1, len(str(max_count)))  # at least 1 char

    # Metric columns: wide enough for header names and formatted values
    #   completion / survival → "0.0000" = 6 chars  → header "Completion" = 10
    #   mean time             → e.g. "148.2"        → header "Mean Time"  =  9
    #   mean reward           → e.g. "-12.3456"     → header "Mean Reward" = 11
    max_time_len    = max(len(f"{r['mean_duration']:.1f}") for r in rows)
    max_reward_len  = max(len(f"{r['mean_reward']:.4f}") for r in rows)

    W_COMPL = max(len("Completion"), 6)       # "0.0000"
    W_SURV  = max(len("Survival"),   6)       # "0.0000"
    W_TIME  = max(len("Mean Time"),  max_time_len)
    W_REW   = max(len("Mean Reward"), max_reward_len)

    # ── Separator ─────────────────────────────────────────────────────
    sep = "  ".join([
        "-" * W_LBL,
        "-" * W_CNT + "  " + "-" * W_CNT + "  " + "-" * W_CNT + "  " + "-" * W_CNT,
        "-" * W_COMPL,
        "-" * W_SURV,
        "-" * W_TIME,
        "-" * W_REW,
    ])

    title_line = f"  Zero-Shot Generalisation Results   (episodes per config: {n_eval_episodes})"
    bar = "=" * max(len(sep), len(title_line) + 2)

    print(f"\n{bar}")
    print(title_line)
    print(bar)

    hdr = (
        f"{'Config':<{W_LBL}}  "
        f"{'S':>{W_CNT}}  {'J':>{W_CNT}}  {'T':>{W_CNT}}  {'R':>{W_CNT}}  "
        f"{'Completion':>{W_COMPL}}  "
        f"{'Survival':>{W_SURV}}  "
        f"{'Mean Time':>{W_TIME}}  "
        f"{'Mean Reward':>{W_REW}}"
    )
    print(hdr)
    print(sep)

    for row in rows:
        suffix = BASELINE_SUFFIX if row.get("is_baseline") else ""
        line = (
            f"{row['label']:<{W_LBL}}  "
            f"{row['n_strikers']:>{W_CNT}}  "
            f"{row['n_jammers']:>{W_CNT}}  "
            f"{row['n_targets']:>{W_CNT}}  "
            f"{row['n_radars']:>{W_CNT}}  "
            f"{row['completion_rate']:>{W_COMPL}.4f}  "
            f"{row['survival_rate']:>{W_SURV}.4f}  "
            f"{row['mean_duration']:>{W_TIME}.1f}  "
            f"{row['mean_reward']:>{W_REW}.4f}"
            f"{suffix}"
        )
        print(line)

    print(sep)
    print()


def _plot_completion_and_survival(rows: List[Dict[str, Any]], n_eval_episodes: int) -> None:
    """Plot completion and survival separately for each TEST_CONFIGS label."""
    if plt is None:
        print("Plotting skipped: matplotlib is not available in this environment.")
        return

    labels = [r["label"] for r in rows]
    completion = [float(r["completion_rate"]) for r in rows]
    survival = [float(r["survival_rate"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].bar(labels, completion, color="tab:blue", alpha=0.9)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Rate")
    axes[0].set_title(f"Completion Rate by Config (n={n_eval_episodes})")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, survival, color="tab:green", alpha=0.9)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Rate")
    axes[1].set_title(f"Survival Rate by Config (n={n_eval_episodes})")
    axes[1].set_xlabel("Configuration Label")
    axes[1].grid(axis="y", alpha=0.3)
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# Main evaluation function
# ──────────────────────────────────────────────────────────────────────

def run_generalisation(
    checkpoint_path: str,
    test_configs: List[Dict[str, Any]],
    n_eval_episodes: int = 200,
    device: Optional[torch.device] = None,
    seed: int = 42,
    include_baseline: bool = True,
    plot_metrics: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load a frozen policy and evaluate it across *test_configs*.

    Parameters
    ----------
    checkpoint_path   : path to a saved .pt checkpoint file
    test_configs      : list of team-composition dicts (see TEST_CONFIGS at top)
    n_eval_episodes   : number of episodes to average per configuration
    device            : torch device (None = auto-detect cuda/cpu)
    seed              : RNG seed for evaluation rollouts
    include_baseline  : prepend the checkpoint's own trained composition

    Returns
    -------
    List of result dicts with keys:
        label, n_strikers, n_jammers, n_targets, n_radars,
        completion_rate, survival_rate, mean_duration, mean_reward, is_baseline
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'─' * 60}")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Device     : {device}")
    print(f"{'─' * 60}")

    # PyTorch 2.6 changed torch.load default to weights_only=True, which
    # rejects custom config dataclasses stored in our trusted local checkpoints.
    _ensure_checkpoint_import_aliases()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    base_env_cfg: EnvConfig   = checkpoint["env_cfg"]
    net_cfg: NetworkConfig    = checkpoint["net_cfg"]
    ppo_template: PPOConfig   = checkpoint.get("ppo_cfg", PPOConfig())
    fofe_cfg: FOFEConfig      = checkpoint.get("fofe_cfg", FOFEConfig(use_fofe=False))

    trained_label = (
        f"{base_env_cfg.n_strikers}s_"
        f"{base_env_cfg.n_jammers}j_"
        f"{base_env_cfg.n_targets}t_"
        f"{base_env_cfg.n_radars}r"
    )

    print(f"  Trained on : {trained_label}")
    print(f"  FOFE       : {'enabled' if fofe_cfg.use_fofe else 'disabled (legacy)'}")
    print(f"  Max steps  : {base_env_cfg.max_steps}")
    print(f"  Episodes   : {n_eval_episodes} per config")
    print(f"{'─' * 60}\n")

    # ── Assemble the list of configs to evaluate ──────────────────────
    configs_to_eval: List[Dict[str, Any]] = list(test_configs)

    if include_baseline:
        # Compare on TOTAL counts (n_targets = known + unknown) so the check
        # works regardless of whether the user specifies n_known_* or n_targets.
        def _total_targets(c: Dict[str, Any]) -> int:
            if "n_known_targets" in c or "n_unknown_targets" in c:
                return c.get("n_known_targets", 0) + c.get("n_unknown_targets", 0)
            return c.get("n_targets", 0)

        def _total_radars(c: Dict[str, Any]) -> int:
            if "n_known_radars" in c or "n_unknown_radars" in c:
                return c.get("n_known_radars", 0) + c.get("n_unknown_radars", 0)
            return c.get("n_radars", 0)

        baseline_key = (
            base_env_cfg.n_strikers,
            base_env_cfg.n_jammers,
            base_env_cfg.n_targets,   # total = known + unknown
            base_env_cfg.n_radars,
        )
        already_present = any(
            (
                c["n_strikers"],
                c["n_jammers"],
                _total_targets(c),
                _total_radars(c),
            ) == baseline_key
            for c in configs_to_eval
        )
        if not already_present:
            configs_to_eval = [{
                "label":             trained_label,
                "n_strikers":        base_env_cfg.n_strikers,
                "n_jammers":         base_env_cfg.n_jammers,
                "n_known_targets":   base_env_cfg.n_known_targets,
                "n_unknown_targets": base_env_cfg.n_unknown_targets,
                "n_known_radars":    base_env_cfg.n_known_radars,
                "n_unknown_radars":  base_env_cfg.n_unknown_radars,
                "_is_baseline":      True,
            }] + configs_to_eval

    # ── Evaluate each configuration ───────────────────────────────────
    results: List[Dict[str, Any]] = []

    for i, test_cfg in enumerate(configs_to_eval):
        label = test_cfg["label"]
        print(f"[{i + 1:2d}/{len(configs_to_eval)}]  {label} ...", end="", flush=True)

        # Build env config with new team composition
        eval_env_cfg = _build_eval_env_cfg(base_env_cfg, test_cfg)
        eval_ppo_cfg = _build_eval_ppo_cfg(ppo_template, device, seed)

        # finalize() propagates fofe_cfg.use_fofe → env_cfg._use_fofe
        eval_exp_cfg = ExperimentConfig(
            env=eval_env_cfg,
            ppo=eval_ppo_cfg,
            net=net_cfg,
            fofe=fofe_cfg,
        ).finalize()

        # Build environment for this team composition
        eval_env = build_env(eval_exp_cfg.env, eval_exp_cfg.ppo)

        # Instantiate policy (same architecture as checkpoint)
        eval_policy = make_combined_policy(
            eval_env,
            hidden=net_cfg.actor_hidden,
            depth=net_cfg.depth,
            fofe_cfg=fofe_cfg if fofe_cfg.use_fofe else None,
        )

        # Load weights — strict=True works because the FOFE actor architecture
        # is independent of team size (permutation-invariant encoding + masking)
        eval_policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
        eval_policy = eval_policy.to(device)

        # Run evaluation rollouts
        metrics = evaluate_current_policy(
            eval_policy,
            eval_exp_cfg.env,
            eval_exp_cfg.ppo,
            n_eval_episodes=n_eval_episodes,
        )

        result: Dict[str, Any] = {
            "label":           label,
            "n_strikers":      eval_env_cfg.n_strikers,
            "n_jammers":       eval_env_cfg.n_jammers,
            "n_targets":       eval_env_cfg.n_targets,
            "n_radars":        eval_env_cfg.n_radars,
            "completion_rate": metrics["eval_task_completion_rate"],
            "survival_rate":   metrics["eval_survival_rate"],
            "mean_duration":   metrics["eval_mean_duration"],
            "mean_reward":     metrics["eval_mean_episode_total_reward"],
            "is_baseline":     test_cfg.get("_is_baseline", False),
        }
        results.append(result)

        print(
            f"\n    completion_rate={result['completion_rate']:.3f}"
            f"\n    survival_rate={result['survival_rate']:.3f}"
            f"\n    mean_duration={result['mean_duration']:.1f}"
        )

    _print_results_table(results, n_eval_episodes)
    if plot_metrics:
        _plot_completion_and_survival(results, n_eval_episodes)
    return results


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Zero-shot generalisation test: evaluate a saved FOFE-MAPPO policy "
            "on different team compositions without any fine-tuning."
        )
    )
    p.add_argument(
        "--checkpoint", required=True, type=str,
        metavar="PATH",
        help="Path to a saved .pt checkpoint (output of run.py or run_curicilum.py).",
    )
    p.add_argument(
        "--n_eval_episodes", type=int, default=50,
        metavar="N",
        help="Episodes to average per test configuration (default: 50).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for evaluation rollouts (default: 42).",
    )
    p.add_argument(
        "--no_baseline", action="store_true",
        help="Skip the trained composition baseline row.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        metavar="DEVICE",
        help="Torch device string, e.g. 'cpu' or 'cuda:0'. Default: auto-detect.",
    )
    p.add_argument(
        "--no_plot", action="store_true",
        help="Disable completion/survival per-label plots.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device) if args.device else None
    run_generalisation(
        checkpoint_path=args.checkpoint,
        test_configs=TEST_CONFIGS,
        n_eval_episodes=args.n_eval_episodes,
        device=device,
        seed=args.seed,
        include_baseline=not args.no_baseline,
        plot_metrics=not args.no_plot,
    )


if __name__ == "__main__":
    main()
