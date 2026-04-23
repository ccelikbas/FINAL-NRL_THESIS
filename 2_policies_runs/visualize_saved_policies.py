from __future__ import annotations

import argparse
import contextlib
import copy
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import torch


# ============================================================================
# User configuration (edit these defaults directly in this file if preferred)
# ============================================================================

# Environment choice: "HF" for high-fidelity radar env, "LF" for standard env.
ENV_MODE = "HF"

# Team composition used for visualization rollouts.
N_STRIKERS = 2
N_JAMMERS = 2
N_TARGETS = 2
N_RADARS = 2

# Number of rollouts to animate for each loaded policy.
N_RUNS_TO_VISUALIZE = 1

# GIF export controls.
SAVE_GIF_ONE_RUN = True
GIF_RUN_INDEX = 1
GIF_OUTPUT_DIR = Path("2_policies_runs/saved_gifs")
GIF_FPS = 12
SHOW_WINDOW_FOR_GIF_RUN = False

# If you do not pass --policies, these are used first.
# Keep empty to rely on --all_saved or USE_ALL_POLICIES_IF_NONE_PROVIDED.
POLICY_FILES: List[str] = [
    "HF_cur.pt"
#     "1x2_FOFE_14_04.pt",
#     "HF_1x1_FOFE.pt",
#     "HF_1x2_FOFE_all_jamm.pt",
]

POLICY_DIR = Path("2_policies_runs/saved_runs")
USE_ALL_POLICIES_IF_NONE_PROVIDED = True

BASE_SEED = 999
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_USE_FOFE_IF_UNKNOWN = True
DEFAULT_ACTOR_HIDDEN = 256
DEFAULT_DEPTH = 3


# ============================================================================
# Package bootstrapping for imports and checkpoint deserialization
# ============================================================================

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_HF_PACKAGE_DIR = _REPO_ROOT / "1_HF_FOFE-MAPPO"
_CHECKPOINT_PKG_ALIAS = "fofe_mappo"


def _ensure_checkpoint_import_aliases() -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    if _CHECKPOINT_PKG_ALIAS in sys.modules:
        return

    pkg = types.ModuleType(_CHECKPOINT_PKG_ALIAS)
    pkg.__path__ = [str(_HF_PACKAGE_DIR)]
    pkg.__package__ = _CHECKPOINT_PKG_ALIAS
    pkg.__file__ = str(_HF_PACKAGE_DIR / "__init__.py")
    sys.modules[_CHECKPOINT_PKG_ALIAS] = pkg


_ensure_checkpoint_import_aliases()

from fofe_mappo.config import (  # noqa: E402
    EnvConfig,
    ExperimentConfig,
    FOFEConfig,
    HFRadarConfig,
    NetworkConfig,
    PPOConfig,
)
from fofe_mappo.models import make_combined_policy  # noqa: E402
from fofe_mappo.rewards import RewardConfig  # noqa: E402
from fofe_mappo.trainer import build_env  # noqa: E402
from fofe_mappo.visualization import TestRunner, animate_rollout  # noqa: E402
from fofe_mappo.HF_visualization import HFTestRunner, hf_animate_rollout  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one or multiple saved policy checkpoints with configurable "
            "entity counts and HF/LF environment mode."
        )
    )
    parser.add_argument(
        "--policies",
        nargs="*",
        default=None,
        help=(
            "Policy filenames or paths. Example: --policies 1x1_FOFE_15_04.pt "
            "HF_1x2_FOFE_all_jamm.pt"
        ),
    )
    parser.add_argument(
        "--all_saved",
        action="store_true",
        help="Use every .pt file in --policy_dir.",
    )
    parser.add_argument("--policy_dir", type=Path, default=POLICY_DIR)

    parser.add_argument("--env_mode", choices=["HF", "LF"], default=ENV_MODE)
    parser.add_argument("--n_strikers", type=int, default=N_STRIKERS)
    parser.add_argument("--n_jammers", type=int, default=N_JAMMERS)
    parser.add_argument("--n_targets", type=int, default=N_TARGETS)
    parser.add_argument("--n_radars", type=int, default=N_RADARS)

    parser.add_argument("--n_runs", type=int, default=N_RUNS_TO_VISUALIZE)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    parser.add_argument(
        "--save_gif_one_run",
        action=argparse.BooleanOptionalAction,
        default=SAVE_GIF_ONE_RUN,
        help="Save a GIF for one selected rollout run.",
    )
    parser.add_argument(
        "--gif_run_index",
        type=int,
        default=GIF_RUN_INDEX,
        help="1-based rollout index to save as GIF.",
    )
    parser.add_argument(
        "--gif_output_dir",
        type=Path,
        default=GIF_OUTPUT_DIR,
        help="Directory where GIF files are saved.",
    )
    parser.add_argument(
        "--gif_fps",
        type=int,
        default=GIF_FPS,
        help="Frames per second for saved GIF.",
    )
    parser.add_argument(
        "--show_window_for_gif_run",
        action=argparse.BooleanOptionalAction,
        default=SHOW_WINDOW_FOR_GIF_RUN,
        help="Also open animation window for the run saved as GIF.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load and validate policies without opening animation windows.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if one selected policy fails to load.",
    )
    return parser


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def _collect_policy_paths(args: argparse.Namespace) -> List[Path]:
    policy_dir = _resolve_path(args.policy_dir)
    if not policy_dir.exists():
        raise FileNotFoundError(f"Policy directory not found: {policy_dir}")

    requested: List[str]
    if args.policies:
        requested = list(args.policies)
    elif args.all_saved:
        requested = [p.name for p in sorted(policy_dir.glob("*.pt"))]
    elif POLICY_FILES:
        requested = list(POLICY_FILES)
    elif USE_ALL_POLICIES_IF_NONE_PROVIDED:
        requested = [p.name for p in sorted(policy_dir.glob("*.pt"))]
    else:
        raise ValueError("No policies selected. Use --policies or --all_saved.")

    if not requested:
        raise ValueError("No policy files resolved.")

    resolved: List[Path] = []
    seen = set()

    for entry in requested:
        raw = Path(entry)
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(policy_dir / raw)
            candidates.append(_resolve_path(raw))

        found = None
        for candidate in candidates:
            if candidate.exists():
                found = candidate.resolve()
                break

        if found is None:
            raise FileNotFoundError(f"Could not find policy: {entry}")

        if found not in seen:
            seen.add(found)
            resolved.append(found)

    return resolved


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    _ensure_checkpoint_import_aliases()
    obj = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint is not a dict: {path}")
    return obj


def _extract_policy_state_dict(checkpoint: Dict[str, Any], path: Path) -> Dict[str, Any]:
    state_dict = checkpoint.get("policy_state_dict")
    if isinstance(state_dict, dict):
        return state_dict

    # Backward-compatible path if a file stores just a raw state_dict mapping.
    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint

    raise KeyError(f"No policy_state_dict found in {path.name}")


def _infer_use_fofe(
    checkpoint: Dict[str, Any],
    policy_state_dict: Dict[str, Any],
    policy_path: Path,
) -> bool:
    ckpt_fofe = checkpoint.get("fofe_cfg")
    if isinstance(ckpt_fofe, FOFEConfig):
        return bool(ckpt_fofe.use_fofe)

    keys = list(policy_state_dict.keys())
    head = " ".join(keys[:25])
    if "self_mlp" in head or "fofe_" in head:
        return True
    if ".net.params" in head:
        return False

    name = policy_path.name.lower()
    if "mappo" in name:
        return False
    if "fofe" in name:
        return True

    return DEFAULT_USE_FOFE_IF_UNKNOWN


def _build_env_cfg(args: argparse.Namespace, checkpoint: Dict[str, Any]) -> EnvConfig:
    base_env = checkpoint.get("env_cfg")
    if isinstance(base_env, EnvConfig):
        env_cfg = copy.deepcopy(base_env)
    else:
        env_cfg = EnvConfig()

    reward_cfg = checkpoint.get("reward_cfg")
    if isinstance(reward_cfg, RewardConfig):
        env_cfg.reward_config = copy.deepcopy(reward_cfg)

    env_cfg.n_strikers = int(args.n_strikers)
    env_cfg.n_jammers = int(args.n_jammers)
    env_cfg.n_known_targets = int(args.n_targets)
    env_cfg.n_unknown_targets = 0
    env_cfg.n_targets = int(args.n_targets)
    env_cfg.n_known_radars = int(args.n_radars)
    env_cfg.n_unknown_radars = 0
    env_cfg.n_radars = int(args.n_radars)
    return env_cfg


def _build_ppo_cfg(args: argparse.Namespace, checkpoint: Dict[str, Any], device: torch.device) -> PPOConfig:
    raw_ppo = checkpoint.get("ppo_cfg")
    if raw_ppo is None:
        raw_ppo = checkpoint.get("ppo_template")

    if isinstance(raw_ppo, PPOConfig):
        ppo_cfg = copy.deepcopy(raw_ppo)
    else:
        ppo_cfg = PPOConfig()

    ppo_cfg.num_envs = 1
    ppo_cfg.seed = int(args.seed)
    ppo_cfg.device = device
    return ppo_cfg


def _build_net_cfg(checkpoint: Dict[str, Any]) -> NetworkConfig:
    raw_net = checkpoint.get("net_cfg")
    if isinstance(raw_net, NetworkConfig):
        return copy.deepcopy(raw_net)

    return NetworkConfig(
        actor_hidden=DEFAULT_ACTOR_HIDDEN,
        critic_hidden=DEFAULT_ACTOR_HIDDEN,
        depth=DEFAULT_DEPTH,
    )


def _build_fofe_cfg(checkpoint: Dict[str, Any], use_fofe: bool) -> FOFEConfig:
    raw_fofe = checkpoint.get("fofe_cfg")
    if isinstance(raw_fofe, FOFEConfig):
        fofe_cfg = copy.deepcopy(raw_fofe)
        fofe_cfg.use_fofe = bool(use_fofe)
        return fofe_cfg
    return FOFEConfig(use_fofe=bool(use_fofe))


def _build_hf_cfg(checkpoint: Dict[str, Any]) -> HFRadarConfig:
    ext_cfg = checkpoint.get("ext_cfg")
    if ext_cfg is not None and hasattr(ext_cfg, "hf_radar"):
        candidate = getattr(ext_cfg, "hf_radar")
        if isinstance(candidate, HFRadarConfig):
            return copy.deepcopy(candidate)
    return HFRadarConfig()


def _validate_counts(args: argparse.Namespace) -> None:
    for name in ["n_strikers", "n_jammers", "n_targets", "n_radars"]:
        value = int(getattr(args, name))
        if value <= 0:
            raise ValueError(f"{name} must be >= 1, got {value}")
    if int(args.n_runs) < 0:
        raise ValueError(f"n_runs must be >= 0, got {args.n_runs}")
    if int(args.gif_fps) <= 0:
        raise ValueError(f"gif_fps must be >= 1, got {args.gif_fps}")
    if int(args.gif_run_index) <= 0:
        raise ValueError(f"gif_run_index must be >= 1, got {args.gif_run_index}")
    if bool(args.save_gif_one_run):
        if int(args.n_runs) <= 0:
            raise ValueError("save_gif_one_run=True requires n_runs >= 1")
        if int(args.gif_run_index) > int(args.n_runs):
            raise ValueError(
                f"gif_run_index ({args.gif_run_index}) must be <= n_runs ({args.n_runs})"
            )


def _build_gif_name(path: Path, args: argparse.Namespace, run_number: int) -> str:
    mode = str(args.env_mode).upper()
    s = int(args.n_strikers)
    j = int(args.n_jammers)
    t = int(args.n_targets)
    r = int(args.n_radars)
    return f"{mode}_S{s}_J{j}_T{t}_R{r}_{path.stem}_run{run_number}.gif"


def _render_animation(
    env_mode: str,
    frames: List[Dict[str, torch.Tensor]],
    env_obj: Any,
    suppress_show: bool,
):
    if env_mode == "HF":
        animate_fn = lambda: hf_animate_rollout(frames, env_obj)
    else:
        animate_fn = lambda: animate_rollout(frames, env_obj)

    if not suppress_show:
        return animate_fn()

    with contextlib.ExitStack() as stack:
        original_show = plt.show

        def _no_show(*_args, **_kwargs):
            return None

        plt.show = _no_show
        stack.callback(lambda: setattr(plt, "show", original_show))
        return animate_fn()


def _keep_only_rollout_axis(ani: Any) -> None:
    """Remove non-rollout axes (reward panel) from animation figure before GIF save."""
    fig = getattr(ani, "_fig", None)
    if fig is None:
        return

    axes = list(fig.axes)
    if len(axes) <= 1:
        return

    for ax in axes[1:]:
        with contextlib.suppress(Exception):
            fig.delaxes(ax)

    # Fallback removal path if any extra axes remain.
    for ax in list(fig.axes)[1:]:
        with contextlib.suppress(Exception):
            ax.remove()

    if not fig.axes:
        return

    primary_ax = fig.axes[0]
    primary_ax.set_position([0.06, 0.07, 0.90, 0.90])
    fig.set_size_inches(10.5, 9.0, forward=True)


def _save_animation_as_gif(ani: Any, gif_path: Path, fps: int) -> None:
    _keep_only_rollout_axis(ani)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(gif_path), writer="pillow", fps=int(fps))
    fig = getattr(ani, "_fig", None)
    if fig is not None:
        plt.close(fig)
    print(f"  Saved GIF: {gif_path}")


def _visualize_single_policy(path: Path, args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint = _load_checkpoint(path, device)
    policy_state = _extract_policy_state_dict(checkpoint, path)

    use_fofe = _infer_use_fofe(checkpoint, policy_state, path)
    env_cfg = _build_env_cfg(args, checkpoint)
    ppo_cfg = _build_ppo_cfg(args, checkpoint, device)
    net_cfg = _build_net_cfg(checkpoint)
    fofe_cfg = _build_fofe_cfg(checkpoint, use_fofe)

    exp_cfg = ExperimentConfig(
        env=env_cfg,
        ppo=ppo_cfg,
        net=net_cfg,
        fofe=fofe_cfg,
    ).finalize()

    env_mode = str(args.env_mode).upper()
    hf_cfg: Optional[HFRadarConfig] = _build_hf_cfg(checkpoint) if env_mode == "HF" else None

    env = build_env(exp_cfg.env, exp_cfg.ppo, hf_radar_cfg=hf_cfg)
    policy = make_combined_policy(
        env,
        hidden=exp_cfg.net.actor_hidden,
        depth=exp_cfg.net.depth,
        fofe_cfg=exp_cfg.fofe if exp_cfg.fofe.use_fofe else None,
    )

    try:
        policy.load_state_dict(policy_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Policy weights failed to load with strict=True. "
            "This usually means architecture mismatch (FOFE/MAPPO or net dims)."
        ) from exc

    print("=" * 72)
    print(f"Policy: {path.name}")
    print(
        f"Mode={env_mode} | FOFE={'on' if exp_cfg.fofe.use_fofe else 'off'} | "
        f"S/J/T/R={exp_cfg.env.n_strikers}/{exp_cfg.env.n_jammers}/{exp_cfg.env.n_targets}/{exp_cfg.env.n_radars}"
    )
    print(f"Rollouts={args.n_runs} | Device={device}")

    if args.dry_run or args.n_runs == 0:
        print("Dry-run/zero-runs mode: checkpoint loaded and policy validated.")
        return

    for run_idx in range(int(args.n_runs)):
        rollout_seed = int(args.seed) + run_idx
        print(f"  Rollout {run_idx + 1}/{args.n_runs} (seed={rollout_seed})")

        if env_mode == "HF":
            tester = HFTestRunner(
                policy,
                env_cfg=exp_cfg.env,
                hf_cfg=hf_cfg if hf_cfg is not None else HFRadarConfig(),
                device=device,
                seed=rollout_seed,
            )
            frames = tester.rollout()
            env_obj = tester.env
        else:
            tester = TestRunner(
                policy,
                env_cfg=exp_cfg.env,
                device=device,
                seed=rollout_seed,
            )
            frames = tester.rollout()
            env_obj = tester.env

        run_number = run_idx + 1
        should_save_gif = bool(args.save_gif_one_run) and (run_number == int(args.gif_run_index))
        if should_save_gif:
            gif_dir = _resolve_path(args.gif_output_dir)
            gif_name = _build_gif_name(path, args, run_number)
            gif_path = gif_dir / gif_name

            # Save path is rendered without an interactive popup for reliability.
            ani = _render_animation(env_mode, frames, env_obj, suppress_show=True)
            _save_animation_as_gif(ani, gif_path, int(args.gif_fps))

            if bool(args.show_window_for_gif_run):
                _render_animation(env_mode, frames, env_obj, suppress_show=False)
        else:
            _render_animation(env_mode, frames, env_obj, suppress_show=False)


def main() -> None:
    args = build_parser().parse_args()
    _validate_counts(args)

    policy_paths = _collect_policy_paths(args)
    print(f"Resolved {len(policy_paths)} policy file(s).")

    failed: List[str] = []
    for policy_path in policy_paths:
        try:
            _visualize_single_policy(policy_path, args)
        except Exception as exc:
            failed.append(policy_path.name)
            print(f"ERROR while processing {policy_path.name}: {type(exc).__name__}: {exc}")
            if args.stop_on_error:
                raise

    if failed:
        print("=" * 72)
        print(f"Completed with {len(failed)} failed policy file(s): {failed}")
    else:
        print("=" * 72)
        print("Completed successfully for all selected policy files.")


if __name__ == "__main__":
    main()
