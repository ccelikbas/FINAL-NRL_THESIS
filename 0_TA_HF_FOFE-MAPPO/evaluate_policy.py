"""
evaluate_policy.py
══════════════════
Evaluate ONE pretrained FOFE-MAPPO checkpoint across a list of test scenarios
and print a KPI table (rows = scenarios, columns = KPIs).

──────────────────────────────────────────────────────────────────────────────
HOW TO USE A PRETRAINED POLICY
──────────────────────────────────────────────────────────────────────────────
The striker AND jammer are trained jointly and saved together as a single
combined policy inside one .pt checkpoint. Both run.py and run_curriculum.py
produce such a file:

    run.py             →  runs/fofe_mappo.pt
    run_curriculum.py  →  runs/curriculum_mappo.pt

Point this script at that ONE file with --checkpoint. The script auto-detects
which producer made it (run.py checkpoints carry an `env_cfg`; curriculum
checkpoints don't) and loads the network / FOFE / reward / extension configs
needed to rebuild the exact policy that was trained.

Run it as a direct script with the project's venv python (there is no `python`
on PATH, and the package folder isn't importable as `-m ...`):

    # from the repo root:
    .\.venv\Scripts\python.exe 0_TA_HF_FOFE-MAPPO\evaluate_policy.py --checkpoint 0_TA_HF_FOFE-MAPPO\runs\2s2-4j.pt

    # or after `cd 0_TA_HF_FOFE-MAPPO`:
    ..\.venv\Scripts\python.exe evaluate_policy.py --checkpoint runs\fofe_mappo.pt --n_episodes 200 --n_repeats 5

──────────────────────────────────────────────────────────────────────────────
HOW TO DEFINE SCENARIOS
──────────────────────────────────────────────────────────────────────────────
Edit the EVAL_SCENARIOS list below. It uses the SAME `CurriculumSection`
dataclass you already use in run_curriculum.py, so you can copy-paste sections
straight across. Field rules are identical:

    None        → inherit from the checkpoint's defaults
    scalar      → fixed for the scenario
    (lo, hi)    → per-environment domain randomization (a single scenario then
                  evaluates over a MIX of configs, exactly like training eval)

`n_iters` is ignored here (it only matters for training); keep it for
copy-paste compatibility. `name` is the row label in the results table.

──────────────────────────────────────────────────────────────────────────────
HOW EVALUATION WORKS
──────────────────────────────────────────────────────────────────────────────
For every scenario we run N_REPEATS independent repeats; each repeat evaluates
the frozen policy over N_EVAL_EPISODES episodes (run in parallel) and yields one
value per KPI. The table then reports mean ± std across the repeats, so you can
see the run-to-run spread of each estimate. Both counts are configurable below
(or via --n_episodes / --n_repeats).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Match the training entry points: force Inductor to compile in-process (avoids
# a Triton 'cubin' KeyError on some Linux+CUDA setups). Harmless elsewhere.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# Windows consoles default to cp1252, which can't encode the table glyphs
# (± × ─). Switch the streams to UTF-8 so printing never crashes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── package bootstrap (so the file runs as a script OR as -m fofe_mappo.*) ──
_THIS_DIR = Path(__file__).resolve().parent
_PKG_NAME = "fofe_mappo"
if __package__ in (None, ""):
    sys.path.insert(0, str(_THIS_DIR.parent))
    if _PKG_NAME not in sys.modules:
        _pkg = types.ModuleType(_PKG_NAME)
        _pkg.__path__ = [str(_THIS_DIR)]
        _pkg.__package__ = _PKG_NAME
        _pkg.__file__ = str(_THIS_DIR / "__init__.py")
        sys.modules[_PKG_NAME] = _pkg
    __package__ = _PKG_NAME

import numpy as np
import torch

from .config import (
    EnvConfig, EnvExtensionsConfig, ExperimentConfig, FOFEConfig,
    NetworkConfig, PPOConfig,
)
from .rewards import RewardConfig
from .models import make_combined_policy
from .trainer import build_env, evaluate_current_policy
# Reuse the EXACT curriculum scenario format + its config-resolution logic so a
# scenario defined here behaves identically to a curriculum section.
from .run_curriculum import CurriculumSection, _section_to_env_cfg, _section_label


# =====================================================================
#  >>>  EDIT YOUR TEST SCENARIOS HERE  <<<
#  Same format as the CURRICULUM list in run_curriculum.py.
# =====================================================================

EVAL_SCENARIOS: List[CurriculumSection] = [
    CurriculumSection(
        name="S1 - Known",
        n_iters=200,                       # ignored during evaluation
        n_strikers=2, n_jammers=(2, 4),
        n_known_targets=3, n_unknown_targets=0,
        n_known_radars=6, n_unknown_radars=0,
        radar_kill_probability=0.05,
        scenario="S2",
    ),
    # CurriculumSection(
    #     name="2sx2j - Known",
    #     n_iters=200,                       # ignored during evaluation
    #     n_strikers=2, n_jammers=2,
    #     n_known_targets=3, n_unknown_targets=0,
    #     n_known_radars=6, n_unknown_radars=0,
    #     radar_kill_probability=0.05,
    #     scenario="S2",
    # ),
    # CurriculumSection(
    #     name="2sx3j - Known",
    #     n_iters=200,                       # ignored during evaluation
    #     n_strikers=2, n_jammers=3,
    #     n_known_targets=3, n_unknown_targets=0,
    #     n_known_radars=6, n_unknown_radars=0,
    #     radar_kill_probability=0.05,
    #     scenario="S2",
    # ),
    # CurriculumSection(
    #     name="2sx4j - Known",
    #     n_iters=200,                       # ignored during evaluation
    #     n_strikers=2, n_jammers=4,
    #     n_known_targets=3, n_unknown_targets=0,
    #     n_known_radars=6, n_unknown_radars=0,
    #     radar_kill_probability=0.05,
    #     scenario="S2",
    # ),
        CurriculumSection(
        name="S2 - Pop-UP",
        n_iters=200,                       # ignored during evaluation
        n_strikers=2, n_jammers=(2, 4),
        n_known_targets=2, n_unknown_targets=1,
        n_known_radars=4, n_unknown_radars=2,
        radar_kill_probability=0.05,
        scenario="S2",
    ),
    # CurriculumSection(
    #     name="2sx2j Pop-Up",
    #     n_iters=200,                       # ignored during evaluation
    #     n_strikers=2, n_jammers=2,
    #     n_known_targets=2, n_unknown_targets=1,
    #     n_known_radars=4, n_unknown_radars=2,
    #     radar_kill_probability=0.05,
    #     scenario="S2",
    # ),
    # CurriculumSection(
    #     name="2sx3j Pop-Up",
    #     n_iters=200,                       # ignored during evaluation
    #     n_strikers=2, n_jammers=3,
    #     n_known_targets=2, n_unknown_targets=1,
    #     n_known_radars=4, n_unknown_radars=2,
    #     radar_kill_probability=0.05,
    #     scenario="S2",
    # ),
    # CurriculumSection(
    #     name="2sx4j Pop-Up",
    #     n_iters=200,                       # ignored during evaluation
    #     n_strikers=2, n_jammers=4,
    #     n_known_targets=2, n_unknown_targets=1,
    #     n_known_radars=4, n_unknown_radars=2,
    #     radar_kill_probability=0.05,
    #     scenario="S2",
    # ),
    CurriculumSection(
        name="S3 - Team Size Sweep",
        n_iters=200,                       # ignored during evaluation
        n_strikers=(1,3), n_jammers=(2,6),
        n_known_targets=2, n_unknown_targets=1,
        n_known_radars=4, n_unknown_radars=2,
        radar_kill_probability=0.05,
        scenario="S2",
    ), 
    CurriculumSection(
        name="S3 - Environment Size Sweep",
        n_iters=200,                       # ignored during evaluation
        n_strikers=2, n_jammers=(2,4),
        n_known_targets=(1,5), n_unknown_targets=0,
        n_known_radars=(4,8), n_unknown_radars=0,
        radar_kill_probability=0.05,
        scenario="S2",
    )
]


# =====================================================================
#  >>>  EVALUATION CONFIG (the "n" you asked about)  <<<
# =====================================================================

N_EVAL_EPISODES = 100      # domain randomised episodes (enviernments) per repeat (run in parallel)
N_REPEATS       = 5        # independent repeats per scenario → mean ± std
BASE_SEED       = 42       # repeat r uses BASE_SEED + r * 1000


# =====================================================================
#  KPI columns (key in evaluate_current_policy output, header, fmt)
# =====================================================================

KPI_COLUMNS = [
    ("completion", "eval_task_completion_rate",       "Completion", "{:.3f}"),
    ("targets",    "eval_targets_destroyed_rate",     "Targets",    "{:.3f}"),
    ("survival",   "eval_survival_rate",              "Survival",   "{:.3f}"),
    ("duration",   "eval_mean_duration",              "Duration",   "{:.1f}"),
    ("reward",     "eval_mean_episode_total_reward",  "Reward",     "{:.2f}"),
]


# =====================================================================
#  IMPLEMENTATION  —  you usually do not need to edit below this line
# =====================================================================

def _ensure_checkpoint_import_aliases() -> None:
    """Expose the `fofe_mappo.*` module names that pickled configs reference.

    Checkpoints store dataclass instances (EnvConfig, RewardConfig, ...) pickled
    under the `fofe_mappo` package alias the training scripts run under. The
    top-level imports above already register fofe_mappo.config / .rewards, but
    we setdefault them explicitly so torch.load(weights_only=False) is robust.
    """
    cfg_mod = sys.modules.get(EnvConfig.__module__)
    rew_mod = sys.modules.get(RewardConfig.__module__)
    if cfg_mod is not None:
        sys.modules.setdefault(f"{_PKG_NAME}.config", cfg_mod)
    if rew_mod is not None:
        sys.modules.setdefault(f"{_PKG_NAME}.rewards", rew_mod)


def _strip_compile_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Drop torch.compile's `_orig_mod.` key prefix so strict loads work.

    On Linux+GPU the trainer compiles the actor/critic, inserting `_orig_mod.`
    into every parameter key. Curriculum checkpoints already strip this, but
    run.py checkpoints may not — stripping here keeps both portable."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


class _LoadedCheckpoint:
    """Everything needed to rebuild the trained policy, from either producer."""

    def __init__(self, path: Path, device: torch.device):
        _ensure_checkpoint_import_aliases()
        ckpt = torch.load(path, map_location=device, weights_only=False)

        if "policy_state_dict" not in ckpt:
            raise KeyError(
                f"{path.name} has no 'policy_state_dict' — is this a "
                f"checkpoint produced by run.py / run_curriculum.py?"
            )

        self.policy_state_dict = _strip_compile_prefix(ckpt["policy_state_dict"])
        self.net_cfg: NetworkConfig = ckpt.get("net_cfg", NetworkConfig())
        self.fofe_cfg: FOFEConfig = ckpt.get("fofe_cfg", FOFEConfig(use_fofe=True))
        self.ext_cfg: EnvExtensionsConfig = ckpt.get("ext_cfg", EnvExtensionsConfig())

        # Producer detection: run.py stores a full env_cfg; curriculum does not.
        env_cfg: Optional[EnvConfig] = ckpt.get("env_cfg")
        if env_cfg is not None:
            self.producer = "run.py"
            self.base_env_cfg = env_cfg
            self.reward_cfg = copy.deepcopy(env_cfg.reward_config)
        else:
            self.producer = "run_curriculum.py"
            # Curriculum trained on config.py's leading EnvConfig() defaults for
            # every field a section didn't override — mirror that here.
            self.base_env_cfg = EnvConfig()
            self.reward_cfg = ckpt.get("reward_cfg", RewardConfig())

        self.hf_radar_cfg = (
            self.ext_cfg.hf_radar if getattr(self.ext_cfg, "use_hf_radar", False) else None
        )


def _build_policy_for_scenario(ckpt: _LoadedCheckpoint, env_cfg: EnvConfig,
                               device: torch.device):
    """Rebuild a policy sized to env_cfg's role split and strict-load weights.

    FOFE parameter shapes are count-invariant, so the load is exact; only the
    striker/jammer role split (which the wrapper uses to slice actions) depends
    on env_cfg, hence the per-scenario rebuild."""
    probe_ppo = PPOConfig(num_envs=1, device=device)
    probe_env = build_env(env_cfg, probe_ppo, hf_radar_cfg=ckpt.hf_radar_cfg)
    policy = make_combined_policy(
        probe_env,
        hidden=ckpt.net_cfg.actor_hidden,
        depth=ckpt.net_cfg.depth,
        fofe_cfg=ckpt.fofe_cfg if ckpt.fofe_cfg.use_fofe else None,
    )
    policy.load_state_dict(ckpt.policy_state_dict, strict=True)
    return policy.to(device)


def _mean_std(values: List[float]) -> tuple[float, float]:
    """Mean and (sample) std over finite values; (nan, nan) if none finite."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def evaluate_scenarios(
    ckpt: _LoadedCheckpoint,
    scenarios: List[CurriculumSection],
    n_episodes: int,
    n_repeats: int,
    base_seed: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Evaluate the frozen policy on every scenario and return result rows."""
    rows: List[Dict[str, Any]] = []

    for i, section in enumerate(scenarios):
        env_cfg = _section_to_env_cfg(
            section, ckpt.base_env_cfg, ckpt.reward_cfg, ckpt.fofe_cfg
        )
        dr_tag = "DR" if env_cfg.dr is not None else "fixed"
        print(f"[{i + 1:2d}/{len(scenarios)}] {section.name:20s} ({dr_tag})  "
              f"{_section_label(section, env_cfg)}", flush=True)

        policy = _build_policy_for_scenario(ckpt, env_cfg, device)

        # repeat -> {kpi_key: value}
        per_repeat: Dict[str, List[float]] = {key: [] for key, *_ in KPI_COLUMNS}
        for r in range(n_repeats):
            eval_ppo = PPOConfig(num_envs=1, device=device, seed=base_seed + r * 1000)
            metrics = evaluate_current_policy(
                policy, env_cfg, eval_ppo,
                n_eval_episodes=n_episodes,
                hf_radar_cfg=ckpt.hf_radar_cfg,
            )
            for key, metric_key, *_ in KPI_COLUMNS:
                per_repeat[key].append(float(metrics.get(metric_key, float("nan"))))
            print(f"        repeat {r + 1}/{n_repeats}  "
                  + "  ".join(
                      f"{key}={per_repeat[key][-1]:.3f}" for key, *_ in KPI_COLUMNS
                  ), flush=True)

        row: Dict[str, Any] = {
            "name": section.name,
            "scenario": env_cfg.scenario,
            "S": env_cfg.n_strikers, "J": env_cfg.n_jammers,
            "T": env_cfg.n_targets, "R": env_cfg.n_radars,
            "dr": env_cfg.dr is not None,
        }
        for key, *_ in KPI_COLUMNS:
            mean, std = _mean_std(per_repeat[key])
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        rows.append(row)

        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


def _print_table(rows: List[Dict[str, Any]], n_episodes: int, n_repeats: int,
                 producer: str) -> None:
    """Print the KPI table: one row per scenario, mean ± std per KPI."""
    # Context columns + one cell per KPI ("0.923 ± 0.031").
    ctx_headers = ["Scenario", "Scn", "S", "J", "T", "R"]

    def _cell(row, key, fmt):
        mean, std = row[f"{key}_mean"], row[f"{key}_std"]
        if not np.isfinite(mean):
            return "n/a"
        return f"{fmt.format(mean)} ± {fmt.format(std)}"

    kpi_headers = [hdr for _, _, hdr, _ in KPI_COLUMNS]
    table_rows = []
    for row in rows:
        name = row["name"] + ("*" if row["dr"] else "")
        ctx = [name, row["scenario"], str(row["S"]), str(row["J"]),
               str(row["T"]), str(row["R"])]
        kpis = [_cell(row, key, fmt) for key, _, _, fmt in KPI_COLUMNS]
        table_rows.append(ctx + kpis)

    headers = ctx_headers + kpi_headers
    widths = [len(h) for h in headers]
    for tr in table_rows:
        for c, cell in enumerate(tr):
            widths[c] = max(widths[c], len(cell))

    def _fmt_row(cells):
        return "  ".join(
            cell.ljust(widths[c]) if c == 0 else cell.rjust(widths[c])
            for c, cell in enumerate(cells)
        )

    sep = "  ".join("-" * w for w in widths)
    title = (f"  Policy Evaluation  (producer: {producer} | "
             f"{n_episodes} episodes × {n_repeats} repeats per scenario)")
    bar = "=" * max(len(sep), len(title))

    print(f"\n{bar}")
    print(title)
    print(bar)
    print(_fmt_row(headers))
    print(sep)
    for tr in table_rows:
        print(_fmt_row(tr))
    print(sep)
    if any(row["dr"] for row in rows):
        print("  * = domain-randomized scenario (counts shown are the maxima)")
    print()


def _write_csv(rows: List[Dict[str, Any]], out_dir: Path, n_episodes: int,
               n_repeats: int) -> Path:
    """Write a flat CSV (mean + std columns per KPI) and return its path."""
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"policy_eval_{stamp}.csv"

    fieldnames = ["name", "scenario", "S", "J", "T", "R", "dr",
                  "n_episodes", "n_repeats"]
    for key, *_ in KPI_COLUMNS:
        fieldnames += [f"{key}_mean", f"{key}_std"]

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: row[k] for k in ("name", "scenario", "S", "J", "T", "R", "dr")}
            out["n_episodes"] = n_episodes
            out["n_repeats"] = n_repeats
            for key, *_ in KPI_COLUMNS:
                out[f"{key}_mean"] = row[f"{key}_mean"]
                out[f"{key}_std"] = row[f"{key}_std"]
            writer.writerow(out)
    return out_path


# =====================================================================
#  CLI
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate one pretrained FOFE-MAPPO checkpoint across the "
        "EVAL_SCENARIOS defined at the top of this file.",
    )
    p.add_argument("--checkpoint", required=True, type=str, metavar="PATH",
                   help="Path to a .pt checkpoint from run.py or run_curriculum.py.")
    p.add_argument("--n_episodes", type=int, default=N_EVAL_EPISODES,
                   help=f"Episodes per repeat (default: {N_EVAL_EPISODES}).")
    p.add_argument("--n_repeats", type=int, default=N_REPEATS,
                   help=f"Repeats per scenario for mean±std (default: {N_REPEATS}).")
    p.add_argument("--seed", type=int, default=BASE_SEED,
                   help=f"Base RNG seed (default: {BASE_SEED}).")
    p.add_argument("--device", type=str, default=None, metavar="DEVICE",
                   help="Torch device, e.g. 'cpu' or 'cuda:0'. Default: auto.")
    p.add_argument("--no_csv", action="store_true",
                   help="Do not write a CSV (console table only).")
    p.add_argument("--out_dir", type=str, default=None,
                   help="CSV output dir (default: <script_dir>/eval_results).")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not EVAL_SCENARIOS:
        raise RuntimeError("EVAL_SCENARIOS is empty — define at least one scenario.")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("─" * 70)
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Device     : {device}")
    ckpt = _LoadedCheckpoint(ckpt_path, device)
    print(f"  Producer   : {ckpt.producer}")
    print(f"  FOFE       : {'enabled' if ckpt.fofe_cfg.use_fofe else 'disabled (legacy)'}")
    print(f"  HF radar   : {'enabled' if ckpt.hf_radar_cfg is not None else 'disabled'}")
    print(f"  Scenarios  : {len(EVAL_SCENARIOS)}")
    print(f"  Eval       : {args.n_episodes} episodes × {args.n_repeats} repeats")
    print("─" * 70)

    rows = evaluate_scenarios(
        ckpt, EVAL_SCENARIOS,
        n_episodes=args.n_episodes,
        n_repeats=args.n_repeats,
        base_seed=args.seed,
        device=device,
    )

    _print_table(rows, args.n_episodes, args.n_repeats, ckpt.producer)

    if not args.no_csv:
        out_dir = Path(args.out_dir) if args.out_dir else (_THIS_DIR / "eval_results")
        csv_path = _write_csv(rows, out_dir, args.n_episodes, args.n_repeats)
        print(f"Saved CSV to: {csv_path}\n")


if __name__ == "__main__":
    main()
