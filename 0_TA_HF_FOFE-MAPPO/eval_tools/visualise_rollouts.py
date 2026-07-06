"""visualise_rollouts.py — render GIF rollouts of ONE trained policy in its FINAL
training configuration, using the exact same visualisation as run_curriculum.py
produces at the end of a run (TestRunner / HFTestRunner + animate_rollout).

WHAT IT DOES
────────────
Given a policy checkpoint and the final-stage environment it was trained on
(scenario, entity counts, kill rate, team-size / kill DOMAIN RANDOMISATION), it
rolls the policy out N times and writes each rollout as an animated GIF.

DOMAIN RANDOMISATION
────────────────────
The visualisation runners build a single concrete env and do NOT apply DR
themselves (that is why run_curriculum drops DR before animating). To still show
the policy across its trained DR distribution, this script SAMPLES a concrete
composition (team sizes / kill / counts) from the FINAL_CONFIG ranges for every
rollout — so each GIF is a different, representative draw from the final training
stage. Set `--no_dr` (or SAMPLE_DR = False) to instead render the fixed
maximum-count configuration once (the run_curriculum style).

OUTPUT
──────
GIFs are written to eval_results/rollout_visualisations/ with stable names
(rollout_01.gif, rollout_02.gif, …). The folder is cleared at the start of every
run, so repeated runs overwrite each other — no date-stamped names.

Run (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\visualise_rollouts.py
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\visualise_rollouts.py --n_rollouts 20
"""
from __future__ import annotations
import argparse, sys, types
from dataclasses import replace
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

from .run_curriculum import CurriculumSection, _section_to_env_cfg, _section_label
from .evaluate_policy import _LoadedCheckpoint, _build_policy_for_scenario, _resolve_policy_path
from .visualization import TestRunner, animate_rollout
from .HF_visualization import HFTestRunner, hf_animate_rollout

# ===================================================================
# CONFIG — edit these (CLI flags override the marked ones)
# ===================================================================

# Policy to visualise. Bare name → runs/, "runs/…/x.pt" → project dir, abs → as-is.
POLICY_PATH = "runs/FINALV1/complete_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt"

# Communication used at evaluation — set it to match how the policy was TRAINED
# (True for the complete/comms model, False for a no-comms baseline).
# None → use the checkpoint's own trained comms setting.
COMMUNICATE: bool | None = True

# FINAL training configuration. EDIT to match the LAST CurriculumSection the
# policy was trained on. Tuples (lo, hi) are DR ranges; scalars are fixed. Below
# mirrors the usual final stage: 2 strikers, jammers domain-randomised in [2, 4],
# fixed targets/radars, fixed kill probability, S2 spawn.
FINAL_CONFIG = CurriculumSection(
    name="final",
    n_iters=1,                       # unused here
    n_strikers=2, n_jammers=(2, 4),  # (lo, hi) → DR
    n_known_targets=2, n_unknown_targets=0,
    n_known_radars=6, n_unknown_radars=0,
    radar_kill_probability=0.25,
    scenario="S2",
)

# Number of GIFs to render (one rollout each).            [CLI: --n_rollouts]
N_ROLLOUTS = 10
# Base seed; rollout i uses BASE_SEED + i for both the DR draw and the layout. [CLI: --base-seed]
BASE_SEED = 999
# Sample a fresh composition from the DR ranges per rollout (True), or render the
# fixed maximum-count config (False, run_curriculum style).   [CLI: --no_dr]
SAMPLE_DR = True

# Output folder (relative paths resolved against the project dir 0_TA_...).  [CLI: --out]
OUT_DIR = "eval_results/rollout_visualisations"

# ===================================================================


def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_THIS_DIR.parent / p)


def _sample_final_section(section: CurriculumSection, rng: np.random.Generator) -> CurriculumSection:
    """Draw a concrete composition from the section's DR ranges.

    Tuple fields (lo, hi) are sampled uniformly (ints inclusive, floats
    continuous); scalar / None fields pass through unchanged. The result has no
    ranges left, so _section_to_env_cfg yields a concrete (dr=None) env config.
    """
    def s_int(v):
        return int(rng.integers(int(v[0]), int(v[1]) + 1)) if isinstance(v, (tuple, list)) else v

    def s_float(v):
        return float(rng.uniform(float(v[0]), float(v[1]))) if isinstance(v, (tuple, list)) else v

    return replace(
        section,
        n_strikers=s_int(section.n_strikers),
        n_jammers=s_int(section.n_jammers),
        n_known_targets=s_int(section.n_known_targets),
        n_unknown_targets=s_int(section.n_unknown_targets),
        n_known_radars=s_int(section.n_known_radars),
        n_unknown_radars=s_int(section.n_unknown_radars),
        max_steps=s_int(section.max_steps),
        radar_kill_probability=s_float(section.radar_kill_probability),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=POLICY_PATH,
                    help="policy .pt to visualise (default: POLICY_PATH)")
    ap.add_argument("--n_rollouts", type=int, default=N_ROLLOUTS,
                    help="number of GIFs to render")
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--no_dr", action="store_true",
                    help="render the fixed max-count config instead of sampling DR")
    args = ap.parse_args()

    sample_dr = SAMPLE_DR and not args.no_dr
    ckpt_path = _resolve_policy_path(args.checkpoint, None)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {ckpt_path.name} on {device} ...")
    ckpt = _LoadedCheckpoint(ckpt_path, device)
    hf_cfg = ckpt.hf_radar_cfg
    print(f"HF radar model: {'ENABLED' if hf_cfg is not None else 'disabled'}  |  "
          f"DR sampling: {'ON' if sample_dr else 'off (fixed max-count config)'}")

    # Fresh output folder — stable names, so re-runs overwrite the previous set.
    out_dir = _resolve_out(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("rollout_*.gif"):
        old.unlink()

    for i in range(args.n_rollouts):
        seed = args.base_seed + i
        rng = np.random.default_rng(seed)
        section = _sample_final_section(FINAL_CONFIG, rng) if sample_dr else FINAL_CONFIG

        env_cfg = _section_to_env_cfg(section, ckpt.base_env_cfg, ckpt.reward_cfg, ckpt.fofe_cfg)
        if COMMUNICATE is not None:
            env_cfg.communicate = bool(COMMUNICATE)
        env_cfg.dr = None  # runners visualise a single concrete env (DR not applied here)

        policy = _build_policy_for_scenario(ckpt, env_cfg, device)
        if hf_cfg is not None:
            runner = HFTestRunner(policy, env_cfg=env_cfg, hf_cfg=hf_cfg, device=device, seed=seed)
            frames = runner.rollout()
            gif_path = out_dir / f"rollout_{i + 1:02d}.gif"
            hf_animate_rollout(frames, runner.env, save_path=str(gif_path))
        else:
            runner = TestRunner(policy, env_cfg=env_cfg, device=device, seed=seed)
            frames = runner.rollout()
            gif_path = out_dir / f"rollout_{i + 1:02d}.gif"
            animate_rollout(frames, runner.env, save_path=str(gif_path))

        print(f"  [{i + 1:02d}/{args.n_rollouts}] {_section_label(section, env_cfg)}  "
              f"seed={seed}  ({len(frames)} frames) → {gif_path.name}")

        del policy, runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nsaved {args.n_rollouts} rollout GIFs → {out_dir}")


if __name__ == "__main__":
    main()
