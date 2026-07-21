"""compare_policies.py — compare TWO policies on the SAME scenario & seed.

Both policies are rolled out on an identical scenario (same composition, same
spawn seed → same target/radar layout) and rendered as two "tactical picture"
subplots placed SIDE BY SIDE:

  * agent trajectories (strikers = solid, jammers = dashed), start = hollow o,
    end = filled marker,
  * targets (survived vs destroyed) and radars,
  * each radar's un-jammed range (dashed) and its jammer-notched effective
    coverage at the end of the episode.

Because the seed is shared, the layout is pixel-for-pixel comparable: any
difference between the left and right panels is due to the policy alone.

Read-only: rolls out two checkpoints, writes a PNG. Nothing in the environment,
trainer, or config is modified.

Configure the run by editing the CONFIG block at the top of the file
(POLICY_A/B_PATH, labels, composition, SEED). CLI flags override them.

Example (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\compare_policies.py
"""
from __future__ import annotations
import argparse, copy, sys, types
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

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .HF_visualization import HFTestRunner
from .evaluate_policy import _LoadedCheckpoint, _build_policy_for_scenario
from .nlr_style import NLR_PRIMARY
# Reuse the tactical-picture rendering + legend from the composition dashboard
# so both tools share identical marker/colour conventions.
from .compare_compositions import draw_run, _legend_handles, _resolve

# ===================================================================
# CONFIG — edit these to control the comparison
# ===================================================================

# The two policies to compare. Relative paths are resolved against this file's
# folder so it works regardless of the current working directory.
POLICY_A_PATH = "runs/FINALV1/complete_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt"
POLICY_B_PATH = "runs/FINALV1/complete_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt"

# Column titles for each policy (shown above each panel).
POLICY_A_LABEL = "Policy A"
POLICY_B_LABEL = "Policy B"

# The single scenario both policies face: team composition + spawn seed.
# The seed fixes the target/radar layout, so both panels share the same map.
N_STRIKERS = 2
N_JAMMERS = 4
SEED = 1002

# Output PNG (relative paths resolved against this file's folder).
OUT_PATH = "eval_results/policy_comparison.png"

# Output resolution (dots per inch). 600 = print-quality for the paper.
DPI = 600

# ===================================================================
# ENVIRONMENT CONFIG — the scenario the policies are evaluated in.
# These override each checkpoint's stored env config so the comparison runs
# in exactly the setup you want (should match how the policies were trained).
# ===================================================================

# Spawn scenario: "S1" or "S2".
SCENARIO = "S2"

# Target / radar counts. "known" = revealed to the agents at spawn,
# "unknown" = hidden until sensed. Totals are known + unknown.
N_KNOWN_TARGETS = 4
N_UNKNOWN_TARGETS = 0
N_KNOWN_RADARS = 6
N_UNKNOWN_RADARS = 0

# Match the training setup: FOFE observation encoding + inter-agent comms.
USE_FOFE = True
COMMUNICATE = True

# ===================================================================


def rollout_policy(ckpt, ns: int, nj: int, seed: int, device):
    """Roll one policy out once on the configured scenario. Returns (frames, env)."""
    env_cfg = copy.deepcopy(ckpt.base_env_cfg)
    env_cfg.n_strikers, env_cfg.n_jammers = ns, nj
    if hasattr(env_cfg, "dr"):
        env_cfg.dr = None  # deterministic layout for a clean picture

    # --- Apply the ENVIRONMENT CONFIG block ---
    # HFTestRunner reads these fields off env_cfg directly to build the env,
    # so set both the known/unknown counts and the resolved totals here
    # (normally EnvConfig.__post_init__ derives the totals; we bypass it).
    env_cfg.scenario = SCENARIO
    env_cfg.n_known_targets, env_cfg.n_unknown_targets = N_KNOWN_TARGETS, N_UNKNOWN_TARGETS
    env_cfg.n_targets = N_KNOWN_TARGETS + N_UNKNOWN_TARGETS
    env_cfg.n_known_radars, env_cfg.n_unknown_radars = N_KNOWN_RADARS, N_UNKNOWN_RADARS
    env_cfg.n_radars = N_KNOWN_RADARS + N_UNKNOWN_RADARS
    env_cfg.communicate = COMMUNICATE
    # FOFE observation encoding. The read-only use_fofe property reflects the
    # private _use_fofe field (normally set by finalize()); set it directly so
    # the env emits FOFE obs, matching the FOFE-trained policy.
    env_cfg._use_fofe = USE_FOFE

    policy = _build_policy_for_scenario(ckpt, env_cfg, device)
    runner = HFTestRunner(policy, env_cfg=env_cfg, hf_cfg=ckpt.hf_radar_cfg,
                          device=device, seed=seed)
    frames = runner.rollout()
    return frames, runner.env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-a", default=POLICY_A_PATH,
                    help="first policy .pt (left panel)")
    ap.add_argument("--checkpoint-b", default=POLICY_B_PATH,
                    help="second policy .pt (right panel)")
    ap.add_argument("--label-a", default=POLICY_A_LABEL)
    ap.add_argument("--label-b", default=POLICY_B_LABEL)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--strikers", type=int, default=N_STRIKERS)
    ap.add_argument("--jammers", type=int, default=N_JAMMERS)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--dpi", type=int, default=DPI,
                    help="output PNG resolution in dots per inch")
    args = ap.parse_args()

    out_path = _resolve(args.out)
    ns, nj, seed = args.strikers, args.jammers, args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"env: scenario={SCENARIO}  {ns}s{nj}j  seed={seed}  "
          f"targets={N_KNOWN_TARGETS}k+{N_UNKNOWN_TARGETS}u  "
          f"radars={N_KNOWN_RADARS}k+{N_UNKNOWN_RADARS}u  "
          f"fofe={USE_FOFE}  communicate={COMMUNICATE}")

    panels = [
        (args.label_a, _resolve(args.checkpoint_a)),
        (args.label_b, _resolve(args.checkpoint_b)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(4.6 * 2, 4.8), squeeze=False)

    for ci, (label, ckpt_path) in enumerate(panels):
        ax = axes[0, ci]
        print(f"loading {ckpt_path.name} on {device} ...")
        ckpt = _LoadedCheckpoint(ckpt_path, device)
        frames, env = rollout_policy(ckpt, ns, nj, seed, device)
        m = draw_run(ax, frames, env, ns, nj)
        ax.set_xlabel("X (km)", fontsize=8)
        if ci == 0:
            ax.set_ylabel("Y (km)", fontsize=8)
        ax.set_title(
            f"{label}\n"
            f"tgt={m['targets_destroyed']:.2f}  ·  surv={m['survival']:.2f}  ·  "
            f"frag={m['frag']:.2f}",
            fontsize=11, fontweight="bold", color=NLR_PRIMARY,
        )
        print(f"  {label}: tgt={m['targets_destroyed']:.3f}  "
              f"surv={m['survival']:.3f}  frag={m['frag']:.3f}")

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=5,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Policy comparison — {ns} strikers · {nj} jammers · seed {seed}\n"
        f"same scenario & layout, one policy per panel",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"saved -> {out_path}  ({args.dpi} dpi)")


if __name__ == "__main__":
    main()
