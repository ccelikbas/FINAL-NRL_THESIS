"""run_multiple_curriculums.py — train SEVERAL policies back-to-back, each with its
OWN curriculum (same section format as run_curriculum.py), for the final
complete-vs-baseline ablation over a weekend.

THE FOUR RUNS (edit the RUNS list below):
  1. complete_S1  : FOFE + communication ON,  S1, 6-8 known radars, 2-4 known targets
  2. complete_S2  : FOFE + communication ON,  S2, 2-3 known + 2-3 unknown radars,
                                                  1-2 known + 1-2 unknown targets
  3. baseline_S1  : zero-padded obs (FOFE OFF) + communication OFF, same curriculum as 1
  4. baseline_S2  : zero-padded obs (FOFE OFF) + communication OFF, same curriculum as 2

Each run follows the V7-proven shape (4500 iters): a short FIXED 2s2j warmup at
kill 0.02 (biases the scarce-escort case), then DR jammers (2,4) while the radar
kill probability anneals 0.02 → 0.05 → 0.10. Baselines use IDENTICAL curricula /
budgets / rewards / nets — the only differences are use_fofe and communicate, so
any performance gap is attributable to FOFE + communication (clean ablation, and
Wilcoxon comparisons in eval_tools stay fair).

Evaluation during training AUTOMATICALLY matches each section's learning config:
train_mappo's periodic eval mirrors the active section's DR (same env_cfg.dr).

OUTPUTS  (one backup folder per run → 4 folders in runs/)
  runs/<run_name>_<YYYYMMDD>.pt        canonical FINAL checkpoint per run (same
                                       keys as run_curriculum.py → all eval_tools
                                       default to it). Rolling-overwritten after
                                       every section as crash insurance.
  runs/<run_name>_<YYYYMMDD>/          backup folder — ONE kept .pt per curriculum
    stage1of5_<name>.pt                stage (never overwritten), the last tagged
    ...                                _FINAL. Evaluate/compare any intermediate
    stage5of5_<name>_FINAL.pt          stage, or recover if a later stage regresses.
  plots/final_curriculums_<YYYYMMDD>/  per-run dashboard / reward / eval-rates
                                       PNGs + combined_eval_rates.png overlaying
                                       all runs.

No animations here (keeps the batch lean) — animate afterwards with the
visualisation tools if wanted.

Run (repo root, project venv) — leave it going over the weekend with sleep
DISABLED on AC power (machine sleep hangs CUDA runs):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\run_multiple_curriculums.py
  # subset / rerun one that failed:
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\run_multiple_curriculums.py --only baseline_S2
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

if __package__ in (None, ""):
    import types
    _this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_this_dir.parent))
    _pkg_name = "fofe_mappo"
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_this_dir)]
        _pkg.__package__ = _pkg_name
        _pkg.__file__ = str(_this_dir / "__init__.py")
        sys.modules[_pkg_name] = _pkg
    __package__ = _pkg_name

import numpy as np
import torch
import matplotlib.pyplot as plt

from .config import (
    DomainRandomization, EnvConfig, EnvExtensionsConfig, ExperimentConfig,
    FOFEConfig, NetworkConfig, PPOConfig,
)
from .nlr_style import (
    apply_nlr_style, NLR_PRIMARY, NLR_SECONDARY, NLR_ACCENT, NLR_DARKGRAY,
)
from .rewards import RewardConfig
from .trainer import train_mappo
# Reuse run_curriculum's section format, resolution logic and plotting so the
# two runners stay behaviourally identical per section.
from .run_curriculum import (
    CurriculumSection, _section_to_env_cfg, _section_label,
    _merge_logs, _state_dict_to_cpu,
    plot_curriculum_dashboard, plot_curriculum_reward, plot_curriculum_eval_rates,
)

apply_nlr_style()

DATE_TAG = datetime.now().strftime("%Y%m%d")


# =====================================================================
#  THE FOUR TRAINING RUNS  —  edit freely
# =====================================================================

@dataclass
class TrainingRun:
    """One full policy training = its own curriculum + architecture flags."""
    name: str                      # → runs/<name>_<date>.pt and plot filenames
    description: str
    use_fofe: bool
    communicate: bool
    curriculum: List[CurriculumSection] = field(default_factory=list)
    # Extra EnvConfig attribute overrides applied to EVERY section of this run
    # (fields CurriculumSection cannot express, e.g. radar_min_sep for S1).
    env_overrides: Dict[str, Any] = field(default_factory=dict)


def _make_curriculum(
    scenario: str,
    n_known_targets, n_unknown_targets,
    n_known_radars, n_unknown_radars,
    communicate: bool,
) -> List[CurriculumSection]:
    """V7-proven shape, 4500 iters total: fixed 2s2j warmup @kill0.02, then DR
    jammers (2,4) while kill anneals 0.02 → 0.05 → 0.10. The WORLD (radar /
    target counts, incl. unknowns) is identical in every section so the policy
    always trains — and is evaluated — on the target scenario distribution."""
    world = dict(
        n_known_targets=n_known_targets, n_unknown_targets=n_unknown_targets,
        n_known_radars=n_known_radars, n_unknown_radars=n_unknown_radars,
        scenario=scenario, communicate=communicate, max_steps=150,
    )
    return [
        CurriculumSection(name="warmup 2s2j k0.025", n_iters=250, #250
                          n_strikers=2, n_jammers=2,
                          radar_kill_probability=0.025, **world),
        CurriculumSection(name="DR j2-4 k0.025", n_iters=750, #750
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.025, **world),
        CurriculumSection(name="DR j2-4 k0.05", n_iters=1000, #1000
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.05, **world),
        CurriculumSection(name="DR j2-4 k0.1", n_iters=1000, #1000
                          n_strikers=2, n_jammers=(2, 4),   
                          radar_kill_probability=0.1, **world),
        CurriculumSection(name="DR j2-4 k0.25", n_iters=2000, #2000
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),
    ]


# Scenario worlds (per your spec):
#   S1: 6-8 known radars, 2-4 known targets (all known).
#   S2: 2-3 known + 2-3 unknown radars, 1-2 known + 1-2 unknown targets.
_S1 = dict(scenario="S2",
           n_known_targets=(2, 4), n_unknown_targets=0,
           n_known_radars=(4, 6), n_unknown_radars=0)
_S2 = dict(scenario="S2",
           n_known_targets=(1, 2), n_unknown_targets=(1, 2),
           n_known_radars=(2, 3), n_unknown_radars=(2, 3))

# S1's default radar_min_sep (0.5) cannot fit >2 radars in the S1 spawn band
# [0.2,0.8]x[0.6,0.8] — layout pregeneration hard-fails (smoke-tested). 0.15 is
# the largest separation that reliably pregenerates 1024 layouts with 7 radars
# (0.20 fails; if you raise the max to 8 radars, drop this to 0.12).
_S1_OVERRIDES = {"radar_min_sep": 0.15}

RUNS: List[TrainingRun] = [
    TrainingRun(
        name="complete_S1_fofe_comm",
        description="Complete policy, scenario 1 — FOFE + communication ON",
        use_fofe=True, communicate=True,
        curriculum=_make_curriculum(**_S1, communicate=True),
        env_overrides=dict(_S1_OVERRIDES),
    ),
    TrainingRun(
        name="complete_S2_fofe_comm",
        description="Complete policy, scenario 2 — FOFE + communication ON",
        use_fofe=True, communicate=True,
        curriculum=_make_curriculum(**_S2, communicate=True),
    ),
    TrainingRun(
        name="baseline_S1_nofofe_nocomm",
        description="Baseline, scenario 1 — zero-padded obs, communication OFF",
        use_fofe=False, communicate=False,
        curriculum=_make_curriculum(**_S1, communicate=False),
        env_overrides=dict(_S1_OVERRIDES),
    ),
    TrainingRun(
        name="baseline_S2_nofofe_nocomm",
        description="Baseline, scenario 2 — zero-padded obs, communication OFF",
        use_fofe=False, communicate=False,
        curriculum=_make_curriculum(**_S2, communicate=False),
    ),
]


# =====================================================================
#  IMPLEMENTATION  —  you usually do not need to edit below this line
# =====================================================================

def _apply_run_env_fixups(
    run: TrainingRun,
    resolved: List[Tuple[CurriculumSection, EnvConfig]],
) -> None:
    """Post-resolution fixups applied to every section's EnvConfig (in place).

    1. run.env_overrides — extra EnvConfig attributes a CurriculumSection cannot
       express (e.g. radar_min_sep=0.15 so S1 can spawn 7 radars).
    2. AGENT-ALLOCATION ALIGNMENT — pad n_strikers/n_jammers allocation to the
       run-wide max so tensor shapes are identical in every section. The
       zero-padded (non-FOFE) critic's input dim depends on the ALLOCATED agent
       count, so without this the carried critic silently re-initialises at any
       section boundary that changes the count (smoke test caught: warmup J2 →
       DR J4 gave a [320,71]-vs-[320,91] size mismatch → random critic restart,
       unfair to the baselines). Sections with a smaller fixed count keep their
       ACTUAL present count via a degenerate DR range (alloc=max slots,
       lo=hi=actual) — the env realises absent slots as not-alive, exactly like
       normal DR. FOFE nets are count-invariant either way; this makes the
       baseline carry strict too.
    """
    for _, env_cfg in resolved:
        for k, v in run.env_overrides.items():
            setattr(env_cfg, k, v)

    max_ns = max(c.n_strikers for _, c in resolved)
    max_nj = max(c.n_jammers for _, c in resolved)
    for _, c in resolved:
        pad_ns = c.n_strikers < max_ns
        pad_nj = c.n_jammers < max_nj
        if not (pad_ns or pad_nj):
            continue
        if c.dr is None:
            c.dr = DomainRandomization()
        if pad_ns:
            if c.dr.n_strikers is None:
                c.dr.n_strikers = (c.n_strikers, c.n_strikers)
            c.n_strikers = max_ns
        if pad_nj:
            if c.dr.n_jammers is None:
                c.dr.n_jammers = (c.n_jammers, c.n_jammers)
            c.n_jammers = max_nj


def _safe(s: str) -> str:
    """Filesystem-safe token from a section name (spaces/dots → underscores)."""
    return "".join(c if (c.isalnum() or c in "-") else "_" for c in str(s)).strip("_")


def _save_checkpoint(save_path: Path, checkpoint: Dict[str, Any], *,
                     net_cfg, fofe_cfg, ext_cfg, reward_cfg,
                     all_logs, curriculum_meta, section_bounds,
                     run_name: str, description: str, completed: int, total: int) -> None:
    """Write the run's .pt with the exact key layout run_curriculum.py uses
    (so every eval_tools script works on it), plus run metadata. Called after
    every section — the file is progressively overwritten as insurance."""
    torch.save(
        {
            "policy_state_dict": checkpoint["policy_state_dict"],
            "critic_state_dict": checkpoint["critic_state_dict"],
            "reward_normalizer_state_dict": checkpoint["reward_normalizer_state_dict"],
            "net_cfg": net_cfg,
            "fofe_cfg": fofe_cfg,
            "ext_cfg": ext_cfg,
            "reward_cfg": reward_cfg,
            "training_logs": all_logs,
            "curriculum": curriculum_meta,
            "section_bounds": section_bounds,
            "run_name": run_name,
            "run_description": description,
            "sections_completed": f"{completed}/{total}",
        },
        save_path,
    )


def _print_run_plan(run: TrainingRun,
                    resolved: List[Tuple[CurriculumSection, EnvConfig]],
                    save_path: Path) -> None:
    total_iters = sum(sec.n_iters for sec, _ in resolved)
    print("\n" + "=" * 78)
    print(f"  RUN '{run.name}'  —  {run.description}")
    print(f"  FOFE {'ON' if run.use_fofe else 'OFF'} | comm {'ON' if run.communicate else 'OFF'} "
          f"| {total_iters} iters | → {save_path}")
    print("-" * 78)
    cursor = 0
    for sec, env_cfg in resolved:
        dr_tag = "DR" if env_cfg.dr is not None else "fixed"
        print(f"  [{cursor:4d},{cursor + sec.n_iters:4d})  {sec.name:20s} "
              f"({dr_tag})  {_section_label(sec, env_cfg)}")
        cursor += sec.n_iters
    print("=" * 78)


def _train_one_run(run: TrainingRun, args, plots_dir: Path) -> Optional[Dict[str, List[float]]]:
    """Train one policy through its curriculum. Returns its merged logs
    (for the combined figure), or None if the run failed."""
    fofe_cfg = FOFEConfig(use_fofe=run.use_fofe)
    ext_cfg = EnvExtensionsConfig()
    hf_radar_cfg = ext_cfg.hf_radar if ext_cfg.use_hf_radar else None
    env_defaults = EnvConfig()
    reward_cfg = RewardConfig()          # current rewards.py = the vetted setup
    net_cfg = NetworkConfig()            # identical nets for complete & baseline

    resolved: List[Tuple[CurriculumSection, EnvConfig]] = [
        (sec, _section_to_env_cfg(sec, env_defaults, reward_cfg, fofe_cfg))
        for sec in run.curriculum
    ]
    _apply_run_env_fixups(run, resolved)
    total_iters = sum(sec.n_iters for sec in run.curriculum)
    curriculum_meta = [
        {"name": s.name, "n_iters": s.n_iters, "label": _section_label(s, c)}
        for s, c in resolved
    ]
    save_path = Path(args.save_dir) / f"{run.name}_{DATE_TAG}.pt"
    # Per-run backup folder holding one kept snapshot per curriculum stage + the
    # FINAL stage (the top-level save_path above stays the canonical rolling final
    # that eval_tools default to).
    backup_dir = Path(args.save_dir) / f"{run.name}_{DATE_TAG}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    _print_run_plan(run, resolved, save_path)

    checkpoint: Optional[Dict[str, Any]] = None
    all_logs: Dict[str, List[float]] = {}
    section_bounds: List[Tuple[str, int, int]] = []
    global_iter = 0
    t0 = time.time()

    for si, (sec, env_cfg) in enumerate(resolved):
        ppo_cfg = PPOConfig(
            num_envs=args.num_envs,
            n_iters=sec.n_iters,
            seed=args.seed + global_iter,
            log_every=args.log_every,
        )
        ppo_cfg.iteration_offset = global_iter
        exp_cfg = ExperimentConfig(
            env=env_cfg, ppo=ppo_cfg, net=net_cfg, fofe=fofe_cfg, ext=ext_cfg,
        ).finalize()

        print(f"\n  ── section '{sec.name}'  iters [{global_iter}, {global_iter + sec.n_iters}) ──")
        base_env, policy, critic, logs, reward_normalizer = train_mappo(
            exp_cfg.env, exp_cfg.ppo, exp_cfg.net, fofe_cfg, checkpoint,
            hf_radar_cfg=hf_radar_cfg,
        )

        _merge_logs(all_logs, logs)
        section_bounds.append((sec.name, global_iter, global_iter + sec.n_iters))
        global_iter += sec.n_iters

        # Carry weights forward on CPU (release GPU before the next section).
        checkpoint = {
            "policy_state_dict": _state_dict_to_cpu(policy.state_dict()),
            "critic_state_dict": _state_dict_to_cpu(critic.state_dict()),
            "reward_normalizer_state_dict": _state_dict_to_cpu(
                reward_normalizer.state_dict() if reward_normalizer is not None else None
            ),
        }
        del base_env, policy, critic, logs, reward_normalizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        is_final = (si + 1 == len(resolved))
        stage_tag = f"stage{si + 1}of{len(resolved)}_{_safe(sec.name)}" + ("_FINAL" if is_final else "")
        stage_path = backup_dir / f"{stage_tag}.pt"
        save_kwargs = dict(
            net_cfg=net_cfg, fofe_cfg=fofe_cfg, ext_cfg=ext_cfg,
            reward_cfg=reward_cfg, all_logs=all_logs,
            curriculum_meta=curriculum_meta, section_bounds=section_bounds,
            run_name=run.name, description=run.description,
            completed=si + 1, total=len(resolved),
        )
        # (1) rolling canonical final (crash insurance; eval_tools default target).
        _save_checkpoint(save_path, checkpoint, **save_kwargs)
        # (2) kept per-stage backup snapshot (never overwritten).
        _save_checkpoint(stage_path, checkpoint, **save_kwargs)
        print(f"  saved → {save_path.name}  +  {backup_dir.name}/{stage_path.name}  "
              f"({si + 1}/{len(resolved)} sections, {(time.time() - t0) / 3600:.1f} h elapsed)")

    # Per-run plots (same three figures run_curriculum produces).
    if not args.no_plot:
        for fn, suffix in ((plot_curriculum_dashboard, "dashboard"),
                           (plot_curriculum_reward, "reward"),
                           (plot_curriculum_eval_rates, "eval_rates")):
            try:
                fn(all_logs, section_bounds,
                   save_path=str(plots_dir / f"{run.name}_{suffix}.png"))
            except Exception as exc:  # noqa: BLE001 — plots must never kill training
                print(f"  plot warning ({suffix}): {type(exc).__name__}: {exc}")

    hrs = (time.time() - t0) / 3600
    print(f"\n  RUN '{run.name}' DONE in {hrs:.1f} h  →  {save_path}")
    return all_logs


def _plot_combined(run_logs: Dict[str, Dict[str, List[float]]], plots_dir: Path) -> None:
    """Overlay the runs' eval curves: one panel per KPI, one line per run."""
    panels = [
        ("eval_survival_rate", "Survival rate"),
        ("eval_task_completion_rate", "Task completion rate"),
        ("eval_targets_destroyed_rate", "Targets-destroyed rate"),
        ("eval_coalition_fragmentation", "Coalition fragmentation (frag)"),
    ]
    colors = [NLR_PRIMARY, NLR_ACCENT, NLR_SECONDARY, NLR_DARKGRAY]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for ci, (name, logs) in enumerate(run_logs.items()):
            ys = logs.get(key) or []
            xs = np.arange(1, len(ys) + 1)
            valid = [(x, y) for x, y in zip(xs, ys) if y is not None and np.isfinite(y)]
            if valid:
                vx, vy = zip(*valid)
                ax.plot(vx, vy, color=colors[ci % len(colors)], lw=1.4, label=name)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Global iteration")
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=7)
    fig.suptitle(f"Final curriculums — combined eval rates ({DATE_TAG})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = plots_dir / "combined_eval_rates.png"
    fig.savefig(out, dpi=130)
    print(f"combined figure → {out}")


def main() -> None:
    ppo_d = PPOConfig()
    p = argparse.ArgumentParser(description="Sequential multi-curriculum trainer "
                                "(final complete/baseline policies). Edit RUNS above.")
    p.add_argument("--num_envs", type=int, default=ppo_d.num_envs,
                   help="Parallel envs (PPOConfig default; sized for the remote "
                        "server). NOTE: the FOFE runs OOM an 8GB laptop GPU even "
                        "at 512 — run the real batch on the server.")
    p.add_argument("--log_every", type=int, default=ppo_d.log_every,
                   help="Eval cadence — eval mirrors the active section's DR.")
    p.add_argument("--seed", type=int, default=ppo_d.seed,
                   help="Same base seed for every run (fair ablation).")
    p.add_argument("--save_dir", type=str, default="runs",
                   help="Checkpoint dir. RELATIVE paths resolve against this file's "
                        "folder (0_TA_HF_FOFE-MAPPO/), NOT the CWD — so 'runs' always "
                        "means 0_TA_HF_FOFE-MAPPO/runs/ where V6/V7 + eval_tools live, "
                        "regardless of where you launch from.")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated run names to execute (default: all). "
                        "E.g. --only baseline_S2_nofofe_nocomm")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="Resolve + print every run's plan (validates all DR ranges), "
                        "then exit without training.")
    p.add_argument("--smoke", type=int, default=None, metavar="N",
                   help="Smoke-test mode: run every section for only N iters at the "
                        "REAL num_envs/worlds (validates the full pipeline incl. GPU "
                        "memory before committing the weekend). Outputs get a "
                        "'_smoke' suffix so real names stay clean.")
    args = p.parse_args()

    # Resolve save_dir against THIS file's folder (0_TA_HF_FOFE-MAPPO), not the
    # CWD, so checkpoints land in the same runs/ as V6/V7 and eval_tools defaults
    # no matter where the script is launched from.
    _save_dir = Path(args.save_dir)
    if not _save_dir.is_absolute():
        _save_dir = Path(__file__).resolve().parent / _save_dir
    args.save_dir = str(_save_dir)

    selected = RUNS
    if args.only:
        wanted = {w.strip() for w in args.only.split(",")}
        selected = [r for r in RUNS if r.name in wanted]
        unknown = wanted - {r.name for r in RUNS}
        if unknown:
            raise SystemExit(f"--only names not found: {sorted(unknown)}; "
                             f"available: {[r.name for r in RUNS]}")

    if args.smoke:
        # Same worlds / num_envs (so memory is representative), tiny iterations,
        # '_smoke'-suffixed outputs.
        selected = [
            replace(run, name=run.name + "_smoke",
                    curriculum=[replace(sec, n_iters=args.smoke) for sec in run.curriculum])
            for run in selected
        ]
        print(f"\n  SMOKE MODE: every section trimmed to {args.smoke} iters.")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    plots_tag = f"final_curriculums_{DATE_TAG}" + ("_smoke" if args.smoke else "")
    plots_dir = Path(__file__).resolve().parent / "plots" / plots_tag
    plots_dir.mkdir(parents=True, exist_ok=True)

    total_iters = sum(sum(s.n_iters for s in r.curriculum) for r in selected)
    print("\n" + "=" * 78)
    print(f"  MULTI-CURRICULUM PLAN  ({DATE_TAG})  —  {len(selected)} run(s), "
          f"{total_iters} total iters")
    print(f"  num_envs {args.num_envs} | eval every {args.log_every} iters "
          f"(mirrors section DR) | seed {args.seed}")
    print(f"  checkpoints → {args.save_dir}/<name>_{DATE_TAG}.pt   plots → {plots_dir}")
    print(f"  rough ETA at ~9 s/iter: {total_iters * 9 / 3600:.0f} h "
          f"(baselines run faster without FOFE)")
    print("  NOTE: disable machine sleep on AC — sleep hangs CUDA runs.")
    print("=" * 78)

    if args.dry_run:
        env_defaults = EnvConfig()
        for run in selected:
            fofe_cfg = FOFEConfig(use_fofe=run.use_fofe)
            resolved = [(sec, _section_to_env_cfg(sec, env_defaults, RewardConfig(), fofe_cfg))
                        for sec in run.curriculum]
            _apply_run_env_fixups(run, resolved)
            _print_run_plan(run, resolved,
                            Path(args.save_dir) / f"{run.name}_{DATE_TAG}.pt")
        print("\n(dry run — nothing trained)")
        return

    run_logs: Dict[str, Dict[str, List[float]]] = {}
    failures: List[str] = []
    for run in selected:
        try:
            logs = _train_one_run(run, args, plots_dir)
            if logs is not None:
                run_logs[run.name] = logs
        except Exception as exc:  # noqa: BLE001 — one failed run must not kill the batch
            failures.append(run.name)
            print(f"\n  !! RUN '{run.name}' FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if run_logs and not args.no_plot:
        try:
            _plot_combined(run_logs, plots_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"combined-plot warning: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 78)
    print("  BATCH SUMMARY")
    for run in selected:
        status = "FAILED" if run.name in failures else ("ok" if run.name in run_logs else "skipped")
        print(f"    {run.name:<28} {status}")
    print("=" * 78)
    if failures:
        print(f"  rerun failed ones with: --only {','.join(failures)}")


if __name__ == "__main__":
    main()
