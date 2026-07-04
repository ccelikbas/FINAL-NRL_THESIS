"""run_multiple_curriculums.py — train the final complete-vs-baseline policies as
TWO LINEAGES, each trained SEQUENTIALLY on scenario 1 THEN continued on scenario 2
(weights carried, so S2 warm-starts from the trained S1 policy — saving the
from-scratch S2 cost). Same section format as run_curriculum.py.

THE TWO LINEAGES (edit the LINEAGES list below):
  1. complete : FOFE + communication ON,  S1 phase → S2 phase
  2. baseline : zero-padded obs (FOFE OFF) + communication OFF, S1 phase → S2 phase

Each lineage yields TWO evaluatable checkpoints — the S1-phase final and the
S2-phase final — so you still get 4 policies (complete_S1, complete_S2,
baseline_S1, baseline_S2). NOTE: the S2 policy is now S1→S2 SEQUENTIAL, not
S2-from-scratch; complete_S2 vs baseline_S2 stays a clean FOFE+comm ablation.

Phases (per lineage):
  S1 phase — the V7-proven shape: fixed 2s2j warmup, then DR jammers (2,4) with the
             radar-kill anneal, in the S1 world (all-known radars/targets).
  S2 phase — CONTINUATION, warm-started: NO warmup (escort already learned), DR
             jammers (2,4), kill re-ramped from a mid value back to the S1 target,
             in the S2 world (known + hidden radars/targets). ~2500 iters.

Weight carry is strict for FOFE (count-invariant) and for the baseline too, as
long as the S1/S2 entity TOTALS match (they do in the shipped worlds); agent
allocation is auto-aligned across the whole lineage. _warn_critic_dim flags any
future divergence (the actor always transfers; only the baseline critic would
re-init).

Evaluation during training AUTOMATICALLY matches each section's learning config:
train_mappo's periodic eval mirrors the active section's DR (same env_cfg.dr).

OUTPUTS  (one backup folder per PHASE → 4 folders in runs/, as before)
  runs/<lineage>_<phase>_<YYYYMMDD>.pt  canonical phase-final checkpoint (e.g.
                                        complete_S1_*, complete_S2_* — 4 total;
                                        same keys as run_curriculum.py → all
                                        eval_tools default to them). Rolling-
                                        overwritten within the phase as insurance.
  runs/<lineage>_<phase>_<YYYYMMDD>/     backup folder — ONE kept .pt per stage of
    stage1ofN_<name>.pt                 that phase (never overwritten), last tagged
    ...                                 _FINAL. The S2 folders' metadata carry the
    stageNofN_<name>_FINAL.pt           cumulative S1→S2 history.
  plots/final_curriculums_<YYYYMMDD>/   per-lineage dashboard/reward/eval-rates
                                        (full S1→S2 span, phase boundaries shaded)
                                        + combined_eval_rates.png overlaying both.

No animations here (keeps the batch lean) — animate afterwards with the
visualisation tools if wanted.

Run (repo root, project venv) — leave it going over the weekend with sleep
DISABLED on AC power (machine sleep hangs CUDA runs):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\run_multiple_curriculums.py
  # one lineage only / rerun a failed one (both phases):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\run_multiple_curriculums.py --only baseline
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
class Phase:
    """One scenario phase of a lineage — its own world + curriculum. Weights carry
    (warm-start) INTO this phase from the previous one. Each phase yields one
    evaluatable checkpoint: runs/<lineage>_<tag>_<date>.pt (+ a backup folder)."""
    tag: str                       # "S1" / "S2" — used in checkpoint + folder names
    curriculum: List[CurriculumSection] = field(default_factory=list)
    # Extra EnvConfig attribute overrides applied to every section of THIS phase
    # (fields CurriculumSection cannot express, e.g. radar_min_sep for S1).
    env_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Lineage:
    """One model trained SEQUENTIALLY across phases (S1 → S2 …), carrying weights
    continuously. The S2 phase therefore warm-starts from the fully-trained S1
    policy — saving the from-scratch S2 cost. Architecture flags (use_fofe,
    communicate) are constant across the lineage — it is one network trained
    through both scenarios."""
    name: str                      # "complete" / "baseline"
    description: str
    use_fofe: bool
    communicate: bool
    phases: List[Phase] = field(default_factory=list)


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

s
def _s2_continuation(
    scenario, n_known_targets, n_unknown_targets,
    n_known_radars, n_unknown_radars, communicate,
) -> List[CurriculumSection]:
    """Scenario-2 CONTINUATION, warm-started from the trained S1 policy: NO 2s2j
    warmup (escort behaviour already learned in S1), DR jammers (2,4) throughout,
    and the radar kill RE-RAMPED from a mid value back to the S1 target (0.25) so
    adaptation to S2's hidden radars/targets isn't done at max lethality cold.
    ~3000 iters (vs a full 5000 from scratch)."""
    world = dict(
        n_known_targets=n_known_targets, n_unknown_targets=n_unknown_targets,
        n_known_radars=n_known_radars, n_unknown_radars=n_unknown_radars,
        scenario=scenario, communicate=communicate, max_steps=150,
    )
    return [
        CurriculumSection(name="S2 DR j2-4 k0.05", n_iters=500,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.05, **world),
        CurriculumSection(name="S2 DR j2-4 k0.1", n_iters=500,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.1, **world),
        CurriculumSection(name="S2 DR j2-4 k0.25", n_iters=2000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),
    ]
s

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

# Two LINEAGES, each trained S1 → S2 (weights carried). Each yields two
# evaluatable checkpoints (…_S1_<date>.pt and …_S2_<date>.pt) → 4 policies total,
# 4 backup folders. S2 warm-starts from the S1-trained policy.
LINEAGES: List[Lineage] = [
    Lineage(
        name="complete", description="Complete model — FOFE + communication ON, trained S1 → S2",
        use_fofe=True, communicate=True,
        phases=[
            Phase("S1", _make_curriculum(**_S1, communicate=True), env_overrides=dict(_S1_OVERRIDES)),
            Phase("S2", _s2_continuation(**_S2, communicate=True)),
        ],
    ),
    Lineage(
        name="baseline", description="Baseline model — zero-padded obs, communication OFF, trained S1 → S2",
        use_fofe=False, communicate=False,
        phases=[
            Phase("S1", _make_curriculum(**_S1, communicate=False), env_overrides=dict(_S1_OVERRIDES)),
            Phase("S2", _s2_continuation(**_S2, communicate=False)),
        ],
    ),
]


# =====================================================================
#  IMPLEMENTATION  —  you usually do not need to edit below this line
# =====================================================================

def _align_agents(resolved: List[Tuple[CurriculumSection, EnvConfig]]) -> None:
    """AGENT-ALLOCATION ALIGNMENT across an ENTIRE lineage (every section of every
    phase), in place. Pads n_strikers/n_jammers allocation to the lineage-wide max
    so tensor shapes are identical end-to-end. The zero-padded (non-FOFE) critic's
    input dim depends on the ALLOCATED agent count, so without this the carried
    critic silently re-initialises at any boundary that changes the count (smoke
    test caught the warmup-J2 → DR-J4 case). Sections with a smaller fixed count
    keep their ACTUAL present count via a degenerate DR range (alloc=max slots,
    lo=hi=actual) — the env realises absent slots as not-alive, exactly like normal
    DR. FOFE nets are count-invariant either way; this makes the baseline carry
    strict too, including across the S1→S2 handoff."""
    max_ns = max(c.n_strikers for _, c in resolved)
    max_nj = max(c.n_jammers for _, c in resolved)
    for _, c in resolved:
        if c.n_strikers < max_ns or c.n_jammers < max_nj:
            if c.dr is None:
                c.dr = DomainRandomization()
        if c.n_strikers < max_ns:
            if c.dr.n_strikers is None:
                c.dr.n_strikers = (c.n_strikers, c.n_strikers)
            c.n_strikers = max_ns
        if c.n_jammers < max_nj:
            if c.dr.n_jammers is None:
                c.dr.n_jammers = (c.n_jammers, c.n_jammers)
            c.n_jammers = max_nj


def _warn_critic_dim(resolved: List[Tuple[CurriculumSection, EnvConfig]], lineage: "Lineage") -> None:
    """The zero-padded (baseline) critic's input dim is (7+E)·n_agents + 3·n_targets
    + 3·n_radars + 1 — it depends on ALLOCATED entity TOTALS. n_agents is aligned by
    _align_agents; targets/radars are NOT force-aligned (that would re-inflate S1's
    radar count past the spawn-separation limit). If the target/radar TOTALS differ
    across the lineage, the baseline critic re-initialises at that boundary (the
    ACTOR still transfers — it is slot-based). FOFE critics are count-invariant, so
    this only matters for the baseline. With the shipped S1/S2 worlds the totals
    match (6 agents / 4 targets / 6 radars) → strict transfer; this just flags a
    future divergence."""
    if lineage.use_fofe:
        return
    totals = {(c.n_targets, c.n_radars) for _, c in resolved}
    if len(totals) > 1:
        print(f"  [!] baseline '{lineage.name}': target/radar TOTALS vary across the "
              f"lineage {sorted(totals)} → the zero-padded CRITIC re-initialises at "
              f"the boundary where they change (the policy/actor still transfers). "
              f"Align S1/S2 counts to the same totals for a strict critic carry.")


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


def _resolve_lineage(lineage: Lineage):
    """Resolve every section of every phase → per-phase [(sec, env_cfg)] with the
    phase's env_overrides applied, then align agent allocation across the whole
    lineage. Returns (phase_resolved, fofe_cfg, ext_cfg, hf_radar_cfg, net_cfg,
    reward_cfg)."""
    fofe_cfg = FOFEConfig(use_fofe=lineage.use_fofe)
    ext_cfg = EnvExtensionsConfig()
    hf_radar_cfg = ext_cfg.hf_radar if ext_cfg.use_hf_radar else None
    env_defaults = EnvConfig()
    reward_cfg = RewardConfig()          # current rewards.py = the vetted setup
    net_cfg = NetworkConfig()            # identical nets for complete & baseline

    phase_resolved: List[Tuple[Phase, List[Tuple[CurriculumSection, EnvConfig]]]] = []
    all_resolved: List[Tuple[CurriculumSection, EnvConfig]] = []
    for phase in lineage.phases:
        res = [(sec, _section_to_env_cfg(sec, env_defaults, reward_cfg, fofe_cfg))
               for sec in phase.curriculum]
        for _, env_cfg in res:            # per-phase EnvConfig overrides
            for k, v in phase.env_overrides.items():
                setattr(env_cfg, k, v)
        phase_resolved.append((phase, res))
        all_resolved.extend(res)
    _align_agents(all_resolved)           # lineage-wide (strict carry across phases)
    _warn_critic_dim(all_resolved, lineage)
    return phase_resolved, fofe_cfg, ext_cfg, hf_radar_cfg, net_cfg, reward_cfg


def _print_lineage_plan(lineage: Lineage, phase_resolved, save_dir: Path) -> None:
    total_iters = sum(sec.n_iters for _, res in phase_resolved for sec, _ in res)
    print("\n" + "=" * 78)
    print(f"  LINEAGE '{lineage.name}'  —  {lineage.description}")
    print(f"  FOFE {'ON' if lineage.use_fofe else 'OFF'} | comm {'ON' if lineage.communicate else 'OFF'} "
          f"| {total_iters} iters | {len(phase_resolved)} phases (weights carried)")
    cursor = 0
    for phase, res in phase_resolved:
        print("-" * 78)
        print(f"  PHASE {phase.tag}  →  {save_dir.name}/{lineage.name}_{phase.tag}_{DATE_TAG}.pt")
        for sec, env_cfg in res:
            dr_tag = "DR" if env_cfg.dr is not None else "fixed"
            print(f"    [{cursor:5d},{cursor + sec.n_iters:5d})  {sec.name:18s} "
                  f"({dr_tag})  {_section_label(sec, env_cfg)}")
            cursor += sec.n_iters
    print("=" * 78)


def _train_one_lineage(lineage: Lineage, args, plots_dir: Path) -> Optional[Dict[str, List[float]]]:
    """Train one lineage SEQUENTIALLY through its phases (S1 → S2 …), carrying the
    checkpoint (CPU weights) continuously so each phase warm-starts from the last.
    Saves one canonical checkpoint + a per-stage backup folder PER PHASE. Returns
    the merged logs spanning all phases (for the combined figure), or None on
    failure."""
    phase_resolved, fofe_cfg, ext_cfg, hf_radar_cfg, net_cfg, reward_cfg = _resolve_lineage(lineage)
    save_dir = Path(args.save_dir)
    _print_lineage_plan(lineage, phase_resolved, save_dir)

    checkpoint: Optional[Dict[str, Any]] = None      # carried across ALL phases
    all_logs: Dict[str, List[float]] = {}            # cumulative (whole lineage)
    section_bounds: List[Tuple[str, int, int]] = []  # cumulative
    curriculum_meta: List[Dict[str, Any]] = []       # cumulative
    global_iter = 0
    t0 = time.time()
    phase_finals: List[Tuple[str, Path]] = []

    for phase, res in phase_resolved:
        backup_dir = save_dir / f"{lineage.name}_{phase.tag}_{DATE_TAG}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        phase_final_path = save_dir / f"{lineage.name}_{phase.tag}_{DATE_TAG}.pt"
        n_sec = len(res)
        print(f"\n  ▶ PHASE {phase.tag} ({lineage.name}) — {n_sec} sections "
              f"→ {phase_final_path.name}")

        for si, (sec, env_cfg) in enumerate(res):
            ppo_cfg = PPOConfig(num_envs=args.num_envs, n_iters=sec.n_iters,
                                seed=args.seed + global_iter, log_every=args.log_every)
            ppo_cfg.iteration_offset = global_iter
            exp_cfg = ExperimentConfig(
                env=env_cfg, ppo=ppo_cfg, net=net_cfg, fofe=fofe_cfg, ext=ext_cfg,
            ).finalize()

            print(f"\n  ── {phase.tag} section '{sec.name}'  "
                  f"iters [{global_iter}, {global_iter + sec.n_iters}) ──")
            base_env, policy, critic, logs, reward_normalizer = train_mappo(
                exp_cfg.env, exp_cfg.ppo, exp_cfg.net, fofe_cfg, checkpoint,
                hf_radar_cfg=hf_radar_cfg,
            )

            _merge_logs(all_logs, logs)
            section_bounds.append((f"{phase.tag}:{sec.name}", global_iter, global_iter + sec.n_iters))
            curriculum_meta.append({"name": f"{phase.tag}:{sec.name}", "n_iters": sec.n_iters,
                                    "label": _section_label(sec, env_cfg)})
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

            is_phase_final = (si + 1 == n_sec)
            stage_tag = f"stage{si + 1}of{n_sec}_{_safe(sec.name)}" + ("_FINAL" if is_phase_final else "")
            save_kwargs = dict(
                net_cfg=net_cfg, fofe_cfg=fofe_cfg, ext_cfg=ext_cfg,
                reward_cfg=reward_cfg, all_logs=all_logs,
                curriculum_meta=list(curriculum_meta), section_bounds=list(section_bounds),
                run_name=f"{lineage.name}_{phase.tag}",
                description=f"{lineage.description}  [phase {phase.tag}]",
                completed=si + 1, total=n_sec,
            )
            # (1) rolling canonical phase-final (crash insurance; eval_tools default).
            _save_checkpoint(phase_final_path, checkpoint, **save_kwargs)
            # (2) kept per-stage backup snapshot (never overwritten).
            _save_checkpoint(backup_dir / f"{stage_tag}.pt", checkpoint, **save_kwargs)
            print(f"  saved → {phase_final_path.name}  +  {backup_dir.name}/{stage_tag}.pt  "
                  f"({phase.tag} {si + 1}/{n_sec}, {(time.time() - t0) / 3600:.1f} h elapsed)")

        phase_finals.append((phase.tag, phase_final_path))

    # Per-lineage plots (span the whole S1→S2 curriculum; section shading marks phases).
    if not args.no_plot:
        for fn, suffix in ((plot_curriculum_dashboard, "dashboard"),
                           (plot_curriculum_reward, "reward"),
                           (plot_curriculum_eval_rates, "eval_rates")):
            try:
                fn(all_logs, section_bounds,
                   save_path=str(plots_dir / f"{lineage.name}_{suffix}.png"))
            except Exception as exc:  # noqa: BLE001 — plots must never kill training
                print(f"  plot warning ({suffix}): {type(exc).__name__}: {exc}")

    hrs = (time.time() - t0) / 3600
    print(f"\n  LINEAGE '{lineage.name}' DONE in {hrs:.1f} h  →  "
          + ", ".join(f"{tag}:{p.name}" for tag, p in phase_finals))
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
    p = argparse.ArgumentParser(description="Sequential multi-curriculum trainer: "
                                "2 lineages (complete/baseline), each trained S1→S2 "
                                "with weights carried. Edit LINEAGES above.")
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
                   help="Comma-separated LINEAGE names to execute (default: all). "
                        "E.g. --only baseline")
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

    selected = LINEAGES
    if args.only:
        wanted = {w.strip() for w in args.only.split(",")}
        selected = [lg for lg in LINEAGES if lg.name in wanted]
        unknown = wanted - {lg.name for lg in LINEAGES}
        if unknown:
            raise SystemExit(f"--only names not found: {sorted(unknown)}; "
                             f"available lineages: {[lg.name for lg in LINEAGES]}")

    if args.smoke:
        # Same worlds / num_envs (so memory is representative), tiny iterations per
        # section, '_smoke'-suffixed lineage names so real outputs stay clean.
        selected = [
            replace(lg, name=lg.name + "_smoke", phases=[
                replace(ph, curriculum=[replace(sec, n_iters=args.smoke) for sec in ph.curriculum])
                for ph in lg.phases
            ])
            for lg in selected
        ]
        print(f"\n  SMOKE MODE: every section trimmed to {args.smoke} iters.")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    plots_tag = f"final_curriculums_{DATE_TAG}" + ("_smoke" if args.smoke else "")
    plots_dir = Path(__file__).resolve().parent / "plots" / plots_tag
    plots_dir.mkdir(parents=True, exist_ok=True)

    total_iters = sum(sec.n_iters for lg in selected for ph in lg.phases for sec in ph.curriculum)
    print("\n" + "=" * 78)
    print(f"  MULTI-CURRICULUM PLAN  ({DATE_TAG})  —  {len(selected)} lineage(s) "
          f"(S1→S2 each), {total_iters} total iters")
    print(f"  num_envs {args.num_envs} | eval every {args.log_every} iters "
          f"(mirrors section DR) | seed {args.seed}")
    print(f"  checkpoints → {args.save_dir}/<lineage>_<phase>_{DATE_TAG}.pt (+ backup folders)")
    print(f"  plots → {plots_dir}")
    print(f"  rough ETA at ~9 s/iter: {total_iters * 9 / 3600:.0f} h "
          f"(baseline runs faster without FOFE)")
    print("  NOTE: disable machine sleep on AC — sleep hangs CUDA runs.")
    print("=" * 78)

    if args.dry_run:
        for lg in selected:
            phase_resolved, *_ = _resolve_lineage(lg)
            _print_lineage_plan(lg, phase_resolved, Path(args.save_dir))
        print("\n(dry run — nothing trained)")
        return

    lineage_logs: Dict[str, Dict[str, List[float]]] = {}
    failures: List[str] = []
    for lg in selected:
        try:
            logs = _train_one_lineage(lg, args, plots_dir)
            if logs is not None:
                lineage_logs[lg.name] = logs
        except Exception as exc:  # noqa: BLE001 — one failed lineage must not kill the batch
            failures.append(lg.name)
            print(f"\n  !! LINEAGE '{lg.name}' FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if lineage_logs and not args.no_plot:
        try:
            _plot_combined(lineage_logs, plots_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"combined-plot warning: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 78)
    print("  BATCH SUMMARY")
    for lg in selected:
        status = "FAILED" if lg.name in failures else ("ok" if lg.name in lineage_logs else "skipped")
        print(f"    {lg.name:<20} {status}")
    print("=" * 78)
    if failures:
        print(f"  rerun failed ones with: --only {','.join(failures)}")


if __name__ == "__main__":
    main()
