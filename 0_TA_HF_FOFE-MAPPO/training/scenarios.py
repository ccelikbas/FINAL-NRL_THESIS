"""scenarios.py — the two scenarios (S1, S2), the four curriculums, and the
shared `Job` type used by the four job specs and by train_master.py.

SCENARIOS (worlds)
  S1 : all-known threat — 2-4 known targets, 4-6 known radars, no hidden entities.
  S2 : partial-obs threat — 1-2 known + 1-2 hidden targets, 3-4 known + 1-2 hidden
       radars. (Both set the EnvConfig `scenario` field to "S2"; the S1/S2
       distinction is carried by the known/unknown ENTITY COUNTS, matching how
       run_multiple_curriculums.py defined them.)

FOUR CURRICULUMS (one per model × scenario) — each is its OWN independent,
separately-editable section list, so the complete and baseline schedules can
DIVERGE within a scenario (e.g. the baseline, lacking FOFE + communication,
often needs a gentler / longer anneal). They currently all start from the same
V7-proven shape:
  complete_s1_curriculum / baseline_s1_curriculum : fixed 2s2j warmup, then DR
       jammers (2,4) while the radar-kill anneals up to 0.25. Trained FROM SCRATCH.
  complete_s2_curriculum / baseline_s2_curriculum : S2 CONTINUATION — no warmup
       (escort already learned in S1), DR jammers (2,4), kill re-ramped from a
       mid value back to 0.25. Trained FROM A CHECKPOINT (see train_master).
Edit any one WITHOUT touching the other three.

What is SHARED per scenario is only the WORLD (entity counts: S1_WORLD / S2_WORLD)
— complete and baseline must train/evaluate on the SAME scenario distribution for
a fair ablation. What differs is the schedule (iters, warmup, kill anneal) and the
model flags (use_fofe, communicate). Each curriculum bakes its own `communicate`.

The S1 and S2 worlds keep matching ENTITY TOTALS (6 agents / 4 targets / 6
radars at their range maxima), so a baseline (non-FOFE) S1→S2 warm-start
transfers the critic strictly, not just the actor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# CurriculumSection is defined once in run_curriculum.py (package root); reuse it
# so a section here behaves identically to a run_curriculum / run_multiple one.
from ..run_curriculum import CurriculumSection


# =====================================================================
#  SCENARIO WORLDS
# =====================================================================
#   S1: 4-6 known radars, 2-4 known targets (all known, no hidden entities).
#   S2: 3-4 known + 1-2 hidden radars, 1-2 known + 1-2 hidden targets.
S1_WORLD: Dict[str, Any] = dict(
    scenario="S2",
    n_known_targets=(2, 4), n_unknown_targets=0,
    n_known_radars=(4, 6), n_unknown_radars=0,
)
S2_WORLD: Dict[str, Any] = dict(
    scenario="S2",
    n_known_targets=(1, 2), n_unknown_targets=(1, 2),
    n_known_radars=(3, 4), n_unknown_radars=(1, 2),
)

# S1's default radar_min_sep (0.5) cannot fit >2 radars in the S1 spawn band
# [0.2,0.8]x[0.6,0.8] — layout pregeneration hard-fails. 0.15 is the largest
# separation that reliably pregenerates 1024 layouts with 7 radars (0.20 fails;
# if you raise the max to 8 radars, drop this to 0.12). Applied to every S1
# section via the job's env_overrides. S2 uses the EnvConfig default.
S1_OVERRIDES: Dict[str, Any] = {"radar_min_sep": 0.15}


# =====================================================================
#  THE FOUR CURRICULUMS  —  edit each INDEPENDENTLY
# =====================================================================
#  Each function below is a standalone section list for one (model × scenario).
#  complete_* and baseline_* are SEPARATE on purpose: change one model's schedule
#  (iters / warmup / kill anneal) without touching the other's. Only the scenario
#  WORLD (S1_WORLD / S2_WORLD, spread via `world`) and `max_steps` are shared —
#  keep those identical across a scenario's two models so the ablation is fair.
#  `communicate` is baked per curriculum (ON for complete, OFF for baseline).
#
#  They currently all start from the same V7-proven shape; diverge as needed.


def complete_s1_curriculum() -> List[CurriculumSection]:
    """COMPLETE × S1 (FOFE + comm ON), FROM SCRATCH. Fixed 2s2j warmup @kill0.025,
    then DR jammers (2,4) while the radar kill anneals 0.025 → 0.25."""
    world = dict(**S1_WORLD, communicate=True, max_steps=150)
    return [
        CurriculumSection(name="warmup 2s2j k0.025", n_iters=250,
                          n_strikers=2, n_jammers=2,
                          radar_kill_probability=0.025, **world),
        CurriculumSection(name="DR j2-4 k0.025", n_iters=750,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.025, **world),  
        CurriculumSection(name="DR j2-4 k0.05", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.05, **world), 
        CurriculumSection(name="DR j2-4 k0.1", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.1, **world), 
        CurriculumSection(name="DR j2-4 k0.15", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.15, **world),
        CurriculumSection(name="DR j2-4 k0.2", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.2, **world),     # 5000 iters to 0.25
        CurriculumSection(name="DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world), 
        CurriculumSection(name="DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world), 
        CurriculumSection(name="DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),
        CurriculumSection(name="DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),
        CurriculumSection(name="DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),    # 10000 iters total
    ]


def baseline_s1_curriculum() -> List[CurriculumSection]:
    """BASELINE × S1 (zero-padded obs, comm OFF), FROM SCRATCH. Same world as
    complete_s1; schedule kept independent so you can lengthen / soften the anneal
    for the harder no-FOFE, no-comm model without affecting the complete run."""
    world = dict(**S1_WORLD, communicate=False, max_steps=150)
    return [
        CurriculumSection(name="warmup 2s2j k0.025", n_iters=250,
                          n_strikers=2, n_jammers=2,
                          radar_kill_probability=0.025, **world),
        CurriculumSection(name="DR j2-4 k0.025", n_iters=750,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.025, **world),
        CurriculumSection(name="DR j2-4 k0.05", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.05, **world),
        CurriculumSection(name="DR j2-4 k0.1", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.1, **world),
        CurriculumSection(name="DR j2-4 k0.15", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.15, **world),
        CurriculumSection(name="DR j2-4 k0.2", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.2, **world), # 5000 iters to 0.25
        CurriculumSection(name="DR j2-4 k0.25", n_iters=5000,   # start 0.25 kill rate
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world), # 10_000 itters total
        CurriculumSection(name="DR j2-4 k0.25", n_iters=5000,
                          n_strikers=2, n_jammers=(2, 4), 
                          radar_kill_probability=0.25, **world), # 15_000 itters total
        CurriculumSection(name="DR j2-4 k0.25", n_iters=5000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world), # 20_000 itters total
        CurriculumSection(name="DR j2-4 k0.25", n_iters=5000,
                          n_strikers=2, n_jammers=(2, 4), 
                          radar_kill_probability=0.25, **world), # 25_000 itters total
        CurriculumSection(name="DR j2-4 k0.25", n_iters=5000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world), # 30_000 itters total
    ]


def complete_s2_curriculum() -> List[CurriculumSection]:
    """COMPLETE × S2 (FOFE + comm ON), CONTINUATION from the S1 checkpoint. No
    2s2j warmup (escort already learned in S1); DR jammers (2,4); radar kill
    re-ramped from a mid value back to the S1 target (0.25)."""
    world = dict(**S2_WORLD, communicate=True, max_steps=150)
    return [
        CurriculumSection(name="S2 DR j2-4 k0.05", n_iters=250,
                          n_strikers=2, n_jammers=2,
                          radar_kill_probability=0.05, **world),
        CurriculumSection(name="S2 DR j2-4 k0.05", n_iters=750,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.05, **world),
        CurriculumSection(name="S2 DR j2-4 k0.1", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.1, **world),
        CurriculumSection(name="S2 DR j2-4 k0.15", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.15, **world),
        CurriculumSection(name="S2 DR j2-4 k0.2", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.2, **world),     # 4000 iters to 0.25
        CurriculumSection(name="S2 DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),    # 5000 iters
        CurriculumSection(name="S2 DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),
        CurriculumSection(name="S2 DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world), 
        CurriculumSection(name="S2 DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),
        CurriculumSection(name="S2 DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),
        CurriculumSection(name="S2 DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),    # 10000 iters total
    ]


def baseline_s2_curriculum() -> List[CurriculumSection]:
    """BASELINE × S2 (zero-padded obs, comm OFF), CONTINUATION from the S1
    checkpoint. Same world as complete_s2; schedule independent so the harder
    no-FOFE, no-comm model can be tuned separately."""
    world = dict(**S2_WORLD, communicate=False, max_steps=150)
    return [
        CurriculumSection(name="S2 DR j2-4 k0.05", n_iters=250,
                          n_strikers=2, n_jammers=2,
                          radar_kill_probability=0.05, **world),
        CurriculumSection(name="S2 DR j2-4 k0.05", n_iters=750,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.05, **world),
        CurriculumSection(name="S2 DR j2-4 k0.1", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.1, **world),
        CurriculumSection(name="S2 DR j2-4 k0.15", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.15, **world),
        CurriculumSection(name="S2 DR j2-4 k0.2", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.2, **world),     # 4000 iters to 0.25
        CurriculumSection(name="S2 DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),    # 5000 iters
        CurriculumSection(name="S2 DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),
        CurriculumSection(name="S2 DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world), 
        CurriculumSection(name="S2 DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),
        CurriculumSection(name="S2 DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),
        CurriculumSection(name="S2 DR j2-4 k0.25", n_iters=1000,
                          n_strikers=2, n_jammers=(2, 4),
                          radar_kill_probability=0.25, **world),    # 10000 iters total
    ]


# =====================================================================
#  SHARED JOB TYPE  (one (model × scenario) training unit)
# =====================================================================

@dataclass
class Job:
    """A single (model × scenario) training unit — declarative, no logic.

    key            : short id used in the master's RUNS list + output filenames
                     (e.g. "complete_s1").
    model          : "complete" | "baseline"  (checkpoint / folder prefix).
    scenario       : "S1" | "S2"              (checkpoint / folder suffix).
    use_fofe       : FOFE encoder on (complete) or off (baseline, zero-padded obs).
    communicate    : inter-agent communication on/off (must match the curriculum).
    curriculum     : the ordered CurriculumSection list to train through.
    env_overrides  : extra EnvConfig field overrides applied to EVERY section
                     (fields CurriculumSection cannot express, e.g. radar_min_sep).
    from_scratch   : True  → S1, trained from random init.
                     False → S2, REQUIRES a warm-start checkpoint (see
                     warmstart_from + the master's checkpoint resolution).
    warmstart_from : the job key an S2 job CONTINUES from when no explicit
                     checkpoint is given (e.g. complete_s2 ← "complete_s1").
    """
    key: str
    model: str
    scenario: str
    use_fofe: bool
    communicate: bool
    description: str
    curriculum: List[CurriculumSection] = field(default_factory=list)
    env_overrides: Dict[str, Any] = field(default_factory=dict)
    from_scratch: bool = True
    warmstart_from: Optional[str] = None
