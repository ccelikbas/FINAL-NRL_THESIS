"""job_complete_s2.py — Complete model × Scenario 2 (FOFE + comm ON, warm-started).

Declarative Job spec only — no training logic. Imported by ../train_master.py,
which owns the engine. Not meant to be run on its own.

S2 is a CONTINUATION: it must warm-start from a checkpoint. With no explicit
checkpoint in the master's RUNS entry it CONTINUES from `warmstart_from`
("complete_s1") — either the complete_s1 trained earlier in the same master run,
or the newest complete_S1 checkpoint already on disk.
"""
from __future__ import annotations

from .scenarios import Job, complete_s2_curriculum

JOB = Job(
    key="complete_s2",
    model="complete",
    scenario="S2",
    use_fofe=True,
    communicate=True,
    description="Complete model — FOFE + communication ON, S2 (partial-obs) continuation",
    curriculum=complete_s2_curriculum(),
    env_overrides={},
    from_scratch=False,
    warmstart_from="complete_s1",
)
