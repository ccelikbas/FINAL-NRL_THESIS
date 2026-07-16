"""job_complete_s1.py — Complete model × Scenario 1 (FOFE + comm ON, from scratch).

Declarative Job spec only — no training logic. Imported by ../train_master.py,
which owns the engine. Not meant to be run on its own.
"""
from __future__ import annotations

from .scenarios import Job, S1_OVERRIDES, complete_s1_curriculum

JOB = Job(
    key="complete_s1",
    model="complete",
    scenario="S1",
    use_fofe=True,
    communicate=True,
    description="Complete model — FOFE + communication ON, S1 (all-known), from scratch",
    curriculum=complete_s1_curriculum(),
    env_overrides=dict(S1_OVERRIDES),
    from_scratch=True,
    warmstart_from=None,
)
