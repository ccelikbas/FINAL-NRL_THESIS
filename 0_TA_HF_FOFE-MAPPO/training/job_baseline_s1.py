"""job_baseline_s1.py — Baseline model × Scenario 1 (FOFE + comm OFF, from scratch).

Declarative Job spec only — no training logic. Imported by ../train_master.py,
which owns the engine. Not meant to be run on its own.
"""
from __future__ import annotations

from .scenarios import Job, S1_OVERRIDES, baseline_s1_curriculum

JOB = Job(
    key="baseline_s1",
    model="baseline",
    scenario="S1",
    use_fofe=False,
    communicate=False,
    description="Baseline model — zero-padded obs (FOFE OFF), communication OFF, S1, from scratch",
    curriculum=baseline_s1_curriculum(),
    env_overrides=dict(S1_OVERRIDES),
    from_scratch=True,
    warmstart_from=None,
)
