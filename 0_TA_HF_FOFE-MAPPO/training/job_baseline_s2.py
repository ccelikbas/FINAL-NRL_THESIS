"""job_baseline_s2.py — Baseline model × Scenario 2 (FOFE + comm OFF, warm-started).

Declarative Job spec only — no training logic. Imported by ../train_master.py,
which owns the engine. Not meant to be run on its own.

S2 is a CONTINUATION: it must warm-start from a checkpoint. With no explicit
checkpoint in the master's RUNS entry it CONTINUES from `warmstart_from`
("baseline_s1") — either the baseline_s1 trained earlier in the same master run,
or the newest baseline_S1 checkpoint already on disk.
"""
from __future__ import annotations

from .scenarios import Job, baseline_s2_curriculum

JOB = Job(
    key="baseline_s2",
    model="baseline",
    scenario="S2",
    use_fofe=False,
    communicate=False,
    description="Baseline model — zero-padded obs (FOFE OFF), communication OFF, S2 continuation",
    curriculum=baseline_s2_curriculum(),
    env_overrides={},
    from_scratch=False,
    warmstart_from="baseline_s1",
)
