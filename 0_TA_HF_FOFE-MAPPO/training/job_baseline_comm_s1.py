"""job_baseline_comm_s1.py — ABLATION: Baseline model with COMMUNICATION ON, S1.

Single-axis communication ablation on the BASELINE architecture. Everything the
baseline_s1 job uses is kept identical — FOFE stays OFF (zero-padded obs), the
baseline_s1 curriculum/schedule is reused UNCHANGED — and ONLY inter-agent
communication is turned ON. The flip is done with `env_overrides={"communicate":
True}`, which the master applies to EVERY section after curriculum resolution
(overriding the communicate=False that baseline_s1_curriculum bakes into each
world). Trained FROM SCRATCH.

Declarative Job spec only — no training logic. Imported by ../train_master.py.
Not meant to be run on its own.
"""
from __future__ import annotations

from .scenarios import Job, S1_OVERRIDES, baseline_s1_curriculum

JOB = Job(
    key="baseline_comm_s1",
    model="baseline_comm",            # → runs/<tag>/baseline_comm_S1_FINAL.pt
    scenario="S1",
    use_fofe=False,                   # FOFE OFF / zero-padded (unchanged from baseline)
    communicate=True,                 # the ablated axis (display; env set via overrides)
    description="ABLATION: Baseline model (FOFE OFF) with COMMUNICATION ON, S1, from scratch",
    curriculum=baseline_s1_curriculum(),
    env_overrides={**S1_OVERRIDES, "communicate": True},
    from_scratch=True,
    warmstart_from=None,
)
