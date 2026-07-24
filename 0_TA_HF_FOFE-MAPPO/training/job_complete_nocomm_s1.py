"""job_complete_nocomm_s1.py — ABLATION: Complete model with COMMUNICATION OFF, S1.

Single-axis communication ablation on the COMPLETE architecture. Everything the
complete_s1 job uses is kept identical — FOFE stays ON, the complete_s1
curriculum/schedule is reused UNCHANGED — and ONLY inter-agent communication is
turned OFF. The flip is done with `env_overrides={"communicate": False}`, which
the master applies to EVERY section after curriculum resolution (overriding the
communicate=True that complete_s1_curriculum bakes into each world). Trained
FROM SCRATCH.

Declarative Job spec only — no training logic. Imported by ../train_master.py.
Not meant to be run on its own.
"""
from __future__ import annotations

from .scenarios import Job, S1_OVERRIDES, complete_s1_curriculum

JOB = Job(
    key="complete_nocomm_s1",
    model="complete_nocomm",          # → runs/<tag>/complete_nocomm_S1_FINAL.pt
    scenario="S1",
    use_fofe=True,                    # FOFE ON (unchanged from the complete model)
    communicate=False,                # the ablated axis (display; env set via overrides)
    description="ABLATION: Complete model (FOFE ON) with COMMUNICATION OFF, S1, from scratch",
    curriculum=complete_s1_curriculum(),
    env_overrides={**S1_OVERRIDES, "communicate": False},
    from_scratch=True,
    warmstart_from=None,
)
