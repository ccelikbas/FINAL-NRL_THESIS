FOFE-MAPPO Strike-EA
====================

Dual-MAPPO with optional FOFE (Flexible Observation Feature Encoding)
based on Wang et al. (2026) — "Cooperative decision-making of UAVs:
A multi-agent reinforcement learning approach".

Architecture
============

When --use_fofe is passed:

  ACTOR (per role — striker / jammer):
    Self-obs [6]        → SelfMLP       → [D_self_out]
    {Agents} [n, 4]     → FOFE_agents   → [D_fofe]     (SEE stack → MLP → MaxPool)
    {Targets} [n, 3]    → FOFE_targets  → [D_fofe]
    {Radars} [n, 3]     → FOFE_radars   → [D_fofe]
                 → Concat → FusionMLP → ActionHead → logits

  CRITIC (per role — centralized):
    {Agents} [A, 7]     → FOFE_agents   → [D_fofe]     (global state, all visible)
    {Targets} [T, 3]    → FOFE_targets  → [D_fofe]
    {Radars} [R, 4]     → FOFE_radars   → [D_fofe]
    + time [1]           → Concat → FusionMLP → ValueHead → [n_role, 1]

  Both actor and critic use permutation-invariant set encoding.
  Visibility masks handle variable entity counts (no top-K, no padding).
  Known/unknown entity visibility uses communication-graph sharing.

When --use_fofe is NOT passed:
  Legacy flat-obs MultiAgentMLP (same as dual_mappo baseline).

How to run
==========

  # FOFE mode:
  python train_mappo.py --use_fofe --n_strikers 2 --n_jammers 2 --n_iters 200

  # Legacy mode (backward compatible):
  python train_mappo.py --n_strikers 2 --n_jammers 2 --n_iters 200

FOFE architecture is configured via FOFEConfig in config.py.
Default values match Wang et al.: SEE dims (96,128), FOFE MLP (512,96),
fusion MLP (256,256).

Key files changed from dual_mappo
==================================
- config.py        : Added FOFEConfig dataclass
- models.py        : Added SEELayer, FOFEBlock, FOFEPolicyNet, FOFEValueNet
- environment.py   : Added _build_fofe_obs(), _build_fofe_critic_state(), use_fofe flag
- trainer.py       : FOFE obs extraction/splitting/flattening in PPO loop
- run.py           : --use_fofe CLI flag
- visualization.py : use_fofe passthrough to TestRunner env
