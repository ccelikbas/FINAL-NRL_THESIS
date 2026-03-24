Dual-MAPPO Strike-EA (Two Independent Multi-Agent PPO Instances)

Refactored from the single MAPPO with dual heads to two fully independent
MAPPO instances — one for strikers, one for jammers.

Architecture
============

STRIKER MAPPO:
  - Actor:  RolePolicyNet  (MultiAgentMLP, share_params=True among strikers)
  - Critic: RoleValueNet   (MLP, centralized — receives full global state)
  - Own PPO clip loss, own GAE, own Adam optimizers

JAMMER MAPPO:
  - Actor:  RolePolicyNet  (MultiAgentMLP, share_params=True among jammers)
  - Critic: RoleValueNet   (MLP, centralized — receives full global state)
  - Own PPO clip loss, own GAE, own Adam optimizers

SHARED INFRASTRUCTURE:
  - One environment (StrikeEA2DEnv) producing combined observations/rewards
  - One TorchRL Collector using CombinedPolicy wrapper
  - CombinedPolicy: routes observations to role-specific actors, merges actions
  - CombinedCritic: runs both critics, merges value estimates
  - Shared hyperparameters (lr, clip_eps, gamma, etc.) from a single PPOConfig
  - Shared reward structure (unchanged from original)

What is included
================
- environment.py   : vectorised Strike-EA 2D environment (UNCHANGED)
- rewards.py       : reward config / shaping settings (UNCHANGED)
- agents.py        : striker / jammer / radar helper classes (UNCHANGED)
- config.py        : experiment/env/ppo/network configs (UNCHANGED)
- normalization.py : reward normalisation (UNCHANGED)
- models.py        : RolePolicyNet, RoleValueNet, CombinedPolicy, CombinedCritic (NEW)
- trainer.py       : Dual-MAPPO training loop with manual PPO loss per role (REWRITTEN)
- utils.py         : Manual GAE, PPO-clip loss, collector helper (REWRITTEN)
- visualization.py : Rollout animation + per-role training plots (UPDATED)
- run.py           : CLI entry point (UPDATED for dual checkpoint format)
- train_mappo.py   : Top-level launcher script (UNCHANGED)

Key differences from single MAPPO
==================================
1. ACTOR: Two completely independent RolePolicyNet instances (not a single
   DualPolicyNet). Each role's agents share parameters within the role, but
   the two roles have NO shared parameters.

2. CRITIC: Two completely independent RoleValueNet instances. Both receive
   the full global state (centralized training), but produce values only
   for their own role's agents.

3. TRAINING: After the Collector yields a batch, rewards/observations/
   values are split by role. GAE is computed independently per role.
   PPO-clip loss is computed and backpropagated independently per role.
   Each role has its own actor optimizer and critic optimizer.

4. LOGGING: Policy loss, value loss, entropy, KL, clip ratio, and
   explained variance are logged separately for striker and jammer,
   plus combined averages for convenience. The training printout shows
   S[...] for striker and J[...] for jammer diagnostics side-by-side.

5. PLOTTING: 3×3 grid with separate panels for striker and jammer
   losses, diagnostics (entropy/KL/clip/EV), plus combined overview.

6. CHECKPOINTS: Saved under "policy_state_dict" and "critic_state_dict"
   (wrapping both roles). NOT compatible with old single-MAPPO checkpoints.

How to run
==========
1) Put the folder "MAPPO" in your workspace.
2) Make sure your environment has PyTorch + TorchRL + TensorDict installed.
3) Run either:
   python train_mappo.py
or
   python -m MAPPO.run

Example
=======
python train_mappo.py --n_strikers 2 --n_jammers 2 --n_targets 3 --n_radars 2 --n_iters 200

Adapting team composition:
  --n_strikers N    Number of striker agents (share one policy network)
  --n_jammers  N    Number of jammer agents  (share one policy network)
  --n_targets  N    Number of targets
  --n_radars   N    Number of radar threats
