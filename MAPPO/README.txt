MAPPO Strike-EA (Multi-Agent PPO)

Converted from the centralized PPO implementation to MAPPO with:
  - Two decentralized actors (one per role: striker, jammer) with parameter sharing within each role
  - Two centralized value heads (one per role), both receiving the full global state

What is included
- environment.py  : vectorised Strike-EA 2D environment (unchanged from centralized version)
- rewards.py      : reward config / shaping settings (unchanged)
- agents.py       : striker / jammer / radar helper classes (unchanged)
- config.py       : experiment/env/ppo/network configs (unchanged)
- models.py       : MAPPO dual-policy actor + dual-value critic (CHANGED)
- trainer.py      : MAPPO training loop using TorchRL (minor changes)
- utils.py        : TorchRL compatibility helpers (unchanged)
- normalization.py: reward normalisation (unchanged)
- visualization.py: rollout animation and training plots (unchanged)
- run.py          : CLI entry point (updated imports)
- ../train_mappo.py : top-level launcher script

How to run
1) Put the folder "mappo_strike_ppo" in your VS Code workspace.
2) Make sure your environment has PyTorch + TorchRL + TensorDict installed.
3) Run either:
   python train_mappo.py
or
   python -m mappo_strike_ppo.run

Example
python train_mappo.py --n_strikers 2 --n_jammers 2 --n_targets 3 --n_radars 2 --n_iters 200

Key architectural differences from centralized PPO
---------------------------------------------------
1. ACTOR (DualPolicyNet):
   - Uses two TorchRL MultiAgentMLPs (centralized=False, share_params=True)
   - Strikers share one policy network; jammers share another
   - Each agent only sees its own ego-centric observation (decentralized execution)
   - Agents of the same role share parameters (homogeneous within role)

2. CRITIC (DualValueNet):
   - Two separate MLP value heads, both receiving the full global state
   - Striker head outputs n_strikers values; jammer head outputs n_jammers values
   - Concatenated to produce [B, n_agents, 1] matching the PPO loss interface
   - Centralised training: the critic sees everything, the actor sees only its own obs

3. TRAINER:
   - Identical PPO loop — the TensorDict interface is preserved so ClipPPOLoss,
     GAE, and Collector all work without changes.

Notes
- The TensorDict key layout is unchanged: ("agents", "observation"), ("agents", "action"),
  ("agents", "reward"), ("agents", "state_value"), "state", "done", "terminated".
- Checkpoints from the centralized version are NOT compatible (different model architectures).
- Requires torchrl.modules.MultiAgentMLP (standard in TorchRL >= 0.3).
