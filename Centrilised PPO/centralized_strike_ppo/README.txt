Centralized Strike-EA PPO (simple version)

What is included
- environment.py  : copied/adapted from your current environment setup
- rewards.py      : copied reward config / shaping settings
- agents.py       : copied striker / jammer / radar helper classes
- config.py       : simple experiment/env/ppo/network configs
- models.py       : one centralized actor + one centralized critic
- trainer.py      : simplified centralized PPO training loop using TorchRL
- utils.py        : TorchRL compatibility helpers from your current setup
- run.py          : CLI entry point inside the package
- ../train_centralized_ppo.py : top-level launcher script

How to run
1) Put the folder "centralized_strike_ppo" in your VS Code workspace.
2) Make sure your environment has PyTorch + TorchRL + TensorDict installed.
3) Run either:
   python train_centralized_ppo.py
or
   python -m centralized_strike_ppo.run

Example
python train_centralized_ppo.py --n_strikers 2 --n_jammers 2 --n_targets 3 --n_radars 2 --n_iters 200

Reward tuning examples
python train_centralized_ppo.py --target_destroyed 80 --timestep_penalty -0.05 --team_spirit 0.5
python train_centralized_ppo.py --striker_progress_scale 5 --jammer_progress_scale 2 --jammer_jam_bonus 1

Notes
- The actor is fully centralized: it receives all controlled-agent observations flattened together and outputs all controlled-agent actions in one forward pass.
- The critic is also centralized and outputs one value per controlled agent from the global state.
- The environment action and observation spaces are preserved from your current setup.
