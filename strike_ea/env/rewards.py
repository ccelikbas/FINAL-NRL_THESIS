from dataclasses import dataclass


@dataclass
class RewardConfig:
    """
    All reward signal weights in one place.

    Positive values encourage behaviour; negative values discourage it.
    """

    # Terminal events
    target_destroyed: float = 10.0   # Per target killed by a striker
    agent_destroyed:  float = -10.0  # Per own agent lost to radar

    # Per-step shaping
    move_closer: float = 1.0   # Reward proportional to proximity to nearest alive target
    jamming:     float = 1.0   # Bonus per jammer actively suppressing a radar
    border:      float = -1.0  # Penalty when agent is near the world boundary
    small_step:  float = -0.01 # Tiny penalty every step (encourages efficiency)
