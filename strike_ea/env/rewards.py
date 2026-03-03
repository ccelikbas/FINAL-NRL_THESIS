from dataclasses import dataclass


@dataclass
class RewardConfig:
    """
    Simplified reward weights — easy to tune from config.py or CLI.

    Team rewards are shared equally among all alive agents.
    Role-specific shaping rewards are given only to the relevant role.

    Positive values encourage behaviour; negative values discourage it.
    """

    # --- Team rewards (shared equally across all alive agents) ---
    target_destroyed: float = 10.0    # Reward when a target is killed (shared)
    timestep_penalty: float = -0.01   # Small per-step cost (encourages efficiency)
    border_penalty:   float = -1.0    # Penalty when any agent enters the boundary zone

    # --- Role-specific shaping ---
    jammer_jamming:    float = 1.0    # Per-jammer reward for actively suppressing a radar
    striker_proximity: float = 0.5    # Per-striker reward for closing distance to nearest target
