from dataclasses import dataclass


@dataclass
class RewardConfig:
    """
    Reward weight configuration for MAPPO.
    
    CRITICAL for convergence: reward signal teaches agents optimal behavior.
    
    Design principles:
    1. SPARSE REWARD (target_destroyed): Binary task success → easy to credit assign
    2. TEAM REWARD: Shared across alive agents → encourages cooperation
    3. SHAPING REWARDS: Per-role guidance → accelerates learning (but can mislead)
    4. PENALTIES: Prevent harmful behaviors → constraints on action space
    
    Tuning guide:
    - If agents ignore the task → increase target_destroyed
    - If agents move randomly → increase shaping rewards (striker_proximity, jammer_jamming)
    - If training is unstable → reduce timestep_penalty, border_penalty (high penalties = large reward variance)
    """

    # ─── TEAM REWARDS (Shared equally across all alive agents) ────────────────
    target_destroyed: float = 10
    # Sparse reward signal when a target is eliminated (main objective)
    # Shared equally among all alive agents (promotes cooperation, no free-rider problem)
    # Formula: reward += target_destroyed × (n_targets_killed / n_alive_agents)
    # Higher = stronger emphasis on task; Lower = learn slower but more robust
    
    timestep_penalty: float = -0.02
    # Small per-step cost given to all agents every step
    # Encourages sample efficiency: "finish episodes faster = better"
    # Must be small (< 0) to avoid dominating target_destroyed signal
    # Typical: -0.001 to -0.1 (adjust if episode length is too short/long)

    border_penalty: float = -1
    # Penalty when agent enters boundary zone (50 km from edges)
    # Prevents agents from camping at map edges to avoid radar threats
    # Uses quadratic scaling: weak far from edge, very strong at edge
    # Formula: penalty = ((border_thresh - dist_to_edge) / border_thresh)^2 × border_penalty × agent_alive
    
    # ─── ROLE-SPECIFIC SHAPING (Given only to relevant role) ──────────────────
    # These are potential-based reward shaping: guide exploration toward goal states
    jammer_jamming: float = 0.5
    # Per-jammer reward for actively suppressing a radar
    # Encourages jammers to position near radars to reduce threat
    # Only jammers receive this; strikers don't
    # Lower if jammers over-focus on jamming instead of helping strikers survive
    
    striker_proximity: float = 1
    # Per-striker reward for closing distance to nearest alive target
    # Encourages strikers to navigate toward objectives
    # Only strikers receive this; jammers use this to coordinate support
    # Formula: reward = (1 - dist_to_nearest_target / max_map_distance) × striker_proximity
    # Lower if strikers are too cautious; Higher if they rush blindly

    # ─── DEATH PENALTIES (Per-agent, applied on the step the agent dies) ──────
    agent_destroyed: float = -10
    # Negative reward when an agent (striker or jammer) is killed by a radar
    # Applied ONLY to the destroyed agent in the step it dies
    # Encourages agents to avoid radar zones and develop stealthy/cautious behavior
    # Higher magnitude = stronger radar avoidance; Lower = more risk-taking