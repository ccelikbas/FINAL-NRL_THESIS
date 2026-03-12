"""Single-environment rollout runner for policy evaluation."""

from __future__ import annotations

from typing import Dict, List

import torch
from torchrl.modules import ProbabilisticActor

try:
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    _EXPLORATION_API = "new"
except ImportError:
    try:
        from tensordict.nn import set_interaction_type, InteractionType
        _EXPLORATION_API = "interaction"
    except ImportError:
        _EXPLORATION_API = None

from strike_ea.env import StrikeEA2DEnv
from strike_ea.config import EnvConfig


class TestRunner:
    """Run a single deterministic rollout and collect per-step snapshots."""

    def __init__(
        self,
        policy=None,
        *,
        device:    torch.device,
        max_steps: int = 220,
        seed:      int = 123,
        env_cfg:   EnvConfig = None,
    ):
        self.device = device
        self.policy = policy.eval() if policy is not None else None
        if env_cfg is None:
            env_cfg = EnvConfig()
        self.env = StrikeEA2DEnv(
            num_envs      = 1,
            max_steps     = max_steps,
            device        = device,
            seed          = seed,
            # World / episodes
            n_strikers    = env_cfg.n_strikers,
            n_jammers     = env_cfg.n_jammers,
            n_targets     = env_cfg.n_targets,
            n_radars      = env_cfg.n_radars,
            dt            = env_cfg.dt,
            world_bounds  = env_cfg.world_bounds,
            # Kinematics
            v_max         = env_cfg.v_max,
            accel_magnitude = env_cfg.accel_magnitude,
            dpsi_max      = env_cfg.dpsi_max,
            h_accel_magnitude_fraction = env_cfg.h_accel_magnitude_fraction,
            # Sensors
            R_obs         = env_cfg.R_obs,
            # Striker
            striker_engage_range = env_cfg.striker_engage_range,
            striker_engage_fov = env_cfg.striker_engage_fov,
            striker_v_min = env_cfg.striker_v_min,
            # Jammer
            jammer_jam_radius = env_cfg.jammer_jam_radius,
            jammer_jam_effect = env_cfg.jammer_jam_effect,
            jammer_v_min = env_cfg.jammer_v_min,
            # Radar
            radar_range   = env_cfg.radar_range,
            radar_kill_probability = env_cfg.radar_kill_probability,
            # Rewards
            border_thresh = env_cfg.border_thresh,
            reward_config = env_cfg.reward_config,
            min_turn_radius = env_cfg.min_turn_radius,
            n_env_layouts = getattr(env_cfg, 'n_env_layouts', 0),
            target_spawn_angle_range = getattr(env_cfg, 'target_spawn_angle_range', (0.0, 360.0)),
        )

    @torch.no_grad()
    def rollout(self) -> List[Dict[str, torch.Tensor]]:
        td     = self.env.reset()
        frames = [self._snapshot()]

        ctx = PolicyEvaluator._deterministic_context()
        with ctx:
            for _ in range(self.env.max_steps):
                td   = self.policy(td)
                td   = self.env.step(td)
                frames.append(self._snapshot())

                done_flag = False
                try:
                    done_flag = bool(td.get(("next", "done")).item())
                except Exception:
                    try:
                        done_flag = bool(td.get("done").item())
                    except Exception:
                        pass

                if done_flag:
                    break

                td = td.get("next")

        return frames

    def _snapshot(self) -> Dict[str, torch.Tensor]:
        env = self.env
        return {
            "agent_pos":      env.agent_pos[0].detach().cpu(),
            "agent_alive":    env.agent_alive[0].detach().cpu(),
            "agent_heading":  env.agent_heading[0].detach().cpu(),
            "target_pos":     env.target_pos[0].detach().cpu(),
            "target_alive":   env.target_alive[0].detach().cpu(),
            "radar_pos":      env.radar_pos[0].detach().cpu(),
            "radar_eff_range": env.radar_eff_range[0].detach().cpu(),
        }


# ---------------------------------------------------------------------------
# Batch evaluation (post-training performance assessment)
# ---------------------------------------------------------------------------

class PolicyEvaluator:
    """Run N independent test episodes and compute aggregate performance metrics.

    Metrics reported:
    - **Task completion rate**: % of episodes where all targets were destroyed.
    - **Mean targets destroyed**: average fraction of targets killed per episode.
    - **Platform survival rate**: average fraction of agents alive at episode end.
    - **Mission duration**: mean episode length (timesteps).
    - **Mean episode reward**: average cumulative reward per episode.
    """

    def __init__(
        self,
        actor,
        *,
        device: torch.device,
        max_steps: int = 200,
        env_cfg: EnvConfig = None,
    ):
        self.actor = actor.eval()
        self.device = device
        self.max_steps = max_steps
        self.env_cfg = env_cfg if env_cfg is not None else EnvConfig()

    @torch.no_grad()
    def evaluate(self, n_episodes: int = 100, seed_offset: int = 10_000) -> Dict:
        """Run *n_episodes* rollouts and return a dict of aggregate metrics.

        Returns a dict with scalar summary stats plus a 'reward_components_per_step'
        sub-dict mapping component names to lists of length max_steps, where each
        entry is the average (over all agents and episodes) reward from that
        component at that timestep.

        Reward metrics now include both per-agent mean (comparable to training
        logs) and total team reward for completeness.
        """
        import numpy as np

        all_rewards: List[float] = []           # total team reward per episode
        all_rewards_per_agent: List[float] = []  # per-agent mean reward per episode
        targets_destroyed_frac: List[float] = []
        agents_alive_frac: List[float] = []
        durations: List[int] = []
        full_completions: int = 0

        n_targets = self.env_cfg.n_targets
        n_agents = self.env_cfg.n_strikers + self.env_cfg.n_jammers

        # Per-step component accumulators: component_name → [max_steps] running sum
        component_names = [
            "target_destroyed", "border_penalty", "timestep_penalty",
            "radar_avoidance", "striker_approach", "jammer_approach",
            "striker_progress", "jammer_progress",
            "jammer_jam_bonus", "formation", "agent_destroyed",
        ]
        step_component_sums = {name: np.zeros(self.max_steps) for name in component_names}
        step_total_sums     = np.zeros(self.max_steps)
        step_counts         = np.zeros(self.max_steps)  # how many episodes contributed to each step

        for ep in range(n_episodes):
            env = StrikeEA2DEnv(
                num_envs=1,
                max_steps=self.max_steps,
                device=self.device,
                seed=seed_offset + ep,
                n_strikers=self.env_cfg.n_strikers,
                n_jammers=self.env_cfg.n_jammers,
                n_targets=self.env_cfg.n_targets,
                n_radars=self.env_cfg.n_radars,
                dt=self.env_cfg.dt,
                world_bounds=self.env_cfg.world_bounds,
                v_max=self.env_cfg.v_max,
                accel_magnitude=self.env_cfg.accel_magnitude,
                dpsi_max=self.env_cfg.dpsi_max,
                h_accel_magnitude_fraction=self.env_cfg.h_accel_magnitude_fraction,
                R_obs=self.env_cfg.R_obs,
                striker_engage_range=self.env_cfg.striker_engage_range,
                striker_engage_fov=self.env_cfg.striker_engage_fov,
                striker_v_min=self.env_cfg.striker_v_min,
                jammer_jam_radius=self.env_cfg.jammer_jam_radius,
                jammer_jam_effect=self.env_cfg.jammer_jam_effect,
                jammer_v_min=self.env_cfg.jammer_v_min,
                radar_range=self.env_cfg.radar_range,
                radar_kill_probability=self.env_cfg.radar_kill_probability,
                border_thresh=self.env_cfg.border_thresh,
                reward_config=self.env_cfg.reward_config,
                min_turn_radius=self.env_cfg.min_turn_radius,
                n_env_layouts=getattr(self.env_cfg, 'n_env_layouts', 0),
                target_spawn_angle_range=getattr(self.env_cfg, 'target_spawn_angle_range', (0.0, 360.0)),
            )

            td = env.reset()
            cumulative_reward = 0.0       # total team reward
            per_agent_rewards = None      # [A] running sum per agent
            steps = 0

            # Use deterministic (mode/greedy) actions for evaluation
            ctx = self._deterministic_context()
            with ctx:
                for t in range(self.max_steps):
                    td = self.actor(td)
                    td = env.step(td)
                    steps += 1

                    # Accumulate per-step team reward
                    try:
                        rew = td.get(("next", env.group, "reward"))  # [1, A, 1]
                        rew_flat = rew.squeeze(0).squeeze(-1)        # [A]
                        step_reward = float(rew.sum().item())
                        cumulative_reward += step_reward
                        if per_agent_rewards is None:
                            per_agent_rewards = rew_flat.clone()
                        else:
                            per_agent_rewards += rew_flat
                        step_total_sums[t] += step_reward
                    except Exception:
                        pass

                    # Read per-component breakdown from env
                    if hasattr(env, 'last_reward_components'):
                        for name in component_names:
                            comp = env.last_reward_components.get(name)
                            if comp is not None:
                                # Sum over agents, take batch element 0
                                step_component_sums[name][t] += float(comp[0].sum().item())

                    step_counts[t] += 1

                    done_flag = False
                    try:
                        done_flag = bool(td.get(("next", "done")).item())
                    except Exception:
                        try:
                            done_flag = bool(td.get("done").item())
                        except Exception:
                            pass

                    if done_flag:
                        break
                    td = td.get("next")

            # End-of-episode stats
            targets_killed = int((~env.target_alive[0]).sum().item())
            agents_surviving = int(env.agent_alive[0].sum().item())

            all_rewards.append(cumulative_reward)
            if per_agent_rewards is not None:
                all_rewards_per_agent.append(float(per_agent_rewards.mean().item()))
            else:
                all_rewards_per_agent.append(0.0)
            targets_destroyed_frac.append(targets_killed / max(n_targets, 1))
            agents_alive_frac.append(agents_surviving / max(n_agents, 1))
            durations.append(steps)
            if targets_killed == n_targets:
                full_completions += 1

        # Aggregate scalar stats
        import statistics
        results = {
            "n_episodes":                n_episodes,
            "mean_episode_total_reward": statistics.mean(all_rewards),
            "std_reward":                statistics.pstdev(all_rewards),
            "task_completion_rate":      full_completions / n_episodes,
            "mean_targets_destroyed":    statistics.mean(targets_destroyed_frac),
            "platform_survival_rate":    statistics.mean(agents_alive_frac),
            "mean_duration":             statistics.mean(durations),
            "std_duration":              statistics.pstdev(durations),
        }

        # Average per-step reward components (only steps that had data)
        safe_counts = np.maximum(step_counts, 1)
        reward_per_step: Dict[str, List[float]] = {}
        for name in component_names:
            reward_per_step[name] = (step_component_sums[name] / safe_counts).tolist()
        reward_per_step["total"] = (step_total_sums / safe_counts).tolist()
        # Also store how many episodes were still running at each step
        reward_per_step["_episode_count"] = step_counts.tolist()

        results["reward_components_per_step"] = reward_per_step
        return results

    @staticmethod
    def _deterministic_context():
        """Return a context manager for deterministic (greedy/mode) action selection.

        Handles different TorchRL API versions:
        - Newer: set_exploration_type(ExplorationType.DETERMINISTIC)
        - Older: set_interaction_type(InteractionType.DETERMINISTIC)
        - Fallback: no-op context (stochastic sampling continues)
        """
        import contextlib
        if _EXPLORATION_API == "new":
            return set_exploration_type(ExplorationType.DETERMINISTIC)
        elif _EXPLORATION_API == "interaction":
            return set_interaction_type(InteractionType.DETERMINISTIC)
        else:
            return contextlib.nullcontext()

    @staticmethod
    def print_report(results: Dict[str, float]):
        """Pretty-print evaluation results to console."""
        print(f"\n{'='*60}")
        print(f"  Policy Evaluation Report  ({int(results['n_episodes'])} episodes)")
        print(f"  (deterministic action selection)")
        print(f"{'='*60}")
        print(f"  Mean Episode Total Reward   {results['mean_episode_total_reward']:>10.2f}  (std {results['std_reward']:.2f})")
        print(f"  Task Completion Rate ...... {results['task_completion_rate']*100:>10.1f}%  (all targets destroyed)")
        print(f"  Mean Targets Destroyed .... {results['mean_targets_destroyed']*100:>10.1f}%")
        print(f"  Platform Survival Rate .... {results['platform_survival_rate']*100:>10.1f}%  (agents alive at end)")
        print(f"  Mean Mission Duration ..... {results['mean_duration']:>10.1f}  steps  (std {results['std_duration']:.1f})")
        print(f"{'='*60}\n")
