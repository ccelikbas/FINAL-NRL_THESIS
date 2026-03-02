"""Single-environment rollout runner for policy evaluation."""

from __future__ import annotations

from typing import Dict, List

import torch
from torchrl.modules import ProbabilisticActor

from strike_ea.env import StrikeEA2DEnv
from strike_ea.config import EnvConfig


class TestRunner:
    """Run a single deterministic rollout and collect per-step snapshots."""

    def __init__(
        self,
        policy:    ProbabilisticActor,
        *,
        device:    torch.device,
        max_steps: int = 220,
        seed:      int = 123,
        env_cfg:   EnvConfig = None,
    ):
        self.device = device
        self.policy = policy.eval()
        if env_cfg is None:
            env_cfg = EnvConfig()
        self.env = StrikeEA2DEnv(
            num_envs      = 1,
            max_steps     = max_steps,
            device        = device,
            seed          = seed,
            n_strikers    = env_cfg.n_strikers,
            n_jammers     = env_cfg.n_jammers,
            n_targets     = env_cfg.n_targets,
            n_radars      = env_cfg.n_radars,
            dt            = env_cfg.dt,
            world_bounds  = env_cfg.world_bounds,
            v_max         = env_cfg.v_max,
            dpsi_max      = env_cfg.dpsi_max,
            R_obs         = env_cfg.R_obs,
            radar_range   = env_cfg.radar_range,
            border_thresh = env_cfg.border_thresh,
            reward_config = env_cfg.reward_config,
        )

    @torch.no_grad()
    def rollout(self) -> List[Dict[str, torch.Tensor]]:
        td     = self.env.reset()
        frames = [self._snapshot()]

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
