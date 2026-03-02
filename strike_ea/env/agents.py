from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class StrikerAgent:
    """Kinetic strike agent: can destroy targets inside its engagement cone."""

    engage_range:   float = 0.12
    engage_fov_deg: float = 60.0
    v_min:          float = 0.0

    def can_engage(
        self,
        rel_vec: torch.Tensor,  # (..., 2) striker→target relative vectors
        heading:  torch.Tensor,  # (...,) agent heading in radians
    ) -> torch.Tensor:          # (...,) bool
        """Return True where a target is within range AND inside the FOV cone."""
        dist = torch.linalg.norm(rel_vec, dim=-1)
        in_range = dist <= self.engage_range

        hx = torch.cos(heading)
        hy = torch.sin(heading)
        hvec = torch.stack([hx, hy], dim=-1)

        rel_hat = rel_vec / dist.unsqueeze(-1).clamp_min(1e-6)
        cos_ang = (hvec * rel_hat).sum(dim=-1).clamp(-1.0, 1.0)
        ang = torch.acos(cos_ang)

        in_fov = ang <= (math.radians(self.engage_fov_deg) * 0.5)
        return in_range & in_fov


@dataclass
class JammerAgent:
    """Electronic-attack agent: suppresses radar effective range when close enough."""

    jam_radius:  float = 0.25
    delta_range: float = 0.10  # How much radar range is reduced per jamming step
    v_min:       float = 0.01

    def jams_radar(self, rel_vec: torch.Tensor) -> torch.Tensor:
        """Return True where jammer is within jam_radius of a radar."""
        dist = torch.linalg.norm(rel_vec, dim=-1)
        return dist <= self.jam_radius


@dataclass
class RadarAgent:
    """Radar agent: detects and kills agents within effective range."""
    
    kill_probability: float = 1.0  # Probability [0, 1] that an agent in radar range is killed per step
    
    def can_kill(self, rng: torch.Generator) -> torch.Tensor:
        """Return a probabilistic kill mask based on kill_probability."""
        # This will be called with the in_radar mask to determine actual kills
        return torch.rand(1, generator=rng) < self.kill_probability
