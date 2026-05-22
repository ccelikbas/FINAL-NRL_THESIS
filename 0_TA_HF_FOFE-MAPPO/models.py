"""
Dual-MAPPO model architecture with optional FOFE observation encoder.

FOFE mode (use_fofe=True):
  Actor receives structured per-channel observations → FOFE encoder → action.
  Critic receives global state as entity sets         → FOFE encoder → value.

Legacy mode (use_fofe=False):
  Actor: flat top-K observation → MultiAgentMLP → action.
  Critic: flat global state vector → MLP → value.

FOFE Actor pipeline:
  ┌─ o_self   [B,A,6]     → SelfMLP      → [D_self_out]
  ├─ {agents} [B,A,E_a,4] → FOFE_agents  → [D_fofe]
  ├─ {targets}[B,A,E_t,3] → FOFE_targets → [D_fofe]
  └─ {radars} [B,A,E_r,3] → FOFE_radars  → [D_fofe]
       → Concat → FusionMLP → ActionHead → [B,A,act_dim,n_choices]

FOFE Critic pipeline (same structure, different weights & entity dims):
  ┌─ time_feat [B,1]           → passed through directly
  ├─ {agents}  [B,A,7]        → FOFE_agents  → [D_fofe]   (global: all agents, all visible)
  ├─ {targets} [B,T,3]        → FOFE_targets → [D_fofe]
  └─ {radars}  [B,R,4]        → FOFE_radars  → [D_fofe]
       → Concat → FusionMLP → ValueHead → [B,n_role,1]
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.modules import MultiAgentMLP

from .config import FOFEConfig
from .environment import StrikeEA2DEnv


# ======================================================================
# Shared distribution (unchanged)
# ======================================================================

class MultiCategorical(torch.distributions.Distribution):
    arg_constraints = {}
    has_rsample = False

    @staticmethod
    def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        return logits.clamp(min=-20.0, max=20.0)

    def __init__(self, logits: torch.Tensor):
        logits = self._sanitize_logits(logits)
        self._cats = torch.distributions.Categorical(logits=logits)
        batch_shape = logits.shape[:-2]
        super().__init__(batch_shape=batch_shape, validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        return self._cats.sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._cats.log_prob(value).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self._cats.entropy().sum(dim=-1)

    @property
    def mode(self) -> torch.Tensor:
        return self._cats.logits.argmax(dim=-1)

    @property
    def deterministic_sample(self) -> torch.Tensor:
        return self.mode


# ======================================================================
# Generic MLP
# ======================================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int):
        super().__init__()
        layers = []
        cur = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(cur, hidden), nn.ReLU()])
            cur = hidden
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# FOFE building blocks  (Wang et al., 2026 — Equations 25-29)
# ======================================================================

class SEELayer(nn.Module):
    """Sequence-Exchangeable Encoding layer (Eq. 26-29).

    Permutation-invariant per-entity transform with two branches:
      - Separated: per-entity gated embedding  (local context)
      - Aggregated: maxpooled embedding broadcast back & gated (global context)

    Dimension flow for hidden dim h:
        Input:  [*, n, d_prev]
        Step 1: LayerNorm          → [*, n, d_prev]     (skip for first layer)
        Step 2: Linear_embed/gate  → [*, n, h] each
        Step 3: x_sep = SiLU(gate) ⊙ embed              [*, n, h]
                x_agg = SiLU(gate) ⊙ Broadcast(MaxPool(embed))  [*, n, h]
        Step 4: Cat([x_sep, x_agg]) → [*, n, 2h]

    Parameters
    ----------
    d_in : int         Input feature dim.
    h : int            Hidden dim for projections.  Output = 2*h.
    use_layernorm : bool   False for first SEE layer (preserve raw input).
    """

    def __init__(self, d_in: int, h: int, use_layernorm: bool = True):
        super().__init__()
        self.linear_embed = nn.Linear(d_in, h)   # Eq. 27 — embedding projection
        self.linear_gate = nn.Linear(d_in, h)    # Eq. 27 — gating projection
        self.norm = nn.LayerNorm(d_in) if use_layernorm else nn.Identity()  # Eq. 26

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : [*, n, d_in]   entity features (masked rows should be zero)
        mask : [*, n]          True = visible entity
        Returns [*, n, 2*h]
        """
        # ---- Eq. 26: LayerNorm (skipped for layer 1) ----
        x_norm = self.norm(x)

        # ---- Eq. 27: shared linear projections ----
        x_embed = self.linear_embed(x_norm)   # [*, n, h]
        z_gate = self.linear_gate(x_norm)     # [*, n, h]

        # ---- Eq. 28: gating + two-branch encoding ----
        gate_act = F.silu(z_gate)             # smooth gating activation

        # Separated branch: per-entity gated embedding
        x_sep = gate_act * x_embed            # [*, n, h]

        # Aggregated branch: maxpool over entities, then broadcast & gate
        mask_exp = mask.unsqueeze(-1)         # [*, n, 1]
        x_embed_masked = x_embed.masked_fill(~mask_exp, float('-inf'))
        x_agg_global = x_embed_masked.max(dim=-2).values  # [*, h]
        # Handle all-masked (no visible entities)
        all_masked = ~mask.any(dim=-1, keepdim=True)       # [*, 1]
        x_agg_global = x_agg_global.masked_fill(all_masked, 0.0)
        x_agg = gate_act * x_agg_global.unsqueeze(-2).expand_as(x_embed)  # [*, n, h]

        # ---- Eq. 29: concatenate separated + aggregated ----
        out = torch.cat([x_sep, x_agg], dim=-1)  # [*, n, 2*h]
        out = torch.where(mask.unsqueeze(-1), out, torch.zeros_like(out))  # zero masked rows (avoids NaN * 0 = NaN)
        return out


class FOFEBlock(nn.Module):
    """FOFE module: SEE stack → per-entity MLP → global MaxPool (Eq. 25).

    Converts a variable-size entity set into a single fixed-size vector.

    Dimension trace (d_in=4, see_dims=(96,128), mlp_dims=(512,96)):
        [n, 4] → SEE1 → [n, 192] → SEE2 → [n, 256]
               → MLP  → [n, 512] → [n, 96]
               → MaxPool → [96]

    Parameters
    ----------
    d_in      : per-entity raw feature dim
    see_dims  : hidden dim per SEE layer; len = number of SEE layers
    mlp_dims  : per-entity MLP dims; last = D_fofe output dim
    """

    def __init__(self, d_in: int, see_dims: Tuple[int, ...], mlp_dims: Tuple[int, ...]):
        super().__init__()
        # ---- SEE layer stack ----
        self.see_layers = nn.ModuleList()
        d_prev = d_in
        for i, h in enumerate(see_dims):
            self.see_layers.append(SEELayer(d_prev, h, use_layernorm=(i > 0)))
            d_prev = 2 * h  # output of SEE_i is [n, 2*h_i]

        # ---- Per-entity MLP (shared across rows via nn.Linear) ----
        layers = []
        cur = d_prev
        for dim in mlp_dims:
            layers.extend([nn.Linear(cur, dim), nn.ReLU()])
            cur = dim
        self.entity_mlp = nn.Sequential(*layers)
        self.output_dim = mlp_dims[-1]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : [*, n, d_in]   (masked rows zeroed)
        mask : [*, n]          True = visible
        Returns [*, D_fofe]
        """
        # Ensure masked rows are exactly zero (guards against residual NaN from inf*0)
        x = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))
        for see in self.see_layers:
            x = see(x, mask)
        x = self.entity_mlp(x)                              # [*, n, D_fofe]
        mask_exp = mask.unsqueeze(-1)
        x_masked = x.masked_fill(~mask_exp, float('-inf'))
        out = x_masked.max(dim=-2).values                    # [*, D_fofe]
        all_masked = ~mask.any(dim=-1, keepdim=True)
        out = out.masked_fill(all_masked, 0.0)
        return out


# ======================================================================
# FOFE-based actor (one per role)
# ======================================================================

class FOFEPolicyNet(nn.Module):
    """FOFE-based decentralized actor for one role.

    Processes 4 observation channels:
      1. Self-obs MLP    (fixed-size: d_self features; base=6, HF=+2 beam state)
      2. FOFE agents     (variable-size set: 5 features per entity
                          — dx_body, dy_body, dist, heading, role)
      3. FOFE targets    (variable-size set: 4 features per entity
                          — dx_body, dy_body, dist, alive)
      4. FOFE radars     (variable-size set: 4+E features per entity
                          — dx_body, dy_body, dist, jammed, *radar_extra;
                          HF env appends a signed beam→radar angle)
    → concatenate → fusion MLP → action head

    All agents of the same role share these weights (parameter sharing).
    """

    def __init__(self, fofe_cfg: FOFEConfig, n_agents: int,
                 act_dim: int, n_choices: int,
                 d_self: int = 6, d_agents: int = 5,
                 d_targets: int = 4, d_radars: int = 4):
        super().__init__()
        self.act_dim = act_dim
        self.n_choices = n_choices
        self.n_agents = n_agents

        # ---- Self-obs MLP ----
        self_layers = []
        cur = d_self
        for dim in fofe_cfg.self_mlp_dims:
            self_layers.extend([nn.Linear(cur, dim), nn.ReLU()])
            cur = dim
        self.self_mlp = nn.Sequential(*self_layers)
        d_self_out = fofe_cfg.self_mlp_dims[-1]

        # ---- FOFE blocks (one per variable-length channel) ----
        self.fofe_agents = FOFEBlock(d_agents, fofe_cfg.agents_see_dims, fofe_cfg.fofe_mlp_dims)
        self.fofe_targets = FOFEBlock(d_targets, fofe_cfg.targets_see_dims, fofe_cfg.fofe_mlp_dims)
        self.fofe_radars = FOFEBlock(d_radars, fofe_cfg.radars_see_dims, fofe_cfg.fofe_mlp_dims)
        d_fofe = fofe_cfg.fofe_mlp_dims[-1]

        # ---- Fusion MLP ----
        d_fuse_in = d_self_out + 3 * d_fofe
        fuse_layers = []
        cur = d_fuse_in
        for dim in fofe_cfg.fusion_mlp_dims:
            fuse_layers.extend([nn.Linear(cur, dim), nn.ReLU()])
            cur = dim
        self.fusion_mlp = nn.Sequential(*fuse_layers)

        # ---- Action head ----
        self.action_head = nn.Linear(cur, act_dim * n_choices)

        # Diagnostic cache (populated each forward, detached, zero cost when unused)
        self._diag_cache: Dict[str, torch.Tensor] = {}

    def forward(self, obs_self, agents_feat, agents_mask,
                targets_feat, targets_mask, radars_feat, radars_mask):
        """
        All inputs: [B, n_role, ...].
        Returns logits [B, n_role, act_dim, n_choices].
        """
        B, A = obs_self.shape[:2]
        # Flatten B×A for shared-weight processing
        sf = obs_self.reshape(B * A, -1)
        af = agents_feat.reshape(B * A, *agents_feat.shape[2:])
        am = agents_mask.reshape(B * A, *agents_mask.shape[2:])
        tf = targets_feat.reshape(B * A, *targets_feat.shape[2:])
        tm = targets_mask.reshape(B * A, *targets_mask.shape[2:])
        rf = radars_feat.reshape(B * A, *radars_feat.shape[2:])
        rm = radars_mask.reshape(B * A, *radars_mask.shape[2:])

        # ---- Process each channel ----
        x_self = self.self_mlp(sf)           # [B*A, D_self_out]
        x_agents = self.fofe_agents(af, am)  # [B*A, D_fofe]
        x_targets = self.fofe_targets(tf, tm)
        x_radars = self.fofe_radars(rf, rm)

        # Cache per-channel outputs for diagnostics (detached = no graph impact)
        self._diag_cache = {
            "x_agents": x_agents.detach(),
            "x_targets": x_targets.detach(),
            "x_radars": x_radars.detach(),
        }

        # ---- Fuse & predict ----
        x = torch.cat([x_self, x_agents, x_targets, x_radars], dim=-1)
        x = self.fusion_mlp(x)
        logits = self.action_head(x)
        return logits.reshape(B, A, self.act_dim, self.n_choices)


# ======================================================================
# FOFE-based critic (one per role)
# ======================================================================

class FOFEValueNet(nn.Module):
    """FOFE-based centralized critic for one role.

    Receives global state decomposed into entity sets (all visible):
      - agent set   [B, A, 7]   (all agents — pos, speed, heading, omega, role, alive)
      - target set  [B, T, 3]   (all targets — pos, alive)
      - radar set   [B, R, 3]   (all radars — pos, jammed)
      - time feat   [B, 1]      (scalar)
    Each set passes through its own FOFE block, results are concatenated
    with time feature, fused, and projected to per-agent values.

    Parameters
    ----------
    fofe_cfg : FOFEConfig
    n_role_agents : int   Number of agents in this role (kept for API compatibility).
    d_agents, d_targets, d_radars : int  Per-entity feature dims.
    """

    def __init__(self, fofe_cfg: FOFEConfig, n_role_agents: int,
                 d_agents: int = 7, d_targets: int = 3, d_radars: int = 3):
        super().__init__()
        self.n_role_agents = n_role_agents

        # ---- FOFE blocks for global state channels ----
        self.fofe_agents = FOFEBlock(d_agents, fofe_cfg.critic_agents_see_dims, fofe_cfg.critic_fofe_mlp_dims)
        self.fofe_targets = FOFEBlock(d_targets, fofe_cfg.critic_targets_see_dims, fofe_cfg.critic_fofe_mlp_dims)
        self.fofe_radars = FOFEBlock(d_radars, fofe_cfg.critic_radars_see_dims, fofe_cfg.critic_fofe_mlp_dims)
        d_fofe = fofe_cfg.critic_fofe_mlp_dims[-1]

        # ---- Fusion MLP ----
        # Input: 3 × D_fofe + 1 (time)
        d_fuse_in = 3 * d_fofe + 1
        fuse_layers = []
        cur = d_fuse_in
        for dim in fofe_cfg.critic_fusion_mlp_dims:
            fuse_layers.extend([nn.Linear(cur, dim), nn.ReLU()])
            cur = dim
        self.fusion_mlp = nn.Sequential(*fuse_layers)

        # ---- Value head: one scalar per role (shared baseline) ----
        self.value_head = nn.Linear(cur, 1)

        # Diagnostic cache (populated each forward, detached, zero cost when unused)
        self._diag_cache: Dict[str, torch.Tensor] = {}

    def forward(self, agent_feat, agent_mask, target_feat, target_mask,
                radar_feat, radar_mask, time_feat):
        """
        All entity inputs: [B, E, d].  Masks: [B, E] bool.
        time_feat: [B, 1].
        Returns [B, 1, 1].
        """
        x_a = self.fofe_agents(agent_feat, agent_mask)    # [B, D_fofe]
        x_t = self.fofe_targets(target_feat, target_mask)  # [B, D_fofe]
        x_r = self.fofe_radars(radar_feat, radar_mask)     # [B, D_fofe]

        # Cache per-channel outputs for diagnostics
        self._diag_cache = {
            "x_agents": x_a.detach(),
            "x_targets": x_t.detach(),
            "x_radars": x_r.detach(),
        }

        x = torch.cat([x_a, x_t, x_r, time_feat], dim=-1)  # [B, 3*D_fofe+1]
        x = self.fusion_mlp(x)
        return self.value_head(x).unsqueeze(-1)  # [B, 1, 1]


# ======================================================================
# Legacy networks (unchanged, for use_fofe=False)
# ======================================================================

class RolePolicyNet(nn.Module):
    """Legacy flat-obs actor with MultiAgentMLP."""
    def __init__(self, obs_dim, act_dim, n_choices, n_agents, hidden=256, depth=3):
        super().__init__()
        self.act_dim = act_dim
        self.n_choices = n_choices
        self.net = MultiAgentMLP(
            n_agent_inputs=obs_dim, n_agent_outputs=act_dim * n_choices,
            n_agents=n_agents, centralized=False, share_params=True,
            depth=depth, num_cells=hidden, activation_class=nn.ReLU,
        )

    def forward(self, obs):
        raw = self.net(obs)
        B, A, _ = raw.shape
        return raw.view(B, A, self.act_dim, self.n_choices)


class RoleValueNet(nn.Module):
    """Legacy flat-state critic."""
    def __init__(self, state_dim, n_agents, hidden=256, depth=3):
        super().__init__()
        self.n_agents = n_agents
        self.net = MLP(state_dim, 1, hidden, depth)

    def forward(self, state):
        return self.net(state).unsqueeze(-1)


# ======================================================================
# FOFE observation key names (used by env, policy, trainer)
# ======================================================================

FOFE_ACTOR_KEYS = (
    "obs_self", "obs_agents_feat", "obs_agents_mask",
    "obs_targets_feat", "obs_targets_mask",
    "obs_radars_feat", "obs_radars_mask",
)

FOFE_CRITIC_KEYS = (
    "crt_agents_feat", "crt_agents_mask",
    "crt_targets_feat", "crt_targets_mask",
    "crt_radars_feat", "crt_radars_mask",
    "crt_time_feat",
)


# ======================================================================
# Combined policy
# ======================================================================

class CombinedPolicy(nn.Module):
    """Wraps striker + jammer actors.  Supports both legacy and FOFE.

    Directional-jammer action layout (HF radar mode, act_dim=3):
      - dims 0,1 (motion):     7-value motion table, all agents. The
        shared Categorical n is max(motion, beam), so indices past
        n_motion_choices on these dims must be masked out.
      - dim 2 (beam ang accel): 9-value beam table (finer near zero for
        smoother control). Only meaningful for jammers; strikers MUST
        keep this dim at the beam-table's zero-acceleration index
        because the env discards their dim-2 entry entirely.

    Both constraints are enforced here by adding a large negative bias
    to invalid logits so the sampled action always lies in the valid
    set, without changing the network architecture or the trainer.
    """

    _LOGIT_MASK_BIAS = -1e9

    def __init__(self, striker_policy, jammer_policy,
                 n_strikers, n_jammers, use_fofe=False,
                 obs_key=("agents", "observation"),
                 action_key=("agents", "action"),
                 deterministic=False,
                 n_motion_choices=None,
                 n_beam_choices=None,
                 beam_zero_idx=None):
        super().__init__()
        self.striker_policy = striker_policy
        self.jammer_policy = jammer_policy
        self.n_strikers = n_strikers
        self.n_jammers = n_jammers
        self.use_fofe = use_fofe
        self.obs_key = obs_key
        self.action_key = action_key
        self.deterministic = deterministic
        self.n_motion_choices = n_motion_choices
        self.n_beam_choices = n_beam_choices
        self.beam_zero_idx = beam_zero_idx

    # --- Action-mask helpers (no-op unless act_dim >= 3 and the env
    #     reported a motion choice count) -----------------------------
    def _needs_action_mask(self, logits: torch.Tensor) -> bool:
        return (
            logits.shape[-2] >= 3
            and self.n_motion_choices is not None
        )

    def _mask_motion_and_bearing(self, logits: torch.Tensor, role: str) -> torch.Tensor:
        """Add a large-negative bias to invalid action indices.
        role: "striker" or "jammer". logits shape: [..., n_role, act_dim, n_choices].
        """
        if not self._needs_action_mask(logits):
            return logits
        nc = logits.shape[-1]
        bias = torch.zeros_like(logits)
        # Motion dims (0, 1): allow only the first n_motion_choices indices.
        if nc > int(self.n_motion_choices):
            bias[..., 0, int(self.n_motion_choices):] = self._LOGIT_MASK_BIAS
            bias[..., 1, int(self.n_motion_choices):] = self._LOGIT_MASK_BIAS
        # Beam dim (2):
        if role == "striker":
            # Pin to the zero-accel index of the beam table — striker
            # dim 2 is discarded by the env, so collapse the distribution.
            zero_idx = int(self.beam_zero_idx) if self.beam_zero_idx is not None else 0
            bias[..., 2, :] = self._LOGIT_MASK_BIAS
            bias[..., 2, zero_idx] = 0.0
        elif role == "jammer":
            if self.n_beam_choices is not None and nc > int(self.n_beam_choices):
                bias[..., 2, int(self.n_beam_choices):] = self._LOGIT_MASK_BIAS
        return logits + bias

    # --- FOFE observation helpers ---
    def _get_fofe_obs(self, td):
        return {k: td.get(("agents", k)) for k in FOFE_ACTOR_KEYS}

    def _split_fofe_obs(self, d, start, end):
        return {k: v[:, start:end] for k, v in d.items()}

    @staticmethod
    def _call_fofe(policy, d):
        return policy(
            d["obs_self"], d["obs_agents_feat"], d["obs_agents_mask"],
            d["obs_targets_feat"], d["obs_targets_mask"],
            d["obs_radars_feat"], d["obs_radars_mask"],
        )

    def forward(self, td):
        ns = self.n_strikers
        if self.use_fofe:
            # ---- FOFE path: extract structured obs per channel ----
            fofe_obs = self._get_fofe_obs(td)
            s_logits = self._call_fofe(self.striker_policy,
                                       self._split_fofe_obs(fofe_obs, 0, ns))
            j_logits = self._call_fofe(self.jammer_policy,
                                       self._split_fofe_obs(fofe_obs, ns, ns + self.n_jammers))
        else:
            # ---- Legacy path: flat observation vector ----
            obs = td.get(self.obs_key)
            s_logits = self.striker_policy(obs[:, :ns])
            j_logits = self.jammer_policy(obs[:, ns:])

        s_logits = self._mask_motion_and_bearing(s_logits, role="striker")
        j_logits = self._mask_motion_and_bearing(j_logits, role="jammer")

        all_logits = MultiCategorical._sanitize_logits(torch.cat([s_logits, j_logits], dim=1))
        dist = MultiCategorical(logits=all_logits)
        action = dist.mode if self.deterministic else dist.sample()

        td.set(self.action_key, action)
        td.set(("agents", "sample_log_prob"), dist.log_prob(action))
        td.set(("agents", "logits"), all_logits)
        return td

    def striker_log_prob_entropy(self, obs_or_dict, action):
        if self.use_fofe:
            logits = self._call_fofe(self.striker_policy, obs_or_dict)
        else:
            logits = self.striker_policy(obs_or_dict)
        logits = self._mask_motion_and_bearing(logits, role="striker")
        logits = MultiCategorical._sanitize_logits(logits)
        dist = MultiCategorical(logits=logits)
        return dist.log_prob(action), dist.entropy()

    def jammer_log_prob_entropy(self, obs_or_dict, action):
        if self.use_fofe:
            logits = self._call_fofe(self.jammer_policy, obs_or_dict)
        else:
            logits = self.jammer_policy(obs_or_dict)
        logits = self._mask_motion_and_bearing(logits, role="jammer")
        logits = MultiCategorical._sanitize_logits(logits)
        dist = MultiCategorical(logits=logits)
        return dist.log_prob(action), dist.entropy()


# ======================================================================
# Combined critic
# ======================================================================

class CombinedCritic(nn.Module):
    """Wraps striker + jammer critics.  Supports both legacy and FOFE."""

    def __init__(self, striker_critic, jammer_critic,
                 n_strikers, n_jammers, use_fofe=False):
        super().__init__()
        self.striker_critic = striker_critic
        self.jammer_critic = jammer_critic
        self.n_strikers = n_strikers
        self.n_jammers = n_jammers
        self.use_fofe = use_fofe

    @staticmethod
    def _broadcast_role_values(values: torch.Tensor, n_role: int, role_name: str) -> torch.Tensor:
        """Normalize role critic outputs to [..., n_role, 1]."""
        if values.ndim < 2 or values.shape[-1] != 1:
            raise RuntimeError(
                f"{role_name} critic must return [..., N, 1], got {tuple(values.shape)}"
            )

        out_n = values.shape[-2]
        if out_n == n_role:
            return values
        if out_n == 1:
            expand_shape = (*values.shape[:-2], n_role, values.shape[-1])
            return values.expand(*expand_shape)
        raise RuntimeError(
            f"{role_name} critic returned N={out_n}, expected N in {{1, {n_role}}}"
        )

    def forward(self, td):
        if self.use_fofe:
            # ---- FOFE critic path: extract global state channels ----
            af = td.get("crt_agents_feat")
            am = td.get("crt_agents_mask")
            tf = td.get("crt_targets_feat")
            tm = td.get("crt_targets_mask")
            rf = td.get("crt_radars_feat")
            rm = td.get("crt_radars_mask")
            t_feat = td.get("crt_time_feat")
            sv = self.striker_critic(af, am, tf, tm, rf, rm, t_feat)
            jv = self.jammer_critic(af, am, tf, tm, rf, rm, t_feat)
        else:
            # ---- Legacy critic path: flat state vector ----
            state = td.get("state")
            sv = self.striker_critic(state)
            jv = self.jammer_critic(state)

        sv = self._broadcast_role_values(sv, self.n_strikers, "striker")
        jv = self._broadcast_role_values(jv, self.n_jammers, "jammer")

        td.set(("agents", "state_value"), torch.cat([sv, jv], dim=-2))
        return td


# ======================================================================
# Factory helpers
# ======================================================================

def make_combined_policy(env: StrikeEA2DEnv, hidden=256, depth=3,
                         fofe_cfg: FOFEConfig | None = None) -> CombinedPolicy:
    use_fofe = fofe_cfg is not None and fofe_cfg.use_fofe

    d_self = int(getattr(env, "d_self", 6))
    # Per-channel actor feature dims (must match env._build_fofe_obs output).
    # See StrikeEA2DEnv._build_fofe_obs for the canonical layout.
    d_agents_actor  = 5
    d_targets_actor = 4
    d_radars_actor  = 4 + int(getattr(env, "_radar_extra_dim", lambda: 0)())

    if use_fofe:
        striker_net = FOFEPolicyNet(
            fofe_cfg, env.n_strikers, env.act_dim, env.n_choices,
            d_self=d_self,
            d_agents=d_agents_actor,
            d_targets=d_targets_actor,
            d_radars=d_radars_actor,
        ).to(env.device)
        jammer_net = FOFEPolicyNet(
            fofe_cfg, env.n_jammers, env.act_dim, env.n_choices,
            d_self=d_self,
            d_agents=d_agents_actor,
            d_targets=d_targets_actor,
            d_radars=d_radars_actor,
        ).to(env.device)
    else:
        striker_net = RolePolicyNet(env.obs_dim, env.act_dim, env.n_choices, env.n_strikers, hidden, depth).to(env.device)
        jammer_net = RolePolicyNet(env.obs_dim, env.act_dim, env.n_choices, env.n_jammers, hidden, depth).to(env.device)

    # If the env exposes motion/beam choice counts (HF directional-jammer
    # model), forward them so the policy can mask invalid indices. Base
    # env doesn't set these → mask becomes a no-op.
    n_motion = getattr(env, "_n_motion_choices", None)
    n_beam = getattr(env, "_n_beam_choices", None)
    beam_zero = getattr(env, "_beam_zero_idx", None)

    return CombinedPolicy(
        striker_net, jammer_net, env.n_strikers, env.n_jammers,
        use_fofe=use_fofe, obs_key=env._obs_key, action_key=env._action_key,
        n_motion_choices=n_motion,
        n_beam_choices=n_beam,
        beam_zero_idx=beam_zero,
    ).to(env.device)


def make_combined_critic(env: StrikeEA2DEnv, hidden=256, depth=3,
                         fofe_cfg: FOFEConfig | None = None) -> CombinedCritic:
    use_fofe = fofe_cfg is not None and fofe_cfg.use_fofe

    if use_fofe:
        # Critic entity dims match _build_fofe_critic_state output:
        #   agents: 7 (x,y,v,ψ,ω,role,alive)
        #   targets: 3 (x,y,alive)
        #   radars:  3 (x,y,jammed) + _critic_radar_extra_dim()
        #            (HF env appends 2 cols: sin/cos of cone centre axis)
        d_radars_critic = 3 + int(getattr(env, "_critic_radar_extra_dim", lambda: 0)())
        striker_critic = FOFEValueNet(fofe_cfg, env.n_strikers, d_agents=7, d_targets=3, d_radars=d_radars_critic).to(env.device)
        jammer_critic = FOFEValueNet(fofe_cfg, env.n_jammers, d_agents=7, d_targets=3, d_radars=d_radars_critic).to(env.device)
    else:
        striker_critic = RoleValueNet(env.state_dim, env.n_strikers, hidden, depth).to(env.device)
        jammer_critic = RoleValueNet(env.state_dim, env.n_jammers, hidden, depth).to(env.device)

    return CombinedCritic(
        striker_critic, jammer_critic, env.n_strikers, env.n_jammers,
        use_fofe=use_fofe,
    ).to(env.device)
