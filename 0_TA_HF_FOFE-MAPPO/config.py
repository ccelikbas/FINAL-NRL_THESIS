from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from .rewards import RewardConfig


# ======================================================================
# FOFE architecture configuration
# ======================================================================

@dataclass
class FOFEConfig:
    """Configuration for Flexible Observation Feature Encoding (FOFE).

    FOFE replaces fixed top-K slot encoding with a permutation-invariant
    set encoder.  Each observation channel (agents, targets, radars) passes
    through its own FOFE block:

        entities [n, d_in] → SEE stack → per-entity MLP → global maxpool → [D_fofe]

    The self-observation bypasses FOFE and uses a small MLP.
    All channel outputs are concatenated and fused by a fusion MLP.

    Dimension trace example (2 SEE layers, h=[96,128], mlp=[512,96]):
        Input:          [n, d_in]   e.g. [n, 4] for agents
        SEE layer 1:    [n, 2×96]  = [n, 192]
        SEE layer 2:    [n, 2×128] = [n, 256]
        Per-entity MLP: [n, 512] → [n, 96]
        MaxPool:        [96]              (fixed regardless of n)

    Parameters
    ----------
    use_fofe : bool
        Master switch.  False = legacy flat-obs MLP.  True = FOFE encoder.

    agents_see_dims : tuple of int
        Hidden dim per SEE layer for agents channel.  Length = N_SEE.
        Each SEE layer i projects [d_prev → h_i], outputs [n, 2*h_i].
        Wang (2026) uses (96, 128).

    targets_see_dims, radars_see_dims : tuple of int
        Same as agents_see_dims but for targets / radars channels.

    fofe_mlp_dims : tuple of int
        Per-entity MLP after the SEE stack, applied to each row.
        Input dim = 2 * last_see_dim.  Output = fofe_mlp_dims[-1] = D_fofe.
        Wang uses (512, 96).

    self_mlp_dims : tuple of int
        MLP for the fixed-size self-obs (6 floats).
        Output dim = self_mlp_dims[-1].

    fusion_mlp_dims : tuple of int
        Fusion MLP after concatenating all channel outputs.
        Input = self_out + 3 × D_fofe.  Wang uses (256, 256).

    critic_agents_see_dims, critic_targets_see_dims, critic_radars_see_dims :
        SEE dims for the critic FOFE (processes global state as entity sets).
        Defaults to same as actor.

    critic_fofe_mlp_dims : tuple of int
        Per-entity MLP dims for critic FOFE blocks.

    critic_self_mlp_dims : tuple of int
        Self-obs MLP dims for critic.  Critic self = concatenation of all
        agent self-features (7 per agent), so input is larger than actor.

    critic_fusion_mlp_dims : tuple of int
        Fusion MLP dims for critic.
    """
    use_fofe: bool = True

    # --- Actor FOFE dims ---
    agents_see_dims:   Tuple[int, ...] = (96,) # one SEE layer
    targets_see_dims:  Tuple[int, ...] = (96,)
    radars_see_dims:   Tuple[int, ...] = (96,)
    fofe_mlp_dims:     Tuple[int, ...] = (128, 64)
    self_mlp_dims:     Tuple[int, ...] = (64, 64)
    fusion_mlp_dims:   Tuple[int, ...] = (256, 256) # reduced from 256

    # --- Critic FOFE dims (default = same as actor) ---
    critic_agents_see_dims:   Tuple[int, ...] = (96,)
    critic_targets_see_dims:  Tuple[int, ...] = (96,)
    critic_radars_see_dims:   Tuple[int, ...] = (96,)
    critic_fofe_mlp_dims:     Tuple[int, ...] = (128, 64)
    critic_fusion_mlp_dims:   Tuple[int, ...] = (256, 256) # reduced from 256


# ======================================================================
# Domain randomization (per-environment, used by the curriculum runner)
# ======================================================================

@dataclass
class DomainRandomization:
    """Per-environment domain-randomization ranges.

    Each field is either ``None`` (the parameter is NOT randomized — the
    env's fixed count/value is used for every environment) or an inclusive
    ``(lo, hi)`` range. When a range is given, EACH parallel environment
    independently samples its own value in ``[lo, hi]`` at every reset, so a
    single training batch (and each evaluation episode) contains a mix of
    configurations — true per-environment domain randomization, not a single
    sample shared per iteration.

    Counts are realised by activating a per-environment subset of entity
    slots ("present" masks) over tensors that the env allocates at the
    maximum (``hi``) size. ``hi`` for every count range must therefore be
    ``<=`` the corresponding ``EnvConfig`` count (which the curriculum runner
    sets to the range maximum).
    """
    n_strikers:        Optional[Tuple[int, int]]   = None
    n_jammers:         Optional[Tuple[int, int]]   = None
    n_known_targets:   Optional[Tuple[int, int]]   = None
    n_unknown_targets: Optional[Tuple[int, int]]   = None
    n_known_radars:    Optional[Tuple[int, int]]   = None
    n_unknown_radars:  Optional[Tuple[int, int]]   = None
    radar_kill_probability: Optional[Tuple[float, float]] = None
    max_steps:         Optional[Tuple[int, int]]   = None

    def active(self) -> bool:
        """True if any field requests randomization."""
        return any(
            v is not None
            for v in (
                self.n_strikers, self.n_jammers,
                self.n_known_targets, self.n_unknown_targets,
                self.n_known_radars, self.n_unknown_radars,
                self.radar_kill_probability, self.max_steps,
            )
        )


# ======================================================================
# Environment configuration
# ======================================================================

@dataclass
class EnvConfig:
    # Team composition
    n_strikers: int = 1
    n_jammers: int = 2
    n_known_targets: int = 1
    n_unknown_targets: int = 0
    n_known_radars: int = 6
    n_unknown_radars: int = 0
    n_targets: int = 0
    n_radars: int = 0

    # World / episode
    world_bounds: Tuple[float, float] = (0.0, 1.0)
    dt: float = 1.0
    max_steps: int = 150
    # Pre-generated radar layouts. n_env_layouts > 0 builds a pool of valid
    # radar positions ONCE at env init (using the slow rejection sampler)
    # and resets cycle through them — avoids running _sample_spaced_radars
    # per-env-per-episode-reset, which is the dominant reset-time cost when
    # multiple radars + a tight min_sep force many rejection retries.
    # Set to 0 for the old random-per-reset behaviour.
    n_env_layouts: int = 256
    radar_min_sep: float = 0.5
    target_spawn_angle_range: Tuple[float, float] = (0, 360)

    # ── Scenario selection ─────────────────────────────────────────
    # "S1" = protected targets (legacy): radars spawned in top band, targets
    #         spawned in a tight annulus around their assigned radar.
    # "S2" = defensive line: targets spawned uniformly in a top band, radars
    #         spawned uniformly (with min separation) in a middle band — radars
    #         form a defensive line between agents and targets.
    # The radar layout pool is still pre-generated at env init for both
    # scenarios; only the sampling bounds + min-sep differ.
    scenario: str = "S2"
    # Minimum pairwise radar separation when scenario == "S2". S2's radar
    # band is much thinner than S1's, so the default is lower to keep
    # rejection sampling tractable.
    s2_radar_min_sep: float = 0.2

    # Kinematics
    v_max: float = 0.02
    accel_magnitude: float = 0.01
    dpsi_max: float = math.radians(12.0)
    h_accel_magnitude_fraction: float = 0.1
    min_turn_radius: float = 0.001

    # Sensors
    R_obs: float = 0.4
    R_comm: float = 0.4

    # Strikers
    striker_engage_range: float = 0.10
    striker_engage_fov: float = 60.0
    striker_v_min: float = 0.005

    # Jammers
    jammer_jam_radius: float = 0.35
    jammer_jam_effect: float = 0.15
    jammer_v_min: float = 0.005

    # Threats
    radar_range: float = 0.20
    radar_kill_probability: float = 1

    # Rewards
    border_thresh: float = 0.05
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    # Per-environment domain randomization (None = disabled → fixed counts).
    # Used by run_curriculum.py. When set, the entity counts above must be the
    # range maxima (tensors are allocated at this size and masked down per-env).
    dr: Optional[DomainRandomization] = None

    # Internal: propagated from FOFEConfig.use_fofe by ExperimentConfig.finalize().
    # Do NOT set directly — change FOFEConfig.use_fofe instead.
    _use_fofe: bool = field(default=False, repr=False)

    @property
    def use_fofe(self) -> bool:
        """Read-only accessor — always reflects FOFEConfig.use_fofe after finalize()."""
        return self._use_fofe

    def __post_init__(self):
        self.n_known_targets = int(self.n_known_targets)
        self.n_unknown_targets = int(self.n_unknown_targets)
        self.n_known_radars = int(self.n_known_radars)
        self.n_unknown_radars = int(self.n_unknown_radars)
        self.radar_min_sep = float(self.radar_min_sep)
        self.s2_radar_min_sep = float(self.s2_radar_min_sep)
        self.scenario = str(self.scenario).upper()

        if self.n_known_targets < 0 or self.n_unknown_targets < 0:
            raise ValueError("n_known_targets and n_unknown_targets must be >= 0")
        if self.n_known_radars < 0 or self.n_unknown_radars < 0:
            raise ValueError("n_known_radars and n_unknown_radars must be >= 0")
        if self.radar_min_sep < 0:
            raise ValueError("radar_min_sep must be >= 0")
        if self.s2_radar_min_sep < 0:
            raise ValueError("s2_radar_min_sep must be >= 0")
        if self.scenario not in ("S1", "S2"):
            raise ValueError(f"scenario must be 'S1' or 'S2', got {self.scenario!r}")

        if self.n_known_targets == 0 and self.n_unknown_targets == 0:
            self.n_targets = int(self.n_targets)
            self.n_known_targets = int(self.n_targets)
            self.n_unknown_targets = 0
        else:
            self.n_targets = int(self.n_known_targets + self.n_unknown_targets)

        if self.n_known_radars == 0 and self.n_unknown_radars == 0:
            self.n_radars = int(self.n_radars)
            self.n_known_radars = int(self.n_radars)
            self.n_unknown_radars = 0
        else:
            self.n_radars = int(self.n_known_radars + self.n_unknown_radars)


@dataclass
class PPOConfig:
    """Shared PPO hyperparameters for both striker and jammer MAPPO."""
    num_envs: int = 2048
    n_iters: int = 1000
    frames_per_batch: Optional[int] = None
    num_epochs: int = 6
    minibatch_size: int = 16384

    gamma: float = 0.99
    lmbda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    normalize_rewards: bool = True

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 1.0
    max_iter_time_s: Optional[float] = 250.0   # safety timeout per iteration (None = disabled)

    seed: int = 0
    log_every: int = 10
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # ── Hardware optimization flags ─────────────────────────────────
    # TF32 matmul on Ampere+ (A100/H100/RTX30/40). Free 10-30% on fp32 matmuls.
    enable_tf32: bool = True
    # cuDNN autotuner — picks best kernel for repeat shapes (stable across iters).
    cudnn_benchmark: bool = True

    # ── Training optimization flags ─────────────────────────────────
    # torch.compile actor + critic nets. First iter is slow (graph trace).
    compile_models: bool = True
    # Mixed-precision autocast for forward + backward (big gain on A100/H100).
    use_amp: bool = True
    # "bfloat16" (recommended, Ampere+; no GradScaler needed) or "float16".
    amp_dtype: str = "bfloat16"

    # ── Diagnostics frequency ──────────────────────────────────────
    # Collect FOFE KPI snapshot every N iterations (was: every minibatch).
    # 1 = every iter (cheap), 5 = every 5 iters, 0 = disabled.
    fofe_kpi_every: int = 1

    def __post_init__(self):
        self.num_envs = int(self.num_envs)
        self.n_iters = int(self.n_iters)
        self.num_epochs = int(self.num_epochs)
        self.minibatch_size = int(self.minibatch_size)


@dataclass
class NetworkConfig:
    actor_hidden: int = 256
    critic_hidden: int = 256
    depth: int = 3


# ======================================================================
# High-fidelity radar model configuration
# ======================================================================

@dataclass
class HFRadarConfig:
    """RF parameters for the high-fidelity angular radar/jammer model.

    Unconstrained radar range is derived from the radar SNR equation:

        SNR = P_t * G_t_lin * G_r_lin * lambda^2 * sigma /
              ((4*pi)^3 * R^4 * k * T0 * B_n * L)

        R_unconstrained = (
            P_t * G_t_lin * G_r_lin * lambda^2 * sigma /
            ((4*pi)^3 * k * T0 * B_n * L)
        )^(1/4)

    Jammed-sector burn-through ranges (from JSR = 1):
        R_main = (((P_t * G_t_lin * sigma) / (4*pi*P_J*G_J_lin)) * R_J^2)^(1/4)
        R_side = ((sigma/(4*pi)) * ((P_t*G_t_lin)/(P_J*G_J_lin))
                  * ((G_t_lin*R_J^2)/G_S_lin))^(1/4)

    Decibel interface:
        Enter gains/loss/SNR threshold in dB everywhere in this config.
        The implementation converts dB values to linear internally.

    Unit handling:
        The SNR equation is evaluated in SI units (meters). The result is then
        converted to normalized world units using:

            R_unc_world = R_unc_meters / meters_per_world_unit

        For this map setup (1 world unit = 1000 km), set:
            meters_per_world_unit = 1_000_000.0

        Optional calibration can then be applied without changing RF params:
            R_unc_world *= normalized_range_scale
        or by setting target_unconstrained_range_world directly.
    """
    # Radar SNR parameters
    radar_tx_power: float = 1e6                 # P_t [W]
    radar_tx_gain: float = 38.0                 # G_t [dB]
    radar_rx_gain: Optional[float] = None       # G_r [dB], defaults to G_t
    wavelength: float = 0.03                    # lambda [m]
    target_rcs: float = 1.0                     # sigma [m^2]
    system_temperature: float = 290.0           # T0 [K]
    receiver_bandwidth: float = 1e6             # B_n [Hz]
    system_losses: float = 4                  # L [dB]
    snr_min: float = 0.0                        # SNR_min [dB]
    boltzmann_constant: float = 1.380649e-23    # k [J/K]

    # World/map scaling
    meters_per_world_unit: float = 1_000_000.0          # 1.0 map unit = 1000 km
    normalized_range_scale: float = 1.0                 # extra multiplier after SI->world conversion
    target_unconstrained_range_world: Optional[float] = None  # if set, overrides normalized_range_scale

    # Radar angular/lobe model parameter
    radar_side_lobe_gain: Optional[float] = None   # G_S [dB], preferred name
    G_S: float = 5                                 # legacy alias [dB]

    # Jammer RF parameters
    jammer_tx_power: Optional[float] = None        # P_J [W], preferred name
    jammer_gain: Optional[float] = None            # G_J [dB], preferred name
    P_J: float = 1e1                              # legacy alias [W]
    G_J: float = 15.0                              # legacy alias [dB]

    # Angular lobe boundaries (degrees, converted to radians internally)
    theta_main_deg: float = 6.0    # full main-lobe width (±1.5° each side)
    theta_side_deg: float = 40.0    # full side-lobe+main-lobe cone width (±4.5° each side)

    # ------------------------------------------------------------------
    # Directional jammer model
    # ------------------------------------------------------------------
    # Each jammer is treated as omnidirectional within its own emission
    # cone and silent everywhere else. The existing radar-centric
    # main/side-lobe physics (above) is unchanged; we only gate it by
    # whether the radar lies inside this cone.
    #
    # jammer_main_lobe_deg : full angular width of the cone (degrees).
    #
    # Beam kinematics (2D, mirrors the agent's heading kinematics but with
    # looser limits — the beam can swing around the jammer faster than the
    # jammer itself can turn). The jammer policy emits an angular
    # acceleration (action dim 2, same 7-value table as motion). State is
    # (beam_angle, beam_rate), where beam_angle is relative to the jammer's
    # own heading.
    #
    # beam_dpsi_max : maximum |beam_rate| per step (radians/step).
    #     Default math.pi → the beam can flip 180° in a single step at
    #     saturation, satisfying the design brief.
    # beam_h_accel_magnitude_fraction : per-step angular acceleration
    #     magnitude expressed as a fraction of beam_dpsi_max (mirrors
    #     EnvConfig.h_accel_magnitude_fraction). Default 0.1 → ~18°/step²
    #     at action=±1, so the beam needs ~10 steps to spin up from rest.
    jammer_main_lobe_deg: float = 120.0
    beam_dpsi_max: float = math.pi
    beam_h_accel_magnitude_fraction: float = 0.1

    # ------------------------------------------------------------------
    # Per-jammer in-cone radar capacity
    # ------------------------------------------------------------------
    # When set, each jammer is only able to actively jam the K closest
    # (Euclidean) radars that fall inside its cone — any additional
    # in-cone radars are treated as un-jammed by that jammer (R_eff =
    # R_unc). This affects R_eff_jar, _jammer_in_cone, _radar_jammed_flag,
    # and (downstream) every reward / obs feature that gates on these.
    #
    # Set to None to disable the cap (jammer affects all in-cone radars,
    # the legacy behaviour).
    jammer_max_jammed_radars: Optional[int] = 2

    def __post_init__(self):
        if self.radar_rx_gain is None:
            self.radar_rx_gain = float(self.radar_tx_gain)
        if self.radar_side_lobe_gain is None:
            self.radar_side_lobe_gain = float(self.G_S)
        if self.jammer_tx_power is None:
            self.jammer_tx_power = float(self.P_J)
        if self.jammer_gain is None:
            self.jammer_gain = float(self.G_J)

        # Keep legacy aliases in sync for backward compatibility.
        self.G_S = float(self.radar_side_lobe_gain)
        self.P_J = float(self.jammer_tx_power)
        self.G_J = float(self.jammer_gain)

        if self.receiver_bandwidth <= 0:
            raise ValueError("receiver_bandwidth must be > 0")
        if self.system_temperature <= 0:
            raise ValueError("system_temperature must be > 0")
        if not math.isfinite(self.radar_tx_gain):
            raise ValueError("radar_tx_gain must be finite (dB)")
        if not math.isfinite(self.radar_rx_gain):
            raise ValueError("radar_rx_gain must be finite (dB)")
        if not math.isfinite(self.system_losses):
            raise ValueError("system_losses must be finite (dB)")
        if not math.isfinite(self.snr_min):
            raise ValueError("snr_min must be finite (dB)")
        if not math.isfinite(self.radar_side_lobe_gain):
            raise ValueError("radar_side_lobe_gain must be finite (dB)")
        if not math.isfinite(self.jammer_gain):
            raise ValueError("jammer_gain must be finite (dB)")
        if self.jammer_tx_power <= 0:
            raise ValueError("jammer_tx_power must be > 0")
        if self.meters_per_world_unit <= 0:
            raise ValueError("meters_per_world_unit must be > 0")
        if self.normalized_range_scale <= 0:
            raise ValueError("normalized_range_scale must be > 0")
        if (self.target_unconstrained_range_world is not None
                and self.target_unconstrained_range_world <= 0):
            raise ValueError("target_unconstrained_range_world must be > 0 when provided")
        if not (0.0 < float(self.jammer_main_lobe_deg) <= 360.0):
            raise ValueError("jammer_main_lobe_deg must be in (0, 360]")
        if not (0.0 < float(self.beam_dpsi_max) <= math.pi + 1e-9):
            raise ValueError("beam_dpsi_max must be in (0, pi] radians/step")
        if not (0.0 <= float(self.beam_h_accel_magnitude_fraction) <= 1.0):
            raise ValueError("beam_h_accel_magnitude_fraction must be in [0, 1]")
        if (self.jammer_max_jammed_radars is not None
                and int(self.jammer_max_jammed_radars) < 1):
            raise ValueError(
                "jammer_max_jammed_radars must be None (unlimited) or >= 1"
            )

    @staticmethod
    def db_to_linear(db_value: float) -> float:
        """Convert dB quantity (power ratio form) to linear."""
        return float(10.0 ** (float(db_value) / 10.0))

    # Backward-compatible aliases used by older code paths.
    @property
    def P_t(self) -> float:
        return float(self.radar_tx_power)

    @property
    def G_t(self) -> float:
        return self.db_to_linear(self.radar_tx_gain)

    @property
    def G_r(self) -> float:
        return self.db_to_linear(self.radar_rx_gain)

    @property
    def sigma(self) -> float:
        return float(self.target_rcs)

    @property
    def L(self) -> float:
        return self.db_to_linear(self.system_losses)

    @property
    def snr_min_linear(self) -> float:
        return self.db_to_linear(self.snr_min)

    @property
    def G_S_linear(self) -> float:
        return self.db_to_linear(self.radar_side_lobe_gain)

    @property
    def G_J_linear(self) -> float:
        return self.db_to_linear(self.jammer_gain)


@dataclass
class EnvExtensionsConfig:
    """Extension flags that select alternative environment implementations."""
    use_hf_radar: bool = True
    hf_radar: HFRadarConfig = field(default_factory=HFRadarConfig)


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    net: NetworkConfig = field(default_factory=NetworkConfig)
    fofe: FOFEConfig = field(default_factory=FOFEConfig)
    ext: EnvExtensionsConfig = field(default_factory=EnvExtensionsConfig)

    def finalize(self):
        if self.ppo.frames_per_batch is None:
            self.ppo.frames_per_batch = int(self.ppo.num_envs * self.env.max_steps)
        # Propagate FOFE flag into EnvConfig so the env knows to emit FOFE obs
        self.env._use_fofe = self.fofe.use_fofe
        return self

