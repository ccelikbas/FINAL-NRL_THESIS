"""
Dual-MAPPO training loop with optional FOFE observation encoding.

When use_fofe=True, the PPO update loop extracts structured per-channel
observations (entity sets + masks) from the TensorDict and passes them
through FOFE-based actor/critic networks instead of flat observation vectors.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------------------------------------------------
# Hardware monitor — samples GPU + CPU stats in a background thread for
# the first N profile iterations so we can sanity-check utilisation.
#
# Three data sources, all optional:
#   - pynvml: GPU utilisation %, memory MB, power W, temp °C (NVIDIA only)
#   - psutil: CPU utilisation %, system RAM GB, process RAM GB
#   - torch.cuda: allocator stats (always available when CUDA is on)
#
# If a source isn't installed we just skip it. Background sampler runs at
# ~10 Hz which is plenty for an iteration that takes seconds.
# ----------------------------------------------------------------------

class _HardwareMonitor:
    def __init__(self, sample_hz: float = 10.0):
        self._dt = 1.0 / max(1.0, float(sample_hz))
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Reset samples
        self._gpu_util: List[float] = []
        self._gpu_mem_mb: List[float] = []
        self._gpu_pwr_w: List[float] = []
        self._gpu_temp_c: List[float] = []
        self._cpu_util: List[float] = []
        self._sys_mem_gb: List[float] = []
        self._proc_mem_gb: List[float] = []

        # Try pynvml
        self._pynvml = None
        self._nv_handle = None
        self._nv_supports_power = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
            # Probe power query support once
            try:
                pynvml.nvmlDeviceGetPowerUsage(self._nv_handle)
                self._nv_supports_power = True
            except Exception:
                self._nv_supports_power = False
        except Exception:
            self._pynvml = None

        # Try psutil
        self._psutil = None
        self._proc = None
        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process()
            # Prime so the next reading isn't 0.0
            psutil.cpu_percent(interval=None)
            self._proc.cpu_percent(interval=None)
        except Exception:
            self._psutil = None

    @property
    def gpu_available(self) -> bool:
        return self._pynvml is not None

    @property
    def cpu_available(self) -> bool:
        return self._psutil is not None

    def device_header(self) -> str:
        """One-line description of detected hardware. Safe to call once at startup."""
        parts: List[str] = []
        if self._pynvml is not None:
            try:
                name = self._pynvml.nvmlDeviceGetName(self._nv_handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._nv_handle)
                total_gb = mem.total / (1024 ** 3)
                parts.append(f"GPU: {name} ({total_gb:.1f} GB)")
            except Exception:
                parts.append("GPU: (pynvml probe failed)")
        elif torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                parts.append(f"GPU: {name} ({total_gb:.1f} GB) [pynvml unavailable — limited stats]")
            except Exception:
                parts.append("GPU: (torch probe failed)")
        else:
            parts.append("GPU: none (CPU-only)")

        if self._psutil is not None:
            try:
                logical = self._psutil.cpu_count(logical=True)
                physical = self._psutil.cpu_count(logical=False)
                vm = self._psutil.virtual_memory()
                total_gb = vm.total / (1024 ** 3)
                parts.append(f"CPU: {physical} cores / {logical} threads, {total_gb:.1f} GB RAM")
            except Exception:
                parts.append("CPU: (psutil probe failed)")
        else:
            parts.append("CPU: (psutil unavailable)")
        return " | ".join(parts)

    def start(self) -> None:
        # Reset sample buffers each start
        self._gpu_util.clear()
        self._gpu_mem_mb.clear()
        self._gpu_pwr_w.clear()
        self._gpu_temp_c.clear()
        self._cpu_util.clear()
        self._sys_mem_gb.clear()
        self._proc_mem_gb.clear()

        if self._pynvml is None and self._psutil is None:
            return  # Nothing to sample
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def _sample_loop(self) -> None:
        while self._running:
            if self._pynvml is not None:
                try:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nv_handle)
                    self._gpu_util.append(float(util.gpu))
                    mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._nv_handle)
                    self._gpu_mem_mb.append(mem.used / (1024 ** 2))
                except Exception:
                    pass
                if self._nv_supports_power:
                    try:
                        mw = self._pynvml.nvmlDeviceGetPowerUsage(self._nv_handle)
                        self._gpu_pwr_w.append(mw / 1000.0)
                    except Exception:
                        pass
                try:
                    # NVML_TEMPERATURE_GPU = 0
                    t_c = self._pynvml.nvmlDeviceGetTemperature(self._nv_handle, 0)
                    self._gpu_temp_c.append(float(t_c))
                except Exception:
                    pass
            if self._psutil is not None:
                try:
                    self._cpu_util.append(float(self._psutil.cpu_percent(interval=None)))
                    vm = self._psutil.virtual_memory()
                    self._sys_mem_gb.append(vm.used / (1024 ** 3))
                    if self._proc is not None:
                        rss = self._proc.memory_info().rss / (1024 ** 3)
                        self._proc_mem_gb.append(rss)
                except Exception:
                    pass
            time.sleep(self._dt)

    def summary(self) -> str:
        def _agg(arr: List[float], label: str, unit: str, fmt: str = "{:.1f}") -> Optional[str]:
            if not arr:
                return None
            mean_v = sum(arr) / len(arr)
            peak_v = max(arr)
            return f"{label} mean={fmt.format(mean_v)}{unit} peak={fmt.format(peak_v)}{unit}"

        parts: List[str] = []
        if self._pynvml is not None:
            for p in (
                _agg(self._gpu_util, "GPU_util", "%"),
                _agg(self._gpu_mem_mb, "GPU_mem", "MB", "{:.0f}"),
                _agg(self._gpu_pwr_w, "GPU_pwr", "W"),
                _agg(self._gpu_temp_c, "GPU_T", "°C", "{:.0f}"),
            ):
                if p is not None:
                    parts.append(p)
        if self._psutil is not None:
            for p in (
                _agg(self._cpu_util, "CPU_util", "%"),
                _agg(self._sys_mem_gb, "SysRAM", "GB", "{:.1f}"),
                _agg(self._proc_mem_gb, "ProcRAM", "GB", "{:.1f}"),
            ):
                if p is not None:
                    parts.append(p)
        # Always-available torch.cuda memory (peak since last reset)
        if torch.cuda.is_available():
            try:
                alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
                parts.append(
                    f"torch_alloc={alloc_mb:.0f}MB peak={peak_mb:.0f}MB reserved={reserved_mb:.0f}MB"
                )
            except Exception:
                pass
        if not parts:
            return "(no monitor backends available — pip install pynvml psutil)"
        return " | ".join(parts)
from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import RewardSum
from torchrl.envs.utils import check_env_specs

from .config import EnvConfig, FOFEConfig, HFRadarConfig, NetworkConfig, PPOConfig
from .environment import StrikeEA2DEnv
from .HF_environment import HFStrikeEA2DEnv
from .models import (
    CombinedCritic,
    CombinedPolicy,
    FOFE_ACTOR_KEYS,
    FOFE_CRITIC_KEYS,
    make_combined_critic,
    make_combined_policy,
)
from .normalization import RewardNormalizer
from .utils import (
    compute_explained_variance,
    compute_gae_sequential,
    make_collector,
    ppo_clip_loss,
    prepare_done_keys,
    value_loss_fn,
)


EVAL_REWARD_COMPONENT_KEYS: Tuple[str, ...] = (
    "target_destroyed",
    "terminal_bonus",
    "border_penalty",
    "timestep_penalty",
    "radar_avoidance",
    "striker_approach",
    "jammer_approach",
    "striker_progress",
    "jammer_progress",
    "jammer_jam_bonus",
    "formation",
    "escort",
    "agent_destroyed",
    "paper_mission",
    "separation_penalty",
    "jammer_coalition_coverage",
    "control_effort",
)


def _finite_mean(values: List[float]) -> float:
    finite_vals = [v for v in values if math.isfinite(v)]
    if not finite_vals:
        return float("nan")
    return float(sum(finite_vals) / len(finite_vals))


# ------------------------------------------------------------------
# Fine-grained profiling helpers (used only for the first N iterations)
# ------------------------------------------------------------------

def _fine_tic(active: bool):
    if not active:
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _fine_lap(acc: Dict[str, float], name: str, t, active: bool):
    if not active or t is None:
        return t
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    now = time.perf_counter()
    acc[name] = acc.get(name, 0.0) + (now - t)
    return now


def _format_buckets(acc: Dict[str, float], top: int = 10) -> str:
    items = sorted(acc.items(), key=lambda kv: -kv[1])
    parts = [f"{k}={v:.3f}s" for k, v in items[:top]]
    other = sum(v for _, v in items[top:])
    if other > 0:
        parts.append(f"other={other:.3f}s")
    return " ".join(parts)


try:
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    _EXPLORATION_API = "new"
except Exception:
    try:
        from tensordict.nn import InteractionType, set_interaction_type
        _EXPLORATION_API = "interaction"
    except Exception:
        _EXPLORATION_API = None


def _deterministic_context():
    if _EXPLORATION_API == "new":
        return set_exploration_type(ExplorationType.DETERMINISTIC)
    if _EXPLORATION_API == "interaction":
        return set_interaction_type(InteractionType.DETERMINISTIC)
    return contextlib.nullcontext()


# ------------------------------------------------------------------
# Environment builder
# ------------------------------------------------------------------

def _env_kwargs(env_cfg: EnvConfig, ppo_cfg: PPOConfig) -> dict:
    """Common keyword arguments shared by StrikeEA2DEnv and HFStrikeEA2DEnv."""
    return dict(
        num_envs=ppo_cfg.num_envs,
        max_steps=env_cfg.max_steps,
        device=ppo_cfg.device,
        seed=ppo_cfg.seed,
        n_strikers=env_cfg.n_strikers,
        n_jammers=env_cfg.n_jammers,
        n_targets=env_cfg.n_targets,
        n_radars=env_cfg.n_radars,
        n_known_targets=env_cfg.n_known_targets,
        n_unknown_targets=env_cfg.n_unknown_targets,
        n_known_radars=env_cfg.n_known_radars,
        n_unknown_radars=env_cfg.n_unknown_radars,
        dt=env_cfg.dt,
        world_bounds=env_cfg.world_bounds,
        v_max=env_cfg.v_max,
        accel_magnitude=env_cfg.accel_magnitude,
        dpsi_max=env_cfg.dpsi_max,
        h_accel_magnitude_fraction=env_cfg.h_accel_magnitude_fraction,
        min_turn_radius=env_cfg.min_turn_radius,
        R_obs=env_cfg.R_obs,
        R_comm=env_cfg.R_comm,
        striker_engage_range=env_cfg.striker_engage_range,
        striker_engage_fov=env_cfg.striker_engage_fov,
        striker_v_min=env_cfg.striker_v_min,
        jammer_jam_radius=env_cfg.jammer_jam_radius,
        jammer_jam_effect=env_cfg.jammer_jam_effect,
        jammer_v_min=env_cfg.jammer_v_min,
        radar_range=env_cfg.radar_range,
        radar_kill_probability=env_cfg.radar_kill_probability,
        border_thresh=env_cfg.border_thresh,
        reward_config=env_cfg.reward_config,
        target_spawn_angle_range=env_cfg.target_spawn_angle_range,
        n_env_layouts=env_cfg.n_env_layouts,
        radar_min_sep=getattr(env_cfg, "radar_min_sep", 0.5),
        scenario=getattr(env_cfg, "scenario", "S1"),
        s2_radar_min_sep=getattr(env_cfg, "s2_radar_min_sep", 0.2),
        s2_target_min_sep=getattr(env_cfg, "s2_target_min_sep", 0.2),
        use_fofe=env_cfg.use_fofe,
        dr=getattr(env_cfg, "dr", None),
    )


def build_env(
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    hf_radar_cfg: Optional[HFRadarConfig] = None,
) -> StrikeEA2DEnv:
    kwargs = _env_kwargs(env_cfg, ppo_cfg)
    if hf_radar_cfg is not None:
        return HFStrikeEA2DEnv(hf_cfg=hf_radar_cfg, **kwargs)
    return StrikeEA2DEnv(**kwargs)


def _safe_check(env) -> None:
    try:
        check_env_specs(env)
        print("check_env_specs: OK")
    except Exception as exc:
        print(f"check_env_specs warning (continuing): {type(exc).__name__}: {exc}")


# ------------------------------------------------------------------
# FOFE observation helpers for training loop
# ------------------------------------------------------------------

def _extract_fofe_actor_obs(td: TensorDict) -> Dict[str, torch.Tensor]:
    """Extract FOFE actor observation channels from a TensorDict."""
    return {k: td.get(("agents", k)) for k in FOFE_ACTOR_KEYS}


def _extract_fofe_critic_obs(td: TensorDict) -> Dict[str, torch.Tensor]:
    """Extract FOFE critic state channels from a TensorDict (root-level keys)."""
    return {k: td.get(k) for k in FOFE_CRITIC_KEYS}


def _actor_role_dim_for_key(key: str, v: torch.Tensor) -> int:
    """Return the role-agent dimension index for one FOFE actor tensor key."""
    if key == "obs_self":
        # [..., n_role, d_self]
        return v.ndim - 2
    if key.endswith("_feat"):
        # [..., n_role, n_entities, d_feat]
        return v.ndim - 3
    if key.endswith("_mask"):
        # [..., n_role, n_entities]
        return v.ndim - 2
    # Fallback for any future actor key following [..., n_role, ...]
    return v.ndim - 2


def _split_fofe_by_role(fofe_dict: Dict[str, torch.Tensor],
                         ns: int, nj: int) -> Tuple[Dict, Dict]:
    """Split FOFE actor obs along agent dim into striker/jammer dicts.

    Actor FOFE obs have shape [*, A, ...] where dim for agent is at position
    determined by the key:
      obs_self:         [*, A, d_self]       → split on dim -2
      obs_agents_feat:  [*, A, E_a, 5 + E]   → split on dim -3
      obs_agents_mask:  [*, A, E_a]          → split on dim -2
      obs_targets_feat: [*, A, E_t, 4]       → split on dim -3
      obs_targets_mask: [*, A, E_t]          → split on dim -2
      obs_radars_feat:  [*, A, E_r, 4 + E]   → split on dim -3
      obs_radars_mask:  [*, A, E_r]          → split on dim -2

    We split using key-based role-axis positions (not size-based heuristics),
    which avoids ambiguity when unrelated dimensions happen to equal ns+nj.
    """
    s_dict, j_dict = {}, {}
    for k, v in fofe_dict.items():
        agent_dim = _actor_role_dim_for_key(k, v)
        s_dict[k] = v.narrow(agent_dim, 0, ns)
        j_dict[k] = v.narrow(agent_dim, ns, nj)
    return s_dict, j_dict


def _transpose_fofe_dict(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Transpose dim 0 and 1 for all tensors in a FOFE dict (B,T <-> T,B)."""
    return {k: v.transpose(0, 1).contiguous() for k, v in d.items()}


def _flatten_fofe_dict(d: Dict[str, torch.Tensor], n_role: int) -> Dict[str, torch.Tensor]:
    """Flatten temporal+batch dims: [T,B,n_role,...] or [N,n_role,...] → [N,n_role,...]."""
    out = {}
    for k, v in d.items():
        role_dim = _actor_role_dim_for_key(k, v)
        trailing = v.shape[role_dim:]
        out[k] = v.reshape(-1, *trailing)
    return out


def _index_fofe_dict(d: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Index first dim of all tensors in a FOFE dict."""
    return {k: v[idx] for k, v in d.items()}


# ------------------------------------------------------------------
# FOFE diagnostic KPI helpers
# ------------------------------------------------------------------

# Canonical list of FOFE log keys — initialised to [] in the logs dict,
# appended to every iteration (NaN when use_fofe=False).
_FOFE_CHANNELS = ("agents", "targets", "radars")
_FOFE_ROLES = ("striker", "jammer")

FOFE_LOG_KEYS: Tuple[str, ...] = tuple(
    f"fofe_{role}_{metric}"
    for role in _FOFE_ROLES
    for metric in (
        # per-channel output norm mean/std (actor)
        *[f"actor_{ch}_mean" for ch in _FOFE_CHANNELS],
        *[f"actor_{ch}_std" for ch in _FOFE_CHANNELS],
        # per-channel output norm mean/std (critic)
        *[f"critic_{ch}_mean" for ch in _FOFE_CHANNELS],
        *[f"critic_{ch}_std" for ch in _FOFE_CHANNELS],
        # collapse fraction (actor)
        *[f"collapse_{ch}" for ch in _FOFE_CHANNELS],
        # channel dominance (actor)
        *[f"dominance_{ch}" for ch in _FOFE_CHANNELS],
        # visible entity counts from masks (actor)
        *[f"visible_{ch}_mean" for ch in _FOFE_CHANNELS],
        # all-masked fraction (actor)
        *[f"all_masked_{ch}" for ch in _FOFE_CHANNELS],
        # SEE gradient norms
        "actor_see_grad_norm",
        "critic_see_grad_norm",
    )
)


@torch.no_grad()
def _channel_mean_std(x: torch.Tensor) -> Tuple[float, float]:
    """Mean and std of L2 norms across samples.  x: [N, D]."""
    norms = x.norm(dim=-1)
    return float(norms.mean().item()), float(norms.std().item())


@torch.no_grad()
def _collapse_fraction(x: torch.Tensor, eps: float = 1e-6) -> float:
    """Fraction of samples where channel output L2 norm < eps."""
    return float((x.norm(dim=-1) < eps).float().mean().item())


@torch.no_grad()
def _channel_dominance(cache: Dict[str, torch.Tensor]) -> Tuple[float, float, float]:
    """Normalised share of each channel's mean norm.  Returns (a, t, r) summing to ~1."""
    na = cache["x_agents"].norm(dim=-1).mean()
    nt = cache["x_targets"].norm(dim=-1).mean()
    nr = cache["x_radars"].norm(dim=-1).mean()
    total = na + nt + nr + 1e-12
    return float(na / total), float(nt / total), float(nr / total)


@torch.no_grad()
def _visible_entity_stats(mask: torch.Tensor) -> Tuple[float, float]:
    """mask: [N, n_role, E] bool.  Returns (mean_visible, all_masked_frac)."""
    visible = mask.float().sum(dim=-1)          # [N, n_role]
    mean_vis = float(visible.mean().item())
    all_masked = float((~mask.any(dim=-1)).float().mean().item())
    return mean_vis, all_masked


def _see_grad_norm(fofe_net) -> float:
    """Total L2 grad norm across all SEE layer params in all three FOFE blocks."""
    total_sq = 0.0
    for block_name in ("fofe_agents", "fofe_targets", "fofe_radars"):
        block = getattr(fofe_net, block_name, None)
        if block is None:
            continue
        for see in block.see_layers:
            for p in see.parameters():
                if p.grad is not None:
                    total_sq += float(p.grad.data.norm(2).item() ** 2)
    return math.sqrt(total_sq)


def _collect_fofe_kpis(
    actor_cache: Dict[str, torch.Tensor],
    critic_cache: Dict[str, torch.Tensor],
    mb_fofe: Dict[str, torch.Tensor],
    actor_net,
    critic_net,
) -> Dict[str, float]:
    """Collect all FOFE KPIs for one role from a single minibatch."""
    d: Dict[str, float] = {}

    # -- actor per-channel output stats --
    for ch in _FOFE_CHANNELS:
        key = f"x_{ch}"
        m, s = _channel_mean_std(actor_cache[key])
        d[f"actor_{ch}_mean"] = m
        d[f"actor_{ch}_std"] = s
        d[f"collapse_{ch}"] = _collapse_fraction(actor_cache[key])

    # -- critic per-channel output stats --
    for ch in _FOFE_CHANNELS:
        key = f"x_{ch}"
        m, s = _channel_mean_std(critic_cache[key])
        d[f"critic_{ch}_mean"] = m
        d[f"critic_{ch}_std"] = s

    # -- channel dominance (actor) --
    dom_a, dom_t, dom_r = _channel_dominance(actor_cache)
    d["dominance_agents"] = dom_a
    d["dominance_targets"] = dom_t
    d["dominance_radars"] = dom_r

    # -- visibility from masks --
    for ch, mask_key in [("agents", "obs_agents_mask"),
                         ("targets", "obs_targets_mask"),
                         ("radars", "obs_radars_mask")]:
        if mask_key in mb_fofe:
            vis, am = _visible_entity_stats(mb_fofe[mask_key])
            d[f"visible_{ch}_mean"] = vis
            d[f"all_masked_{ch}"] = am
        else:
            d[f"visible_{ch}_mean"] = float("nan")
            d[f"all_masked_{ch}"] = float("nan")

    # -- SEE gradient norms (read after backward + clip, before step) --
    d["actor_see_grad_norm"] = _see_grad_norm(actor_net)
    d["critic_see_grad_norm"] = _see_grad_norm(critic_net)

    return d


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_current_policy(
    policy: CombinedPolicy,
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    n_eval_episodes: int = 200,
    hf_radar_cfg: Optional[HFRadarConfig] = None,
) -> Dict[str, float]:
    # Run all episodes in parallel: num_envs = n_eval_episodes so every env
    # completes exactly one episode, each with a different random layout.
    # This is ~n_eval_episodes× faster than the old single-env sequential loop.
    n_eval_episodes = max(1, int(n_eval_episodes))
    eval_ppo = PPOConfig(
        num_envs=n_eval_episodes,
        device=ppo_cfg.device,
        seed=ppo_cfg.seed + 10_000,
    )
    eval_env = build_env(env_cfg, eval_ppo, hf_radar_cfg=hf_radar_cfg)

    was_training = policy.training
    policy.eval()
    policy.deterministic = True

    ep_total_rewards: List[float] = []
    ep_survival: List[float] = []
    ep_duration: List[float] = []
    ep_completion: List[float] = []
    ep_targets_destroyed: List[float] = []
    ep_component_rewards: Dict[str, List[float]] = {
        key: [] for key in EVAL_REWARD_COMPONENT_KEYS
    }

    # Track which envs have already finished so we stop stepping them.
    # done_mask[b] = True once env b has emitted a done signal.
    done_mask = torch.zeros(n_eval_episodes, dtype=torch.bool, device=ppo_cfg.device)
    # Accumulate per-component rewards per env across steps.
    component_sums: Dict[str, torch.Tensor] = {
        key: torch.zeros(n_eval_episodes, device=ppo_cfg.device)
        for key in EVAL_REWARD_COMPONENT_KEYS
    }

    td = eval_env.reset()
    stats_by_env: Dict[int, Any] = {}

    for _ in range(env_cfg.max_steps):
        td = policy(td)
        td_next = eval_env.step(td)

        # Accumulate component rewards for envs still running
        for comp_key in EVAL_REWARD_COMPONENT_KEYS:
            comp_tensor = eval_env.last_reward_components[comp_key]  # [B, A, 1] or [B]
            per_env = comp_tensor.reshape(n_eval_episodes, -1).sum(dim=-1)
            component_sums[comp_key] += per_env * (~done_mask).float()

        # Determine which envs just fired done for the first time this step
        step_done = td_next.get(("next", "done"))  # [B, 1] or [B]
        step_done = step_done.reshape(n_eval_episodes).bool()
        newly_done = step_done & ~done_mask
        done_mask = done_mask | step_done

        # Collect stats immediately for newly-done envs.
        # The env pushes stats inside _step then zeroes reward accumulators.
        # If we wait until the end, re-stepping done envs fires the stat again
        # with reward=0 and an inflated step_count, overwriting the real values.
        if newly_done.any():
            for s in eval_env.pop_episode_stats():
                env_idx = int(s.get("env_idx", -1))
                if env_idx >= 0 and newly_done[env_idx] and env_idx not in stats_by_env:
                    stats_by_env[env_idx] = s

        if done_mask.all():
            break
        td = td_next.get("next")

    for b in range(n_eval_episodes):
        s = stats_by_env.get(b)
        if s is not None:
            ep_total_rewards.append(float(s.get("episode_total_reward", float("nan"))))
            ep_survival.append(float(s.get("survival_frac", float("nan"))))
            ep_duration.append(float(s.get("duration", float("nan"))))
            ep_completion.append(1.0 if bool(s.get("mission_complete", False)) else 0.0)
            ep_targets_destroyed.append(float(s.get("targets_frac", float("nan"))))
        else:
            ep_total_rewards.append(float("nan"))
            ep_survival.append(float("nan"))
            ep_duration.append(float("nan"))
            ep_completion.append(float("nan"))
            ep_targets_destroyed.append(float("nan"))
        for comp_key in EVAL_REWARD_COMPONENT_KEYS:
            ep_component_rewards[comp_key].append(float(component_sums[comp_key][b].item()))

    policy.deterministic = False
    if was_training:
        policy.train()

    metrics = {
        "eval_mean_episode_total_reward": float(sum(ep_total_rewards) / len(ep_total_rewards)),
        "eval_survival_rate": float(sum(ep_survival) / len(ep_survival)),
        "eval_mean_duration": float(sum(ep_duration) / len(ep_duration)),
        "eval_task_completion_rate": float(sum(ep_completion) / len(ep_completion)),
        "eval_targets_destroyed_rate": _finite_mean(ep_targets_destroyed),
    }
    for comp_key in EVAL_REWARD_COMPONENT_KEYS:
        metrics[f"eval_component_{comp_key}"] = _finite_mean(ep_component_rewards[comp_key])
    return metrics


# ------------------------------------------------------------------
# Rollout data reshaping helpers
# ------------------------------------------------------------------

def _infer_time_batch(td: TensorDict, num_envs: int) -> Tuple[int, int, bool]:
    obs = td.get(("agents", "observation"))
    if obs.ndim == 3:
        return 1, obs.shape[0], False
    if obs.shape[1] == num_envs:
        return obs.shape[0], obs.shape[1], False
    elif obs.shape[0] == num_envs:
        return obs.shape[1], obs.shape[0], True
    else:
        raise ValueError(f"Cannot infer layout: obs shape={obs.shape}, num_envs={num_envs}")


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------

def train_mappo(
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    net_cfg: NetworkConfig,
    fofe_cfg: Optional[FOFEConfig] = None,
    checkpoint: Optional[Dict[str, Any]] = None,
    hf_radar_cfg: Optional[HFRadarConfig] = None,
) -> Tuple[StrikeEA2DEnv, CombinedPolicy, CombinedCritic, Dict[str, List[float]], Optional[RewardNormalizer]]:
    device = ppo_cfg.device
    ns = env_cfg.n_strikers
    nj = env_cfg.n_jammers
    use_fofe = fofe_cfg is not None and fofe_cfg.use_fofe

    # ── Hardware optimization flags (set once, before any GPU work) ──
    if torch.cuda.is_available():
        if bool(getattr(ppo_cfg, "enable_tf32", False)):
            try:
                torch.set_float32_matmul_precision("high")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("TF32 matmul + cuDNN allow_tf32: ENABLED")
            except Exception as exc:
                print(f"TF32 enable failed (continuing): {type(exc).__name__}: {exc}")
        if bool(getattr(ppo_cfg, "cudnn_benchmark", False)):
            try:
                torch.backends.cudnn.benchmark = True
                print("cuDNN benchmark: ENABLED")
            except Exception as exc:
                print(f"cuDNN benchmark enable failed (continuing): {type(exc).__name__}: {exc}")

    # ── AMP (autocast) setup ─────────────────────────────────────────
    # bfloat16 has the same exponent range as fp32 → no GradScaler needed,
    # numerically safe for PPO ratios. fp16 needs GradScaler to handle the
    # narrower range without underflow.
    _amp_enabled = bool(getattr(ppo_cfg, "use_amp", False)) and torch.cuda.is_available()
    _amp_dtype_str = str(getattr(ppo_cfg, "amp_dtype", "bfloat16")).lower()
    _amp_dtype_map = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16":  torch.float16,  "fp16": torch.float16,
    }
    _amp_dtype = _amp_dtype_map.get(_amp_dtype_str, torch.bfloat16)
    # Auto-fallback bfloat16 → float16 on hardware that doesn't support bf16
    # (anything older than Ampere — laptop GPUs are often Turing/Pascal).
    if _amp_enabled and _amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("AMP: bfloat16 not supported on this GPU — falling back to float16 + GradScaler.")
        _amp_dtype = torch.float16
    _use_grad_scaler = _amp_enabled and _amp_dtype == torch.float16
    if _amp_enabled:
        try:
            _grad_scaler = torch.amp.GradScaler("cuda") if _use_grad_scaler else None
        except Exception:
            _grad_scaler = torch.cuda.amp.GradScaler() if _use_grad_scaler else None
        print(f"AMP autocast: ENABLED (dtype={_amp_dtype}, GradScaler={'on' if _use_grad_scaler else 'off'})")
    else:
        _grad_scaler = None

    def _autocast():
        if _amp_enabled:
            return torch.autocast(device_type="cuda", dtype=_amp_dtype)
        return contextlib.nullcontext()

    # ── Build environment ────────────────────────────────────────────
    base_env = build_env(env_cfg, ppo_cfg, hf_radar_cfg=hf_radar_cfg)
    env = TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env._reward_key], out_keys=[(base_env.group, "episode_reward")]),
    )
    _safe_check(env)

    # ── Build networks (FOFE or legacy depending on fofe_cfg) ────────
    policy = make_combined_policy(base_env, hidden=net_cfg.actor_hidden,
                                  depth=net_cfg.depth, fofe_cfg=fofe_cfg)
    critic = make_combined_critic(base_env, hidden=net_cfg.critic_hidden,
                                  depth=net_cfg.depth, fofe_cfg=fofe_cfg)

    # ── Reward normalizer ────────────────────────────────────────────
    reward_normalizer: Optional[RewardNormalizer] = None
    if bool(ppo_cfg.normalize_rewards):
        reward_normalizer = RewardNormalizer(
            num_envs=ppo_cfg.num_envs, gamma=ppo_cfg.gamma, device=device,
        )

    # ── Load checkpoint (BEFORE torch.compile) ───────────────────────
    # Loading must happen before compile: torch.compile wraps the module
    # so state_dict keys gain an `_orig_mod.` prefix, which would mismatch
    # the stripped-prefix state_dicts carried between curriculum sections
    # and the on-disk checkpoint.
    if checkpoint is not None:
        try:
            if "policy_state_dict" in checkpoint:
                policy.load_state_dict(checkpoint["policy_state_dict"])
            if "critic_state_dict" in checkpoint:
                critic.load_state_dict(checkpoint["critic_state_dict"])
            if reward_normalizer is not None and checkpoint.get("reward_normalizer_state_dict") is not None:
                reward_normalizer.load_state_dict(checkpoint["reward_normalizer_state_dict"])
        except Exception as exc:
            print(f"checkpoint load warning (continuing): {type(exc).__name__}: {exc}")

    # ── Optional torch.compile ────────────────────────────────────────
    # Compiles the underlying actor/critic nets in place. The wrappers
    # (CombinedPolicy / CombinedCritic) still call them through .striker_*
    # / .jammer_* attributes, so swap is transparent.
    #
    # NOTE: torch.compile() returns lazily — the actual graph is traced and
    # compiled on the FIRST forward call, NOT at torch.compile() time. So a
    # try/except wrapping torch.compile() does not catch backend failures
    # (e.g. missing Triton on Windows). We probe Triton availability upfront
    # and skip compile cleanly if it can't work. The probe is conservative;
    # on Linux with a recent torch install, Triton is bundled.
    if bool(getattr(ppo_cfg, "compile_models", False)) and torch.cuda.is_available():
        try:
            import triton  # noqa: F401
            _triton_ok = True
        except Exception:
            _triton_ok = False
            print(
                "torch.compile: SKIPPED — Triton not available (expected on Windows; "
                "torch.compile's Inductor backend requires Triton). On Linux + GPU, "
                "Triton ships with torch ≥ 2.0 and you'll get the perf gain automatically."
            )
        if _triton_ok:
            try:
                policy.striker_policy = torch.compile(policy.striker_policy)
                policy.jammer_policy  = torch.compile(policy.jammer_policy)
                critic.striker_critic = torch.compile(critic.striker_critic)
                critic.jammer_critic  = torch.compile(critic.jammer_critic)
                print("torch.compile: ENABLED (policy + critic). First iter will be slow (graph trace).")
            except Exception as exc:
                print(f"torch.compile setup failed (continuing uncompiled): {type(exc).__name__}: {exc}")

    # ── Collector ────────────────────────────────────────────────────
    collector = make_collector(env, policy, ppo_cfg.frames_per_batch, ppo_cfg.n_iters, device)

    # ── Per-role optimizers ──────────────────────────────────────────
    striker_actor_optim = optim.Adam(policy.striker_policy.parameters(), lr=ppo_cfg.actor_lr)
    striker_critic_optim = optim.Adam(critic.striker_critic.parameters(), lr=ppo_cfg.critic_lr)
    jammer_actor_optim = optim.Adam(policy.jammer_policy.parameters(), lr=ppo_cfg.actor_lr)
    jammer_critic_optim = optim.Adam(critic.jammer_critic.parameters(), lr=ppo_cfg.critic_lr)

    # ── Logging dict ─────────────────────────────────────────────────
    logs: Dict[str, List[float]] = {
        "train_mean_episode_total_reward": [],
        "striker_loss_policy": [], "striker_loss_value": [],
        "striker_entropy": [], "striker_approx_kl": [],
        "striker_clip_ratio": [], "striker_explained_variance": [],
        "jammer_loss_policy": [], "jammer_loss_value": [],
        "jammer_entropy": [], "jammer_approx_kl": [],
        "jammer_clip_ratio": [], "jammer_explained_variance": [],
        "loss_policy": [], "loss_value": [], "entropy": [],
        "clip_ratio": [], "explained_variance": [],
        "eval_mean_episode_total_reward": [], "eval_survival_rate": [],
        "eval_mean_duration": [], "eval_task_completion_rate": [],
        "eval_targets_destroyed_rate": [],
        "reward_norm_running_mean": [], "reward_norm_running_std": [],
        "raw_reward_mean": [], "raw_reward_std": [],
        "normalized_reward_mean": [], "normalized_reward_std": [],
        "iter_time_s": [],           # total wall time per iteration (incl. eval)
        "iter_time_excl_eval_s": [], # training-only time per iteration (eval subtracted)
        "eval_time_s": [],           # time spent inside evaluate_current_policy (0 on non-eval iters)
    }
    for comp_key in EVAL_REWARD_COMPONENT_KEYS:
        logs[f"train_component_{comp_key}"] = []
        logs[f"eval_component_{comp_key}"] = []
    for fk in FOFE_LOG_KEYS:
        logs[fk] = []

    # ==================================================================
    # MAIN TRAINING LOOP
    # ==================================================================
    _PROFILE_ITERS = int(getattr(ppo_cfg, "profile_iters", 3))
    _ITER_OFFSET = int(getattr(ppo_cfg, "iteration_offset", 0))
    _first_eval_profiled = False  # also print profile on the first eval iteration
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t_iter_start = time.perf_counter()   # true start of iter 0 = before collector first blocks

    # ── Fine-grained profiling (first _PROFILE_ITERS iterations) ──────
    # Enables per-sub-step timing in both the env _step (rollout side)
    # and the trainer's prep + PPO update sections.  Adds ~cuda.sync
    # overhead per lap, so turn off after the window.
    _env_profile_supported = hasattr(base_env, "set_profile_active")
    if _env_profile_supported and _PROFILE_ITERS > 0:
        base_env.set_profile_active(True)

    # ── Hardware monitor (first _PROFILE_ITERS iterations) ────────────
    # Background-thread samples GPU util/mem/power/temp and CPU util/RAM
    # at ~10 Hz so we can spot under-utilisation. Starts before iter 0
    # begins collecting, restarts at the end of each profiled iter.
    _hw_monitor: Optional[_HardwareMonitor] = None
    if _PROFILE_ITERS > 0:
        _hw_monitor = _HardwareMonitor()
        print(f"  [HW probe] {_hw_monitor.device_header()}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _hw_monitor.start()

    for it, td in enumerate(collector):
        # Sync GPU so the clock is read after all pending ops have finished,
        # then record when the batch was delivered by the collector.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t_batch_ready = time.perf_counter()
        _t_rollout_s = _t_batch_ready - _t_iter_start   # time collector spent on rollout
        _timed_out = False

        # ── Fine-profile flags for this iteration ───────────────────
        _fine = (_ITER_OFFSET + it) < _PROFILE_ITERS
        _fine_acc: Dict[str, float] = {}
        _env_buckets: Dict[str, float] = (
            base_env.pop_profile_buckets() if (_fine and _env_profile_supported) else {}
        )

        _tf = _fine_tic(_fine)
        td = td.to(device)
        _tf = _fine_lap(_fine_acc, "prep_to_device", _tf, _fine)

        # ── Reward normalization ─────────────────────────────────────
        if reward_normalizer is not None:
            norm_stats = reward_normalizer.normalize_rollout_td(
                td, reward_key=("next",) + base_env._reward_key, done_key=("next", "done"),
            )
        else:
            raw_r = td.get(("next",) + base_env._reward_key)
            raw_mean = float(raw_r.mean().item()) if raw_r.numel() > 0 else float("nan")
            raw_std = float(raw_r.std(unbiased=False).item()) if raw_r.numel() > 0 else float("nan")
            norm_stats = {"running_mean": float("nan"), "running_std": float("nan"),
                          "raw_reward_mean": raw_mean, "raw_reward_std": raw_std,
                          "normalized_reward_mean": raw_mean, "normalized_reward_std": raw_std}

        prepare_done_keys(td, base_env)
        _tf = _fine_lap(_fine_acc, "prep_reward_norm", _tf, _fine)

        # ── Compute values for current and next states ───────────────
        with torch.no_grad():
            try:
                critic(td)
            except Exception as e:
                print(f"WARNING critic(td) failed: {e}")
        _tf = _fine_lap(_fine_acc, "prep_critic_curr", _tf, _fine)
        with torch.no_grad():
            try:
                nxt = td.get("next").to(device)
                critic(nxt)
                td.set("next", nxt)
            except Exception as e:
                print(f"WARNING critic(nxt) failed: {e}")
        _tf = _fine_lap(_fine_acc, "prep_critic_next", _tf, _fine)

        # ── Extract tensors ──────────────────────────────────────────
        obs_all = td.get(("agents", "observation"))
        actions_all = td.get(("agents", "action"))
        old_log_probs_all = td.get(("agents", "sample_log_prob"))
        rewards_all = td.get(("next", "agents", "reward"))
        values_all = td.get(("agents", "state_value"))
        next_values_all = td.get(("next", "agents", "state_value"))
        dones = td.get(("next", "done"))

        # ── Extract FOFE obs if enabled ──────────────────────────────
        if use_fofe:
            fofe_actor_obs = _extract_fofe_actor_obs(td)
            fofe_critic_obs = _extract_fofe_critic_obs(td)

        # ── Handle time dimension ────────────────────────────────────
        has_time = obs_all.ndim >= 4
        if has_time:
            T, B, transposed = _infer_time_batch(td, ppo_cfg.num_envs)
            if transposed:
                obs_all = obs_all.transpose(0, 1)
                actions_all = actions_all.transpose(0, 1)
                old_log_probs_all = old_log_probs_all.transpose(0, 1)
                rewards_all = rewards_all.transpose(0, 1)
                values_all = values_all.transpose(0, 1)
                next_values_all = next_values_all.transpose(0, 1)
                dones = dones.transpose(0, 1)
                if use_fofe:
                    fofe_actor_obs = _transpose_fofe_dict(fofe_actor_obs)
                    fofe_critic_obs = _transpose_fofe_dict(fofe_critic_obs)

        # ── Split by role ────────────────────────────────────────────
        s_obs = obs_all[..., :ns, :]
        j_obs = obs_all[..., ns:, :]
        s_act = actions_all[..., :ns, :]
        j_act = actions_all[..., ns:, :]
        s_old_lp = old_log_probs_all[..., :ns]
        j_old_lp = old_log_probs_all[..., ns:]
        s_rew = rewards_all[..., :ns, :]
        j_rew = rewards_all[..., ns:, :]
        s_val = values_all[..., :ns, :]
        j_val = values_all[..., ns:, :]
        s_nval = next_values_all[..., :ns, :]
        j_nval = next_values_all[..., ns:, :]

        if use_fofe:
            s_fofe_actor, j_fofe_actor = _split_fofe_by_role(fofe_actor_obs, ns, nj)
        _tf = _fine_lap(_fine_acc, "prep_extract_split", _tf, _fine)

        # ── GAE ──────────────────────────────────────────────────────
        with torch.no_grad():
            if has_time:
                s_adv, s_ret = compute_gae_sequential(s_rew, s_val, s_nval, dones, ppo_cfg.gamma, ppo_cfg.lmbda)
                j_adv, j_ret = compute_gae_sequential(j_rew, j_val, j_nval, dones, ppo_cfg.gamma, ppo_cfg.lmbda)
            else:
                from .utils import compute_gae
                s_adv, s_ret = compute_gae(s_rew, s_val, s_nval, dones, ppo_cfg.gamma, ppo_cfg.lmbda)
                j_adv, j_ret = compute_gae(j_rew, j_val, j_nval, dones, ppo_cfg.gamma, ppo_cfg.lmbda)
        _tf = _fine_lap(_fine_acc, "prep_gae", _tf, _fine)

        # ── Flatten everything to [N, ...] ───────────────────────────
        s_obs_f = s_obs.reshape(-1, ns, s_obs.shape[-1])
        j_obs_f = j_obs.reshape(-1, nj, j_obs.shape[-1])
        s_act_f = s_act.reshape(-1, ns, s_act.shape[-1])
        j_act_f = j_act.reshape(-1, nj, j_act.shape[-1])
        s_old_lp_f = s_old_lp.reshape(-1, ns)
        j_old_lp_f = j_old_lp.reshape(-1, nj)
        s_adv_f = s_adv.reshape(-1, ns, 1)
        j_adv_f = j_adv.reshape(-1, nj, 1)
        s_ret_f = s_ret.reshape(-1, ns, 1)
        j_ret_f = j_ret.reshape(-1, nj, 1)
        s_val_f = s_val.reshape(-1, ns, 1)
        j_val_f = j_val.reshape(-1, nj, 1)

        state_raw = td.get("state")
        if has_time and transposed:
            state_raw = state_raw.transpose(0, 1)
        state_f = state_raw.reshape(-1, state_raw.shape[-1])

        # ── Flatten FOFE observations ────────────────────────────────
        if use_fofe:
            s_fofe_f = _flatten_fofe_dict(s_fofe_actor, ns)
            j_fofe_f = _flatten_fofe_dict(j_fofe_actor, nj)
            crt_fofe_f = {k: v.reshape(-1, *v.shape[2:]) if v.ndim > 2 else v.reshape(-1, *v.shape[1:])
                          for k, v in fofe_critic_obs.items()}

        n_samples = s_obs_f.shape[0]
        _tf = _fine_lap(_fine_acc, "prep_flatten", _tf, _fine)

        # ── PPO update for each role ─────────────────────────────────
        _t_ppo_start = time.perf_counter()
        s_pol_acc = s_val_acc = s_ent_acc = s_kl_acc = s_clip_acc = 0.0
        j_pol_acc = j_val_acc = j_ent_acc = j_kl_acc = j_clip_acc = 0.0
        n_updates = 0
        # FOFE KPI snapshot — collected ONCE per iteration on the first
        # minibatch of the first epoch (was: every minibatch, ~3 s/iter cost).
        # Gated by ppo_cfg.fofe_kpi_every: 1 = every iter, 5 = every 5 iters,
        # 0 = disabled.
        _fofe_kpi_s: Dict[str, float] = {}
        _fofe_kpi_j: Dict[str, float] = {}
        _fofe_every = int(getattr(ppo_cfg, "fofe_kpi_every", 1))
        _global_iter_1based = _ITER_OFFSET + it + 1
        _capture_fofe_this_iter = bool(
            use_fofe and _fofe_every > 0 and (_global_iter_1based % _fofe_every == 0)
        )
        _fofe_captured_s = False
        _fofe_captured_j = False

        for _ in range(ppo_cfg.num_epochs):
            if _timed_out:
                break
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, ppo_cfg.minibatch_size):
                idx = perm[start : start + ppo_cfg.minibatch_size]
                if idx.numel() == 0:
                    continue
                if (ppo_cfg.max_iter_time_s is not None
                        and (time.perf_counter() - _t_batch_ready) > ppo_cfg.max_iter_time_s):
                    _timed_out = True
                    print(
                        f"  [TIMEOUT] Iter {it + 1}: {time.perf_counter() - _t_batch_ready:.1f}s in PPO updates "
                        f"> {ppo_cfg.max_iter_time_s}s limit — skipping remaining minibatches "
                        f"({n_updates} updates completed)."
                    )
                    break

                _tfm = _fine_tic(_fine)

                # ── Striker PPO update ───────────────────────────────
                mb_s_act = s_act_f[idx]
                mb_s_old_lp = s_old_lp_f[idx]
                mb_s_adv = s_adv_f[idx]
                mb_s_ret = s_ret_f[idx]

                if use_fofe:
                    # FOFE path: pass structured obs dict
                    mb_s_fofe = _index_fofe_dict(s_fofe_f, idx)
                    _tfm = _fine_lap(_fine_acc, "ppo_mb_index", _tfm, _fine)
                    with _autocast():
                        s_new_lp, s_entropy = policy.striker_log_prob_entropy(mb_s_fofe, mb_s_act)
                else:
                    mb_s_obs = s_obs_f[idx]
                    _tfm = _fine_lap(_fine_acc, "ppo_mb_index", _tfm, _fine)
                    with _autocast():
                        s_new_lp, s_entropy = policy.striker_log_prob_entropy(mb_s_obs, mb_s_act)

                # PPO loss in fp32 for numerical stability of ratio/exp.
                s_loss_info = ppo_clip_loss(
                    s_new_lp.float(), mb_s_old_lp, mb_s_adv, s_entropy.float(),
                    ppo_cfg.clip_eps, ppo_cfg.entropy_coef,
                )
                _tfm = _fine_lap(_fine_acc, "ppo_s_actor_fwd", _tfm, _fine)

                if use_fofe:
                    # FOFE critic: pass entity sets
                    mb_crt = _index_fofe_dict(crt_fofe_f, idx)
                    with _autocast():
                        s_pred_val = critic.striker_critic(
                            mb_crt["crt_agents_feat"], mb_crt["crt_agents_mask"],
                            mb_crt["crt_targets_feat"], mb_crt["crt_targets_mask"],
                            mb_crt["crt_radars_feat"], mb_crt["crt_radars_mask"],
                            mb_crt["crt_time_feat"],
                        )
                else:
                    mb_state = state_f[idx]
                    with _autocast():
                        s_pred_val = critic.striker_critic(mb_state)

                s_pred_val = critic._broadcast_role_values(s_pred_val.float(), ns, "striker")

                s_vloss = value_loss_fn(s_pred_val, mb_s_ret)
                _tfm = _fine_lap(_fine_acc, "ppo_s_critic_fwd", _tfm, _fine)

                striker_actor_optim.zero_grad(set_to_none=True)
                if _grad_scaler is not None:
                    _grad_scaler.scale(s_loss_info["loss_total"]).backward()
                    _grad_scaler.unscale_(striker_actor_optim)
                else:
                    s_loss_info["loss_total"].backward()
                nn.utils.clip_grad_norm_(policy.striker_policy.parameters(), ppo_cfg.max_grad_norm)
                _tfm = _fine_lap(_fine_acc, "ppo_s_actor_bwd", _tfm, _fine)

                # -- FOFE KPI snapshot (striker): ONCE per iter, on the first
                #    eligible minibatch. Grads must be read between backward
                #    and optim.step() (zero_grad on the next mb wipes them).
                if _capture_fofe_this_iter and not _fofe_captured_s:
                    _fofe_kpi_s = _collect_fofe_kpis(
                        policy.striker_policy._diag_cache,
                        critic.striker_critic._diag_cache,
                        mb_s_fofe, policy.striker_policy, critic.striker_critic,
                    )
                    _fofe_captured_s = True
                _tfm = _fine_lap(_fine_acc, "ppo_fofe_kpi", _tfm, _fine)

                if _grad_scaler is not None:
                    _grad_scaler.step(striker_actor_optim)
                else:
                    striker_actor_optim.step()

                striker_critic_optim.zero_grad(set_to_none=True)
                if _grad_scaler is not None:
                    _grad_scaler.scale(s_vloss).backward()
                    _grad_scaler.unscale_(striker_critic_optim)
                else:
                    s_vloss.backward()
                nn.utils.clip_grad_norm_(critic.striker_critic.parameters(), ppo_cfg.max_grad_norm)
                if _grad_scaler is not None:
                    _grad_scaler.step(striker_critic_optim)
                else:
                    striker_critic_optim.step()
                _tfm = _fine_lap(_fine_acc, "ppo_s_step_and_critic_bwd", _tfm, _fine)

                s_pol_acc += float(s_loss_info["loss_policy"].item())
                s_val_acc += float(s_vloss.item())
                s_ent_acc += float(s_loss_info["entropy_mean"].item())
                s_kl_acc += s_loss_info["approx_kl"]
                s_clip_acc += s_loss_info["clip_fraction"]
                _tfm = _fine_lap(_fine_acc, "ppo_s_stats", _tfm, _fine)

                # ── Jammer PPO update ────────────────────────────────
                mb_j_act = j_act_f[idx]
                mb_j_old_lp = j_old_lp_f[idx]
                mb_j_adv = j_adv_f[idx]
                mb_j_ret = j_ret_f[idx]

                if use_fofe:
                    mb_j_fofe = _index_fofe_dict(j_fofe_f, idx)
                    with _autocast():
                        j_new_lp, j_entropy = policy.jammer_log_prob_entropy(mb_j_fofe, mb_j_act)
                else:
                    mb_j_obs = j_obs_f[idx]
                    with _autocast():
                        j_new_lp, j_entropy = policy.jammer_log_prob_entropy(mb_j_obs, mb_j_act)

                j_loss_info = ppo_clip_loss(
                    j_new_lp.float(), mb_j_old_lp, mb_j_adv, j_entropy.float(),
                    ppo_cfg.clip_eps, ppo_cfg.entropy_coef,
                )
                _tfm = _fine_lap(_fine_acc, "ppo_j_actor_fwd", _tfm, _fine)

                if use_fofe:
                    # Reuse mb_crt from striker (same global state)
                    with _autocast():
                        j_pred_val = critic.jammer_critic(
                            mb_crt["crt_agents_feat"], mb_crt["crt_agents_mask"],
                            mb_crt["crt_targets_feat"], mb_crt["crt_targets_mask"],
                            mb_crt["crt_radars_feat"], mb_crt["crt_radars_mask"],
                            mb_crt["crt_time_feat"],
                        )
                else:
                    with _autocast():
                        j_pred_val = critic.jammer_critic(state_f[idx])

                j_pred_val = critic._broadcast_role_values(j_pred_val.float(), nj, "jammer")

                j_vloss = value_loss_fn(j_pred_val, mb_j_ret)
                _tfm = _fine_lap(_fine_acc, "ppo_j_critic_fwd", _tfm, _fine)

                jammer_actor_optim.zero_grad(set_to_none=True)
                if _grad_scaler is not None:
                    _grad_scaler.scale(j_loss_info["loss_total"]).backward()
                    _grad_scaler.unscale_(jammer_actor_optim)
                else:
                    j_loss_info["loss_total"].backward()
                nn.utils.clip_grad_norm_(policy.jammer_policy.parameters(), ppo_cfg.max_grad_norm)
                _tfm = _fine_lap(_fine_acc, "ppo_j_actor_bwd", _tfm, _fine)

                # -- FOFE KPI snapshot (jammer): ONCE per iter, gated by
                #    _capture_fofe_this_iter / fofe_kpi_every (see striker).
                if _capture_fofe_this_iter and not _fofe_captured_j:
                    _fofe_kpi_j = _collect_fofe_kpis(
                        policy.jammer_policy._diag_cache,
                        critic.jammer_critic._diag_cache,
                        mb_j_fofe, policy.jammer_policy, critic.jammer_critic,
                    )
                    _fofe_captured_j = True
                _tfm = _fine_lap(_fine_acc, "ppo_fofe_kpi", _tfm, _fine)

                if _grad_scaler is not None:
                    _grad_scaler.step(jammer_actor_optim)
                else:
                    jammer_actor_optim.step()

                jammer_critic_optim.zero_grad(set_to_none=True)
                if _grad_scaler is not None:
                    _grad_scaler.scale(j_vloss).backward()
                    _grad_scaler.unscale_(jammer_critic_optim)
                else:
                    j_vloss.backward()
                nn.utils.clip_grad_norm_(critic.jammer_critic.parameters(), ppo_cfg.max_grad_norm)
                if _grad_scaler is not None:
                    _grad_scaler.step(jammer_critic_optim)
                    # GradScaler.update() should be called once per minibatch
                    # AFTER all step() calls — here, after the last optimizer.
                    _grad_scaler.update()
                else:
                    jammer_critic_optim.step()
                _tfm = _fine_lap(_fine_acc, "ppo_j_step_and_critic_bwd", _tfm, _fine)

                j_pol_acc += float(j_loss_info["loss_policy"].item())
                j_val_acc += float(j_vloss.item())
                j_ent_acc += float(j_loss_info["entropy_mean"].item())
                j_kl_acc += j_loss_info["approx_kl"]
                j_clip_acc += j_loss_info["clip_fraction"]
                _tfm = _fine_lap(_fine_acc, "ppo_j_stats", _tfm, _fine)

                n_updates += 1

        # ── Update collector policy weights ──────────────────────────
        try:
            collector.update_policy_weights_()
        except Exception:
            print("policy weight update failed (continuing)")

        _t_post_start = time.perf_counter()

        # ── Compute explained variance per role ──────────────────────
        s_ev = compute_explained_variance(s_ret_f, s_val_f)
        j_ev = compute_explained_variance(j_ret_f, j_val_f)

        # ── Training episode stats ───────────────────────────────────
        train_ep_stats = base_env.pop_episode_stats()
        if train_ep_stats:
            train_mean_episode_total_reward = (
                sum(s["episode_total_reward"] for s in train_ep_stats) / len(train_ep_stats)
            )
        else:
            train_mean_episode_total_reward = float("nan")

        train_component_means = {comp_key: float("nan") for comp_key in EVAL_REWARD_COMPONENT_KEYS}
        if train_ep_stats:
            for comp_key in EVAL_REWARD_COMPONENT_KEYS:
                comp_vals = [
                    float(s.get("episode_component_reward", {}).get(comp_key, float("nan")))
                    for s in train_ep_stats
                ]
                train_component_means[comp_key] = _finite_mean(comp_vals)

        # ── Record logs ──────────────────────────────────────────────
        div = max(1, n_updates)
        logs["train_mean_episode_total_reward"].append(train_mean_episode_total_reward)
        logs["striker_loss_policy"].append(s_pol_acc / div)
        logs["striker_loss_value"].append(s_val_acc / div)
        logs["striker_entropy"].append(s_ent_acc / div)
        logs["striker_approx_kl"].append(s_kl_acc / div)
        logs["striker_clip_ratio"].append(s_clip_acc / div)
        logs["striker_explained_variance"].append(s_ev)
        logs["jammer_loss_policy"].append(j_pol_acc / div)
        logs["jammer_loss_value"].append(j_val_acc / div)
        logs["jammer_entropy"].append(j_ent_acc / div)
        logs["jammer_approx_kl"].append(j_kl_acc / div)
        logs["jammer_clip_ratio"].append(j_clip_acc / div)
        logs["jammer_explained_variance"].append(j_ev)
        logs["loss_policy"].append((s_pol_acc + j_pol_acc) / (2.0 * div))
        logs["loss_value"].append((s_val_acc + j_val_acc) / (2.0 * div))
        logs["entropy"].append((s_ent_acc + j_ent_acc) / (2.0 * div))
        logs["clip_ratio"].append((s_clip_acc + j_clip_acc) / (2.0 * div))
        ev_avg = float("nan")
        if math.isfinite(s_ev) and math.isfinite(j_ev):
            ev_avg = (s_ev + j_ev) / 2.0
        elif math.isfinite(s_ev):
            ev_avg = s_ev
        elif math.isfinite(j_ev):
            ev_avg = j_ev
        logs["explained_variance"].append(ev_avg)
        logs["reward_norm_running_mean"].append(norm_stats["running_mean"])
        logs["reward_norm_running_std"].append(norm_stats["running_std"])
        logs["raw_reward_mean"].append(norm_stats["raw_reward_mean"])
        logs["raw_reward_std"].append(norm_stats["raw_reward_std"])
        logs["normalized_reward_mean"].append(norm_stats["normalized_reward_mean"])
        logs["normalized_reward_std"].append(norm_stats["normalized_reward_std"])
        for comp_key in EVAL_REWARD_COMPONENT_KEYS:
            logs[f"train_component_{comp_key}"].append(train_component_means[comp_key])

        # ── Record FOFE KPIs ──────────────────────────────────────────
        if use_fofe and _fofe_kpi_s and _fofe_kpi_j:
            for metric_key, val in _fofe_kpi_s.items():
                logs[f"fofe_striker_{metric_key}"].append(val)
            for metric_key, val in _fofe_kpi_j.items():
                logs[f"fofe_jammer_{metric_key}"].append(val)
        else:
            for fk in FOFE_LOG_KEYS:
                logs[fk].append(float("nan"))

        # ── Evaluation ───────────────────────────────────────────────
        global_iter_1based = _ITER_OFFSET + it + 1
        do_eval = bool(ppo_cfg.log_every) and (global_iter_1based % ppo_cfg.log_every == 0)
        if do_eval:
            _t_eval_start = time.perf_counter()
            eval_metrics = evaluate_current_policy(policy, env_cfg, ppo_cfg, hf_radar_cfg=hf_radar_cfg)
            _t_eval_s = time.perf_counter() - _t_eval_start
            logs["eval_mean_episode_total_reward"].append(eval_metrics["eval_mean_episode_total_reward"])
            logs["eval_survival_rate"].append(eval_metrics["eval_survival_rate"])
            logs["eval_mean_duration"].append(eval_metrics["eval_mean_duration"])
            logs["eval_task_completion_rate"].append(eval_metrics["eval_task_completion_rate"])
            logs["eval_targets_destroyed_rate"].append(eval_metrics["eval_targets_destroyed_rate"])
            for comp_key in EVAL_REWARD_COMPONENT_KEYS:
                logs[f"eval_component_{comp_key}"].append(eval_metrics[f"eval_component_{comp_key}"])
        else:
            logs["eval_mean_episode_total_reward"].append(float("nan"))
            logs["eval_survival_rate"].append(float("nan"))
            logs["eval_mean_duration"].append(float("nan"))
            logs["eval_task_completion_rate"].append(float("nan"))
            logs["eval_targets_destroyed_rate"].append(float("nan"))
            for comp_key in EVAL_REWARD_COMPONENT_KEYS:
                logs[f"eval_component_{comp_key}"].append(float("nan"))

        # Sync GPU so all remaining async ops finish before we read the clock.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t_iter_end = time.perf_counter()

        # ── True wall-clock breakdown ─────────────────────────────────
        # rollout : collector blocking (before this iteration's body ran)
        # prep    : td.to(device) + norm + critic inference + GAE + reshape
        # ppo     : all forward/backward passes across epochs/minibatches
        # post    : EV + episode stats + logging  (NOT eval — eval tracked separately)
        # eval    : evaluate_current_policy (0 s on non-eval iterations)
        _t_prep_s     = _t_ppo_start  - _t_batch_ready
        _t_ppo_s      = _t_post_start - _t_ppo_start
        _t_eval_s_val = _t_eval_s if do_eval else 0.0
        _t_post_s     = _t_iter_end   - _t_post_start - _t_eval_s_val
        _iter_total_s = _t_iter_end   - _t_iter_start   # true wall time incl. rollout + eval
        logs["iter_time_s"].append(_iter_total_s)
        logs["iter_time_excl_eval_s"].append(_iter_total_s - _t_eval_s_val)
        logs["eval_time_s"].append(_t_eval_s_val)

        # ── Profile print ─────────────────────────────────────────────
        # Print for the first _PROFILE_ITERS iterations AND for the first
        # eval iteration (so both non-eval and eval timings appear in the log).
        _do_profile = ((_ITER_OFFSET + it) < _PROFILE_ITERS) or (do_eval and not _first_eval_profiled)
        if _do_profile:
            if do_eval:
                _first_eval_profiled = True
            print(
                f"  [PROFILE iter {global_iter_1based}] "
                f"total={_iter_total_s:.2f}s | "
                f"rollout={_t_rollout_s:.2f}s | "
                f"prep(norm+critic+GAE+reshape)={_t_prep_s:.2f}s | "
                f"ppo_updates={_t_ppo_s:.2f}s | "
                f"post(EV+stats+log)={_t_post_s:.2f}s"
                + (f" | eval={_t_eval_s_val:.2f}s" if do_eval else "")
                + (" [TIMED OUT]" if _timed_out else "")
            )

        # ── Hardware monitor snapshot ─────────────────────────────────
        # Stop the background sampler, print mean/peak for this iter,
        # then restart it if more profile iters remain. Tied to the same
        # _PROFILE_ITERS window as the fine-grained timing buckets.
        if _hw_monitor is not None and (_ITER_OFFSET + it) < _PROFILE_ITERS:
            _hw_monitor.stop()
            print(f"  [HW iter {global_iter_1based}] {_hw_monitor.summary()}")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if (_ITER_OFFSET + it + 1) < _PROFILE_ITERS:
                _hw_monitor.start()
            else:
                _hw_monitor = None  # window done — release the thread

        # ── Fine-grained breakdown (only for first _PROFILE_ITERS iters) ──
        if _fine:
            env_buckets = dict(_env_buckets)  # already popped at iter start
            env_total = sum(env_buckets.values())
            prep_buckets = {k: v for k, v in _fine_acc.items() if k.startswith("prep_")}
            ppo_buckets  = {k: v for k, v in _fine_acc.items() if k.startswith("ppo_")}
            prep_total = sum(prep_buckets.values())
            ppo_total  = sum(ppo_buckets.values())
            print(
                f"  [FINE iter {global_iter_1based}] "
                f"rollout_env_sum={env_total:.2f}s ({_format_buckets(env_buckets, top=14)})"
            )
            print(
                f"  [FINE iter {global_iter_1based}] "
                f"prep_sum={prep_total:.2f}s ({_format_buckets(prep_buckets, top=8)})"
            )
            print(
                f"  [FINE iter {global_iter_1based}] "
                f"ppo_sum={ppo_total:.2f}s ({_format_buckets(ppo_buckets, top=14)}) "
                f"[n_updates={n_updates}]"
            )

        # ── Disable env fine-profiling once window is over ────────────
        if _env_profile_supported and (_ITER_OFFSET + it + 1) >= _PROFILE_ITERS \
                and getattr(base_env, "_profile_active", False):
            base_env.set_profile_active(False)

        # ── Reset iteration start for next iteration ──────────────────
        _t_iter_start = _t_iter_end

        # ── Print ────────────────────────────────────────────────────
        if do_eval:
            norm_log = ""
            if bool(ppo_cfg.normalize_rewards):
                norm_log = (
                    f" | raw_s {norm_stats['raw_reward_std']:.4f}"
                    f" | norm_s {norm_stats['normalized_reward_std']:.4f}"
                    f" | ret_std {norm_stats['running_std']:.4f}"
                    f" | ret_m {norm_stats['running_mean']:.4f}"
                )
            print(
                f"Iter {global_iter_1based:4d} | "
                f"train_ret {logs['train_mean_episode_total_reward'][-1]: .3f} | "
                f"eval_ret {logs['eval_mean_episode_total_reward'][-1]: .3f} | "
                f"comp {logs['eval_task_completion_rate'][-1]:.2f} | "
                f"tgt {logs['eval_targets_destroyed_rate'][-1]:.2f} | "
                f"surv {logs['eval_survival_rate'][-1]:.2f} | "
                f"dur {logs['eval_mean_duration'][-1]:.1f} | "
                f"S[pi {logs['striker_loss_policy'][-1]:.4f} "
                f"V {logs['striker_loss_value'][-1]:.4f} "
                f"H {logs['striker_entropy'][-1]:.4f} "
                f"ev {logs['striker_explained_variance'][-1]:.4f}] | "
                f"J[pi {logs['jammer_loss_policy'][-1]:.4f} "
                f"V {logs['jammer_loss_value'][-1]:.4f} "
                f"H {logs['jammer_entropy'][-1]:.4f} "
                f"ev {logs['jammer_explained_variance'][-1]:.4f}]"
                f"{norm_log}"
            )
            if use_fofe and _fofe_kpi_s and _fofe_kpi_j:
                print(
                    f"  FOFE["
                    f"S a/t/r dom {_fofe_kpi_s.get('dominance_agents',0):.2f}/"
                    f"{_fofe_kpi_s.get('dominance_targets',0):.2f}/"
                    f"{_fofe_kpi_s.get('dominance_radars',0):.2f} "
                    f"col_t {_fofe_kpi_s.get('collapse_targets',0):.3f} "
                    f"vis_t {_fofe_kpi_s.get('visible_targets_mean',0):.1f} "
                    f"grad {_fofe_kpi_s.get('actor_see_grad_norm',0):.4f} | "
                    f"J a/t/r dom {_fofe_kpi_j.get('dominance_agents',0):.2f}/"
                    f"{_fofe_kpi_j.get('dominance_targets',0):.2f}/"
                    f"{_fofe_kpi_j.get('dominance_radars',0):.2f} "
                    f"col_t {_fofe_kpi_j.get('collapse_targets',0):.3f} "
                    f"vis_t {_fofe_kpi_j.get('visible_targets_mean',0):.1f} "
                    f"grad {_fofe_kpi_j.get('actor_see_grad_norm',0):.4f}]"
                )

        if it + 1 >= ppo_cfg.n_iters:
            break

    try:
        collector.shutdown()
    except Exception:
        pass

    return base_env, policy, critic, logs, reward_normalizer
