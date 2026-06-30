"""reward_scale_analysis.py — measure the ACTUAL per-component reward magnitudes
of a trained policy, to see whether the shaping terms drown out the mission/other
rewards. Read-only.

Rolls out `--num_envs` independent episodes at a fixed (n_strikers,n_jammers),
accumulates each reward component (from env.last_reward_components) per episode
while the env is still active, and prints a table ranked by |mean episode total|
with each component's share of the total absolute reward, plus per-step means.

Run (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\reward_scale_analysis.py \\
      --checkpoint runs\\composition_agnostic_k2.pt --n_strikers 2 --n_jammers 4
"""
from __future__ import annotations
import argparse, sys, types
from collections import defaultdict
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PKG_NAME = "fofe_mappo"
if __package__ in (None, ""):
    sys.path.insert(0, str(_THIS_DIR.parent))
    if _PKG_NAME not in sys.modules:
        _pkg = types.ModuleType(_PKG_NAME)
        _pkg.__path__ = [str(_THIS_DIR)]
        _pkg.__package__ = _PKG_NAME
        _pkg.__file__ = str(_THIS_DIR / "__init__.py")
        sys.modules[_PKG_NAME] = _pkg
    __package__ = _PKG_NAME

import numpy as np
import torch
from .HF_environment import HFStrikeEA2DEnv
from .evaluate_policy import _LoadedCheckpoint, _build_policy_for_scenario

# Components I (the assistant) added/own — flagged in the table.
MINE = {"escort", "target_cover"}


def build_hf_env(ec, hf_cfg, num_envs, device, seed):
    return HFStrikeEA2DEnv(
        hf_cfg=hf_cfg, num_envs=num_envs, max_steps=ec.max_steps, device=device, seed=seed,
        n_strikers=ec.n_strikers, n_jammers=ec.n_jammers, n_targets=ec.n_targets, n_radars=ec.n_radars,
        n_known_targets=ec.n_known_targets, n_unknown_targets=ec.n_unknown_targets,
        n_known_radars=ec.n_known_radars, n_unknown_radars=ec.n_unknown_radars,
        dt=ec.dt, world_bounds=ec.world_bounds, v_max=ec.v_max, accel_magnitude=ec.accel_magnitude,
        dpsi_max=ec.dpsi_max, h_accel_magnitude_fraction=ec.h_accel_magnitude_fraction,
        min_turn_radius=ec.min_turn_radius, R_obs=ec.R_obs, R_comm=ec.R_comm,
        communicate=getattr(ec, "communicate", True),
        n_other_agent_obs_slots=getattr(ec, "n_other_agent_obs_slots", 3),
        n_radar_obs_slots=getattr(ec, "n_radar_obs_slots", 2),
        n_target_obs_slots=getattr(ec, "n_target_obs_slots", 2),
        striker_engage_range=ec.striker_engage_range, striker_engage_fov=ec.striker_engage_fov,
        striker_v_min=ec.striker_v_min, jammer_jam_radius=ec.jammer_jam_radius,
        jammer_jam_effect=ec.jammer_jam_effect, jammer_v_min=ec.jammer_v_min,
        radar_range=ec.radar_range, radar_kill_probability=ec.radar_kill_probability,
        border_thresh=ec.border_thresh, reward_config=ec.reward_config,
        target_spawn_angle_range=ec.target_spawn_angle_range, n_env_layouts=ec.n_env_layouts,
        radar_min_sep=getattr(ec, "radar_min_sep", 0.5), scenario=getattr(ec, "scenario", "S1"),
        s2_radar_min_sep=getattr(ec, "s2_radar_min_sep", 0.2),
        s2_target_min_sep=getattr(ec, "s2_target_min_sep", 0.2),
        use_fofe=getattr(ec, "use_fofe", False),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/composition_agnostic_k2.pt")
    ap.add_argument("--n_strikers", type=int, default=2)
    ap.add_argument("--n_jammers", type=int, default=4)
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _LoadedCheckpoint(Path(args.checkpoint), device)
    ec = ckpt.base_env_cfg
    ec.n_strikers, ec.n_jammers = args.n_strikers, args.n_jammers
    if hasattr(ec, "dr"):
        ec.dr = None
    rc = ec.reward_config
    print(f"ckpt={args.checkpoint}  scenario={getattr(ec,'scenario','?')}  "
          f"ns={ec.n_strikers} nj={ec.n_jammers}  num_envs={args.num_envs}")
    print(f"escort: ell={rc.escort_kernel_length} kappa={rc.escort_capacity} w_s={rc.escort_striker_scale} "
          f"w_j={rc.escort_jammer_scale} w_over={getattr(rc,'escort_over_scale','?')} "
          f"w_a={getattr(rc,'jammer_escort_approach_scale','?')} | target_cover w_st={rc.target_cover_scale} "
          f"kt={rc.target_cover_capacity} | striker_approach w_lin={rc.striker_approach_w_lin} "
          f"target_destroyed={rc.target_destroyed} timestep={rc.timestep_penalty}\n")

    policy = _build_policy_for_scenario(ckpt, ec, device)
    policy.eval(); policy.deterministic = True
    B = args.num_envs
    env = build_hf_env(ec, ckpt.hf_radar_cfg, B, device, args.seed)

    with torch.no_grad():
        td = env.reset()
        done = torch.zeros(B, dtype=torch.bool, device=device)
        ep_len = torch.zeros(B, device=device)
        comp = defaultdict(lambda: torch.zeros(B, device=device))   # signed Σ over agents, per env
        absc = defaultdict(lambda: torch.zeros(B, device=device))   # Σ|·| over agents, per env
        for _ in range(env.max_steps):
            td = policy(td); td = env.step(td)
            active = (~done).float()
            for k, v in env.last_reward_components.items():
                comp[k] += v.sum(dim=-1) * active
                absc[k] += v.abs().sum(dim=-1) * active
            ep_len += active
            done = done | td.get(("next", "done")).reshape(B).bool()
            if bool(done.all()):
                break
            td = td.get("next")

    mean_len = float(ep_len.mean())
    # mean per-episode signed total + mean per-episode Σ|·| (magnitude) per component
    rows = []
    for k in comp:
        ep_signed = float(comp[k].mean())
        ep_absmag = float(absc[k].mean())
        rows.append((k, ep_signed, ep_absmag))
    total_absmag = sum(r[2] for r in rows) or 1.0
    rows.sort(key=lambda r: -r[2])

    print(f"mean episode length = {mean_len:.1f} steps   (n={B} episodes, {args.n_strikers}s{args.n_jammers}j)\n")
    print(f"{'component':<28}{'ep_signed':>12}{'ep_|mag|':>12}{'%|reward|':>11}{'per-step':>11}  owner")
    print("-" * 86)
    for k, sgn, mag in rows:
        share = 100.0 * mag / total_absmag
        per_step = sgn / max(mean_len, 1.0)
        owner = "<< MINE" if k in MINE else ""
        print(f"{k:<28}{sgn:>12.2f}{mag:>12.2f}{share:>10.1f}%{per_step:>11.4f}  {owner}")
    mine_share = 100.0 * sum(r[2] for r in rows if r[0] in MINE) / total_absmag
    print("-" * 86)
    print(f"MY shaping (escort+target_cover) = {mine_share:.1f}% of total |reward| per episode")


if __name__ == "__main__":
    main()
