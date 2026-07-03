"""summarize_policy.py — print a brief, readable overview of a trained checkpoint:
its training curriculum, reward space, observation space, network, and final eval
metrics. Read-only (never modifies the checkpoint).

Run (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\summarize_policy.py \\
      --checkpoint runs\\2s2-4jV7.pt
(--checkpoint accepts an absolute path, a repo-root-relative path, or a bare name
that resolves under 0_TA_HF_FOFE-MAPPO/runs/.)
"""
from __future__ import annotations
import argparse, re, sys, types
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent           # .../0_TA_HF_FOFE-MAPPO/eval_tools
_PKG_DIR = _THIS_DIR.parent                            # .../0_TA_HF_FOFE-MAPPO
_RUNS_DIR = _PKG_DIR / "runs"
_PKG_NAME = "fofe_mappo"
if __package__ in (None, ""):
    sys.path.insert(0, str(_PKG_DIR.parent))
    if _PKG_NAME not in sys.modules:
        _pkg = types.ModuleType(_PKG_NAME)
        _pkg.__path__ = [str(_PKG_DIR), str(_THIS_DIR)]
        _pkg.__package__ = _PKG_NAME
        _pkg.__file__ = str(_THIS_DIR / "__init__.py")
        sys.modules[_PKG_NAME] = _pkg
    __package__ = _PKG_NAME

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from .config import PPOConfig
from .trainer import build_env
from .evaluate_policy import _LoadedCheckpoint

# ═══════════════════════════════════════════════════════════════════
#  >>>  EDIT THIS — the policy to summarise  <<<
#  Bare name → runs/ ; or a repo-root-relative / absolute path.
#  (The --checkpoint CLI flag, if given, overrides this.)
# ═══════════════════════════════════════════════════════════════════
POLICY_PATH = "2s2-4jV6.pt"

W = 66  # display width


def _rule(title=""):
    if not title:
        return "─" * W
    return f"── {title} " + "─" * max(0, W - len(title) - 4)


def _resolve(p):
    p = Path(p)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    if (_RUNS_DIR / p).exists():
        return _RUNS_DIR / p
    return (_PKG_DIR / p) if (_PKG_DIR / p).exists() else (_PKG_DIR.parent / p)


def _net_d_self(state_dict):
    for k, v in (state_dict or {}).items():
        if k.endswith("self_mlp.0.weight") and hasattr(v, "ndim") and v.ndim == 2:
            return int(v.shape[1])
    return None


def _counts_from_label(label):
    """(n_strikers, n_jammers_max, n_radars, n_targets) from a curriculum label
    like 'S2 J[2-4] kT2 kR6 kill0.1 steps150 S2 commON'."""
    def g(pat):
        m = re.search(pat, label or "")
        return int(m.group(1)) if m else None
    mj = re.search(r"J\[?(\d+)(?:-(\d+))?", label or "")
    nj = int(mj.group(2) or mj.group(1)) if mj else None   # DR range -> allocate at max
    return g(r"^S(\d+)"), nj, g(r"kR\[?(\d+)"), g(r"kT\[?(\d+)")


def _last(logs, key):
    v = logs.get(key) if isinstance(logs, dict) else None
    if isinstance(v, (list, tuple)) and len(v):
        return v[-1]
    return None


def _fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=POLICY_PATH,
                    help=f"policy .pt (default: POLICY_PATH = {POLICY_PATH!r})")
    args = ap.parse_args()
    path = _resolve(args.checkpoint)

    raw = torch.load(path, map_location="cpu", weights_only=False)
    ck = _LoadedCheckpoint(path, torch.device("cpu"))
    ec = ck.base_env_cfg
    rc = raw.get("reward_cfg") or ec.reward_config   # prefer the BAKED reward config
    net = raw.get("net_cfg")
    fofe = raw.get("fofe_cfg")
    logs = raw.get("training_logs", {}) or {}
    curr = raw.get("curriculum", []) or []
    bounds = raw.get("section_bounds", []) or []
    use_fofe = bool(getattr(fofe, "use_fofe", False))
    size_mb = path.stat().st_size / 1e6

    print("═" * W)
    print(f"  POLICY OVERVIEW — {path.name}")
    print("═" * W)
    print(f"file       : {path}")
    print(f"             {size_mb:.1f} MB")
    print(f"producer   : {getattr(ck, 'producer', '?')}    FOFE: {'ON' if use_fofe else 'OFF'}"
          f"    reward-norm: {'ON' if raw.get('reward_normalizer_state_dict') else 'OFF'}")

    # ── training / curriculum ──
    print("\n" + _rule("TRAINING / CURRICULUM"))
    total = sum(int(s.get("n_iters", 0)) for s in curr) if curr else None
    print(f"total: {total if total is not None else '?'} iters across {len(curr)} section(s)")
    for i, s in enumerate(curr):
        lo, hi = (bounds[i][1], bounds[i][2]) if i < len(bounds) else (None, None)
        rng = f"[{lo:>5},{hi:>5})" if lo is not None else " " * 13
        print(f"  {rng}  {s.get('label', s.get('name', '?'))}")
    surv = _last(logs, "eval_survival_rate"); comp = _last(logs, "eval_task_completion_rate")
    tgt = _last(logs, "eval_targets_destroyed_rate"); frag = _last(logs, "eval_coalition_fragmentation")
    print(f"final eval : survival {_fmt(surv)} | completion {_fmt(comp)} | "
          f"targets {_fmt(tgt)} | frag {_fmt(frag, 3)}")

    # ── reward space (active terms) ──
    print("\n" + _rule("REWARD SPACE (non-zero terms)"))
    g = lambda n, d=0.0: float(getattr(rc, n, d))
    print(f"mission     target_destroyed {g('target_destroyed'):g}  terminal {g('terminal_bonus'):g}  "
          f"timestep {g('timestep_penalty'):g}  agent_destroyed {g('agent_destroyed'):g}  "
          f"team_spirit {g('team_spirit'):g}")
    if g('escort_striker_scale') or g('escort_jammer_scale') or g('escort_over_scale') or g('jammer_escort_approach_scale'):
        kt = str(getattr(rc, 'escort_kernel_type', 'exp')).lower()
        kern = (f"sigmoid R={g('escort_kernel_radius'):g} s={g('escort_kernel_softness'):g}"
                if kt == "sigmoid" else f"exp ℓ={g('escort_kernel_length'):g}")
        print(f"escort      kernel={kern}  κ={g('escort_capacity'):g}  |  "
              f"w_s {g('escort_striker_scale'):g}  w_j {g('escort_jammer_scale'):g}  "
              f"w_over {g('escort_over_scale'):g}  w_a {g('jammer_escort_approach_scale'):g}")
    if g('target_cover_scale'):
        print(f"target-cover  w_st {g('target_cover_scale'):g}  κ_t {g('target_cover_capacity'):g}  "
              f"ℓ_t {g('target_cover_kernel_length'):g}  τ_t {g('target_cover_commit_temp'):g}")
    fparts = []
    if g('striker_formation_scale'):
        fparts.append(f"striker {g('striker_formation_scale'):g} (ref {g('striker_formation_ref_dist'):g})")
    if g('jammer_formation_scale'):
        fparts.append(f"jammer {g('jammer_formation_scale'):g} (ref {g('jammer_formation_ref_dist'):g})")
    if fparts:
        print("formation   " + "   ".join(fparts))
    aparts = []
    if g('striker_approach_w_lin'):
        aparts.append(f"striker {g('striker_approach_w_lin'):g}")
    if g('jammer_approach_w_lin'):
        aparts.append(f"jammer {g('jammer_approach_w_lin'):g}")
    if aparts:
        print("approach    " + "   ".join(aparts))
    hf = [("beam_align", g('jammer_beam_alignment_scale')), ("beam_on_radar", g('jammer_beam_on_radar_bonus')),
          ("hf_exposed", g('hf_margin_exposed_penalty')), ("hf_protected", g('hf_margin_protected_penalty'))]
    hf = [f"{n} {v:g}" for n, v in hf if v]
    if hf:
        print("hf/jam      " + "  ".join(hf))
    eff = [("accel", g('accel_effort_scale')), ("angular", g('angular_effort_scale')), ("beam", g('beam_accel_effort_scale'))]
    eff = [f"{n} {v:g}" for n, v in eff if v]
    if eff:
        print("control     " + "  ".join(eff))

    # ── observation space ──
    print("\n" + _rule("OBSERVATION SPACE"))
    net_ds = _net_d_self(raw.get("policy_state_dict"))
    try:
        ec._use_fofe = use_fofe
        # counts (esp. n_radars) drive the critic state dim; the reconstructed
        # base_env_cfg can be wrong, so take them from the final curriculum label.
        if curr:
            ns, nj, nr, nt = _counts_from_label(curr[-1].get("label", ""))
            if ns: ec.n_strikers = ns
            if nj: ec.n_jammers = nj
            if nr: ec.n_radars = ec.n_known_radars = nr; ec.n_unknown_radars = 0
            if nt: ec.n_targets = ec.n_known_targets = nt; ec.n_unknown_targets = 0
        if hasattr(ec, "dr"):
            ec.dr = None
        env = build_env(ec, PPOConfig(num_envs=2, device="cpu"), hf_radar_cfg=ck.hf_radar_cfg)
        d_self = env.d_self; se = env._self_extra_dim()
        n_oa, oa_e = env.n_other_agent_obs_slots, env._other_agent_extra_dim()
        n_rd, rd_e = env.n_radar_obs_slots, env._radar_extra_dim()
        n_tg = env.n_target_obs_slots
        obs_dim, state_dim, ce = env._compute_obs_dim(), env._compute_state_dim(), env._critic_agent_extra_dim()
        print(f"encoding   : {'FOFE (permutation-invariant)' if use_fofe else 'flat / zero-padded MLP'}")
        print(f"actor obs  = {obs_dim} floats")
        print(f"  self ({d_self})        : x,y,speed,heading,heading_rate,t_norm"
              + (f" + {se} extra" if se else ""))
        print(f"  other-agent × {n_oa}    : dx,dy,dist,heading,role" + (f" + {oa_e} extra" if oa_e else ""))
        print(f"  radar × {n_rd}          : dx,dy,dist,jammed" + (f" + {rd_e} extra" if rd_e else ""))
        print(f"  target × {n_tg}         : dx,dy,dist,alive")
        print(f"critic state = {state_dim} floats  (per-agent 7+{ce}, targets 3×T, radars 3×R, +1)")
        # striker soft-count feature detection (self-extra col 2 == c_i/kappa)
        baked = net_ds if net_ds is not None else d_self
        has_feat = baked is not None and baked >= 9
        note = "" if (net_ds is None or net_ds == d_self) else f"  [!] checkpoint d_self={net_ds} ≠ current-code {d_self}"
        print(f"striker soft-count feature (c_i/κ) : {'PRESENT' if has_feat else 'ABSENT'}"
              f"  (d_self={baked}){note}")
    except Exception as exc:  # noqa: BLE001
        print(f"encoding   : {'FOFE' if use_fofe else 'flat MLP'}   (env build failed: {type(exc).__name__})")
        if net_ds is not None:
            print(f"self obs dim (from network) = {net_ds}  "
                  f"→ striker soft-count feature: {'PRESENT' if net_ds >= 9 else 'ABSENT'}")

    # ── network ──
    if net is not None:
        print("\n" + _rule("NETWORK"))
        print(f"actor_hidden {getattr(net, 'actor_hidden', '?')}   "
              f"critic_hidden {getattr(net, 'critic_hidden', '?')}   depth {getattr(net, 'depth', '?')}")
        if use_fofe and fofe is not None:
            print(f"FOFE see-dims: agents {getattr(fofe, 'agents_see_dims', '?')}  "
                  f"targets {getattr(fofe, 'targets_see_dims', '?')}  radars {getattr(fofe, 'radars_see_dims', '?')}")
    print("═" * W)


if __name__ == "__main__":
    main()
