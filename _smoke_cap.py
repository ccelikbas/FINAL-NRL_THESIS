"""Smoke test for the jammer_max_jammed_radars capacity cap.

Builds a scenario with one jammer and FOUR radars laid out so that all
four fall inside the jammer's wide cone (jammer_main_lobe_deg = 120).
Verifies:
  - K=2 (default): _jammer_in_cone keeps exactly the 2 closest radars
  - K=1: keeps exactly the 1 closest
  - K=None (unlimited): keeps all 4 (legacy behaviour)
  - K=10 (>= R): keeps all 4 (no-op)
  - _radar_jammed_flag agrees with the restricted set
  - crt_radars_feat sin/cos columns are non-zero only on jammed radars
"""
from __future__ import annotations

import math
import os
import sys
import importlib.util

ROOT = os.path.abspath(os.path.dirname(__file__))
PKG_DIR = os.path.join(ROOT, "0_TA_HF_FOFE-MAPPO")
PKG_NAME = "tapkg"

init_spec = importlib.util.spec_from_file_location(
    PKG_NAME, os.path.join(PKG_DIR, "__init__.py"),
    submodule_search_locations=[PKG_DIR],
)
pkg = importlib.util.module_from_spec(init_spec)
sys.modules[PKG_NAME] = pkg
init_spec.loader.exec_module(pkg)

for sub in ("config", "rewards", "environment", "HF_environment"):
    s = importlib.util.spec_from_file_location(
        f"{PKG_NAME}.{sub}", os.path.join(PKG_DIR, sub + ".py")
    )
    m = importlib.util.module_from_spec(s)
    sys.modules[f"{PKG_NAME}.{sub}"] = m
    s.loader.exec_module(m)

import torch  # noqa: E402

cfg_mod = sys.modules[f"{PKG_NAME}.config"]
hf_mod = sys.modules[f"{PKG_NAME}.HF_environment"]


def build_env(K):
    hf_cfg = cfg_mod.HFRadarConfig(jammer_max_jammed_radars=K)
    env = hf_mod.HFStrikeEA2DEnv(
        hf_cfg=hf_cfg,
        num_envs=1,
        max_steps=10,
        device=torch.device("cpu"),
        seed=0,
        n_strikers=1,
        n_jammers=1,
        n_targets=1,
        n_radars=4,
        n_known_targets=1,
        n_unknown_targets=0,
        n_known_radars=4,
        n_unknown_radars=0,
        use_fofe=True,
    )
    return env


def setup_geom_and_inspect(env, label):
    # Place jammer at origin pointing along +x. Place 4 radars at
    # increasing distances along +x so all sit inside the 120 deg cone.
    env.reset()
    env.agent_pos[0, env.n_strikers, 0] = 0.5
    env.agent_pos[0, env.n_strikers, 1] = 0.5
    env.agent_heading[0, env.n_strikers] = 0.0
    env.jammer_bearing[0, 0] = 0.0   # beam along +x
    # 4 radars to the +x side, increasing distance
    for i, dx in enumerate([0.05, 0.10, 0.15, 0.20]):
        env.radar_pos[0, i, 0] = 0.5 + dx
        env.radar_pos[0, i, 1] = 0.5
    env._update_geometry_cache()   # refreshes _c_dist_ar and _jammer_in_cone

    jic = env._jammer_in_cone[0, 0].tolist()             # [R] bool
    jammed = env._radar_jammed_flag()[0].tolist()        # [R] float
    crt_feat = env._build_fofe_critic_state()["crt_radars_feat"][0]
    sin_c = crt_feat[..., 3].tolist()
    cos_c = crt_feat[..., 4].tolist()
    norm = [s * s + c * c for s, c in zip(sin_c, cos_c)]
    print(f"-- {label} --")
    print(f"  cap = {env._jammer_max_jammed_radars}")
    print(f"  _jammer_in_cone[0, 0]: {jic}")
    print(f"  _radar_jammed_flag   : {jammed}")
    print(f"  crt sin^2+cos^2      : {[f'{n:.3f}' for n in norm]}")
    return jic, jammed, norm


def main():
    # Distances along +x from jammer: 0.05, 0.10, 0.15, 0.20  →  closest is r=0
    print("=" * 60)
    env_k2 = build_env(K=2)
    jic2, j2, n2 = setup_geom_and_inspect(env_k2, "K = 2 (default)")
    # Expect exactly the 2 closest radars (indices 0 and 1) to be active.
    assert jic2 == [True, True, False, False], f"K=2: got {jic2}"
    assert [j > 0.5 for j in j2] == [True, True, False, False]
    # sin/cos populated only on jammed radars.
    for i, n in enumerate(n2):
        if j2[i] > 0.5:
            assert abs(n - 1.0) < 1e-5
        else:
            assert n < 1e-10

    print()
    env_k1 = build_env(K=1)
    jic1, j1, n1 = setup_geom_and_inspect(env_k1, "K = 1")
    assert jic1 == [True, False, False, False], f"K=1: got {jic1}"

    print()
    env_un = build_env(K=None)
    jicu, ju, nu = setup_geom_and_inspect(env_un, "K = None (unlimited)")
    assert jicu == [True, True, True, True], f"K=None: got {jicu}"

    print()
    env_k10 = build_env(K=10)
    jic10, j10, n10 = setup_geom_and_inspect(env_k10, "K = 10 (>= R = 4, no-op)")
    assert jic10 == [True, True, True, True], f"K=10: got {jic10}"

    print()
    print("=" * 60)
    print("ALL OK")


if __name__ == "__main__":
    main()
