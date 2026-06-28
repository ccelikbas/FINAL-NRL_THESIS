"""Diagnostic: roll out a checkpoint and inspect coalition structure.

Confirms the failure mode behind frag≈0 (are both strikers on one target? are
jammers committed to / straddling strikers?). Saves a trajectory figure and
prints numeric summaries. Read-only — does not modify any checkpoint.

Run:  .\.venv\Scripts\python.exe 0_TA_HF_FOFE-MAPPO\_diag_softcommit.py --checkpoint runs\fofe_mappo_softcommit.pt
"""
from __future__ import annotations
import argparse, sys, types
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .environment import coalition_fragmentation
from .HF_visualization import HFTestRunner
from .evaluate_policy import _LoadedCheckpoint, _build_policy_for_scenario


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/fofe_mappo_softcommit.pt")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--radius", type=float, default=0.2)  # coalition_radius
    ap.add_argument("--out", default="runs/diag_softcommit.png")
    ap.add_argument("--n_jammers", type=int, default=None,
                    help="Override active jammer count for evaluation (fixed composition; "
                         "clears DR). Legacy actor obs is fixed-slot so the same policy loads.")
    ap.add_argument("--n_strikers", type=int, default=None,
                    help="Override striker count for evaluation (fixed composition).")
    ap.add_argument("--title", default=None, help="Optional figure suptitle prefix (e.g. config label).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _LoadedCheckpoint(Path(args.checkpoint), device)
    env_cfg = ckpt.base_env_cfg
    if args.n_jammers is not None or args.n_strikers is not None:
        if hasattr(env_cfg, "dr"):
            env_cfg.dr = None                # disable DR so the counts are exactly as set
    if args.n_jammers is not None:
        env_cfg.n_jammers = args.n_jammers   # fixed composition for eval
    if args.n_strikers is not None:
        env_cfg.n_strikers = args.n_strikers
    rc = env_cfg.reward_config
    ns, nj = env_cfg.n_strikers, env_cfg.n_jammers
    print(f"ckpt={args.checkpoint} producer={ckpt.producer} ns={ns} nj={nj} "
          f"scenario={getattr(env_cfg,'scenario','?')}")
    print(f"escort: ell={rc.escort_kernel_length} kappa={rc.escort_capacity} "
          f"w_s={rc.escort_striker_scale} w_j={rc.escort_jammer_scale} "
          f"tau={getattr(rc,'escort_commit_temp','n/a')}")

    policy = _build_policy_for_scenario(ckpt, env_cfg, device)

    n_panels = min(args.seeds, 6)
    fig, axes = plt.subplots(2, n_panels, figsize=(3.2 * n_panels, 6.2))
    if n_panels == 1:
        axes = axes.reshape(2, 1)

    ep_frags, ss_dists, diff_target_frac, imbalances, bal_fracs = [], [], [], [], []
    for si in range(args.seeds):
        runner = HFTestRunner(policy, env_cfg=env_cfg, hf_cfg=ckpt.hf_radar_cfg,
                              device=device, seed=1000 + si)
        frames = runner.rollout()

        fr_frag, fr_ssd, fr_difftgt, fr_imbal = [], [], [], []
        s_traj = [[] for _ in range(ns)]
        j_traj = [[] for _ in range(nj)]
        for fr in frames:
            ap_ = fr["agent_pos"]            # [A,2]
            al_ = fr["agent_alive"].bool()   # [A]
            tp_ = fr["target_pos"]           # [T,2]
            ta_ = fr["target_alive"].bool()  # [T]
            f, _ = coalition_fragmentation(ap_, args.radius, alive_mask=al_)
            fr_frag.append(float(f))
            s = ap_[:ns]; j = ap_[ns:]
            for k in range(ns): s_traj[k].append(s[k].numpy())
            for k in range(nj): j_traj[k].append(j[k].numpy())
            if ns == 2:
                fr_ssd.append(float(torch.norm(s[0] - s[1])))
            # which alive target is each striker nearest -> different targets?
            if ta_.any() and ns == 2:
                tp_alive = tp_[ta_]
                d0 = torch.norm(tp_alive - s[0], dim=-1)
                d1 = torch.norm(tp_alive - s[1], dim=-1)
                fr_difftgt.append(int(torch.argmin(d0) != torch.argmin(d1)))
            # jammers-per-striker BALANCE: assign each alive jammer to its nearest
            # alive striker; imbalance = max-min count (0 = even split e.g. 2-2;
            # 2 = a 1-3 split; 4 = 0-4). Only meaningful with ≥2 strikers alive.
            s_al = al_[:ns]; j_al = al_[ns:]
            if ns >= 2 and int(s_al.sum()) >= 2 and int(j_al.sum()) >= 1:
                d_js = torch.cdist(j[j_al], s[s_al])           # [nj_alive, ns_alive]
                counts = torch.bincount(d_js.argmin(dim=1), minlength=int(s_al.sum()))
                fr_imbal.append(float(counts.max() - counts.min()))

        ep_frags.append(np.mean(fr_frag))
        if fr_ssd: ss_dists.append(np.mean(fr_ssd))
        if fr_difftgt: diff_target_frac.append(np.mean(fr_difftgt))
        if fr_imbal:
            imbalances.append(float(np.mean(fr_imbal)))
            bal_fracs.append(float(np.mean([1.0 if x == 0 else 0.0 for x in fr_imbal])))

        if si < n_panels:
            axt = axes[0, si]
            # trajectories
            s_colors = ["tab:blue", "tab:cyan"]
            j_colors = ["tab:red", "tab:orange"]
            for k in range(ns):
                arr = np.array(s_traj[k])
                axt.plot(arr[:, 0], arr[:, 1], "-", color=s_colors[k % 2], lw=1.5,
                         label=f"striker{k}")
                axt.plot(arr[-1, 0], arr[-1, 1], "o", color=s_colors[k % 2], ms=6)
            for k in range(nj):
                arr = np.array(j_traj[k])
                axt.plot(arr[:, 0], arr[:, 1], "--", color=j_colors[k % 2], lw=1.2,
                         label=f"jammer{k}")
                axt.plot(arr[-1, 0], arr[-1, 1], "s", color=j_colors[k % 2], ms=5)
            tp0 = frames[0]["target_pos"].numpy()
            axt.plot(tp0[:, 0], tp0[:, 1], "kX", ms=11, label="targets")
            rp0 = frames[0]["radar_pos"].numpy()
            axt.plot(rp0[:, 0], rp0[:, 1], "k^", ms=6, mfc="none", label="radars")
            axt.set_title(f"seed {1000+si} | fraḡ={ep_frags[-1]:.2f}", fontsize=9)
            axt.set_aspect("equal"); axt.tick_params(labelsize=6)
            if si == 0: axt.legend(fontsize=5, loc="best")
            axf = axes[1, si]
            axf.plot(fr_frag, color="tab:green")
            axf.axhline(2/3, ls=":", color="gray", lw=0.8)
            axf.set_ylim(-0.05, 1.05); axf.set_title("frag vs step", fontsize=8)
            axf.tick_params(labelsize=6)

    _label = (args.title + "  " if args.title else "") + f"{ns}s{nj}j"
    fig.suptitle(
        f"{_label}  |  episode-mean frag={np.mean(ep_frags):.3f} "
        f"| mean striker-striker dist={np.mean(ss_dists) if ss_dists else float('nan'):.3f} "
        f"| frac time strikers on DIFFERENT targets={np.mean(diff_target_frac) if diff_target_frac else float('nan'):.2f}",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"\nepisode-mean frag per seed: {[round(x,3) for x in ep_frags]}")
    print(f"OVERALL episode-mean frag       = {np.mean(ep_frags):.3f}  (target ~0.667 for two pairs)")
    print(f"mean striker-striker distance   = {np.mean(ss_dists) if ss_dists else float('nan'):.3f}  (coalition_radius={args.radius})")
    print(f"frac of time on DIFFERENT tgts  = {np.mean(diff_target_frac) if diff_target_frac else float('nan'):.3f}")
    print(f"jammers/striker IMBALANCE        = {np.mean(imbalances) if imbalances else float('nan'):.3f}  (max-min count; 0=even e.g. 2-2, 2=a 1-3 split)")
    print(f"frac time BALANCED (even) split  = {np.mean(bal_fracs) if bal_fracs else float('nan'):.3f}  (want ~1.0)")
    print(f"saved figure -> {args.out}")


if __name__ == "__main__":
    main()
