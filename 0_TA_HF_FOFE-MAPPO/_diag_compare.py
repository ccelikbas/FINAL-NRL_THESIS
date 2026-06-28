"""Side-by-side validation of ONE policy across team compositions.

Rolls out the same checkpoint at several (n_strikers, n_jammers) configs and plots
agent paths + frag-vs-step side by side. Read-only.

Run: .\.venv\Scripts\python.exe 0_TA_HF_FOFE-MAPPO\_diag_compare.py --checkpoint runs\composition_agnostic.pt
"""
from __future__ import annotations
import argparse, copy, sys, types
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

# κ=2 (strikers prefer a 2-jammer escort)
CONFIGS = [(2, 4), (2, 2), (1, 2), (1, 1)]
EXPECT = {
    (2, 4): "2 formations (1s+2j ea)",   # two groups of 3 → frag≈0.60
    (2, 2): "1 clump (4: 2s share 2j)",  # one group of 4 → frag≈0
    (1, 2): "1 formation (1s+2j)",       # one group of 3 → frag≈0
    (1, 1): "1 formation (1s+1j)",       # one group of 2, under-escorted → frag≈0
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/composition_agnostic.pt")
    ap.add_argument("--seed", type=int, default=1002)
    ap.add_argument("--radius", type=float, default=0.2)
    ap.add_argument("--out", default="runs/diag_compare_4cfg.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _LoadedCheckpoint(Path(args.checkpoint), device)
    base = ckpt.base_env_cfg

    n = len(CONFIGS)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 6.6))
    s_colors = ["tab:blue", "tab:cyan", "tab:green", "navy"]
    j_colors = ["tab:red", "tab:orange", "tab:purple", "brown"]

    for ci, (ns, nj) in enumerate(CONFIGS):
        env_cfg = copy.deepcopy(base)
        env_cfg.n_strikers, env_cfg.n_jammers = ns, nj
        if hasattr(env_cfg, "dr"):
            env_cfg.dr = None
        policy = _build_policy_for_scenario(ckpt, env_cfg, device)
        runner = HFTestRunner(policy, env_cfg=env_cfg, hf_cfg=ckpt.hf_radar_cfg,
                              device=device, seed=args.seed)
        frames = runner.rollout()

        fr_frag = []
        s_traj = [[] for _ in range(ns)]
        j_traj = [[] for _ in range(nj)]
        for fr in frames:
            ap_ = fr["agent_pos"]
            al_ = fr["agent_alive"].bool()
            f, _ = coalition_fragmentation(ap_, args.radius, alive_mask=al_)
            fr_frag.append(float(f))
            for k in range(ns): s_traj[k].append(ap_[k].numpy())
            for k in range(nj): j_traj[k].append(ap_[ns + k].numpy())

        axt = axes[0, ci]
        for k in range(ns):
            a = np.array(s_traj[k])
            axt.plot(a[:, 0], a[:, 1], "-", color=s_colors[k % 4], lw=1.7,
                     label=f"striker{k}")
            axt.plot(a[-1, 0], a[-1, 1], "o", color=s_colors[k % 4], ms=6)
        for k in range(nj):
            a = np.array(j_traj[k])
            axt.plot(a[:, 0], a[:, 1], "--", color=j_colors[k % 4], lw=1.3,
                     label=f"jammer{k}")
            axt.plot(a[-1, 0], a[-1, 1], "s", color=j_colors[k % 4], ms=5)
        tp = frames[0]["target_pos"].numpy()
        axt.plot(tp[:, 0], tp[:, 1], "kX", ms=12, label="targets")
        rp = frames[0]["radar_pos"].numpy()
        axt.plot(rp[:, 0], rp[:, 1], "k^", ms=6, mfc="none", label="radars")
        axt.set_title(f"{ns}s{nj}j  →  expect {EXPECT[(ns, nj)]}\nseed-mean frag={np.mean(fr_frag):.2f}",
                      fontsize=9)
        axt.set_aspect("equal"); axt.tick_params(labelsize=6)
        axt.legend(fontsize=5, loc="best")

        axf = axes[1, ci]
        axf.plot(fr_frag, color="tab:green")
        axf.axhline(2 / 3, ls=":", color="gray", lw=0.8)
        axf.set_ylim(-0.05, 1.05)
        axf.set_title("frag vs step (0.667=two pairs)", fontsize=8)
        axf.tick_params(labelsize=6)

    fig.suptitle(
        f"Composition-agnostic validation — {Path(args.checkpoint).name}  (seed {args.seed}; "
        f"top=paths, bottom=frag/step)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
