"""escort_reward_analysis.py — quantify how big the ESCORT reward is versus every
other reward component for a trained checkpoint, and break the escort into its
sub-terms. Read-only (never modifies a checkpoint).

Answers:
  • How large is the total escort reward vs all other rewards? (episode & per-step)
  • How large are the escort sub-terms vs each other?
        striker_under  = -w_s   · Σ_s (κ-c_s)+          (escort_striker_scale)
        jammer_under   = -w_j   · Σ_s (κ-c_s)+  (×n_j)  (escort_jammer_scale)
        over           = -w_over· Σ_s (c_s-κ)+  (×n_j)  (escort_over_scale)
        attraction     = -w_a   · soft-nearest dist     (jammer_escort_approach_scale)

Robustness / honesty about episode length:
  • The component + sub-term BARS are per-episode totals averaged over MANY episodes
    across several seeds (--n_seeds), shown with the seed-to-seed spread (error bars).
  • The per-timestep panel shows a handful of INDIVIDUAL episodes (each ending at its
    own length) instead of a cross-env mean — because most episodes finish early
    (mission complete) and a mean-over-active-envs is survivor-biased in the tail.
  • An episode-length histogram is included so the length distribution is explicit.

The escort sub-terms are recomputed OFFLINE from agent positions using the exact env
formula and the checkpoint's OWN baked kernel (exp or sigmoid); the printed
offline-vs-env escort ratio should be ≈1.000 (a correctness check).

The eval world is auto-parsed from the baked curriculum label (scenario / striker /
jammer / target / radar counts / kill prob / steps); override with the CLI flags.
Outputs (figure + CSV) go to  0_TA_HF_FOFE-MAPPO/escort_analysis/.

Run (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\escort_reward_analysis.py \\
      --checkpoint 0_TA_HF_FOFE-MAPPO\\runs\\2s4j_V3.pt --n_seeds 5
"""
from __future__ import annotations
import argparse, re, sys, types
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluate_policy import _LoadedCheckpoint, _build_policy_for_scenario
from .reward_scale_analysis import build_hf_env

ESCORT_KEY = "escort"
SUBS = ("striker_under", "jammer_under", "over", "attraction")


def parse_label(label: str) -> dict:
    def g(pat, cast=int, default=None):
        m = re.search(pat, label or "")
        return cast(m.group(1)) if m else default
    return dict(
        n_strikers=g(r"^S(\d+)"), n_jammers=g(r"\bJ(\d+)"),
        n_known_targets=g(r"kT\[?(\d+)"), n_known_radars=g(r"kR\[?(\d+)"),
        radar_kill_probability=g(r"kill([\d.]+)", float), max_steps=g(r"steps(\d+)"),
        scenario=(re.search(r"(S\d)\s+comm", label).group(1) if re.search(r"(S\d)\s+comm", label) else None),
    )


def escort_kernel(d, rc):
    if str(getattr(rc, "escort_kernel_type", "exp")).lower() == "sigmoid":
        R = float(getattr(rc, "escort_kernel_radius", 0.15))
        s = max(float(getattr(rc, "escort_kernel_softness", 0.03)), 1e-6)
        return torch.sigmoid((R - d) / s)
    return torch.exp(-d / max(float(rc.escort_kernel_length), 1e-6))


def escort_subterms(pos, alive, ns, rc):
    """Exact env escort, split into the 4 sub-terms. Returns (total[B], parts{B})."""
    w_s = float(rc.escort_striker_scale); w_j = float(rc.escort_jammer_scale)
    w_over = float(getattr(rc, "escort_over_scale", 0.0))
    w_a = float(getattr(rc, "jammer_escort_approach_scale", 0.0))
    kappa = float(rc.escort_capacity)
    s, j = pos[:, :ns, :], pos[:, ns:, :]
    sa, ja = alive[:, :ns].float(), alive[:, ns:].float()
    d = torch.cdist(s, j)
    k = escort_kernel(d, rc) * ja[:, None, :]
    c = k.sum(-1)
    anyj = (ja.sum(-1, keepdim=True) > 0)
    strik = torch.where(anyj, -w_s * (kappa - c).clamp(min=0), torch.zeros_like(c)) * sa
    unmet = (kappa - c).clamp(min=0).sum(-1, keepdim=True)
    over = (c - kappa).clamp(min=0).sum(-1, keepdim=True)
    eps = 1e-6
    invd = torch.where(sa[:, :, None] > 0, 1.0 / (d + eps), torch.zeros_like(d))
    wsum = invd.sum(1, keepdim=True).clamp_min(eps)
    shaped = ((invd / wsum) * d).sum(1)
    parts = {
        "striker_under": strik.sum(-1),
        "jammer_under":  (-w_j * unmet * ja).sum(-1),
        "over":          (-w_over * over * ja).sum(-1),
        "attraction":    (-w_a * shaped * ja).sum(-1),
    }
    total = sum(parts.values())
    return total, parts


def get_alive(env):
    for n in ("agent_alive", "alive", "_alive"):
        a = getattr(env, n, None)
        if a is not None:
            return a.bool()


def rollout_seed(ckpt, ec, rc, ns, B, seed, device, capture_ts=False):
    """One seed = B episodes. Returns per-episode MEAN component/subterm dicts,
    episode lengths, offline/env ratio, and (if capture_ts) per-step escort [T,B]."""
    policy = _build_policy_for_scenario(ckpt, ec, device)
    policy.eval(); policy.deterministic = True
    env = build_hf_env(ec, ckpt.hf_radar_cfg, B, device, seed)
    T = env.max_steps
    comp_signed = defaultdict(lambda: torch.zeros(B, device=device))
    comp_abs = defaultdict(lambda: torch.zeros(B, device=device))
    sub_signed = defaultdict(lambda: torch.zeros(B, device=device))
    off_esc = torch.zeros(B, device=device); env_esc = torch.zeros(B, device=device)
    ep_len = torch.zeros(B, device=device)
    ts_esc = np.full((T, B), np.nan) if capture_ts else None
    ts_comp = defaultdict(lambda: np.zeros(T)) if capture_ts else None   # Σ over active envs, per step
    ts_cnt = np.zeros(T) if capture_ts else None                          # #active envs, per step
    with torch.no_grad():
        td = env.reset()
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for t in range(T):
            td = policy(td); td = env.step(td)
            active = (~done)
            af = active.float()
            for k, v in env.last_reward_components.items():
                comp_signed[k] += v.sum(-1) * af
                comp_abs[k] += v.abs().sum(-1) * af
            tot, parts = escort_subterms(env.agent_pos, get_alive(env), ns, rc)
            for k, v in parts.items():
                sub_signed[k] += v * af
            off_esc += tot * af
            env_esc += env.last_reward_components[ESCORT_KEY].sum(-1) * af
            if capture_ts:
                e = (tot * af).detach().cpu().numpy()
                e[~active.detach().cpu().numpy()] = np.nan
                ts_esc[t] = e
                for k, v in env.last_reward_components.items():
                    ts_comp[k][t] = float((v.sum(-1) * af).sum())
                ts_cnt[t] = float(af.sum())
            ep_len += af
            done = done | td.get(("next", "done")).reshape(B).bool()
            if bool(done.all()):
                break
            td = td.get("next")
    cs = {k: float(comp_signed[k].mean()) for k in comp_signed}
    ca = {k: float(comp_abs[k].mean()) for k in comp_abs}
    ss = {k: float(sub_signed[k].mean()) for k in sub_signed}
    ratio = float(off_esc.sum() / env_esc.sum()) if float(env_esc.sum()) != 0 else float("nan")
    ts = None
    if capture_ts:
        ts = {"esc": ts_esc, "comp": {k: ts_comp[k] for k in ts_comp}, "cnt": ts_cnt}
    return cs, ca, ss, ep_len.detach().cpu().numpy(), ratio, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="0_TA_HF_FOFE-MAPPO/runs/2s4j_V3.pt")
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--out_dir", default=str(_THIS_DIR / "escort_analysis"))
    for k in ("n_strikers", "n_jammers", "n_known_radars", "n_known_targets", "max_steps"):
        ap.add_argument("--" + k, type=int, default=None)
    ap.add_argument("--radar_kill_probability", type=float, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _LoadedCheckpoint(Path(args.checkpoint), device)
    ec = ckpt.base_env_cfg
    rc = ec.reward_config
    raw = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    label = (raw.get("curriculum") or [{}])[0].get("label", "") if isinstance(raw.get("curriculum"), list) else ""
    cfg = parse_label(label)

    def pick(name):
        v = getattr(args, name, None)
        return v if v is not None else cfg.get(name)
    ns = pick("n_strikers") or ec.n_strikers
    nj = pick("n_jammers") or ec.n_jammers
    nkr = pick("n_known_radars") or ec.n_known_radars
    nkt = pick("n_known_targets") or ec.n_known_targets
    ms = pick("max_steps") or ec.max_steps
    kill = pick("radar_kill_probability")
    ec.n_strikers, ec.n_jammers = ns, nj
    ec.n_known_radars = nkr; ec.n_unknown_radars = 0; ec.n_radars = nkr
    ec.n_known_targets = nkt; ec.n_unknown_targets = 0; ec.n_targets = nkt
    ec.max_steps = ms
    if kill is not None:
        ec.radar_kill_probability = kill
    if cfg.get("scenario"):
        ec.scenario = cfg["scenario"]
    if hasattr(ec, "dr"):
        ec.dr = None
    ec._use_fofe = bool(getattr(ckpt.fofe_cfg, "use_fofe", False)) if getattr(ckpt, "fofe_cfg", None) else getattr(ec, "_use_fofe", False)

    print(f"checkpoint = {args.checkpoint}")
    print(f"baked label = {label!r}")
    print(f"eval world : scenario={ec.scenario} ns={ns} nj={nj} radars={nkr} targets={nkt} "
          f"kill={ec.radar_kill_probability} steps={ms} fofe={ec._use_fofe}")
    print(f"escort kernel: type={getattr(rc,'escort_kernel_type','exp')} R={getattr(rc,'escort_kernel_radius','-')} "
          f"s={getattr(rc,'escort_kernel_softness','-')} kappa={rc.escort_capacity}  |  "
          f"w_s={rc.escort_striker_scale} w_j={rc.escort_jammer_scale} "
          f"w_over={getattr(rc,'escort_over_scale','-')} w_a={getattr(rc,'jammer_escort_approach_scale','-')}")
    print(f"averaging over {args.n_seeds} seeds x {args.num_envs} envs = {args.n_seeds*args.num_envs} episodes\n")

    # ---- multi-seed rollouts ----
    all_cs, all_ca, all_ss, lengths, ratios = [], [], [], [], []
    ts_esc0 = None
    for i in range(args.n_seeds):
        cs, ca, ss, ep_len, ratio, ts = rollout_seed(
            ckpt, ec, rc, ns, args.num_envs, args.seed0 + i, device, capture_ts=(i == 0))
        all_cs.append(cs); all_ca.append(ca); all_ss.append(ss)
        lengths.append(ep_len); ratios.append(ratio)
        if i == 0:
            ts_esc0 = ts
        print(f"  seed {args.seed0+i}: mean ep len={ep_len.mean():.1f}  escort/ep={cs.get(ESCORT_KEY,0):.2f}  ratio={ratio:.3f}")
    lengths = np.concatenate(lengths)

    keys = sorted(all_ca[0], key=lambda k: -np.mean([c[k] for c in all_ca]))
    ca_mean = {k: np.mean([c[k] for c in all_ca]) for k in keys}
    ca_std = {k: np.std([c[k] for c in all_ca]) for k in keys}
    cs_mean = {k: np.mean([c[k] for c in all_cs]) for k in keys}
    total_abs = sum(ca_mean.values()) or 1.0
    ss_mean = {k: np.mean([s[k] for s in all_ss]) for k in SUBS}
    ss_std = {k: np.std([s[k] for s in all_ss]) for k in SUBS}
    mean_len = float(lengths.mean()); med_len = float(np.median(lengths))

    print(f"\noffline/env escort ratio = {np.mean(ratios):.3f} (want 1.000)")
    print(f"episode length: mean={mean_len:.1f}  median={med_len:.1f}  p90={np.percentile(lengths,90):.0f}  max={lengths.max():.0f}\n")
    print(f"{'component':<26}{'ep_signed':>11}{'ep_|mag|':>11}{'+/-std':>8}{'%|R|':>7}")
    print("-" * 63)
    for k in keys:
        mark = "  <<< ESCORT" if k == ESCORT_KEY else ""
        print(f"{k:<26}{cs_mean[k]:>11.2f}{ca_mean[k]:>11.2f}{ca_std[k]:>8.2f}{100*ca_mean[k]/total_abs:>6.1f}%{mark}")
    print("-" * 63)
    print(f"TOTAL escort = {100*ca_mean[ESCORT_KEY]/total_abs:.1f}% of all |reward|;  "
          f"escort {cs_mean[ESCORT_KEY]:.2f}/ep  vs  non-escort {sum(cs_mean[k] for k in keys if k!=ESCORT_KEY):.2f}/ep")
    print(f"\n{'escort sub-term':<16}{'ep_signed':>11}{'+/-std':>8}{'% escort':>10}")
    print("-" * 45)
    esc_tot = sum(ss_mean.values()) or -1e-9
    for k in SUBS:
        print(f"{k:<16}{ss_mean[k]:>11.3f}{ss_std[k]:>8.3f}{100*ss_mean[k]/esc_tot:>9.0f}%")

    # ---- figure (2x2) ----
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.checkpoint).stem
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # (0,0) components, mean +/- std over seeds
    kk = [k for k in keys if ca_mean[k] > 1e-6][::-1]
    ax[0, 0].barh(kk, [ca_mean[k] for k in kk], xerr=[ca_std[k] for k in kk],
                  color=["tab:red" if k == ESCORT_KEY else "tab:gray" for k in kk], capsize=2)
    ax[0, 0].set_title(f"|episode reward| by component  (mean±std over {args.n_seeds} seeds)\n"
                       f"escort = {100*ca_mean[ESCORT_KEY]/total_abs:.0f}% of all |reward|", fontsize=10)
    ax[0, 0].tick_params(labelsize=8); ax[0, 0].set_xlabel("mean |reward| per episode")

    # (0,1) escort sub-terms
    ax[0, 1].bar(SUBS, [ss_mean[k] for k in SUBS], yerr=[ss_std[k] for k in SUBS], capsize=3,
                 color=["tab:orange", "tab:red", "tab:purple", "tab:blue"])
    ax[0, 1].axhline(0, color="k", lw=0.6)
    ax[0, 1].set_title("escort sub-terms (signed, per episode)", fontsize=10)
    ax[0, 1].tick_params(axis="x", labelsize=9, rotation=15); ax[0, 1].set_ylabel("reward / episode")
    for i, k in enumerate(SUBS):
        ax[0, 1].text(i, ss_mean[k], f"{100*ss_mean[k]/esc_tot:.0f}%", ha="center",
                      va="top" if ss_mean[k] < 0 else "bottom", fontsize=8)

    # (1,0) individual episodes (escort/step), each ending at its own length
    ts_esc = ts_esc0["esc"]
    T = ts_esc.shape[0]
    lens0 = np.array([np.max(np.where(~np.isnan(ts_esc[:, b]))[0]) + 1 if np.any(~np.isnan(ts_esc[:, b])) else 0
                      for b in range(ts_esc.shape[1])])
    order = np.argsort(lens0)
    picks = order[np.linspace(0, len(order) - 1, 8).astype(int)]
    for b in picks:
        L = lens0[b]
        ax[1, 0].plot(np.arange(L), ts_esc[:L, b], lw=1.0, alpha=0.8, label=f"ep len {L}")
    # representative mean only where >=50% of envs still active
    frac = np.mean(~np.isnan(ts_esc), axis=1)
    cut = int(np.argmax(frac < 0.5)) if np.any(frac < 0.5) else T
    mean_active = np.nanmean(ts_esc[:cut], axis=1)
    ax[1, 0].plot(np.arange(cut), mean_active, "k", lw=2.5, label="mean (≥50% active)")
    ax[1, 0].axvline(med_len, ls=":", color="gray"); ax[1, 0].text(med_len, ax[1, 0].get_ylim()[0], " median len", fontsize=7)
    ax[1, 0].set_title("escort reward / step — INDIVIDUAL episodes (seed 0)", fontsize=10)
    ax[1, 0].set_xlabel("timestep"); ax[1, 0].legend(fontsize=6, ncol=2)

    # (1,1) episode-length histogram
    ax[1, 1].hist(lengths, bins=30, color="tab:gray", edgecolor="k", alpha=0.8)
    ax[1, 1].axvline(mean_len, color="tab:red", lw=2, label=f"mean {mean_len:.0f}")
    ax[1, 1].axvline(med_len, color="tab:green", lw=2, label=f"median {med_len:.0f}")
    ax[1, 1].set_title(f"episode-length distribution ({len(lengths)} episodes)\n"
                       "most finish early = mission complete; tail = hard layouts", fontsize=10)
    ax[1, 1].set_xlabel("episode length (steps)"); ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"Escort reward analysis — {name}  (kernel={getattr(rc,'escort_kernel_type','exp')} "
                 f"R={getattr(rc,'escort_kernel_radius','-')}; {args.n_seeds} seeds × {args.num_envs} envs)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = out_dir / f"escort_analysis_{name}.png"
    fig.savefig(fig_path, dpi=120)

    # ---- SEPARATE figure: ALL reward components per timestep (escort in context) ----
    comp_ts = ts_esc0["comp"]; cnt = ts_esc0["cnt"]
    win = int(np.argmax((cnt / args.num_envs) < 0.5)) if np.any((cnt / args.num_envs) < 0.5) else T
    win = max(win, 5)
    xs = np.arange(win); denom = np.clip(cnt[:win], 1, None)
    per_step = {k: comp_ts[k][:win] / denom for k in comp_ts}
    # rank dense components by mean |per-step| over the representative window; keep top 8 + escort
    ranked = sorted(per_step, key=lambda k: -np.mean(np.abs(per_step[k])))
    show = [k for k in ranked if np.mean(np.abs(per_step[k])) > 1e-4][:8]
    if ESCORT_KEY not in show:
        show.append(ESCORT_KEY)
    fig2, bx = plt.subplots(1, 1, figsize=(11, 6))
    for k in show:
        bx.plot(xs, per_step[k], lw=2.6 if k == ESCORT_KEY else 1.4,
                color="tab:red" if k == ESCORT_KEY else None,
                zorder=5 if k == ESCORT_KEY else 2, label=k)
    bx.axhline(0, color="k", lw=0.6)
    bx.axvline(med_len, ls=":", color="gray"); bx.text(med_len, bx.get_ylim()[0], " median ep len", fontsize=8)
    bx.set_title(f"Reward components per timestep — {name}  (mean over active envs, "
                 f"shown while ≥50% active)\nescort (bold red) now ~{100*ca_mean[ESCORT_KEY]/total_abs:.0f}% of |reward|",
                 fontsize=11)
    bx.set_xlabel("timestep"); bx.set_ylabel("reward / step"); bx.legend(fontsize=9, ncol=2)
    fig2.tight_layout()
    fig2_path = out_dir / f"escort_analysis_{name}_components_per_step.png"
    fig2.savefig(fig2_path, dpi=120)

    csv_path = out_dir / f"escort_analysis_{name}.csv"
    with open(csv_path, "w") as f:
        f.write("component,ep_signed,ep_absmag,absmag_std,pct_abs\n")
        for k in keys:
            f.write(f"{k},{cs_mean[k]:.4f},{ca_mean[k]:.4f},{ca_std[k]:.4f},{100*ca_mean[k]/total_abs:.2f}\n")
        f.write("\nescort_subterm,ep_signed,std,pct_escort\n")
        for k in SUBS:
            f.write(f"{k},{ss_mean[k]:.4f},{ss_std[k]:.4f},{100*ss_mean[k]/esc_tot:.1f}\n")
        f.write(f"\nepisode_length_mean,{mean_len:.2f}\nepisode_length_median,{med_len:.2f}\n")
    print(f"\nsaved figure      -> {fig_path}")
    print(f"saved components  -> {fig2_path}")
    print(f"saved CSV         -> {csv_path}")


if __name__ == "__main__":
    main()
