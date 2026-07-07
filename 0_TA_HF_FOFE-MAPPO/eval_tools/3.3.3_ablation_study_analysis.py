"""ablation_study_analysis.py — ablation LaTeX table for ONE scenario.

Reuses the evaluation engine in evaluate_policy.py (paired rollouts + Wilcoxon
signed-rank test) and emits a single LaTeX table for FOUR policies on ONE
scenario, with the KPIs as ROWS and the policies as COLUMNS:

    KPI  |  Complete (value)  |  Baseline (value, p)  |  No-Com (value, p)  |  No-FOFE (value, p)

Complete is the reference (value only). Each other policy shows its value and a
Wilcoxon p-value versus Complete. Same inputs/engine as the mission / team tools;
only the layout differs (KPIs down the side, policies across the top).

Duration is SUCCESS-CONDITIONED (see SUCCESS_CONDITIONED_KPIS): each policy's value
is its mean duration over its OWN successful missions, and each p tests Complete
vs that policy over the episodes BOTH completed (removes the early-death confound).

Run (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\ablation_study_analysis.py
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\ablation_study_analysis.py --n_episodes 100
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Dict, List

_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR.parent
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

import numpy as np
import torch

from .evaluate_policy import (
    PolicyInput, evaluate_comparison, _LoadedCheckpoint, _resolve_policy_path,
    _print_config, _print_policy_diagnostics, _p_used, _wilcoxon_pair, _ALT, KPIS,
)
from .run_curriculum import CurriculumSection

# =====================================================================
#  >>>  TEST CONFIG  (how many runs, seeding, significance)  <<<
# =====================================================================

# Number of RUNS = paired episodes per policy (the test N).   [CLI: --n_episodes]
N_EPISODES = 500
# Parallel envs per rollout chunk (lower this if you hit GPU OOM). [CLI: --chunk]
CHUNK_EPISODES = 300
# Base RNG seed; all policies share it so episodes are paired 1:1.   [CLI: --seed]
BASE_SEED = 42
# Significance level.                                               [CLI: --alpha]
ALPHA = 0.05
# Multiplicity correction across the comparison columns: "none" or "holm". [--p_adjust]
P_ADJUST = "none"

# =====================================================================
#  >>>  POLICIES  (the COLUMNS)  <<<
#  MAIN_POLICY = "Complete" (reference, value-only column). Each COMPARISON is one
#  (value, p) column pair, tested vs Complete. The column headers are the policy
#  `name`s. communicate=... must match how each ablation was TRAINED.
#
#  NOTE: No-Com / No-FOFE below are PLACEHOLDERS pointing at existing checkpoints
#  so the tool runs out of the box — REPLACE their policy_file with your actual
#  separately-trained ablation checkpoints (No-Com = FOFE on / comms off,
#  No-FOFE = FOFE off / comms on). FOFE on/off is baked into the checkpoint.
# =====================================================================

MAIN_POLICY = PolicyInput(
    name="Complete",
    policy_file="runs/FINALV1/complete_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt",
    communicate=True,
)
COMPARISON_POLICIES: List[PolicyInput] = [
    PolicyInput(
        name="Baseline",
        policy_file="runs/FINALV1/baseline_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt",
        communicate=False,
    ),
    PolicyInput(  # PLACEHOLDER — replace with your FOFE-on / comms-off checkpoint
        name="No-Com",
        policy_file="runs/FINALV1/complete_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt",
        communicate=False,
    ),
    PolicyInput(  # PLACEHOLDER — replace with your FOFE-off / comms-on checkpoint
        name="No-FOFE",
        policy_file="runs/FINALV1/baseline_S1_20260704/stage5of5_DR_j2-4_k0_25_FINAL.pt",
        communicate=True,
    ),
]

# =====================================================================
#  >>>  SCENARIO  (exactly ONE — the ablation is for a single scenario)  <<<
# =====================================================================

EVAL_SCENARIOS: List[CurriculumSection] = [
    CurriculumSection(
        name="S2",
        n_iters=1,
        n_strikers=2, n_jammers=(2, 4),
        n_known_targets=(2, 4), n_unknown_targets=0,
        n_known_radars=(4, 6), n_unknown_radars=0,
        radar_kill_probability=0.25,
        scenario="S2",
    ),
]

# KPI rows (top → bottom). Keys must exist in evaluate_policy.KPIS.
TABLE_KPIS = ["targets", "survival", "duration"]

# KPIs restricted to SUCCESSFUL episodes (mission complete == all targets destroyed).
# Each policy's value = mean over its OWN successful episodes; each p = one-sided
# test vs Complete over the episodes BOTH completed. Removes the early-death
# confound so duration's lower-is-better holds.
SUCCESS_CONDITIONED_KPIS = {"duration"}
SUCCESS_KEY = "completion"

# LaTeX caption / label / output.
CAPTION = "Ablation-study results for Scenario~S2."
LABEL = "tab:ablation-study-s2"
OUT_PATH = "eval_results/ablation_study.tex"

# =====================================================================

_KPI_BY_KEY = {s.key: s for s in KPIS}


def _fmt_val(key: str, mean: float) -> str:
    if not np.isfinite(mean):
        return "--"
    return _KPI_BY_KEY[key].fmt.format(mean)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _own_success_mean(scn, policy, key, samples) -> float:
    """Mean of a KPI over the policy's OWN successful (mission-complete) episodes."""
    arr = np.asarray(samples[scn.name][policy.name][key], dtype=float)
    succ = np.asarray(samples[scn.name][policy.name][SUCCESS_KEY], dtype=float) >= 0.5
    fin = succ & np.isfinite(arr)
    return float(arr[fin].mean()) if fin.any() else float("nan")


def _value(scn, policy, key, summary, samples) -> float:
    if key in SUCCESS_CONDITIONED_KPIS:
        return _own_success_mean(scn, policy, key, samples)
    return summary[(scn.name, policy.name, key)][0]


def _pairwise_success_p(scn, key, main, comp, samples) -> float:
    """One-sided Wilcoxon (KPI's own direction) of Complete vs comp over episodes
    BOTH completed."""
    sm = np.asarray(samples[scn.name][main.name][SUCCESS_KEY], dtype=float) >= 0.5
    sc = np.asarray(samples[scn.name][comp.name][SUCCESS_KEY], dtype=float) >= 0.5
    mv = np.asarray(samples[scn.name][main.name][key], dtype=float)
    cv = np.asarray(samples[scn.name][comp.name][key], dtype=float)
    fin = sm & sc & np.isfinite(mv) & np.isfinite(cv)
    if not fin.any():
        return float("nan")
    return _wilcoxon_pair(mv[fin], cv[fin], _ALT[_KPI_BY_KEY[key].direction])["pvalue"]


def _pvalue(scn, key, main, comp, tests, samples, p_adjust) -> float:
    if key in SUCCESS_CONDITIONED_KPIS:
        return _pairwise_success_p(scn, key, main, comp, samples)
    res = tests.get((scn.name, comp.name, key))
    return _p_used(res, p_adjust) if res is not None else float("nan")


def build_latex(scn, main, comparisons, summary, tests, samples, p_adjust) -> str:
    K = len(comparisons)
    colspec = "l" + "c" * (1 + 2 * K)         # KPI + Complete(1) + 2 per comparison

    lines: List[str] = [
        r"\begin{table}[htbp]",
        r"    \centering",
        rf"    \caption{{{CAPTION}}}",
        rf"    \label{{{LABEL}}}",
        r"    \small",
        rf"    \begin{{tabular}}{{{colspec}}}",
        r"        \toprule",
    ]

    # Header row 1: KPI | Complete | <each comparison> (spanning 2).
    hdr = [r"        \textbf{KPI}",
           rf"        & \multicolumn{{1}}{{c}}{{\textbf{{{main.name}}}}}"]
    for c in comparisons:
        hdr.append(rf"        & \multicolumn{{2}}{{c}}{{\textbf{{{c.name}}}}}")
    lines.append("\n".join(hdr) + r" \\")

    # cmidrules: Complete (2-2), then each comparison pair.
    lines.append(r"        \cmidrule(lr){2-2}")
    col = 3
    for _c in comparisons:
        lines.append(rf"        \cmidrule(lr){{{col}-{col + 1}}}")
        col += 2

    # Header row 2: Value | (Value, p) per comparison.
    h2 = [r"        & \textbf{Value}"]
    for _c in comparisons:
        h2.append(r"        & \textbf{Value} & \textbf{$p$}")
    lines.append("\n".join(h2) + r" \\")
    lines.append(r"        \midrule")

    # Body: one row per KPI.
    for key in TABLE_KPIS:
        cells = [_KPI_BY_KEY[key].label,
                 _fmt_val(key, _value(scn, main, key, summary, samples))]
        for c in comparisons:
            cells.append(_fmt_val(key, _value(scn, c, key, summary, samples)))
            cells.append(_fmt_p(_pvalue(scn, key, main, c, tests, samples, p_adjust)))
        lines.append("        " + " & ".join(cells) + r" \\")

    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ablation LaTeX table (KPIs × policies) for ONE scenario: "
        "Complete vs Baseline / No-Com / No-FOFE via the Wilcoxon signed-rank test.")
    p.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                   help="Default .pt for any policy that leaves policy_file=None.")
    p.add_argument("--n_episodes", type=int, default=N_EPISODES)
    p.add_argument("--chunk", type=int, default=CHUNK_EPISODES)
    p.add_argument("--seed", type=int, default=BASE_SEED)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--p_adjust", type=str, default=P_ADJUST, choices=["none", "holm"])
    p.add_argument("--device", type=str, default=None, metavar="DEVICE")
    p.add_argument("--out", type=str, default=OUT_PATH)
    return p


def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_PKG_DIR / p)


def main() -> None:
    args = _build_parser().parse_args()
    if len(EVAL_SCENARIOS) != 1:
        raise RuntimeError("ablation table expects exactly ONE scenario in EVAL_SCENARIOS.")
    if not COMPARISON_POLICIES:
        raise RuntimeError("need at least one comparison policy for the ablation columns.")
    scn = EVAL_SCENARIOS[0]

    default_ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    _print_config(MAIN_POLICY, COMPARISON_POLICIES, EVAL_SCENARIOS,
                  args.n_episodes, args.seed, args.alpha, args.p_adjust, device)
    cache: Dict[str, _LoadedCheckpoint] = {}
    _print_policy_diagnostics(MAIN_POLICY, COMPARISON_POLICIES, EVAL_SCENARIOS,
                              default_ckpt_path, device, cache)

    summary, tests, samples = evaluate_comparison(
        EVAL_SCENARIOS, MAIN_POLICY, COMPARISON_POLICIES, default_ckpt_path,
        n_episodes=args.n_episodes, chunk=args.chunk, base_seed=args.seed,
        p_adjust=args.p_adjust, device=device, cache=cache,
    )

    latex = build_latex(scn, MAIN_POLICY, COMPARISON_POLICIES,
                        summary, tests, samples, args.p_adjust)

    # Effective n per policy for the success-conditioned KPIs.
    if SUCCESS_CONDITIONED_KPIS:
        print(f"[note] {', '.join(sorted(SUCCESS_CONDITIONED_KPIS))} value = mean over "
              f"each policy's OWN successful episodes; p = Complete vs policy over "
              f"jointly-successful episodes. Successful episodes / {args.n_episodes}:")
        for pol in [MAIN_POLICY] + COMPARISON_POLICIES:
            succ = np.asarray(samples[scn.name][pol.name][SUCCESS_KEY], dtype=float) >= 0.5
            print(f"         {pol.name:<10}: {int(succ.sum())}")
    print("\n" + latex)

    out = _resolve_out(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table to: {out}")


if __name__ == "__main__":
    main()
