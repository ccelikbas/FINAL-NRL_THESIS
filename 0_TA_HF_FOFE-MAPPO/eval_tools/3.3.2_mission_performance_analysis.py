"""mission_performance_analysis.py — mission-performance table across SCENARIOS.

Reuses the evaluation engine in evaluate_policy.py (paired rollouts + Wilcoxon
signed-rank test) and emits a single LaTeX table comparing a COMPLETE policy
against a BASELINE policy on a set of scenarios, for three KPIs:

    Targets destroyed · Survival · Duration     (each: Complete, Baseline, p-value)

Rows are the scenarios in EVAL_SCENARIOS (e.g. S1, S2). Edit the CONFIG block
below (same inputs as evaluate_policy.py); only the scenarios and the output
table format differ from the emergent-team-behaviour tool.

Run (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\mission_performance_analysis.py
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\mission_performance_analysis.py --n_episodes 100
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Reuse the whole evaluation engine + config defaults from evaluate_policy.
from .evaluate_policy import (
    PolicyInput, evaluate_comparison, _LoadedCheckpoint, _resolve_policy_path,
    _print_config, _print_policy_diagnostics, _p_used, KPIS,
    N_EPISODES, CHUNK_EPISODES, BASE_SEED, ALPHA, P_ADJUST,
)
from .run_curriculum import CurriculumSection

# =====================================================================
#  >>>  POLICIES  (Complete = reference column, Baseline = compared)  <<<
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
]

# =====================================================================
#  >>>  SCENARIOS  (the ROWS of the table)  <<<
#  Same CurriculumSection format as evaluate_policy.py. The row label is the
#  section `name` (printed in math mode, e.g. $S1$). Communication is per-policy.
# =====================================================================

EVAL_SCENARIOS: List[CurriculumSection] = [
    CurriculumSection(
        name="S1",
        n_iters=1,
        n_strikers=2, n_jammers=(2, 4),
        n_known_targets=(2, 4), n_unknown_targets=0,
        n_known_radars=(4, 6), n_unknown_radars=0,
        radar_kill_probability=0.25,
        scenario="S2",
    ),
    # CurriculumSection(
    #     name="S2",
    #     n_iters=1,
    #     n_strikers=2, n_jammers=(2, 4),
    #     n_known_targets=(2, 4), n_unknown_targets=0,
    #     n_known_radars=(4, 6), n_unknown_radars=0,
    #     radar_kill_probability=0.25,
    #     scenario="S2",
    # ),
]

# KPIs shown (in column order). Keys must exist in evaluate_policy.KPIS.
TABLE_KPIS = ["targets", "survival", "duration"]

# Output .tex (relative paths resolved against the project dir 0_TA_...).
OUT_PATH = "eval_results/mission_performance.tex"

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


def _cells(scn, key, main, base, summary, tests, p_adjust) -> Tuple[str, str, str]:
    """(complete, baseline, p) strings for one (scenario, KPI). Baseline/p are ''
    when there is no comparison policy."""
    c = _fmt_val(key, summary[(scn.name, main.name, key)][0])
    if base is None:
        return c, "", ""
    b = _fmt_val(key, summary[(scn.name, base.name, key)][0])
    res = tests.get((scn.name, base.name, key))
    p = _fmt_p(_p_used(res, p_adjust)) if res is not None else "--"
    return c, b, p


_HEADER = r"""\begin{table}[htbp]
    \centering
    \caption{Performance comparison across scenarios (n=__N__).}
    \label{tab:mission-performence}
    \small
    \resizebox{\textwidth}{!}{%
        \begin{tabular}{l*{9}{c}}
            \toprule
            \textbf{Configuration}
            & \multicolumn{3}{c}{\textbf{Targets destroyed}}
            & \multicolumn{3}{c}{\textbf{Survival}}
            & \multicolumn{3}{c}{\textbf{Duration}} \\

            \cmidrule(lr){2-4}
            \cmidrule(lr){5-7}
            \cmidrule(lr){8-10}

            & Complete & Baseline & $p$-value
            & Complete & Baseline & $p$-value
            & Complete & Baseline & $p$-value \\
            \midrule

"""

_FOOTER = r"""            \bottomrule
        \end{tabular}%
    }
\end{table}
"""


def build_latex(scenarios, main, comparison, summary, tests, n_episodes, p_adjust) -> str:
    base = comparison[0] if comparison else None
    blocks = []
    for scn in scenarios:
        t = _cells(scn, "targets", main, base, summary, tests, p_adjust)
        s = _cells(scn, "survival", main, base, summary, tests, p_adjust)
        d = _cells(scn, "duration", main, base, summary, tests, p_adjust)
        blocks.append(
            f"            ${scn.name}$\n"
            f"            & {t[0]} & {t[1]} & {t[2]}\n"
            f"            & {s[0]} & {s[1]} & {s[2]}\n"
            f"            & {d[0]} & {d[1]} & {d[2]} \\\\\n"
        )
    return _HEADER.replace("__N__", str(n_episodes)) + "\n".join(blocks) + _FOOTER


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mission-performance LaTeX table (Targets/Survival/Duration) "
        "for a Complete vs Baseline policy across EVAL_SCENARIOS.")
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
    if not EVAL_SCENARIOS:
        raise RuntimeError("EVAL_SCENARIOS is empty — define at least one scenario.")

    default_ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    _print_config(MAIN_POLICY, COMPARISON_POLICIES, EVAL_SCENARIOS,
                  args.n_episodes, args.seed, args.alpha, args.p_adjust, device)
    cache: Dict[str, _LoadedCheckpoint] = {}
    _print_policy_diagnostics(MAIN_POLICY, COMPARISON_POLICIES, EVAL_SCENARIOS,
                              default_ckpt_path, device, cache)

    summary, tests, _samples = evaluate_comparison(
        EVAL_SCENARIOS, MAIN_POLICY, COMPARISON_POLICIES, default_ckpt_path,
        n_episodes=args.n_episodes, chunk=args.chunk, base_seed=args.seed,
        p_adjust=args.p_adjust, device=device, cache=cache,
    )

    latex = build_latex(EVAL_SCENARIOS, MAIN_POLICY, COMPARISON_POLICIES,
                        summary, tests, args.n_episodes, args.p_adjust)
    print("\n" + latex)

    out = _resolve_out(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table to: {out}")


if __name__ == "__main__":
    main()
