"""kpi_correlation_analysis.py — how strongly are the mission KPIs coupled?

Reuses the SAME evaluation engine as 3.3.2 (evaluate_policy.evaluate_comparison,
paired rollouts under common random numbers) to collect PER-EPISODE KPIs, then
measures to what extent the four headline KPIs move together:

    Targets destroyed · Survival · Duration · Coalition fragmentation

For every (scenario, model) it rolls out N_EPISODES episodes, giving four
length-N per-episode vectors, and computes the 4x4 SPEARMAN rank-correlation
matrix between them. The matrices are rendered as a 2x2 grid of heatmaps
(rows = scenarios S1/S2, cols = Complete/Baseline) and written out as a high-DPI
PNG; the raw correlation and p-value values are also dumped to CSV.

──────────────────────────────────────────────────────────────────────────────
WHAT IS COMPUTED (and why Spearman)
──────────────────────────────────────────────────────────────────────────────
The unit of observation is one EPISODE. Each episode yields a value for each
KPI, so across N episodes each KPI is a length-N vector. The correlation between
two KPIs is computed ACROSS these episodes: e.g. do episodes with high survival
also tend to have low fragmentation?

We use the SPEARMAN rank correlation ρ rather than Pearson r. The raw per-episode
KPIs are not normal — targets/survival are bounded fractions, duration is
right-skewed, fragmentation is a bounded index — and Spearman (a Pearson
correlation on the RANKS) only assumes a MONOTONIC relationship, so it is robust
to that skew and to outliers. ρ ranges from −1 (perfectly opposed) through 0 (no
monotonic association) to +1 (move together). Each cell is annotated with ρ and a
two-sided significance star (*** p<0.001, ** p<0.01, * p<0.05) from the exact
Spearman test.

Duration is NOT restricted to successful episodes here (that would make
"targets destroyed" constant — every successful episode destroys all targets — and
so undefined). Instead all episodes are kept: a failed mission simply carries its
true (run-to-horizon) duration, which preserves the variance of every KPI. This is
a deliberate difference from the 3.3.2 mission-performance table, where duration is
success-conditioned only because it is being COMPARED between two methods.

Read-only: never modifies a checkpoint or the environment/engine.

Run (repo root, project venv):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\3.3.7_kpi_correlation_analysis.py
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\eval_tools\\3.3.7_kpi_correlation_analysis.py --n_episodes 100
"""
from __future__ import annotations

import argparse
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import spearmanr, pearsonr

# Reuse the evaluation engine (paired rollouts, KPI defs, path resolution) from
# evaluate_policy — exactly as 3.3.2 does. We only read the per-episode `samples`.
from .evaluate_policy import (
    PolicyInput, evaluate_comparison, _LoadedCheckpoint,
    _print_policy_diagnostics, KPIS, SEED_STRIDE,
)
from .run_curriculum import CurriculumSection
# NLR house palette (auto-applied to matplotlib on import).
from .nlr_style import (
    NLR_TERRA, NLR_TERRA_20, NLR_TERRA_50, NLR_LIGHTBLUE, NLR_LIGHTBLUE_20,
    NLR_LIGHTBLUE_50, NLR_DARKBLUE, NLR_DARKGRAY,
)

# Diverging colormap for correlation (−1 … 0 … +1): terra ← white → dark blue.
NLR_DIV = LinearSegmentedColormap.from_list(
    "nlr_div", [NLR_TERRA, NLR_TERRA_50, NLR_TERRA_20, "#ffffff",
                NLR_LIGHTBLUE_20, NLR_LIGHTBLUE_50, NLR_DARKBLUE])
NLR_DIV.set_bad("#eeeeee")

# =====================================================================
#  >>>  TEST CONFIG  <<<
# =====================================================================
# Number of paired episodes rolled out per policy per scenario.   [CLI: --n_episodes]
N_EPISODES = 600
# Parallel envs per rollout chunk (lower if you hit GPU OOM).          [CLI: --chunk]
CHUNK_EPISODES = 300
# Base RNG seed; both policies share it so episodes are paired 1:1.     [CLI: --seed]
BASE_SEED = 42
# Correlation method: "spearman" (rank, robust) or "pearson" (linear). [CLI: --method]
CORR_METHOD = "spearman"

# =====================================================================
#  >>>  POLICIES  (a (Complete, Baseline) pair per scenario)  <<<
#  Same four inputs as 3.3.2: each scenario row carries its own pair.
# =====================================================================
MAIN_LABEL = "Complete"
BASE_LABEL = "Baseline"


@dataclass
class ScenarioPolicies:
    """The (Complete, Baseline) policy pair evaluated for ONE scenario."""
    main: Optional[PolicyInput] = None
    base: Optional[PolicyInput] = None


SCENARIO_POLICIES: Dict[str, ScenarioPolicies] = {
    "S1": ScenarioPolicies(
        main=PolicyInput(name=MAIN_LABEL, policy_file="runs/FINALV2/complete_stage7of8_DR_j2-4_k0_25.pt", communicate=True),
        base=PolicyInput(name=BASE_LABEL, policy_file="runs/FINALV2/Final_Baseline_Cont_4.pt", communicate=False),
    ),
    "S2": ScenarioPolicies(
        main=PolicyInput(name=MAIN_LABEL, policy_file="runs/FINALV1/complete_S2_20260704/stage3of3_S2_DR_j2-4_k0_25_FINAL.pt", communicate=True),
        base=PolicyInput(name=BASE_LABEL, policy_file="runs/FINALV2/S2_Baseline_stage9of9_S2_DR_j2-4_k0_25_FINAL.pt", communicate=False),
    ),
}

# =====================================================================
#  >>>  SCENARIOS  (same worlds as 3.3.2)  <<<
# =====================================================================
EVAL_SCENARIOS: List[CurriculumSection] = [
    CurriculumSection(
        name="S1",
        n_iters=1,
        n_strikers=2, n_jammers=4,
        n_known_targets=(2, 4), n_unknown_targets=0,
        n_known_radars=(4, 6), n_unknown_radars=0,
        radar_kill_probability=0.25,
        scenario="S2",
    ),
    CurriculumSection(
        name="S2",
        n_iters=1,
        n_strikers=2, n_jammers=(2, 4),
        n_known_targets=(1, 2), n_unknown_targets=(1, 2),
        n_known_radars=(3, 4), n_unknown_radars=(1, 2),
        radar_kill_probability=0.25,
        scenario="S2",
    ),
]

# The four KPIs to correlate: (engine sample key, display label). Keys must exist
# in evaluate_policy.KPIS (per-episode arrays returned in `samples`).
CORR_KPIS: List[Tuple[str, str]] = [
    ("targets",       "Targets\ndestroyed"),
    ("survival",      "Survival"),
    ("duration",      "Duration"),
    ("fragmentation", "Coalition\nfrag."),
]

# Outputs (relative paths resolved against the project dir 0_TA_...).
OUT_PATH = "eval_results/kpi_correlation.png"
DPI = 600

# =====================================================================

_VALID_KEYS = {s.key for s in KPIS}
for _k, _ in CORR_KPIS:
    if _k not in _VALID_KEYS:
        raise RuntimeError(f"CORR_KPIS key '{_k}' is not a KPI in evaluate_policy.KPIS "
                           f"({sorted(_VALID_KEYS)}).")


def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_PKG_DIR / p)


# =====================================================================
#  Correlation
# =====================================================================

def correlation_matrix(vectors: List[np.ndarray], method: str
                       ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pairwise correlation + p-value matrices for K per-episode KPI vectors.

    `vectors` is a list of K equal-length arrays (one per KPI). Episodes with a
    non-finite value in ANY KPI are dropped (listwise) so every pair uses the
    SAME episode set. Returns (corr KxK, pval KxK, n_used). A KPI with zero
    variance yields NaN in its row/column (correlation undefined).
    """
    K = len(vectors)
    n = min(v.size for v in vectors)
    data = np.column_stack([np.asarray(v[:n], dtype=float) for v in vectors])
    finite = np.all(np.isfinite(data), axis=1)
    data = data[finite]
    n_used = data.shape[0]

    corr = np.full((K, K), np.nan)
    pval = np.full((K, K), np.nan)
    stat_fn = spearmanr if method == "spearman" else pearsonr
    for i in range(K):
        xi = data[:, i]
        const_i = np.ptp(xi) == 0.0
        for j in range(i, K):
            if i == j:
                corr[i, j], pval[i, j] = 1.0, 0.0
                continue
            xj = data[:, j]
            if n_used < 3 or const_i or np.ptp(xj) == 0.0:
                r, p = np.nan, np.nan
            else:
                r, p = stat_fn(xi, xj)
                r, p = float(r), float(p)
            corr[i, j] = corr[j, i] = r
            pval[i, j] = pval[j, i] = p
    return corr, pval, n_used


# =====================================================================
#  Plotting
# =====================================================================

def _cell_text_color(value: float, norm: Normalize) -> str:
    """Black or white annotation text, chosen for legibility on the cell colour."""
    if not np.isfinite(value):
        return NLR_DARKGRAY
    r, g, b, _ = NLR_DIV(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#ffffff" if luminance < 0.55 else "#14202b"


def _draw_corr(ax, corr, pval, labels, title, norm) -> None:
    K = len(labels)
    ax.imshow(np.ma.masked_invalid(corr), cmap=NLR_DIV, norm=norm,
              aspect="equal", interpolation="nearest")
    ax.set_xticks(range(K)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11, pad=6)
    ax.tick_params(length=0)
    for i in range(K):
        for j in range(K):
            v = corr[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8,
                        color=NLR_DARKGRAY)
                continue
            txt = f"{v:+.2f}".replace("+1.00", "1.00")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color=_cell_text_color(v, norm))


def plot_grid(matrices, scenarios, out_png, method) -> None:
    """2x2 grid: rows = scenarios, cols = Complete / Baseline. A missing policy
    slot renders an empty 'no policy' panel."""
    labels = [lbl for _, lbl in CORR_KPIS]
    cols = [(MAIN_LABEL, "main"), (BASE_LABEL, "base")]
    norm = Normalize(vmin=-1.0, vmax=1.0)

    nrow, ncol = len(scenarios), len(cols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.6 * nrow),
                             squeeze=False)
    for ri, scn in enumerate(scenarios):
        for ci, (col_label, slot) in enumerate(cols):
            ax = axes[ri, ci]
            entry = matrices.get((scn.name, slot))
            if entry is None:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{scn.name} · {col_label}\n(no policy)",
                        ha="center", va="center", fontsize=10, color=NLR_DARKGRAY,
                        transform=ax.transAxes)
                continue
            corr, pval, n_used = entry
            _draw_corr(ax, corr, pval, labels,
                       f"{scn.name} · {col_label}  (n={n_used})", norm)

    sm = plt.cm.ScalarMappable(cmap=NLR_DIV, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    cb.set_label(f"{method.capitalize()} correlation", fontsize=10)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved correlation heatmaps -> {out_png}")


def write_csv(matrices, scenarios, out_csv, method) -> None:
    keys = [k for k, _ in CORR_KPIS]
    with open(out_csv, "w") as f:
        f.write("scenario,model,method,kpi_a,kpi_b,n,correlation,p_value\n")
        cols = [(MAIN_LABEL, "main"), (BASE_LABEL, "base")]
        for scn in scenarios:
            for col_label, slot in cols:
                entry = matrices.get((scn.name, slot))
                if entry is None:
                    continue
                corr, pval, n_used = entry
                for i, ka in enumerate(keys):
                    for j, kb in enumerate(keys):
                        if j < i:
                            continue
                        f.write(f"{scn.name},{col_label},{method},{ka},{kb},"
                                f"{n_used},{corr[i, j]:.6f},{pval[i, j]:.6f}\n")
    print(f"saved correlation CSV     -> {out_csv}")


# =====================================================================
#  Config banner + CLI
# =====================================================================

def _slot_txt(pol: Optional[PolicyInput]) -> str:
    if pol is None or not pol.policy_file:
        return "(empty — panel left blank)"
    return str(pol.policy_file)


def _print_config(scenario_policies, scenarios, n_episodes, base_seed, method) -> None:
    print("─" * 78)
    print("  KPI CORRELATION — per-episode Spearman/Pearson across KPIs")
    print("─" * 78)
    print(f"  KPIs correlated    : {', '.join(k for k, _ in CORR_KPIS)}")
    print(f"  Method             : {method}")
    print("  Per-scenario policy pairs:")
    for scn in scenarios:
        pair = scenario_policies.get(scn.name)
        m = pair.main if pair else None
        b = pair.base if pair else None
        print(f"      [{scn.name}]  {MAIN_LABEL}: {_slot_txt(m)}")
        print(f"      {' ' * len(scn.name)}    {BASE_LABEL}: {_slot_txt(b)}")
    print(f"  Scenarios          : {len(scenarios)}  "
          f"({', '.join(s.name for s in scenarios)})")
    print(f"  N (episodes)       : {n_episodes}   base seed: {base_seed}")
    print("─" * 78)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="KPI correlation heatmaps (Targets/Survival/Duration/"
        "Fragmentation) per (scenario, model), reusing the 3.3.2 paired-rollout "
        "engine.")
    p.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                   help="Default .pt for any policy that leaves policy_file=None.")
    p.add_argument("--n_episodes", type=int, default=N_EPISODES)
    p.add_argument("--chunk", type=int, default=CHUNK_EPISODES)
    p.add_argument("--seed", type=int, default=BASE_SEED)
    p.add_argument("--method", type=str, default=CORR_METHOD,
                   choices=["spearman", "pearson"])
    p.add_argument("--device", type=str, default=None, metavar="DEVICE")
    p.add_argument("--out", type=str, default=OUT_PATH)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not EVAL_SCENARIOS:
        raise RuntimeError("EVAL_SCENARIOS is empty — define at least one scenario.")

    default_ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    _print_config(SCENARIO_POLICIES, EVAL_SCENARIOS, args.n_episodes, args.seed,
                  args.method)

    def _present(pol: Optional[PolicyInput]) -> Optional[PolicyInput]:
        if pol is not None and (pol.policy_file or default_ckpt_path is not None):
            return pol
        return None

    cache: Dict[str, _LoadedCheckpoint] = {}
    samples: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    slots: Dict[Tuple[str, str], Optional[PolicyInput]] = {}

    # Drive the engine one scenario at a time (each scenario has its OWN pair),
    # mirroring 3.3.2 so the seeding matches (base_seed + si*SEED_STRIDE).
    for si, scn in enumerate(EVAL_SCENARIOS):
        pair = SCENARIO_POLICIES.get(scn.name)
        main = _present(pair.main) if pair else None
        base = _present(pair.base) if pair else None
        slots[(scn.name, "main")] = main
        slots[(scn.name, "base")] = base

        present = [p for p in (main, base) if p is not None]
        if not present:
            print(f"\n[{si + 1}/{len(EVAL_SCENARIOS)}] Scenario '{scn.name}': "
                  f"no policies configured — leaving its panels blank.", flush=True)
            continue

        _print_policy_diagnostics(present[0], present[1:], [scn],
                                  default_ckpt_path, device, cache)
        seed = args.seed + si * SEED_STRIDE
        _summary, _tests, sub = evaluate_comparison(
            [scn], present[0], present[1:], default_ckpt_path,
            n_episodes=args.n_episodes, chunk=args.chunk, base_seed=seed,
            p_adjust="none", device=device, cache=cache,
        )
        samples.update(sub)

    # ── per (scenario, model) correlation matrices ──
    keys = [k for k, _ in CORR_KPIS]
    matrices: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, int]] = {}
    print("\n" + "=" * 78)
    print(f"  {args.method.upper()} CORRELATION MATRICES (per-episode, across KPIs)")
    print("=" * 78)
    for scn in EVAL_SCENARIOS:
        scn_samples = samples.get(scn.name, {})
        for slot in ("main", "base"):
            pol = slots.get((scn.name, slot))
            if pol is None or pol.name not in scn_samples:
                continue
            vectors = [np.asarray(scn_samples[pol.name][k], dtype=float) for k in keys]
            corr, pval, n_used = correlation_matrix(vectors, args.method)
            matrices[(scn.name, slot)] = (corr, pval, n_used)
            print(f"\n  {scn.name} · {pol.name}  (n={n_used})")
            header = "            " + "".join(f"{k[:8]:>10s}" for k in keys)
            print(header)
            for i, ka in enumerate(keys):
                cells = "".join(
                    f"{corr[i, j]:+.2f}".rjust(10)
                    if np.isfinite(corr[i, j]) else f"{'n/a':>10s}"
                    for j in range(len(keys)))
                print(f"  {ka[:10]:>10s}{cells}")

    if not matrices:
        raise RuntimeError("No correlation matrices produced — check the policy paths.")

    out_png = _resolve_out(args.out)
    plot_grid(matrices, EVAL_SCENARIOS, out_png, args.method)
    write_csv(matrices, EVAL_SCENARIOS, out_png.with_suffix(".csv"), args.method)


if __name__ == "__main__":
    main()
