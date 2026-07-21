"""train_master.py — one easy control file to train the four (model × scenario)
policies, in any combination, any number of times, from scratch or a checkpoint.

This REPLACES run_multiple_curriculums.py (which is kept unchanged). The WHAT-to-
train lives in training/ (scenarios + four Job specs); this file owns the ENGINE
and the run control.

──────────────────────────────────────────────────────────────────────────────
THE FOUR JOBS
──────────────────────────────────────────────────────────────────────────────
    complete_s1 · baseline_s1   → trained FROM SCRATCH (S1, all-known world)
    complete_s2 · baseline_s2   → trained FROM A CHECKPOINT (S2 continuation)

──────────────────────────────────────────────────────────────────────────────
HOW YOU CONTROL IT  —  edit the RUNS list below
──────────────────────────────────────────────────────────────────────────────
RUNS is an ORDERED list; each `Run` is executed in turn. A Run picks which jobs
to train, a `tag` (→ its own runs/<tag>/ folder so nothing overwrites), a seed,
and — optionally — explicit warm-start checkpoints. This expresses everything:
    • all four once            → one Run(jobs=ALL_JOBS, ...)
    • all four, twice (stats)  → two Runs with DIFFERENT tags (and seeds)
    • a single model           → Run(jobs=["complete_s2"], ...)

──────────────────────────────────────────────────────────────────────────────
WHERE AN S2 JOB'S CHECKPOINT COMES FROM  (two ways — this is the key bit)
──────────────────────────────────────────────────────────────────────────────
For each S2 job the warm-start checkpoint is resolved in this order:
  0. FORCED SCRATCH — `from_scratch=True` on the Run trains every job in it from
     random init, ignoring everything below (use to train an S2 job from scratch).
  1. EXPLICIT — `checkpoints={"complete_s2": "runs/V3/complete_S1_FINAL.pt"}`
     on the Run wins (path relative to 0_TA_HF_FOFE-MAPPO/, or absolute).
  2. CHAINED  — otherwise it CONTINUES from its S1 sibling trained EARLIER in
     this same master run (e.g. put "complete_s1" then "complete_s2" in one Run,
     or an earlier Run — no path needed; this is the run_multiple behaviour).
  3. ON DISK  — otherwise the newest matching <model>_S1 checkpoint already saved
     under runs/ is used.
  4. else it trains from scratch, with a loud warning (unusual for S2).

──────────────────────────────────────────────────────────────────────────────
OUTPUTS  (organised; repeated runs never overwrite)
──────────────────────────────────────────────────────────────────────────────
  runs/<tag>/<model>_<scenario>_FINAL.pt        canonical checkpoint (rolling;
                                                same keys as run_curriculum.py, so
                                                every eval_tools script reads it)
  runs/<tag>/<model>_<scenario>/stageNofN_*.pt  per-stage backups (never overwritten)
  plots/<tag>/<job>_{dashboard,reward,eval_rates}.png
If runs/<tag>/ already holds a job's FINAL from a previous session, the whole
run is redirected to runs/<tag>_2/ (…_3, …) so data is never lost.

Run (repo root, project venv) — disable machine sleep on AC (sleep hangs CUDA):
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\train_master.py
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\train_master.py --dry_run
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\train_master.py --smoke 3
  .venv\\Scripts\\python.exe 0_TA_HF_FOFE-MAPPO\\train_master.py --only V3
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# Windows consoles default to cp1252, which can't encode the plan glyphs (→ — ▶).
# Switch the streams to UTF-8 so printing never crashes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

if __package__ in (None, ""):
    import types
    _this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_this_dir.parent))
    _pkg_name = "fofe_mappo"
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_this_dir)]
        _pkg.__package__ = _pkg_name
        _pkg.__file__ = str(_this_dir / "__init__.py")
        sys.modules[_pkg_name] = _pkg
    __package__ = _pkg_name

import torch

from .config import (
    DomainRandomization, EnvConfig, EnvExtensionsConfig, ExperimentConfig,
    FOFEConfig, NetworkConfig, PPOConfig,
)
from .nlr_style import apply_nlr_style
from .rewards import RewardConfig
from .trainer import train_mappo
from .run_curriculum import (
    CurriculumSection, _section_to_env_cfg, _section_label,
    _merge_logs, _state_dict_to_cpu,
    plot_curriculum_dashboard, plot_curriculum_reward, plot_curriculum_eval_rates,
)
# The four (model × scenario) job specs + the shared Job type.
from .training.scenarios import Job
from .training.job_complete_s1 import JOB as _JOB_COMPLETE_S1
from .training.job_baseline_s1 import JOB as _JOB_BASELINE_S1
from .training.job_complete_s2 import JOB as _JOB_COMPLETE_S2
from .training.job_baseline_s2 import JOB as _JOB_BASELINE_S2

apply_nlr_style()

# Job registry (key → Job) and the convenience "all four" list.
_JOBS: Dict[str, Job] = {
    j.key: j for j in (_JOB_COMPLETE_S1, _JOB_BASELINE_S1, _JOB_COMPLETE_S2, _JOB_BASELINE_S2)
}
ALL_JOBS: List[str] = ["complete_s1", "baseline_s1", "complete_s2", "baseline_s2"]


# =====================================================================
#  >>>  RUN CONTROL  —  THIS is the part you edit  <<<
# =====================================================================

@dataclass
class Run:
    """One batch of training. Executed in order; each writes to runs/<tag>/.

    tag         : folder name under runs/ (and plots/). Use a DIFFERENT tag for
                  each repeat so nothing overwrites (needed for statistics).
    jobs        : ordered list of job keys to train (subset of ALL_JOBS).
    seed        : base RNG seed for this run (repeats should differ, e.g. 0, 1).
    checkpoints : optional {job_key: checkpoint_path} to FORCE an S2 job's warm-
                  start (path relative to 0_TA_HF_FOFE-MAPPO/, or absolute).
                  Omit to CHAIN from the S1 sibling trained earlier / on disk.
    from_scratch: True → force EVERY job in this Run to train from random init,
                  ignoring checkpoints / chaining / on-disk S1 (overrides the
                  Job's own from_scratch). Use to train an S2 job from scratch.
    """
    tag: str
    jobs: List[str] = field(default_factory=lambda: list(ALL_JOBS))
    seed: int = 42
    checkpoints: Dict[str, str] = field(default_factory=dict)
    from_scratch: bool = False


# EDIT THIS. Each Run runs top-to-bottom. See the header for the checkpoint rules.
RUNS: List[Run] = [
    # ── Train all four models once (S1 from scratch; S2 chains from the S1
    #    trained just above it in this same run — no checkpoint paths needed). ──
    # Run(tag="V2", jobs="complete_s2", seed=0),

    # ── Example: a SECOND independent repeat for statistics. Different tag +
    #    seed → its own runs/V3b/ folder, nothing from "V3" is overwritten. ──
    # Run(tag="V3b", jobs=ALL_JOBS, seed=1),

    # ── Example: only the complete S1→S2 lineage (chained, no paths). ──
    # Run(tag="V3_complete_S1", jobs=["complete_s1"], seed=0),

    Run(tag="V5_test_complete_S2", jobs=["complete_s2"], seed=0, from_scratch=True),
    # Run(tag="V5_test_baseline_S2", jobs=["baseline_s2"], seed=0, from_scratch=True),
]


# =====================================================================
#  IMPLEMENTATION  —  the engine.  You usually do not edit below here.
# =====================================================================

_PKG_DIR = Path(__file__).resolve().parent


def _align_agents(resolved: List[Tuple[CurriculumSection, EnvConfig]]) -> None:
    """AGENT-ALLOCATION ALIGNMENT across a job's sections, in place. Pads
    n_strikers/n_jammers allocation to the job-wide max so tensor shapes are
    identical section-to-section. The zero-padded (non-FOFE) critic's input dim
    depends on the ALLOCATED agent count, so without this the carried critic
    silently re-initialises at any boundary that changes the count (e.g. the
    warmup-J2 → DR-J4 step). Sections with a smaller fixed count keep their
    ACTUAL present count via a degenerate DR range (alloc=max, lo=hi=actual).
    FOFE nets are count-invariant either way."""
    max_ns = max(c.n_strikers for _, c in resolved)
    max_nj = max(c.n_jammers for _, c in resolved)
    for _, c in resolved:
        if c.n_strikers < max_ns or c.n_jammers < max_nj:
            if c.dr is None:
                c.dr = DomainRandomization()
        if c.n_strikers < max_ns:
            if c.dr.n_strikers is None:
                c.dr.n_strikers = (c.n_strikers, c.n_strikers)
            c.n_strikers = max_ns
        if c.n_jammers < max_nj:
            if c.dr.n_jammers is None:
                c.dr.n_jammers = (c.n_jammers, c.n_jammers)
            c.n_jammers = max_nj


def _warn_critic_dim(resolved: List[Tuple[CurriculumSection, EnvConfig]], job: Job) -> None:
    """The zero-padded (baseline) critic's input dim depends on ALLOCATED entity
    TOTALS. n_agents is aligned by _align_agents; targets/radars are NOT force-
    aligned. If the target/radar TOTALS differ across the job's sections the
    baseline critic re-initialises at that boundary (the ACTOR still transfers —
    it is slot-based). FOFE critics are count-invariant, so this only matters for
    the baseline. Within a single scenario the world is constant → no divergence;
    this flags any future edit that breaks it."""
    if job.use_fofe:
        return
    totals = {(c.n_targets, c.n_radars) for _, c in resolved}
    if len(totals) > 1:
        print(f"  [!] baseline '{job.key}': target/radar TOTALS vary across the "
              f"curriculum {sorted(totals)} → the zero-padded CRITIC re-initialises "
              f"at the boundary where they change (the policy/actor still transfers).")


def _safe(s: str) -> str:
    """Filesystem-safe token from a section name (spaces/dots → underscores)."""
    return "".join(c if (c.isalnum() or c in "-") else "_" for c in str(s)).strip("_")


def _resolve_ckpt_path(spec: str) -> Path:
    """Resolve a user-given checkpoint path (absolute, or relative to the package
    dir 0_TA_HF_FOFE-MAPPO/ — so 'runs/V3/complete_S1_FINAL.pt' works)."""
    p = Path(spec)
    return p if p.is_absolute() else (_PKG_DIR / p)


def _find_prior_s1(save_root: Path, model: str) -> Optional[Path]:
    """Newest on-disk S1 checkpoint for a model, searched under runs/ (this
    layout's runs/<tag>/<model>_S1_FINAL.pt AND legacy flat <model>_S1_*.pt)."""
    cands = list(save_root.glob(f"**/{model}_S1_FINAL.pt"))
    cands += list(save_root.glob(f"{model}_S1_*.pt"))          # legacy flat names
    cands = [p for p in cands if p.is_file()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _load_initial_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a saved .pt as warm-start weights (policy/critic/reward-norm)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "policy_state_dict": ck["policy_state_dict"],
        "critic_state_dict": ck["critic_state_dict"],
        "reward_normalizer_state_dict": ck.get("reward_normalizer_state_dict"),
    }


def _resolve_warmstart(job: Job, run: Run, produced: Dict[str, Path],
                       save_root: Path) -> Tuple[Optional[Path], str]:
    """Resolve an S2 job's warm-start checkpoint (see the module header for the
    order). Returns (path_or_None, human_source)."""
    # 0. forced from scratch on the Run — overrides all fallbacks below.
    if run.from_scratch:
        return None, "from scratch (forced by Run)"
    # 1. explicit path on the Run.
    spec = run.checkpoints.get(job.key)
    if spec:
        p = _resolve_ckpt_path(spec)
        if not p.exists():
            raise FileNotFoundError(
                f"job '{job.key}': explicit checkpoint not found: {p}")
        return p, f"explicit ({p})"
    # 2. chained from the S1 sibling trained earlier THIS master run.
    if job.warmstart_from and job.warmstart_from in produced:
        return produced[job.warmstart_from], f"chained from {job.warmstart_from} (this run)"
    # 3. newest matching S1 on disk.
    if not job.from_scratch:
        prior = _find_prior_s1(save_root, job.model)
        if prior is not None:
            return prior, f"newest on disk ({prior})"
        print(f"  [!] {job.key}: no explicit / chained / on-disk {job.model}_S1 "
              f"checkpoint found — training S2 FROM SCRATCH (unusual).")
    # 4. from scratch.
    return None, "from scratch"


def _resolve_job(job: Job):
    """Resolve every section → [(sec, env_cfg)] with the job's env_overrides
    applied, align agent allocation across the job, warn on critic divergence.
    Returns (resolved, fofe_cfg, ext_cfg, hf_radar_cfg, net_cfg, reward_cfg)."""
    fofe_cfg = FOFEConfig(use_fofe=job.use_fofe)
    ext_cfg = EnvExtensionsConfig()
    hf_radar_cfg = ext_cfg.hf_radar if ext_cfg.use_hf_radar else None
    env_defaults = EnvConfig()
    reward_cfg = RewardConfig()          # current rewards.py = the vetted setup
    net_cfg = NetworkConfig()            # identical nets for complete & baseline

    resolved: List[Tuple[CurriculumSection, EnvConfig]] = [
        (sec, _section_to_env_cfg(sec, env_defaults, reward_cfg, fofe_cfg))
        for sec in job.curriculum
    ]
    for _, env_cfg in resolved:          # per-job EnvConfig overrides
        for k, v in job.env_overrides.items():
            setattr(env_cfg, k, v)
    _align_agents(resolved)
    _warn_critic_dim(resolved, job)
    return resolved, fofe_cfg, ext_cfg, hf_radar_cfg, net_cfg, reward_cfg


def _save_checkpoint(save_path: Path, checkpoint: Dict[str, Any], *,
                     net_cfg, fofe_cfg, ext_cfg, reward_cfg,
                     all_logs, curriculum_meta, section_bounds,
                     run_name: str, description: str, completed: int, total: int) -> None:
    """Write the .pt with the exact key layout run_curriculum.py uses (so every
    eval_tools script works on it), plus run metadata. Called after every section
    — the file is progressively overwritten as crash insurance."""
    torch.save(
        {
            "policy_state_dict": checkpoint["policy_state_dict"],
            "critic_state_dict": checkpoint["critic_state_dict"],
            "reward_normalizer_state_dict": checkpoint["reward_normalizer_state_dict"],
            "net_cfg": net_cfg,
            "fofe_cfg": fofe_cfg,
            "ext_cfg": ext_cfg,
            "reward_cfg": reward_cfg,
            "training_logs": all_logs,
            "curriculum": curriculum_meta,
            "section_bounds": section_bounds,
            "run_name": run_name,
            "run_description": description,
            "sections_completed": f"{completed}/{total}",
        },
        save_path,
    )


def _print_job_plan(job: Job, resolved, out_dir: Path, warm_src: str, seed: int) -> None:
    total_iters = sum(sec.n_iters for sec, _ in resolved)
    print("\n" + "-" * 78)
    print(f"  JOB '{job.key}'  —  {job.description}")
    print(f"  FOFE {'ON' if job.use_fofe else 'OFF'} | comm {'ON' if job.communicate else 'OFF'} "
          f"| {total_iters} iters | {len(resolved)} sections | seed {seed}")
    print(f"  warm-start : {warm_src}")
    print(f"  output     : {out_dir.name}/{job.model}_{job.scenario}_FINAL.pt (+ backup folder)")
    cursor = 0
    for sec, env_cfg in resolved:
        dr_tag = "DR" if env_cfg.dr is not None else "fixed"
        print(f"    [{cursor:5d},{cursor + sec.n_iters:5d})  {sec.name:18s} "
              f"({dr_tag})  {_section_label(sec, env_cfg)}")
        cursor += sec.n_iters
    print("-" * 78)


def _train_job(job: Job, warmstart_path: Optional[Path], out_dir: Path,
               plots_dir: Path, seed: int, args) -> Optional[Path]:
    """Train one job through its curriculum, warm-starting from `warmstart_path`
    if given. Saves a rolling canonical checkpoint + per-stage backups under
    out_dir, and per-job plots. Returns the canonical checkpoint path."""
    resolved, fofe_cfg, ext_cfg, hf_radar_cfg, net_cfg, reward_cfg = _resolve_job(job)

    backup_dir = out_dir / f"{job.model}_{job.scenario}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    canonical = out_dir / f"{job.model}_{job.scenario}_FINAL.pt"
    n_sec = len(resolved)

    checkpoint: Optional[Dict[str, Any]] = None
    if warmstart_path is not None:
        checkpoint = _load_initial_checkpoint(warmstart_path)
        print(f"  warm-start: '{job.key}' continues from {warmstart_path.name}")

    all_logs: Dict[str, List[float]] = {}
    section_bounds: List[Tuple[str, int, int]] = []
    curriculum_meta: List[Dict[str, Any]] = []
    global_iter = 0
    t0 = time.time()

    for si, (sec, env_cfg) in enumerate(resolved):
        ppo_cfg = PPOConfig(num_envs=args.num_envs, n_iters=sec.n_iters,
                            seed=seed + global_iter, log_every=args.log_every)
        ppo_cfg.iteration_offset = global_iter
        exp_cfg = ExperimentConfig(
            env=env_cfg, ppo=ppo_cfg, net=net_cfg, fofe=fofe_cfg, ext=ext_cfg,
        ).finalize()

        print(f"\n  ── {job.key} section '{sec.name}'  "
              f"iters [{global_iter}, {global_iter + sec.n_iters}) ──")
        base_env, policy, critic, logs, reward_normalizer = train_mappo(
            exp_cfg.env, exp_cfg.ppo, exp_cfg.net, fofe_cfg, checkpoint,
            hf_radar_cfg=hf_radar_cfg,
        )

        _merge_logs(all_logs, logs)
        section_bounds.append((f"{job.scenario}:{sec.name}", global_iter, global_iter + sec.n_iters))
        curriculum_meta.append({"name": f"{job.scenario}:{sec.name}", "n_iters": sec.n_iters,
                                "label": _section_label(sec, env_cfg)})
        global_iter += sec.n_iters

        # Carry weights forward on CPU (release GPU before the next section).
        checkpoint = {
            "policy_state_dict": _state_dict_to_cpu(policy.state_dict()),
            "critic_state_dict": _state_dict_to_cpu(critic.state_dict()),
            "reward_normalizer_state_dict": _state_dict_to_cpu(
                reward_normalizer.state_dict() if reward_normalizer is not None else None
            ),
        }
        del base_env, policy, critic, logs, reward_normalizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        is_final = (si + 1 == n_sec)
        stage_tag = f"stage{si + 1}of{n_sec}_{_safe(sec.name)}" + ("_FINAL" if is_final else "")
        save_kwargs = dict(
            net_cfg=net_cfg, fofe_cfg=fofe_cfg, ext_cfg=ext_cfg, reward_cfg=reward_cfg,
            all_logs=all_logs, curriculum_meta=list(curriculum_meta),
            section_bounds=list(section_bounds),
            run_name=f"{job.model}_{job.scenario}",
            description=f"{job.description}  [tag {out_dir.name}]",
            completed=si + 1, total=n_sec,
        )
        _save_checkpoint(canonical, checkpoint, **save_kwargs)                 # rolling canonical
        _save_checkpoint(backup_dir / f"{stage_tag}.pt", checkpoint, **save_kwargs)  # kept backup
        print(f"  saved → {out_dir.name}/{canonical.name}  +  "
              f"{backup_dir.name}/{stage_tag}.pt  "
              f"({si + 1}/{n_sec}, {(time.time() - t0) / 3600:.1f} h elapsed)")

    if not args.no_plot:
        for fn, suffix in ((plot_curriculum_dashboard, "dashboard"),
                           (plot_curriculum_reward, "reward"),
                           (plot_curriculum_eval_rates, "eval_rates")):
            try:
                fn(all_logs, section_bounds,
                   save_path=str(plots_dir / f"{job.key}_{suffix}.png"))
            except Exception as exc:  # noqa: BLE001 — plots must never kill training
                print(f"  plot warning ({suffix}): {type(exc).__name__}: {exc}")

    print(f"  JOB '{job.key}' DONE in {(time.time() - t0) / 3600:.1f} h  →  {canonical}")
    return canonical


def _unique_out_dir(save_root: Path, tag: str, job_keys: List[str]) -> Path:
    """runs/<tag>, redirected to <tag>_2, <tag>_3 … if that folder already holds
    a FINAL for any job in THIS run (never overwrite a prior session's data)."""
    def collides(d: Path) -> bool:
        return d.exists() and any(
            (d / f"{_JOBS[k].model}_{_JOBS[k].scenario}_FINAL.pt").exists() for k in job_keys)
    out = save_root / tag
    if not collides(out):
        return out
    i = 2
    while collides(save_root / f"{tag}_{i}"):
        i += 1
    bumped = save_root / f"{tag}_{i}"
    print(f"  [!] runs/{tag}/ already holds a FINAL for one of these jobs — "
          f"redirecting this run to runs/{bumped.name}/ so nothing is overwritten.")
    return bumped


def _maybe_smoke(job: Job, smoke: Optional[int]) -> Job:
    """Return a copy of the job with every section trimmed to `smoke` iters
    (pipeline test at the REAL worlds), or the job unchanged when smoke is off."""
    if not smoke:
        return job
    trimmed = [replace(sec, n_iters=smoke) for sec in job.curriculum]
    return replace(job, curriculum=trimmed)


def _build_parser() -> argparse.ArgumentParser:
    ppo_d = PPOConfig()
    p = argparse.ArgumentParser(
        description="Master trainer for the four (model × scenario) policies. "
        "Edit the RUNS list at the top of this file.")
    p.add_argument("--num_envs", type=int, default=ppo_d.num_envs,
                   help="Parallel envs (PPOConfig default; the FOFE runs OOM an "
                        "8GB laptop GPU even at 512 — run the real batch on the server).")
    p.add_argument("--log_every", type=int, default=ppo_d.log_every,
                   help="Eval cadence — eval mirrors the active section's DR.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override every Run's seed with this value (default: use each Run's seed).")
    p.add_argument("--save_dir", type=str, default="runs",
                   help="Checkpoint root. RELATIVE paths resolve against this file's "
                        "folder (0_TA_HF_FOFE-MAPPO/), NOT the CWD.")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated Run TAGS to execute (default: all RUNS). E.g. --only V3")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="Resolve + print every run/job plan (validates DR ranges + "
                        "warm-start resolution), then exit without training.")
    p.add_argument("--smoke", type=int, default=None, metavar="N",
                   help="Smoke-test: run every section for only N iters at the REAL "
                        "num_envs/worlds (validates the full pipeline incl. GPU memory). "
                        "Outputs get a '_smoke' tag so real runs stay clean.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if not RUNS:
        print("\n  RUNS is empty — add a Run to the RUNS list at the top of the file.")
        return
    unknown = [k for run in RUNS for k in run.jobs if k not in _JOBS]
    if unknown:
        raise ValueError(f"Unknown job key(s) {sorted(set(unknown))}; valid: {ALL_JOBS}")

    save_root = Path(args.save_dir)
    if not save_root.is_absolute():
        save_root = _PKG_DIR / save_root
    save_root.mkdir(parents=True, exist_ok=True)

    selected = list(RUNS)
    if args.only:
        wanted = {w.strip() for w in args.only.split(",")}
        selected = [r for r in selected if r.tag in wanted]
    if not selected:
        print(f"\n  Nothing to run — --only {args.only} matched no Run tags.")
        return

    total_iters = sum(sec.n_iters for r in selected for k in r.jobs for sec in _JOBS[k].curriculum)
    smoke_note = f" (SMOKE {args.smoke} iters/section)" if args.smoke else ""
    print("\n" + "=" * 78)
    print(f"  TRAIN MASTER — {len(selected)} run(s), "
          f"{sum(len(r.jobs) for r in selected)} job(s){smoke_note}")
    print(f"  num_envs {args.num_envs} | eval every {args.log_every} iters | "
          f"checkpoints → {save_root}")
    if not args.smoke:
        print(f"  rough ETA at ~9 s/iter: {total_iters * 9 / 3600:.0f} h "
              f"(baseline runs faster without FOFE)")
    print("  NOTE: disable machine sleep on AC — sleep hangs CUDA runs.")
    print("=" * 78)

    produced: Dict[str, Path] = {}   # job_key → saved canonical path (this master run)
    failures: List[str] = []         # "tag/job" labels

    for run in selected:
        tag = run.tag + ("_smoke" if args.smoke else "")
        seed = args.seed if args.seed is not None else run.seed
        out_dir = _unique_out_dir(save_root, tag, run.jobs) if not args.dry_run else save_root / tag
        plots_dir = _PKG_DIR / "plots" / tag

        print("\n" + "=" * 78)
        print(f"  RUN '{tag}'  —  jobs: {', '.join(run.jobs)}  | seed {seed}  → {out_dir}")
        print("=" * 78)

        for jobkey in run.jobs:
            job = _maybe_smoke(_JOBS[jobkey], args.smoke)
            try:
                warm_path, warm_src = _resolve_warmstart(job, run, produced, save_root)
            except FileNotFoundError as exc:
                print(f"  !! {tag}/{jobkey} SKIPPED: {exc}")
                failures.append(f"{tag}/{jobkey}")
                continue

            if args.dry_run:
                resolved, *_ = _resolve_job(job)
                _print_job_plan(job, resolved, out_dir, warm_src, seed)
                # Predict this job's checkpoint so a later S2 plan shows it chaining.
                produced[jobkey] = out_dir / f"{job.model}_{job.scenario}_FINAL.pt"
                continue

            plots_dir.mkdir(parents=True, exist_ok=True)
            resolved, *_ = _resolve_job(job)
            _print_job_plan(job, resolved, out_dir, warm_src, seed)
            try:
                canonical = _train_job(job, warm_path, out_dir, plots_dir, seed, args)
                produced[jobkey] = canonical      # later S2 jobs can chain from this
            except Exception as exc:  # noqa: BLE001 — one failed job must not kill the batch
                failures.append(f"{tag}/{jobkey}")
                print(f"\n  !! {tag}/{jobkey} FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if args.dry_run:
        print("\n(dry run — nothing trained)")
        return

    print("\n" + "=" * 78)
    print("  BATCH SUMMARY")
    for run in selected:
        tag = run.tag + ("_smoke" if args.smoke else "")
        for jobkey in run.jobs:
            label = f"{tag}/{jobkey}"
            status = "FAILED" if label in failures else ("ok" if jobkey in produced else "done")
            print(f"    {label:<32} {status}")
    print("=" * 78)
    if failures:
        print(f"  {len(failures)} job(s) failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
