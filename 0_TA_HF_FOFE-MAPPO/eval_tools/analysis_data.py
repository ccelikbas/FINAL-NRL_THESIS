"""analysis_data.py — the shared DATA DROP of the 3.3.x analyses.

Every 3.3.x analysis script keeps its own outputs (LaTeX tables, console tables,
its diagnostic figures) exactly as before, and ADDITIONALLY dumps the numbers it
just computed to a plain JSON file here. Nothing else changes: the analyses stay
the single source of truth for HOW the numbers are produced (rollout counts,
seeds, pairing, success-conditioning, which statistical test), and the config in
each script stays the place where that is set.

The point is that everything downstream — the presentation figures in
presentation/ above all — can then be built WITHOUT re-running a single rollout:
run the analysis once, then rebuild/restyle its charts as often as you like from
the saved file. Re-running the analysis is what refreshes the data; the chart
scripts only ever read it.

Layout (relative to the project dir 0_TA_HF_FOFE-MAPPO):

    eval_results/analysis_data/<name>.json      the numbers  (this module)
    eval_results/analysis_data/<name>.npz       bulk arrays, when a script has any

Each JSON has the same top level:

    {
      "name":      "3.3.2_mission_performance",   file/analysis id
      "source":    "3.3.2_mission_performance_analysis.py",
      "saved_at":  "2026-09-04T12:34:56",         local time, ISO-8601
      "meta":      {...},   run config: n_episodes, seed, alpha, ci_level, policies
      "data":      {...}    the analysis-specific payload
    }

`data` is whatever the analysis computed, converted to JSON-safe types (numpy
scalars/arrays → Python numbers/lists, tuples → lists, sets → sorted lists). NaN
and Infinity are written as the JSON literals `NaN` / `Infinity`, which Python's
own json module reads back as floats — so a missing value survives the round trip
as a NaN instead of turning into None.

Usage in an analysis script:

    from .analysis_data import save_analysis, save_arrays
    save_analysis("3.3.2_mission_performance", payload, meta=meta,
                  source=Path(__file__).name)

Usage in a figure script:

    from .analysis_data import load_analysis
    d = load_analysis("3.3.2_mission_performance")["data"]
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

# Where the dumps live, relative to the project dir (0_TA_HF_FOFE-MAPPO).
DATA_DIR = "eval_results/analysis_data"

_PKG_DIR = Path(__file__).resolve().parent.parent


def data_dir(out_dir: Optional[str] = None) -> Path:
    """The absolute analysis-data directory (created on demand by the writers)."""
    p = Path(out_dir) if out_dir else Path(DATA_DIR)
    return p if p.is_absolute() else (_PKG_DIR / p)


# ---------------------------------------------------------------------
#  JSON conversion
# ---------------------------------------------------------------------

def jsonable(obj: Any) -> Any:
    """Recursively convert an analysis result to JSON-safe types.

    numpy scalars → int/float/bool, numpy arrays → nested lists, tuples/sets →
    lists (sets sorted, so a dump is reproducible), Path → str, dict keys → str.
    Anything else that json cannot serialise falls back to repr(), so a dump never
    fails just because a payload picked up an exotic object."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return [jsonable(v) for v in sorted(obj, key=str)]
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "_asdict"):                     # namedtuple
        return jsonable(obj._asdict())
    if hasattr(obj, "__dict__") and not callable(obj):   # dataclass / simple object
        return {str(k): jsonable(v) for k, v in vars(obj).items()}
    return repr(obj)


def policy_record(pol) -> Optional[Dict[str, Any]]:
    """The identity of one evaluated policy, for the `meta` block: which
    checkpoint it is and whether it was rolled out with communication on."""
    if pol is None:
        return None
    return {"name": getattr(pol, "name", None),
            "policy_file": jsonable(getattr(pol, "policy_file", None)),
            "communicate": getattr(pol, "communicate", None)}


def scenario_record(scn) -> Dict[str, Any]:
    """The scenario/composition a block of numbers was measured on, so a figure
    can label its axes (and a reader can check the world) without the config."""
    keys = ("name", "n_strikers", "n_jammers", "n_known_targets",
            "n_unknown_targets", "n_known_radars", "n_unknown_radars",
            "radar_kill_probability", "scenario")
    return {k: jsonable(getattr(scn, k, None)) for k in keys}


# ---------------------------------------------------------------------
#  Writing
# ---------------------------------------------------------------------

def save_analysis(name: str, data: Any, *, meta: Optional[Mapping] = None,
                  source: Optional[str] = None,
                  out_dir: Optional[str] = None) -> Path:
    """Write one analysis' numbers to <data_dir>/<name>.json and return the path.

    `data` is the payload (any nested structure of dicts/lists/numbers/arrays);
    `meta` is the run config that produced it (episodes, seed, alpha, CI level,
    policies …), kept separate so a figure script can print provenance."""
    out = data_dir(out_dir) / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "name": name,
        "source": source,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "meta": jsonable(meta or {}),
        "data": jsonable(data),
    }
    # allow_nan=True (the default) writes NaN/Infinity as bare literals, which
    # json.load reads back as floats — a missing value stays a NaN, not None.
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Saved analysis data to: {out}")
    return out


def save_arrays(name: str, arrays: Mapping[str, Any], *,
                out_dir: Optional[str] = None) -> Path:
    """Write bulk numeric series to <data_dir>/<name>.npz (compressed).

    For anything too big to belong in the JSON — training curves, per-episode KPI
    samples. Keys become the npz member names."""
    out = data_dir(out_dir) / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in arrays.items()})
    print(f"Saved analysis arrays to: {out}")
    return out


# ---------------------------------------------------------------------
#  Reading
# ---------------------------------------------------------------------

def load_analysis(name: str, *, out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Read back a dump written by save_analysis (the whole document: name /
    source / saved_at / meta / data). Raises FileNotFoundError with the command
    to regenerate it when the analysis has not been run yet."""
    path = data_dir(out_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No analysis data at {path}.\nRun the matching eval_tools/{name}*.py "
            f"analysis first — it writes this file at the end of its run.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_arrays(name: str, *, out_dir: Optional[str] = None):
    """Read back the .npz written by save_arrays (an NpzFile; index by key)."""
    path = data_dir(out_dir) / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No analysis arrays at {path}.")
    return np.load(path, allow_pickle=False)
