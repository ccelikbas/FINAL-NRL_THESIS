"""NLR house-style plot palette and helpers.

Importing this module automatically applies the NLR plot style to
Matplotlib (and any Seaborn / Pandas plot that delegates to Matplotlib),
so every plot in the project uses the NLR palette without per-call
overrides.

Usage
-----
    from .nlr_style import (
        apply_nlr_style,
        NLR_PRIMARY, NLR_SECONDARY, NLR_ACCENT,
        NLR_GRAY, NLR_DARKGRAY,
        NLR_COLORS, nlr_color,
    )

Palette (NLR huisstijl)
-----------------------
    NLR_LIGHTBLUE      #19aee9
    NLR_DARKBLUE       #004d7d   <- primary / main curves
    NLR_TERRA          #ed7914   <- accent
    NLR_LIGHTBLUE_50   #94ceec
    NLR_LIGHTBLUE_20   #d8ecf8
    NLR_TERRA_50       #f8b985
    NLR_TERRA_20       #fde5cf
    NLR_GRAY           #9d9d9d   <- gridlines / reference
    NLR_DARKGRAY       #555555   <- axes / text
"""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

# ---------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------
NLR_LIGHTBLUE     = "#19aee9"
NLR_DARKBLUE      = "#004d7d"
NLR_TERRA         = "#ed7914"
NLR_LIGHTBLUE_50  = "#94ceec"
NLR_LIGHTBLUE_20  = "#d8ecf8"
NLR_TERRA_50      = "#f8b985"
NLR_TERRA_20      = "#fde5cf"
NLR_GRAY          = "#9d9d9d"
NLR_DARKGRAY      = "#555555"

NLR_COLORS = {
    "lightblue":     NLR_LIGHTBLUE,
    "darkblue":      NLR_DARKBLUE,
    "terra":         NLR_TERRA,
    "lightblue_50":  NLR_LIGHTBLUE_50,
    "lightblue_20":  NLR_LIGHTBLUE_20,
    "terra_50":      NLR_TERRA_50,
    "terra_20":      NLR_TERRA_20,
    "gray":          NLR_GRAY,
    "darkgray":      NLR_DARKGRAY,
}

# Semantic roles (donkerblauw=primair, lichtblauw=secundair, terra=accent).
NLR_PRIMARY   = NLR_DARKBLUE
NLR_SECONDARY = NLR_LIGHTBLUE
NLR_ACCENT    = NLR_TERRA
NLR_REFERENCE = NLR_DARKGRAY     # totals / reference lines
NLR_GRID      = NLR_GRAY         # gridlines

# Default property cycle for line plots. Primary first, then accent and
# secondary, then tints so dashboards with many series remain readable.
NLR_CYCLE = [
    NLR_DARKBLUE,
    NLR_TERRA,
    NLR_LIGHTBLUE,
    NLR_DARKGRAY,
    NLR_LIGHTBLUE_50,
    NLR_TERRA_50,
    NLR_GRAY,
    NLR_LIGHTBLUE_20,
    NLR_TERRA_20,
]

# Map common Matplotlib color shortcuts to NLR equivalents. Useful when
# remapping legacy ``color="tab:blue"`` style calls.
_MPL_TO_NLR = {
    "tab:blue":   NLR_DARKBLUE,
    "tab:orange": NLR_TERRA,
    "tab:green":  NLR_LIGHTBLUE,
    "tab:red":    NLR_TERRA,
    "tab:purple": NLR_LIGHTBLUE_50,
    "tab:brown":  NLR_DARKGRAY,
    "tab:pink":   NLR_TERRA_50,
    "tab:gray":   NLR_GRAY,
    "tab:olive":  NLR_GRAY,
    "tab:cyan":   NLR_LIGHTBLUE_50,
    "C0": NLR_DARKBLUE,
    "C1": NLR_TERRA,
    "C2": NLR_LIGHTBLUE,
    "C3": NLR_DARKGRAY,
    "C4": NLR_LIGHTBLUE_50,
    "C5": NLR_TERRA_50,
    "C6": NLR_GRAY,
    "C7": NLR_LIGHTBLUE_20,
    "C8": NLR_TERRA_20,
    "C9": NLR_DARKGRAY,
}


def nlr_color(key: str) -> str:
    """Return the NLR hex code for a palette key or Matplotlib shortcut.

    Unknown keys are returned unchanged, so this is also safe as a
    pass-through when a caller may supply any color string.
    """
    if key in NLR_COLORS:
        return NLR_COLORS[key]
    return _MPL_TO_NLR.get(key, key)


# ---------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------
_APPLIED = False


def apply_nlr_style() -> None:
    """Apply NLR house-style defaults to Matplotlib rcParams.

    Called automatically on import; safe to call multiple times.
    Affects the default color cycle, axes/text/tick colors, and grid
    appearance so every subsequent plot uses the NLR palette without
    per-plot color overrides.
    """
    global _APPLIED
    mpl.rcParams["axes.prop_cycle"]  = cycler(color=NLR_CYCLE)
    mpl.rcParams["axes.edgecolor"]   = NLR_DARKGRAY
    mpl.rcParams["axes.labelcolor"]  = NLR_DARKGRAY
    mpl.rcParams["axes.titlecolor"]  = NLR_DARKGRAY
    mpl.rcParams["text.color"]       = NLR_DARKGRAY
    mpl.rcParams["xtick.color"]      = NLR_DARKGRAY
    mpl.rcParams["ytick.color"]      = NLR_DARKGRAY
    mpl.rcParams["xtick.labelcolor"] = NLR_DARKGRAY
    mpl.rcParams["ytick.labelcolor"] = NLR_DARKGRAY
    mpl.rcParams["grid.color"]       = NLR_GRAY
    mpl.rcParams["grid.alpha"]       = 0.4
    mpl.rcParams["grid.linestyle"]   = "-"
    mpl.rcParams["grid.linewidth"]   = 0.6
    mpl.rcParams["legend.edgecolor"] = NLR_GRAY
    mpl.rcParams["legend.labelcolor"] = NLR_DARKGRAY
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"]   = "white"
    _APPLIED = True


# Auto-apply on import: any module that imports this gets NLR style
# applied before its first plot call.
apply_nlr_style()
