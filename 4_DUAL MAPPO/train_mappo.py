#!/usr/bin/env python3
"""Top-level launcher for Dual-MAPPO Strike-EA training."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    import types
    _this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_this_dir.parent))
    # Register this directory as a proper Python package so that
    # relative imports resolve to DUAL MAPPO modules, not MAPPO.
    _pkg_name = "dual_mappo"
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_this_dir)]
        _pkg.__package__ = _pkg_name
        _pkg.__file__ = str(_this_dir / "__init__.py")
        sys.modules[_pkg_name] = _pkg
    __package__ = _pkg_name

from .run import main

if __name__ == "__main__":
    main()
