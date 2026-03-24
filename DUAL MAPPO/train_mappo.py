#!/usr/bin/env python3
"""Top-level launcher for Dual-MAPPO Strike-EA training."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from MAPPO.run import main
else:
    from .run import main

if __name__ == "__main__":
    main()
