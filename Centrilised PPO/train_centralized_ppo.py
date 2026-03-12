from pathlib import Path
import runpy
import sys

pkg_root = Path(__file__).resolve().parent
sys.path.insert(0, str(pkg_root))
runpy.run_module('centralized_strike_ppo.run', run_name='__main__')
