from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / 'scripts' / 'bench_v2'))

from shared_core import cli_main

if __name__ == '__main__':
    cli_main(default_plan='CountInVoxelPlan', default_tool='warp-md')
