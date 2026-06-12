from __future__ import annotations

import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    from .traj_py import qm_cli

    args = ["warp-qm", *(sys.argv[1:] if argv is None else argv)]
    return int(qm_cli(args))


if __name__ == "__main__":
    raise SystemExit(main())
