from __future__ import annotations

import subprocess
import sys
from typing import Optional

from . import polymer_build


def run_cli(argv: Optional[list[str]] = None) -> int:
    command = [polymer_build._binary(), *(argv if argv is not None else sys.argv[1:])]
    result = subprocess.run(command, check=False)
    return int(result.returncode)


def main(argv: Optional[list[str]] = None) -> int:
    try:
        return run_cli(argv)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
