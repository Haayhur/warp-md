import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from warp_md.pack import water_pdb

ROOT = Path(__file__).resolve().parents[3]
PYTHON_SRC = str(ROOT / "python")

try:
    from warp_md import traj_py  # type: ignore

    HAS_PACK_RUN = hasattr(traj_py, "pack_from_json")
except Exception:
    HAS_PACK_RUN = False


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = PYTHON_SRC + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "warp_md.pack_cli", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_pack_cli_help() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--output" in result.stdout
    assert "--format" in result.stdout


@pytest.mark.skipif(not HAS_PACK_RUN, reason="pack bindings unavailable")
def test_pack_cli_runs_json_config(tmp_path: Path) -> None:
    out = tmp_path / "packed.pdb"
    cfg = {
        "structures": [
            {"path": water_pdb("tip3p"), "count": 2},
        ],
        "box": {"size": [18.0, 18.0, 18.0], "shape": "orthorhombic"},
        "output": {"path": str(out), "format": "pdb"},
    }
    cfg_path = tmp_path / "pack.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = _run("--config", str(cfg_path))
    assert result.returncode == 0, result.stderr
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "ATOM" in text or "HETATM" in text

