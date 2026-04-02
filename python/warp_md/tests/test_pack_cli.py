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
    assert "warp-pack solution" in result.stdout
    assert "warp-pack run" in result.stdout
    assert "mode-specific options" in result.stdout


def test_pack_cli_solution_print_config_uses_bundled_salt(tmp_path: Path) -> None:
    solute = tmp_path / "solute.pdb"
    solute.write_text(
        "ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
        encoding="utf-8",
    )
    result = _run(
        "solution",
        "--solute",
        str(solute),
        "--box",
        "40",
        "--solvent",
        "tip3p",
        "--salt",
        "cacl2",
        "--salt-molar",
        "0.15",
        "--output",
        str(tmp_path / "system.pdb"),
        "--print-config",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    paths = [Path(item["path"]).name for item in payload["structures"]]
    assert "ca.pdb" in paths
    assert "cl.pdb" in paths


def test_pack_cli_solution_supports_custom_catalog(tmp_path: Path) -> None:
    solute = tmp_path / "solute.pdb"
    acetate = tmp_path / "acetate.pdb"
    catalog = tmp_path / "catalog.json"
    solute.write_text(
        "ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
        encoding="utf-8",
    )
    acetate.write_text(
        "HETATM    1  C1  ACT A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
        encoding="utf-8",
    )
    catalog.write_text(
        json.dumps(
            {
                "ions": [
                    {
                        "species": "OAc-",
                        "aliases": ["acetate"],
                        "template": str(acetate),
                        "formula_symbol": "OAc",
                        "charge_e": -1,
                        "mass_amu": 59.044,
                    }
                ],
                "salts": [
                    {
                        "name": "naoac",
                        "aliases": ["sodium acetate"],
                        "species": {"Na+": 1, "OAc-": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = _run(
        "solution",
        "--solute",
        str(solute),
        "--box",
        "30",
        "--salt",
        "naoac",
        "--salt-molar",
        "0.1",
        "--catalog",
        str(catalog),
        "--output",
        str(tmp_path / "system.pdb"),
        "--print-config",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert any(Path(item["path"]).name == "acetate.pdb" for item in payload["structures"])


def test_pack_cli_solution_print_config_supports_bundled_polyatomic_salt(tmp_path: Path) -> None:
    solute = tmp_path / "solute.pdb"
    solute.write_text(
        "ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
        encoding="utf-8",
    )
    result = _run(
        "solution",
        "--solute",
        str(solute),
        "--box",
        "40",
        "--solvent",
        "tip3p",
        "--salt",
        "mgso4",
        "--salt-molar",
        "0.15",
        "--output",
        str(tmp_path / "system.pdb"),
        "--print-config",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    paths = {Path(item["path"]).name for item in payload["structures"]}
    assert "mg.pdb" in paths
    assert "so4.pdb" in paths


def test_pack_cli_solution_print_recipe_includes_templates_and_achieved_molarity(tmp_path: Path) -> None:
    solute = tmp_path / "solute.pdb"
    solute.write_text(
        "ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
        encoding="utf-8",
    )
    result = _run(
        "solution",
        "--solute",
        str(solute),
        "--box",
        "40",
        "--solvent",
        "tip3p",
        "--salt",
        "mgso4",
        "--salt-molar",
        "0.15",
        "--output",
        str(tmp_path / "system.pdb"),
        "--print-recipe",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["salt"]["achieved_molar"] is not None
    assert payload["templates"]["ions"]["SO4^2-"].endswith("so4.pdb")


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


@pytest.mark.skipif(not HAS_PACK_RUN, reason="pack bindings unavailable")
def test_pack_cli_output_override_infers_format_from_suffix(tmp_path: Path) -> None:
    cfg = {
        "structures": [
            {"path": water_pdb("tip3p"), "count": 1},
        ],
        "box": {"size": [18.0, 18.0, 18.0], "shape": "orthorhombic"},
    }
    cfg_path = tmp_path / "pack.json"
    out = tmp_path / "packed.cif"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = _run("--config", str(cfg_path), "--output", str(out))
    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert "data_warp_pack" in out.read_text(encoding="utf-8")

