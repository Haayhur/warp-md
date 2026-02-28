from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_validation_manifest_has_expected_notebooks() -> None:
    root = _repo_root()
    manifest_path = root / "internal" / "validation" / "manifest.json"
    gates_path = root / "internal" / "validation" / "gates.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gates = json.loads(gates_path.read_text(encoding="utf-8"))

    notebooks = manifest.get("notebooks", [])
    assert len(notebooks) >= 10

    slugs = [n["slug"] for n in notebooks]
    assert len(slugs) == len(set(slugs))

    # Tiered gate policy must exist.
    assert "tiers" in gates
    assert "accuracy" in gates
    assert "speed" in gates

    for nb in notebooks:
        assert "runner" in nb
        assert "baseline" in nb
        if nb["runner"] == "mdanalysis":
            assert nb["baseline"] == "mdanalysis"
            assert nb.get("analyses")

    # Pack lanes are split into accuracy and speed modes.
    pack_small = next(n for n in notebooks if n["slug"] == "11_packmol_parity_box_fill")
    pack_large = next(n for n in notebooks if n["slug"] == "11_packmol_speed_large")
    assert pack_small["dataset"]["validation_mode"] == "accuracy"
    assert pack_small["dataset"]["enforce_speed_gates"] is False
    assert pack_large["dataset"]["validation_mode"] == "speed"
    assert pack_large["dataset"]["enforce_speed_gates"] is True


def test_validation_notebooks_generated_from_manifest(tmp_path: Path) -> None:
    root = _repo_root()
    manifest_path = root / "internal" / "validation" / "manifest.json"
    generator = root / "scripts" / "validation" / "generate_colab_notebooks.py"

    subprocess.check_call(
        [
            sys.executable,
            str(generator),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=root,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = [n["slug"] for n in manifest.get("notebooks", [])]

    for slug in expected:
        nb_path = tmp_path / f"{slug}.ipynb"
        assert nb_path.exists()
        doc = json.loads(nb_path.read_text(encoding="utf-8"))
        assert doc["nbformat"] == 4
        assert any(cell.get("cell_type") == "code" for cell in doc.get("cells", []))
