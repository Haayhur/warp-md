import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from warp_md.pack import water_pdb
from warp_md.pack.config import Box, OutputSpec, PackConfig, Structure
from warp_md.pack.export import export
from warp_md.pack.result import PackResult

try:
    from warp_md import traj_py  # type: ignore

    HAS_PACK_EXPORT = hasattr(traj_py, "pack_write_output")
except Exception:
    HAS_PACK_EXPORT = False


class DummyResult:
    def __init__(self):
        self.coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        self.box = (10.0, 10.0, 10.0)
        self.name = ["H1", "O1"]
        self.element = ["H", "O"]
        self.resname = ["MOL", "WAT"]
        self.resid = [1, 2]
        self.chain = ["A", "A"]
        self.segid = ["", "SEG"]
        self.charge = [0.0, -0.8]
        self.mol_id = [1, 2]
        self.bonds = [(0, 1)]
        self.record_kind = ["ATOM", "HETATM"]
        self.ter_after = [0]


def test_packconfig_to_dict_includes_flags():
    cfg = PackConfig(
        structures=[Structure("water.pdb", count=2)],
        box=Box((10.0, 10.0, 10.0)),
        output=OutputSpec("out.pdb", "pdb"),
        add_amber_ter=True,
        amber_ter_preserve=True,
        hexadecimal_indices=True,
        use_short_tol=True,
        short_tol_dist=0.5,
        short_tol_scale=2.0,
        movebadrandom=True,
        randominitialpoint=True,
        fbins=2.0,
        restart_from="all.restart",
        restart_to="all.out.restart",
        relax_steps=5,
        relax_step=0.25,
    )
    payload = cfg.to_dict()
    assert payload["add_amber_ter"] is True
    assert payload["amber_ter_preserve"] is True
    assert payload["hexadecimal_indices"] is True
    assert payload["use_short_tol"] is True
    assert payload["short_tol_dist"] == 0.5
    assert payload["short_tol_scale"] == 2.0
    assert payload["movebadrandom"] is True
    assert payload["randominitialpoint"] is True
    assert payload["fbins"] == 2.0
    assert payload["restart_from"] == "all.restart"
    assert payload["restart_to"] == "all.out.restart"
    assert payload["relax_steps"] == 5
    assert payload["relax_step"] == 0.25


def test_structure_to_dict_includes_restart():
    s = Structure(
        "water.pdb",
        count=3,
        restart_from="water.in.restart",
        restart_to="water.out.restart",
    )
    payload = s.to_dict()
    assert payload["restart_from"] == "water.in.restart"
    assert payload["restart_to"] == "water.out.restart"


def test_export_pdb_writes_conect_and_ter(tmp_path):
    assert HAS_PACK_EXPORT, "pack_write_output binding unavailable"
    out = tmp_path / "out.pdb"
    export(DummyResult(), "pdb", str(out))
    text = out.read_text(encoding="utf-8")
    assert "ATOM" in text
    assert "HETATM" in text
    assert "TER" in text
    assert "CONECT" in text
    assert text.strip().endswith("END")


def test_export_pdb_write_conect_flag(tmp_path):
    assert HAS_PACK_EXPORT, "pack_write_output binding unavailable"
    out = tmp_path / "out_noconect.pdb"
    export(DummyResult(), "pdb", str(out), write_conect=False)
    text = out.read_text(encoding="utf-8")
    assert "CONECT" not in text


def test_export_pdb_add_box_sides(tmp_path):
    assert HAS_PACK_EXPORT, "pack_write_output binding unavailable"
    out = tmp_path / "out_box.pdb"
    export(DummyResult(), "pdb", str(out), add_box_sides=True, box_sides_fix=2.0)
    text = out.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0].startswith("CRYST1")


def test_export_crd_writes_lines(tmp_path):
    assert HAS_PACK_EXPORT, "pack_write_output binding unavailable"
    out = tmp_path / "out.crd"
    export(DummyResult(), "crd", str(out))
    text = out.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0].startswith("* TITLE")
    assert lines[3].strip().endswith("EXT")


def test_export_accepts_none_optional_fields(tmp_path):
    assert HAS_PACK_EXPORT, "pack_write_output binding unavailable"
    out = tmp_path / "out_none_optional.pdb"
    result = PackResult(
        coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.9572, 0.0], [0.9266, -0.2396, 0.0]], dtype=np.float32),
        box=(12.0, 12.0, 12.0),
        name=["O", "H1", "H2"],
        element=["O", "H", "H"],
        resname=["SOL", "SOL", "SOL"],
        resid=[1, 1, 1],
        chain=["A", "A", "A"],
        charge=[-0.834, 0.417, 0.417],
        mol_id=[1, 1, 1],
        segid=["", "", ""],
        bonds=[(0, 1), (0, 2)],
        record_kind=["ATOM", "ATOM", "ATOM"],
        ter_after=None,
    )
    export(result, "pdb", str(out))
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "ATOM" in text


def test_water_pdb_paths_exist():
    for model in ["spce", "tip3p", "tip4pew", "tip5p", "tip4p-ew", "spc/e"]:
        path = Path(water_pdb(model))
        assert path.exists()
