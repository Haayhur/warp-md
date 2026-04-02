import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from warp_md.pack import (
    available_ion_species,
    available_salt_names,
    ion_pdb,
    salt_recipe,
    water_pdb,
)
from warp_md.pack.config import Box, OutputSpec, PackConfig, Structure
from warp_md.pack.export import export
from warp_md.pack import data as pack_data
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
    written = export(DummyResult(), "pdb", str(out))
    assert written["path"] == str(out)
    assert written["fallback_applied"] is False
    text = out.read_text(encoding="utf-8")
    assert "CRYST1" in text
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


def test_export_pdb_add_box_sides_accepts_none_fix(tmp_path):
    assert HAS_PACK_EXPORT, "pack_write_output binding unavailable"
    out = tmp_path / "out_box_none_fix.pdb"
    export(DummyResult(), "pdb", str(out), add_box_sides=True, box_sides_fix=None)
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


def test_export_falls_back_to_mmcif_when_strict_pdb_overflows(tmp_path):
    assert HAS_PACK_EXPORT, "pack_write_output binding unavailable"
    out = tmp_path / "strict_overflow.pdb"
    result = DummyResult()
    result.resname[0] = "LONGRES"
    written = export(result, "pdb-strict", str(out))
    assert written["fallback_applied"] is True
    assert written["format"] == "mmcif"
    fallback = Path(written["path"])
    assert fallback.exists()
    assert fallback.suffix == ".cif"
    assert "data_warp_pack" in fallback.read_text(encoding="utf-8")


def test_water_pdb_paths_exist():
    for model in ["spce", "tip3p", "tip4pew", "tip5p", "tip4p-ew", "spc/e"]:
        path = Path(water_pdb(model))
        assert path.exists()


def test_ion_pdb_paths_exist():
    for species in ["na+", "cl-", "k+", "ca2+"]:
        path = Path(ion_pdb(species))
        assert path.exists()


def test_available_ion_species_returns_canonical_names():
    species = available_ion_species()
    assert "Na+" in species
    assert "Cl-" in species
    assert "K+" in species
    assert "Ca2+" in species


def test_available_salt_names_and_recipe():
    names = available_salt_names()
    assert "nacl" in names
    assert "cacl2" in names
    cacl2 = salt_recipe("calcium chloride")
    assert cacl2["formula"] == "CaCl2"
    assert cacl2["species"] == {"Ca2+": 1, "Cl-": 2}


def test_ion_registry_overlay_supports_external_templates(tmp_path, monkeypatch):
    mg = tmp_path / "mg.pdb"
    br = tmp_path / "br.pdb"
    registry = tmp_path / "ions_overlay.json"
    mg.write_text(
        "HETATM    1 MG   MG2 A   1       0.000   0.000   0.000  1.00  0.00          MG\nEND\n",
        encoding="utf-8",
    )
    br.write_text(
        "HETATM    1 BR   BR- A   1       0.000   0.000   0.000  1.00  0.00          BR\nEND\n",
        encoding="utf-8",
    )
    registry.write_text(
        (
            "{\n"
            '  "ions": [\n'
            '    {"species": "Mg2+", "aliases": ["mg2+"], "template": "mg.pdb", "formula_symbol": "Mg", "charge_e": 2, "mass_amu": 24.305},\n'
            '    {"species": "Br-", "aliases": ["br-"], "template": "br.pdb", "formula_symbol": "Br", "charge_e": -1, "mass_amu": 79.904}\n'
            "  ]\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WARP_MD_ION_REGISTRY", str(registry))
    pack_data._ion_registry.cache_clear()
    try:
        assert Path(ion_pdb("mg2+")) == mg.resolve()
        assert Path(ion_pdb("Br-")) == br.resolve()
        species = available_ion_species()
        assert "Mg2+" in species
        assert "Br-" in species
        assert "Na+" in species
    finally:
        pack_data._ion_registry.cache_clear()


def test_salt_registry_overlay_supports_custom_names(tmp_path, monkeypatch):
    registry = tmp_path / "salts_overlay.json"
    registry.write_text(
        (
            "{\n"
            '  "salts": [\n'
            '    {"name": "mgcl2", "aliases": ["MgCl2", "magnesium chloride"], "formula": "MgCl2", "species": {"Mg2+": 1, "Cl-": 2}}\n'
            "  ]\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WARP_MD_SALT_REGISTRY", str(registry))
    pack_data._salt_registry.cache_clear()
    try:
        names = available_salt_names()
        assert "mgcl2" in names
        assert salt_recipe("magnesium chloride")["species"] == {"Mg2+": 1, "Cl-": 2}
        assert "nacl" in names
    finally:
        pack_data._salt_registry.cache_clear()
