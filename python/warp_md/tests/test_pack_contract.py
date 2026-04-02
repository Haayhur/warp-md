from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from warp_md import pack_contract

try:
    from warp_md import traj_py  # type: ignore
except Exception:  # pragma: no cover - import gate
    traj_py = None


HAS_NATIVE_PACK_AGENT = bool(traj_py) and hasattr(traj_py, "pack_agent_run")

ROOT = Path(__file__).resolve().parents[3]
PYTHON_SRC = str(ROOT / "python")

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_PACK_AGENT,
    reason="native pack agent bindings unavailable",
)


def _run_cli(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = PYTHON_SRC + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "warp_md.pack_cli", *args],
        capture_output=True,
        text=True,
        input=stdin,
        env=env,
    )


def _write_solute(path: Path) -> None:
    path.write_text(
        "ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  O   MOL A   1       1.500   0.000   0.000  1.00  0.00           O\n"
        "END\n",
        encoding="utf-8",
    )


def _write_training_oligomer(path: Path) -> None:
    path.write_text(
        "ATOM      1  C1  HDA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C2  RPT A   2       3.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C3  TLA A   3       6.000   0.000   0.000  1.00  0.00           C\n"
        "END\n",
        encoding="utf-8",
    )


def _write_prmtop(path: Path, total_charge_e: float) -> None:
    amber_charge = total_charge_e * 18.2223
    path.write_text(
        "%FLAG ATOM_NAME\n"
        "%FORMAT(20a4)\n"
        "C O\n"
        "%FLAG RESIDUE_LABEL\n"
        "%FORMAT(20a4)\n"
        "MOL\n"
        "%FLAG RESIDUE_POINTER\n"
        "%FORMAT(10I8)\n"
        "1\n"
        "%FLAG ATOMIC_NUMBER\n"
        "%FORMAT(10I8)\n"
        "6 8\n"
        "%FLAG CHARGE\n"
        "%FORMAT(5E16.8)\n"
        f"{amber_charge:16.8E} {0.0:16.8E}\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_pack_schema_request() -> None:
    result = _run_cli("schema")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["properties"]["schema_version"]["default"] == "warp-pack.agent.v1"


def test_pack_capabilities_json() -> None:
    payload = pack_contract.pack_capabilities()
    assert payload["schema_version"] == "warp-pack.agent.v1"
    assert payload["streaming_supported"] is True
    assert payload["polymer_build_supported"] is True
    assert payload["preferred_solute_input"] == "components"
    assert payload["supported_solute_inputs"] == ["components", "solute", "polymer_build"]
    assert payload["supported_morphology_modes"] == [
        "single_chain_solution",
        "amorphous_bulk",
        "backbone_aligned_bulk",
    ]
    assert payload["supported_charge_sources"] == ["charge_manifest", "prmtop"]
    assert "pdb-strict" in payload["supported_output_formats"]
    assert payload["supported_output_controls"] == [
        "format",
        "write_conect",
        "preserve_topology_graph",
        "md_package",
    ]
    assert "tip3p" in payload["supported_solvent_models"]
    assert "Ca2+" in payload["supported_ion_species"]
    assert "nacl" in payload["supported_salt_names"]
    assert "cacl2" in payload["supported_salt_names"]
    assert "salt.name" in payload["supported_salt_inputs"]
    assert "salt.formula" in payload["supported_salt_inputs"]
    assert "neutralize.with" in payload["supported_ion_controls"]


def test_pack_example_polymer_build_handoff() -> None:
    payload = pack_contract.example_request("polymer_build_handoff")
    assert payload["polymer_build"]["build_manifest"] == "outputs/pmma_50mer.build.json"
    assert payload["environment"]["solvent"]["model"] == "tip3p"
    assert payload["environment"]["ions"]["salt"]["name"] == "nacl"
    assert payload["environment"]["ions"]["neutralize"]["enabled"] is True
    assert payload["outputs"]["format"] == "pdb-strict"
    assert payload["outputs"]["write_conect"] is True
    assert payload["outputs"]["preserve_topology_graph"] is True
    assert payload["outputs"]["md_package"].endswith(".md-ready.json")


def test_pack_example_components_bulk() -> None:
    payload = pack_contract.example_request("components_amorphous_bulk")
    assert payload["components"][0]["source"]["kind"] == "polymer_build"
    assert payload["environment"]["morphology"]["mode"] == "amorphous_bulk"


def test_pack_validate_rejects_removed_inline_polymer(tmp_path: Path) -> None:
    training = tmp_path / "training.pdb"
    _write_training_oligomer(training)

    payload = {
        "version": "warp-pack.agent.v1",
        "polymer": {
            "param_source": {
                "mode": "oligomer_training",
                "artifact": str(training),
                "nmer": 3,
            },
            "target": {
                "mode": "linear_homopolymer",
                "n_repeat": 4,
                "sequence": "AAAA",
                "conformation": {"mode": "extended"},
            },
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 30.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": True, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": str(tmp_path / "system.pdb"),
            "manifest": str(tmp_path / "system_manifest.json"),
        },
    }

    result = pack_contract.validate_request_payload(payload)
    assert result["valid"] is False
    assert result["errors"][0]["code"] == "E_UNSUPPORTED_FEATURE"
    assert result["errors"][0]["path"] == "/polymer"


def test_pack_validate_rejects_non_uniform_homopolymer_sequence(tmp_path: Path) -> None:
    training = tmp_path / "training.pdb"
    _write_training_oligomer(training)

    payload = {
        "version": "warp-pack.agent.v1",
        "polymer": {
            "param_source": {
                "mode": "oligomer_training",
                "artifact": str(training),
                "nmer": 3,
            },
            "target": {
                "mode": "linear_homopolymer",
                "n_repeat": 4,
                "sequence": "ABBA",
                "conformation": {"mode": "extended"},
            },
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 30.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": False, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": str(tmp_path / "system.pdb"),
            "manifest": str(tmp_path / "system_manifest.json"),
        },
    }

    result = pack_contract.validate_request_payload(payload)
    assert result["valid"] is False
    assert result["errors"][0]["path"] == "/polymer"


def test_run_build_request_solute_writes_manifest(tmp_path: Path) -> None:
    solute = tmp_path / "solute.pdb"
    charge_manifest = tmp_path / "solute_charge.json"
    coords = tmp_path / "out" / "system.pdb"
    manifest = tmp_path / "out" / "system_manifest.json"
    md_package = tmp_path / "out" / "system_manifest.md-ready.json"
    _write_solute(solute)
    _write_json(
        charge_manifest,
        {
            "version": "warp-pack.charge-manifest.v1",
            "net_charge_e": -2.0,
        },
    )

    payload = {
        "version": "warp-pack.agent.v1",
        "run_id": "solute-solvate-001",
        "solute": {
            "path": str(solute),
            "kind": "small_molecule",
            "charge_manifest": str(charge_manifest),
        },
        "environment": {
            "box": {"mode": "padding", "padding_angstrom": 8.0, "shape": "cubic"},
            "solvent": {"mode": "explicit", "model": "tip3p"},
            "ions": {"neutralize": {"enabled": True}, "salt": {"name": "nacl", "molar": 0.05}},
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": str(coords),
            "manifest": str(manifest),
            "md_package": str(md_package),
            "format": "pdb-strict",
            "write_conect": True,
            "preserve_topology_graph": True,
        },
    }

    exit_code, envelope = pack_contract.run_build_request(payload)
    assert exit_code == 0
    assert envelope["status"] == "ok"
    assert coords.exists()
    assert manifest.exists()
    assert md_package.exists()

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["neutralization_policy_applied"] == "charge_manifest.net_charge_e"
    assert manifest_payload["net_charge_before_neutralization"] == -2.0
    assert manifest_payload["target_salt_name"] == "nacl"
    assert manifest_payload["charge_source_kinds"] == ["net_charge_e"]
    assert manifest_payload["water_count"] > 0
    assert manifest_payload["final_box_vectors_angstrom"][0][0] > 0.0
    assert manifest_payload["output_metadata"]["coordinates"]["format"] == "pdb-strict"
    assert manifest_payload["output_metadata"]["coordinates"]["write_conect"] is True
    assert manifest_payload["output_metadata"]["md_package"]["path"] == str(md_package)

    md_package_payload = json.loads(md_package.read_text(encoding="utf-8"))
    assert md_package_payload["coordinates"]["format"] == "pdb-strict"
    assert md_package_payload["coordinates"]["write_conect"] is True
    assert md_package_payload["pack_manifest"]["path"] == str(manifest)


def test_run_components_bulk_with_prmtop_fallback(tmp_path: Path) -> None:
    solute = tmp_path / "solute.pdb"
    prmtop = tmp_path / "solute.prmtop"
    coords = tmp_path / "components_bulk.pdb"
    manifest = tmp_path / "components_bulk_manifest.json"
    _write_solute(solute)
    _write_prmtop(prmtop, 1.0)

    payload = {
        "version": "warp-pack.agent.v1",
        "run_id": "components-bulk-001",
        "components": [
            {
                "name": "chain_a",
                "count": 2,
                "source": {
                    "kind": "artifact",
                    "artifact": {
                        "path": str(solute),
                        "kind": "polymer_chain",
                        "topology": str(prmtop),
                    },
                },
            }
        ],
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": [40.0, 40.0, 40.0], "shape": "orthorhombic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": True, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "amorphous_bulk"},
        },
        "outputs": {
            "coordinates": str(coords),
            "manifest": str(manifest),
        },
    }

    exit_code, envelope = pack_contract.run_build_request(payload)
    assert exit_code == 0
    assert envelope["status"] == "ok"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["morphology_mode"] == "amorphous_bulk"
    assert manifest_payload["charge_source_kinds"] == ["prmtop.total_charge"]
    assert manifest_payload["net_charge_before_neutralization"] == 2.0
    assert manifest_payload["achieved_salt_counts_by_species"]["Cl-"] == 2


def test_run_build_request_supports_bundled_cacl2_by_name(tmp_path: Path) -> None:
    solute = tmp_path / "solute.pdb"
    coords = tmp_path / "system.pdb"
    manifest = tmp_path / "system_manifest.json"
    _write_solute(solute)

    payload = {
        "version": "warp-pack.agent.v1",
        "run_id": "solute-cacl2-001",
        "solute": {
            "path": str(solute),
            "kind": "small_molecule",
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 50.0, "shape": "cubic"},
            "solvent": {"mode": "explicit", "model": "tip3p"},
            "ions": {"neutralize": False, "salt": {"name": "cacl2", "molar": 0.15}},
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": str(coords),
            "manifest": str(manifest),
        },
    }

    exit_code, envelope = pack_contract.run_build_request(payload)
    assert exit_code == 0
    assert envelope["status"] == "ok"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["ion_counts"]["Ca2+"] == 11
    assert manifest_payload["ion_counts"]["Cl-"] == 22
    assert manifest_payload["target_salt_name"] == "cacl2"
    assert manifest_payload["target_salt_formula"] == "CaCl2"


def test_run_build_request_supports_explicit_salt_species_map(tmp_path: Path) -> None:
    solute = tmp_path / "solute.pdb"
    coords = tmp_path / "system.pdb"
    manifest = tmp_path / "system_manifest.json"
    _write_solute(solute)

    payload = {
        "version": "warp-pack.agent.v1",
        "run_id": "solute-species-salt-001",
        "solute": {
            "path": str(solute),
            "kind": "small_molecule",
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 50.0, "shape": "cubic"},
            "solvent": {"mode": "explicit", "model": "tip3p"},
            "ions": {
                "neutralize": False,
                "salt": {"species": {"Ca2+": 1, "Cl-": 2}, "molar": 0.15},
            },
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": str(coords),
            "manifest": str(manifest),
        },
    }

    exit_code, envelope = pack_contract.run_build_request(payload)
    assert exit_code == 0
    assert envelope["status"] == "ok"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["target_salt_formula"] == "CaCl2"
    assert manifest_payload["ion_counts"]["Ca2+"] == 11
    assert manifest_payload["ion_counts"]["Cl-"] == 22


def test_run_build_request_neutralize_with_prefers_requested_counterion(tmp_path: Path) -> None:
    solute = tmp_path / "solute.pdb"
    charge_manifest = tmp_path / "solute_charge.json"
    coords = tmp_path / "system.pdb"
    manifest = tmp_path / "system_manifest.json"
    _write_solute(solute)
    _write_json(
        charge_manifest,
        {
            "version": "warp-pack.charge-manifest.v1",
            "net_charge_e": 2.0,
        },
    )

    payload = {
        "version": "warp-pack.agent.v1",
        "run_id": "neutralize-with-001",
        "solute": {
            "path": str(solute),
            "kind": "small_molecule",
            "charge_manifest": str(charge_manifest),
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 30.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": {"enabled": True, "with": "Cl-"}},
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": str(coords),
            "manifest": str(manifest),
        },
    }

    exit_code, envelope = pack_contract.run_build_request(payload)
    assert exit_code == 0
    assert envelope["status"] == "ok"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["ion_counts"]["Cl-"] == 2


def test_run_build_request_strict_pdb_falls_back_to_mmcif(tmp_path: Path) -> None:
    solute = tmp_path / "solute_long_resname.cif"
    coords = tmp_path / "system.pdb"
    manifest = tmp_path / "system_manifest.json"
    solute.write_text(
        "data_test\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "ATOM 1 C C1 LONGRES A 1 0.0 0.0 0.0\n",
        encoding="utf-8",
    )

    payload = {
        "version": "warp-pack.agent.v1",
        "run_id": "strict-fallback-001",
        "solute": {
            "path": str(solute),
            "kind": "small_molecule",
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 30.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": False, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": str(coords),
            "manifest": str(manifest),
            "format": "pdb-strict",
        },
    }

    exit_code, envelope = pack_contract.run_build_request(payload)
    assert exit_code == 0
    actual_coords = Path(envelope["artifacts"]["coordinates"])
    assert actual_coords.suffix == ".cif"
    assert actual_coords.exists()
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["output_metadata"]["coordinates"]["path"] == str(actual_coords)
    assert manifest_payload["output_metadata"]["coordinates"]["format"] == "mmcif"


def test_validate_accepts_fixed_box_density_with_explicit_solvent(tmp_path: Path) -> None:
    solute = tmp_path / "solute.pdb"
    _write_solute(solute)
    payload = {
        "version": "warp-pack.agent.v1",
        "components": [
            {
                "name": "solute",
                "count": 1,
                "source": {
                    "kind": "artifact",
                    "artifact": {
                        "path": str(solute),
                        "kind": "small_molecule",
                    },
                },
            }
        ],
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": [40.0, 40.0, 40.0], "shape": "orthorhombic"},
            "solvent": {"mode": "explicit", "model": "tip3p"},
            "ions": {"neutralize": False, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "amorphous_bulk", "target_density_g_cm3": 0.9},
        },
        "outputs": {
            "coordinates": str(tmp_path / "bulk.pdb"),
            "manifest": str(tmp_path / "bulk_manifest.json"),
        },
    }

    result = pack_contract.validate_request_payload(payload)
    assert result["valid"] is True


def test_cli_run_streams_ndjson_success(tmp_path: Path) -> None:
    solute = tmp_path / "solute.pdb"
    charge_manifest = tmp_path / "solute_charge.json"
    request = tmp_path / "request.json"
    coords = tmp_path / "system.pdb"
    manifest = tmp_path / "system_manifest.json"
    _write_solute(solute)
    _write_json(
        charge_manifest,
        {
            "version": "warp-pack.charge-manifest.v1",
            "net_charge_e": -1.0,
        },
    )
    _write_json(
        request,
        {
            "version": "warp-pack.agent.v1",
            "run_id": "stream-ok",
            "solute": {
                "path": str(solute),
                "kind": "complex",
                "charge_manifest": str(charge_manifest),
            },
            "environment": {
                "box": {"mode": "padding", "padding_angstrom": 8.0, "shape": "cubic"},
                "solvent": {"mode": "explicit", "model": "tip3p"},
                "ions": {"neutralize": True, "salt_molar": 0.05, "cation": "Na+", "anion": "Cl-"},
                "morphology": {"mode": "single_chain_solution"},
            },
            "outputs": {
                "coordinates": str(coords),
                "manifest": str(manifest),
            },
        },
    )

    result = _run_cli("run", str(request), "--stream", "ndjson")
    assert result.returncode == 0
    events = [json.loads(line) for line in result.stderr.splitlines() if line.strip()]
    assert events[0]["event"] == "run_started"
    progress_phases = [event["phase"] for event in events if event["event"] == "phase_progress"]
    assert "world_build" in progress_phases
    assert "solvation" in progress_phases
    assert "ionization" in progress_phases
    assert "packing" in progress_phases
    assert "manifest" in progress_phases
