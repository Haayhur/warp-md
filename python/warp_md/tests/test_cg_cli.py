from __future__ import annotations

import json
from pathlib import Path

import pytest

from warp_md import cg_cli
from warp_md import cg_contract


def test_cg_cli_validate_returns_zero_for_valid_payload(monkeypatch, capsys, tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text('{"schema_version":"warp-cg.agent.v1","name":"benzene","smiles":"c1ccccc1"}', encoding="utf-8")

    def fake_validate(_payload: dict) -> dict:
        return {"valid": True, "status": "ok"}

    monkeypatch.setattr(cg_cli, "validate_request_payload", fake_validate)

    exit_code = cg_cli.run_cli(["validate", str(request)])

    assert exit_code == 0
    assert '"valid": true' in capsys.readouterr().out.lower()


def test_cg_cli_validate_returns_non_zero_for_invalid_payload(monkeypatch, capsys, tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text('{"schema_version":"warp-cg.agent.v1","name":"","smiles":""}', encoding="utf-8")

    def fake_validate(_payload: dict) -> dict:
        return {
            "valid": False,
            "status": "error",
            "error": {"code": "warp_cg.invalid_request", "message": "invalid request"},
        }

    monkeypatch.setattr(cg_cli, "validate_request_payload", fake_validate)

    exit_code = cg_cli.run_cli(["validate", str(request)])

    assert exit_code == 2
    assert '"valid": false' in capsys.readouterr().out.lower()


def test_cg_contract_native_run_writes_mapping_and_itp(tmp_path: Path) -> None:
    if cg_contract.traj_py is None:
        pytest.skip("native warp-md bindings unavailable")

    payload = {
        "schema_version": "warp-cg.agent.v1",
        "name": "benzene",
        "smiles": "c1ccccc1",
        "output": {
            "out_dir": str(tmp_path),
            "write_mapping_json": True,
            "write_topology_itp": True,
            "write_topology_top": True,
        },
    }

    exit_code, result = cg_contract.run_cg_request(payload)

    assert exit_code == 0
    assert result["status"] == "ok"
    assert result["bead_count"] == 3
    artifact_kinds = {artifact["kind"] for artifact in result["artifacts"]}
    assert "martini_mapping_json" in artifact_kinds
    assert "martini_topology_itp" in artifact_kinds
    assert "martini_topology_top" in artifact_kinds


def test_cg_contract_native_capabilities_report_bonded_terms() -> None:
    if cg_contract.traj_py is None:
        pytest.skip("native warp-md bindings unavailable")

    capabilities = cg_contract.cg_capabilities()

    assert capabilities["optimization"]["methods"] == ["bayesian_optimization", "pso"]
    assert capabilities["optimization"]["terms"] == ["bonds", "angles", "dihedrals"]


def test_cg_cli_build_capabilities_exposes_biomolecular_surface(capsys) -> None:
    if cg_contract.traj_py is None:
        pytest.skip("native warp-md bindings unavailable")

    exit_code = cg_cli.run_cli(["build", "capabilities"])

    assert exit_code == 0
    payload = capsys.readouterr().out
    assert '"tool": "warp-cg build"' in payload
    assert "typed leaflet regions" in payload
    assert "coordinate-less inserted solutes" in payload


def test_cg_cli_build_schema_includes_membrane_and_inserted_fields(capsys) -> None:
    if cg_contract.traj_py is None:
        pytest.skip("native warp-md bindings unavailable")

    exit_code = cg_cli.run_cli(["build", "schema"])

    assert exit_code == 0
    payload = capsys.readouterr().out
    assert '"membranes"' in payload
    assert '"proteins"' in payload
    assert '"solutes"' in payload
    assert '"regions"' in payload


def test_cg_cli_build_validate_accepts_native_example(tmp_path: Path) -> None:
    if cg_contract.traj_py is None:
        pytest.skip("native warp-md bindings unavailable")

    request = tmp_path / "build_request.json"
    request.write_text(
        json.dumps(cg_contract.build_example_request()),
        encoding="utf-8",
    )

    exit_code = cg_cli.run_cli(["build", "validate", str(request)])

    assert exit_code == 0


def test_cg_cli_simulate_capabilities_and_schema(capsys) -> None:
    if cg_contract.traj_py is None:
        pytest.skip("native warp-md bindings unavailable")

    assert cg_cli.run_cli(["simulate", "capabilities"]) == 0
    caps = capsys.readouterr().out
    assert '"tool": "warp-cg simulate"' in caps
    assert "planning_manifest_status_only" in caps

    assert cg_cli.run_cli(["simulate", "schema", "--kind", "request"]) == 0
    schema = capsys.readouterr().out
    assert '"SimulateRequest"' in schema
    assert '"protocol"' in schema


def test_cg_cli_simulate_example_validate_and_plan(tmp_path: Path, capsys) -> None:
    if cg_contract.traj_py is None:
        pytest.skip("native warp-md bindings unavailable")

    request = tmp_path / "simulate.json"
    request.write_text(
        json.dumps(cg_contract.simulate_example_request("gromacs")),
        encoding="utf-8",
    )

    assert cg_cli.run_cli(["simulate", "validate", str(request)]) == 0
    assert '"valid": true' in capsys.readouterr().out.lower()

    assert cg_cli.run_cli(["simulate", "plan", str(request)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["engine"] == "gromacs"
    assert plan["commands"][0]["program"] == "gmx"
    assert plan["commands"][0]["args"][0] == "grompp"


def test_cg_cli_simulate_status_detects_checkpoint(tmp_path: Path, capsys) -> None:
    if cg_contract.traj_py is None:
        pytest.skip("native warp-md bindings unavailable")

    (tmp_path / "nvt.cpt").write_text("checkpoint", encoding="utf-8")
    (tmp_path / "nvt.log").write_text("Finished mdrun", encoding="utf-8")

    assert cg_cli.run_cli(["simulate", "status", str(tmp_path)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["restart_capable"] is True
    assert status["last_checkpoint"] == "nvt.cpt"
    assert status["completed_stages"] == ["nvt"]
