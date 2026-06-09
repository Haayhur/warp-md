from __future__ import annotations

from pathlib import Path

import pytest

from warp_md import cg_cli
from warp_md import cg_contract


def test_cg_cli_validate_returns_zero_for_valid_payload(monkeypatch, capsys, tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text('{"schema_version":"warp-cg.agent.v2","name":"benzene","smiles":"c1ccccc1"}', encoding="utf-8")

    def fake_validate(_payload: dict) -> dict:
        return {"valid": True, "status": "ok"}

    monkeypatch.setattr(cg_cli, "validate_request_payload", fake_validate)

    exit_code = cg_cli.run_cli(["validate", str(request)])

    assert exit_code == 0
    assert '"valid": true' in capsys.readouterr().out.lower()


def test_cg_cli_validate_returns_non_zero_for_invalid_payload(monkeypatch, capsys, tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text('{"schema_version":"warp-cg.agent.v2","name":"","smiles":""}', encoding="utf-8")

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
        "schema_version": "warp-cg.agent.v2",
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
