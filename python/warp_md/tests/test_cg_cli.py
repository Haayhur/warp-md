from __future__ import annotations

import json
from pathlib import Path

import pytest

from warp_md import cg_cli
from warp_md import cg_contract
from warp_md import cg_martini_openmm_evaluator


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


def test_cg_cli_forcefield_inspect(monkeypatch, capsys) -> None:
    def fake_inspect(kind: str = "martini3") -> dict:
        return {
            "schema_version": "warp-cg.forcefield-manifest.v1",
            "kind": kind,
            "files": [{"path": "martini_v3.0.0.itp"}],
        }

    monkeypatch.setattr(cg_cli, "cg_forcefield_inspect", fake_inspect)

    exit_code = cg_cli.run_cli(["forcefield", "inspect", "--kind", "martini3"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"kind": "martini3"' in output
    assert "martini_v3.0.0.itp" in output


def test_cg_contract_native_forcefield_install_writes_bundled_files(tmp_path: Path) -> None:
    if cg_contract.traj_py is None:
        pytest.skip("native warp-md bindings unavailable")

    manifest = cg_contract.cg_forcefield_inspect("martini3")
    assert manifest["kind"] == "martini3"
    assert {entry["path"] for entry in manifest["files"]} >= {
        "LICENSE",
        "NOTICE.md",
        "martini_v3.0.0.itp",
    }

    dest = tmp_path / "forcefields" / "martini3"
    installed = cg_contract.cg_forcefield_install(str(dest), kind="martini3", overwrite=True)

    assert installed["kind"] == "martini3"
    assert (dest / "martini_v3.0.0.itp").is_file()
    assert (dest / "NOTICE.md").is_file()
    assert (dest / "warp_cg_forcefield_manifest.json").is_file()


def test_cg_cli_martini_openmm_runner_dry_run(capsys, tmp_path: Path) -> None:
    exit_code = cg_cli.run_cli(
        [
            "runner",
            "martini-openmm",
            "--gro",
            "system.gro",
            "--top",
            "system.top",
            "--outdir",
            str(tmp_path / "run"),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert Path(payload["trajectory"]).exists()


def test_martini_openmm_evaluator_applies_template_replacements(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "system.gro").write_text("dummy gro\n", encoding="utf-8")
    (template / "system.top").write_text('#include "molecule.itp"\n', encoding="utf-8")
    (template / "molecule.itp").write_text("bond {{bond.group_1_length_nm}}\n", encoding="utf-8")

    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": "warp-cg.martini-openmm-runner.v1",
                "base_dir": str(tmp_path),
                "kind": "martini_openmm",
                "template_dir": str(template),
                "gro": "system.gro",
                "top": "system.top",
                "replacements": [
                    {
                        "path": "molecule.itp",
                        "parameter": "bond.group_1_length_nm",
                        "format": ".3f",
                    }
                ],
                "protocol": {
                    "dry_run": True,
                    "prefix": "eq_npt",
                    "trajectory_format": "xtc",
                },
            }
        ),
        encoding="utf-8",
    )
    evaluation_dir = tmp_path / "evaluation_000000"
    evaluation_dir.mkdir()
    candidate = evaluation_dir / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "schema_version": "warp-cg.objective-request.v1",
                "candidate": {
                    "named_parameters": [
                        {
                            "name": "bond.group_1_length_nm",
                            "value": 0.42,
                            "normalized_value": 0.5,
                            "min": 0.1,
                            "max": 1.0,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    result = evaluation_dir / "result.json"

    exit_code = cg_martini_openmm_evaluator.evaluate(spec, candidate, result)

    assert exit_code == 0
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["metrics"]["runner.replacements"] == 1.0
    assert "candidate_trajectory" in payload
    assert "0.420" in (evaluation_dir / "molecule.itp").read_text(encoding="utf-8")


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
