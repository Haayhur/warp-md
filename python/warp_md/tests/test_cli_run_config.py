import json
from pathlib import Path

import numpy as np

from warp_md import cli_run, contract


class _DummyPlan:
    def __init__(self, output):
        self._output = output

    def run(self, traj, system, chunk_frames=None, device="auto"):
        return self._output


class _FailingPlan:
    def run(self, traj, system, chunk_frames=None, device="auto"):
        raise RuntimeError("plan runtime failure")


def _write_config(tmp_path: Path, payload: dict) -> Path:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(payload))
    return cfg_path


def _write_topology(path: Path) -> None:
    path.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "END\n",
        encoding="utf-8",
    )


def test_run_config_emits_json_envelope(monkeypatch, tmp_path, capsys) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "output_dir": str(tmp_path / "outputs"),
            "analyses": [{"name": "rg", "selection": "protein"}],
        },
    )

    load_traj_calls: list[int] = []

    monkeypatch.setattr(cli_run, "_load_system", lambda spec: object())
    monkeypatch.setattr(
        cli_run,
        "_load_trajectory",
        lambda spec, system: (load_traj_calls.append(1), object())[1],
    )
    monkeypatch.setattr(
        cli_run,
        "PLAN_BUILDERS",
        {"rg": lambda system, spec: _DummyPlan(np.array([1.0, 2.0]))},
    )

    code = cli_run.main(["run", str(cfg_path)])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 0
    assert envelope["status"] == "ok"
    assert envelope["exit_code"] == 0
    assert envelope["analysis_count"] == 1
    assert envelope["results"][0]["analysis"] == "rg"
    assert envelope["results"][0]["status"] == "ok"
    assert envelope["results"][0]["kind"] == "array"
    assert envelope["results"][0]["shape"] == [2]
    assert envelope["results"][0]["artifact"]["format"] == "npz"
    assert (
        envelope["results"][0]["artifact"]["description"]
        == contract.get_plan_schema("rg")["outputs"][0]["description"]
    )
    assert envelope["results"][0]["artifact"]["bytes"] > 0
    assert len(envelope["results"][0]["artifact"]["sha256"]) == 64
    plot_rec = envelope["results"][0]["artifact"]["plot_recommendations"][0]
    assert plot_rec["artifact"] == envelope["results"][0]["out"]
    assert plot_rec["plot_type"] == "line"
    assert plot_rec["x"]["field"] == "index"
    assert plot_rec["y"]["field"] == "rg_nm"
    assert plot_rec["y"]["units"] == "nm"
    companions = envelope["results"][0]["artifact"]["companions"]
    assert any(item["role"] == "npz_companion_manifest" for item in companions)
    csv_companion = next(item for item in companions if item["format"] == "csv")
    assert csv_companion["source_key"] == "rg_nm"
    assert Path(csv_companion["path"]).exists()
    assert Path(csv_companion["path"]).read_text().splitlines()[0] == "rg_nm"
    assert Path(envelope["results"][0]["out"]).exists()
    assert len(load_traj_calls) == 1


def test_run_config_skips_csv_companion_for_string_arrays(
    monkeypatch, tmp_path, capsys
) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "output_dir": str(tmp_path / "outputs"),
            "analyses": [{"name": "dssp"}],
        },
    )

    monkeypatch.setattr(cli_run, "_load_system", lambda spec: object())
    monkeypatch.setattr(cli_run, "_load_trajectory", lambda spec, system: object())
    monkeypatch.setattr(
        cli_run,
        "PLAN_BUILDERS",
        {"dssp": lambda system, spec: _DummyPlan(np.array(["ALA", "GLY"]))},
    )

    code = cli_run.main(["run", str(cfg_path)])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 0
    assert envelope["status"] == "ok"
    companions = envelope["results"][0]["artifact"]["companions"]
    assert [item["format"] for item in companions] == ["json"]

    manifest_path = Path(companions[0]["path"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["arrays"][0]["dtype"].startswith("<U")
    assert manifest["arrays"][0]["csv_skipped"] == "non_numeric_dtype"
    assert not (manifest_path.parent / "structure.csv").exists()

    with np.load(envelope["results"][0]["out"]) as data:
        assert data["structure"].tolist() == ["ALA", "GLY"]


class _NativePlotRenderer:
    def warp_md_agent_render_plots(self, payload_json, out_dir):
        out_path = Path(out_dir) / "rg_0_0.svg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("<svg></svg>\n")
        return {
            "status": "ok",
            "plot_count": 1,
            "artifacts": [{"path": str(out_path), "format": "svg"}],
            "skipped": [],
        }


def test_plot_command_renders_svg_from_result_envelope(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli_run.contract, "_native", lambda: _NativePlotRenderer())
    result_path = tmp_path / "warp_md_result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "results": [
                    {
                        "analysis": "rg",
                        "out": str(tmp_path / "rg.npz"),
                        "status": "ok",
                        "artifact": {
                            "path": str(tmp_path / "rg.npz"),
                            "format": "npz",
                            "plot_recommendations": [
                                {
                                    "plot_type": "line",
                                    "x": {
                                        "field": "index",
                                        "units": "frame",
                                        "source": "implicit_index",
                                    },
                                    "y": {"field": "rg_nm", "units": "nm"},
                                    "title": "Radius of gyration",
                                }
                            ],
                            "companions": [],
                        },
                    }
                ],
            }
        )
    )
    out_dir = tmp_path / "plots"

    code = cli_run.main(["plot", str(result_path), "--out-dir", str(out_dir)])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 0
    assert envelope["status"] == "ok"
    assert envelope["plot_count"] == 1
    svg_path = Path(envelope["artifacts"][0]["path"])
    assert svg_path.exists()
    assert svg_path.read_text().startswith("<svg")


def test_schema_command_supports_plot_manifest(capsys) -> None:
    code = cli_run.main(["schema", "--kind", "plot-manifest"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    assert payload["title"] == "PlotManifest"


def test_bundle_plan_command_outputs_expanded_request(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(contract, "_native", lambda: None)
    top = tmp_path / "topology.pdb"
    _write_topology(top)
    traj = tmp_path / "traj.xtc"
    traj.write_bytes(b"placeholder")
    energy = tmp_path / "energy.csv"
    energy.write_text("time,potential_energy\n0,-1.0\n", encoding="utf-8")
    cfg_path = _write_config(
        tmp_path,
        {
            "system": str(top),
            "trajectory": str(traj),
            "inputs": {"energy_table": {"path": str(energy)}},
            "analyses": [{"name": "rg", "selection": "all"}],
        },
    )

    code = cli_run.main(["bundle-plan", "standard_md_report", str(cfg_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    assert payload["bundle"] == "standard_md_report"
    assert any(item["name"] == "rg" for item in payload["analyses"])
    assert "request" in payload


def test_inspect_inputs_reports_external_table_and_missing_traj(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(contract, "_native", lambda: None)
    top = tmp_path / "topology.pdb"
    _write_topology(top)
    state = tmp_path / "state.xvg"
    state.write_text("# state\n@ xaxis label \"time\"\n0 300 0.99\n", encoding="utf-8")
    cfg_path = _write_config(
        tmp_path,
        {
            "system": str(top),
            "trajectory": str(tmp_path / "missing.xtc"),
            "inputs": {"state_table": {"path": str(state), "format": "xvg"}},
            "analyses": [{"name": "rg", "selection": "all"}],
        },
    )

    code = cli_run.main(["inspect-inputs", str(cfg_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 2
    assert payload["external_tables"]["state_table"]["columns"] == ["column_1", "column_2", "column_3"]
    assert any(error["code"] == "E_INPUT_MISSING" for error in payload["errors"])


def test_run_config_dry_run_skips_execution(monkeypatch, tmp_path, capsys) -> None:
    output_dir = tmp_path / "dry_outputs"
    cfg_path = _write_config(
        tmp_path,
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "output_dir": str(output_dir),
            "analyses": [{"name": "rg", "selection": "protein"}],
        },
    )

    monkeypatch.setattr(cli_run, "_load_system", lambda spec: object())

    def _unexpected_traj(*args, **kwargs):
        raise AssertionError("trajectory load should not run in dry-run mode")

    monkeypatch.setattr(cli_run, "_load_trajectory", _unexpected_traj)
    monkeypatch.setattr(
        cli_run,
        "PLAN_BUILDERS",
        {"rg": lambda system, spec: _DummyPlan(np.array([1.0]))},
    )

    code = cli_run.main(["run", str(cfg_path), "--dry-run"])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 0
    assert envelope["status"] == "dry_run"
    assert envelope["exit_code"] == 0
    assert envelope["analysis_count"] == 1
    assert envelope["results"][0]["analysis"] == "rg"
    assert envelope["results"][0]["status"] == "dry_run"
    assert not Path(envelope["results"][0]["out"]).exists()
    assert not output_dir.exists()


def test_run_config_accepts_topology_traj_aliases(monkeypatch, tmp_path, capsys) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "topology": "topology.pdb",
            "traj": "traj.xtc",
            "analyses": [{"name": "chain-rg", "selection": "protein"}],
        },
    )

    monkeypatch.setattr(cli_run, "_load_system", lambda spec: object())
    monkeypatch.setattr(
        cli_run,
        "PLAN_BUILDERS",
        {"chain_rg": lambda system, spec: _DummyPlan(np.array([0.1]))},
    )

    code = cli_run.main(["run", str(cfg_path), "--dry-run"])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 0
    assert envelope["status"] == "dry_run"
    assert envelope["exit_code"] == 0
    assert envelope["results"][0]["analysis"] == "chain_rg"


def test_run_config_docking_defaults_to_json_artifact(monkeypatch, tmp_path, capsys) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "output_dir": str(tmp_path / "outputs"),
            "analyses": [
                {
                    "name": "docking",
                    "receptor_mask": "protein and not resname LIG",
                    "ligand_mask": "resname LIG",
                }
            ],
        },
    )

    monkeypatch.setattr(cli_run, "_load_system", lambda spec: object())

    code = cli_run.main(["run", str(cfg_path), "--dry-run"])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 0
    assert envelope["status"] == "dry_run"
    assert envelope["exit_code"] == 0
    assert envelope["results"][0]["analysis"] == "docking"
    assert envelope["results"][0]["out"].endswith(".json")


def test_run_config_preserves_requested_analysis_count_on_partial_success(monkeypatch, tmp_path, capsys) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "output_dir": str(tmp_path / "outputs"),
            "fail_fast": False,
            "analyses": [
                {"name": "rg", "selection": "protein"},
                {"name": "rmsd", "selection": "backbone"},
            ],
        },
    )

    monkeypatch.setattr(cli_run, "_load_system", lambda spec: object())
    monkeypatch.setattr(cli_run, "_load_trajectory", lambda spec, system: object())

    def _builder(_system, spec):
        if spec["name"] == "rg":
            return _FailingPlan()
        return _DummyPlan(np.array([2.0]))

    monkeypatch.setattr(cli_run, "PLAN_BUILDERS", {"rg": _builder, "rmsd": _builder})

    code = cli_run.main(["run", str(cfg_path)])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 0
    assert envelope["status"] == "ok"
    assert envelope["analysis_count"] == 2
    assert len(envelope["results"]) == 1
    assert envelope["results"][0]["analysis"] == "rmsd"
    assert any("rg failed" in warning for warning in envelope["warnings"])
