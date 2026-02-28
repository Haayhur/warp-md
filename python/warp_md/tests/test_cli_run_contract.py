import json
from pathlib import Path

import numpy as np

from warp_md import cli_run


class _FailingPlan:
    def run(self, traj, system, chunk_frames=None, device="auto"):
        raise RuntimeError("plan runtime failure")


class _SuccessPlan:
    def __init__(self, output):
        self._output = output

    def run(self, traj, system, chunk_frames=None, device="auto"):
        return self._output


def _write_config(tmp_path: Path, payload: dict) -> Path:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(payload))
    return cfg_path


def test_run_validation_error_contract(tmp_path, capsys) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg"}],
        },
    )

    code = cli_run.main(["run", str(cfg_path)])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 2
    assert envelope["status"] == "error"
    assert envelope["exit_code"] == 2
    assert envelope["error"]["code"] == "E_CONFIG_VALIDATION"
    assert "traceback" not in envelope["error"]


def test_run_debug_errors_include_traceback(monkeypatch, tmp_path, capsys) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg", "selection": "protein"}],
        },
    )

    def _bad_system(_spec):
        raise RuntimeError("cannot load topology")

    monkeypatch.setattr(cli_run, "_load_system", _bad_system)

    code = cli_run.main(["run", str(cfg_path), "--debug-errors"])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 4
    assert envelope["status"] == "error"
    assert envelope["error"]["code"] == "E_SYSTEM_LOAD"
    assert "traceback" in envelope["error"]


def test_run_analysis_spec_error_exit_code(monkeypatch, tmp_path, capsys) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg", "selection": "protein"}],
        },
    )

    monkeypatch.setattr(cli_run, "_load_system", lambda spec: object())
    monkeypatch.setattr(cli_run, "_load_trajectory", lambda spec, system: object())

    def _bad_builder(system, spec):
        raise ValueError("bad analysis options")

    monkeypatch.setattr(cli_run, "PLAN_BUILDERS", {"rg": _bad_builder})

    code = cli_run.main(["run", str(cfg_path)])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 3
    assert envelope["status"] == "error"
    assert envelope["error"]["code"] == "E_ANALYSIS_SPEC"


def test_run_runtime_error_exit_code(monkeypatch, tmp_path, capsys) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg", "selection": "protein"}],
        },
    )

    monkeypatch.setattr(cli_run, "_load_system", lambda spec: object())
    monkeypatch.setattr(cli_run, "_load_trajectory", lambda spec, system: object())
    monkeypatch.setattr(
        cli_run,
        "PLAN_BUILDERS",
        {"rg": lambda system, spec: _FailingPlan()},
    )

    code = cli_run.main(["run", str(cfg_path)])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 4
    assert envelope["status"] == "error"
    assert envelope["error"]["code"] == "E_RUNTIME_EXEC"


def test_run_ndjson_stream_events(monkeypatch, tmp_path, capsys) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg", "selection": "protein"}],
        },
    )

    monkeypatch.setattr(cli_run, "_load_system", lambda spec: object())

    code = cli_run.main(["run", str(cfg_path), "--dry-run", "--stream", "ndjson"])
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines() if line.strip()]

    assert code == 0
    assert [item["event"] for item in events] == [
        "run_started",
        "analysis_started",
        "analysis_completed",
        "run_completed",
    ]
    assert events[-1]["final_envelope"]["status"] == "dry_run"
    assert events[-1]["final_envelope"]["exit_code"] == 0
    assert events[0]["completed"] == 0
    assert events[0]["total"] == 1
    assert events[0]["progress_pct"] == 0.0
    assert events[2]["completed"] == 1
    assert events[2]["total"] == 1
    assert events[2]["progress_pct"] == 100.0


def test_run_ndjson_stream_failure_event(tmp_path, capsys) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg"}],
        },
    )

    code = cli_run.main(["run", str(cfg_path), "--stream", "ndjson"])
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines() if line.strip()]

    assert code == 2
    assert events[-1]["event"] == "run_failed"
    assert events[-1]["final_envelope"]["status"] == "error"


def test_run_ndjson_analysis_failed_event(monkeypatch, tmp_path, capsys) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg", "selection": "protein"}],
        },
    )

    monkeypatch.setattr(cli_run, "_load_system", lambda spec: object())
    monkeypatch.setattr(cli_run, "_load_trajectory", lambda spec, system: object())

    def _bad_builder(system, spec):
        raise ValueError("bad analysis options")

    monkeypatch.setattr(cli_run, "PLAN_BUILDERS", {"rg": _bad_builder})

    code = cli_run.main(["run", str(cfg_path), "--stream", "ndjson"])
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines() if line.strip()]

    assert code == 3
    assert any(event["event"] == "analysis_failed" for event in events)
    assert events[-1]["event"] == "run_failed"


def test_single_analysis_emits_envelope(monkeypatch, tmp_path, capsys) -> None:
    out_path = tmp_path / "single_rg.npz"
    monkeypatch.setattr(cli_run, "_load_system_from_args", lambda args: object())
    monkeypatch.setattr(cli_run, "_load_traj_from_args", lambda args, system: object())
    monkeypatch.setattr(
        cli_run,
        "build_plan_from_args",
        lambda args, system: _SuccessPlan(np.array([1.0, 2.0])),
    )

    code = cli_run.main(
        [
            "rg",
            "--topology",
            "topology.pdb",
            "--traj",
            "traj.xtc",
            "--selection",
            "protein",
            "--out",
            str(out_path),
        ]
    )
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 0
    assert envelope["status"] == "ok"
    assert envelope["exit_code"] == 0
    assert envelope["analysis_count"] == 1
    assert envelope["results"][0]["analysis"] == "rg"
    assert envelope["results"][0]["artifact"]["format"] == "npz"
    assert out_path.exists()


def test_single_analysis_error_envelope(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        cli_run,
        "_load_system_from_args",
        lambda args: (_ for _ in ()).throw(ValueError("bad topology")),
    )

    code = cli_run.main(
        [
            "rg",
            "--topology",
            "topology.pdb",
            "--traj",
            "traj.xtc",
            "--selection",
            "protein",
            "--out",
            str(tmp_path / "single_err.npz"),
        ]
    )
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 4
    assert envelope["status"] == "error"
    assert envelope["error"]["code"] == "E_SYSTEM_LOAD"


def test_single_analysis_deprecated_summary_flags_emit_warning(monkeypatch, tmp_path, capsys) -> None:
    out_path = tmp_path / "single_warn.npz"
    monkeypatch.setattr(cli_run, "_load_system_from_args", lambda args: object())
    monkeypatch.setattr(cli_run, "_load_traj_from_args", lambda args, system: object())
    monkeypatch.setattr(
        cli_run,
        "build_plan_from_args",
        lambda args, system: _SuccessPlan(np.array([1.0])),
    )

    code = cli_run.main(
        [
            "rg",
            "--topology",
            "topology.pdb",
            "--traj",
            "traj.xtc",
            "--selection",
            "protein",
            "--out",
            str(out_path),
            "--no-summary",
            "--summary-format",
            "text",
        ]
    )
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 0
    assert envelope["status"] == "ok"
    assert any("--no-summary is deprecated" in warning for warning in envelope["warnings"])
    assert any("--summary-format is deprecated" in warning for warning in envelope["warnings"])


def test_atlas_fetch_command_success(monkeypatch, tmp_path, capsys) -> None:
    out_path = tmp_path / "16pk_A_total.zip"

    def _fake_download(**kwargs):
        assert kwargs["dataset"] == "ATLAS"
        assert kwargs["kind"] == "total"
        assert kwargs["pdb_chain"] == "16pk_A"
        assert kwargs["out"] == str(out_path)
        return {
            "status": "ok",
            "dataset": "atlas",
            "kind": "total",
            "pdb_chain": "16pk_A",
            "url": "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/total/16pk_A",
            "out": str(out_path),
            "bytes": 10,
            "sha256": "abc",
        }

    monkeypatch.setattr(cli_run, "download_atlas_trajectory", _fake_download)
    code = cli_run.main(
        [
            "atlas-fetch",
            "--dataset",
            "ATLAS",
            "--kind",
            "total",
            "--pdb-chain",
            "16pk_A",
            "--out",
            str(out_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    assert payload["kind"] == "total"
    assert payload["pdb_chain"] == "16pk_A"


def test_atlas_fetch_command_error(monkeypatch, capsys) -> None:
    def _fail_download(**kwargs):
        raise cli_run.AtlasApiError("The zip file was not found")

    monkeypatch.setattr(cli_run, "download_atlas_trajectory", _fail_download)
    code = cli_run.main(
        [
            "atlas-fetch",
            "--dataset",
            "ATLAS",
            "--kind",
            "total",
            "--pdb-chain",
            "missing_A",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 4
    assert payload["status"] == "error"
    assert payload["exit_code"] == 4
    assert payload["error"]["code"] == "E_ATLAS_FETCH"
