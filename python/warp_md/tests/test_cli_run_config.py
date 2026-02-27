import json
from pathlib import Path

import numpy as np

from warp_md import cli_run


class _DummyPlan:
    def __init__(self, output):
        self._output = output

    def run(self, traj, system, chunk_frames=None, device="auto"):
        return self._output


def _write_config(tmp_path: Path, payload: dict) -> Path:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(payload))
    return cfg_path


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
    assert envelope["results"][0]["artifact"]["bytes"] > 0
    assert len(envelope["results"][0]["artifact"]["sha256"]) == 64
    assert Path(envelope["results"][0]["out"]).exists()
    assert len(load_traj_calls) == 1


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
