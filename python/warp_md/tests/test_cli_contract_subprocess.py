import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON_SRC = str(ROOT / "python")


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = PYTHON_SRC + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "warp_md.cli", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_subprocess_run_missing_config_returns_exit_2() -> None:
    result = _run_cli("run", "/tmp/warpmd-does-not-exist.json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 2
    assert payload["error"]["code"] == "E_CONFIG_LOAD"


def test_subprocess_run_validation_error_returns_exit_2(tmp_path) -> None:
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(
        json.dumps(
            {
                "system": "topology.pdb",
                "trajectory": "traj.xtc",
                "analyses": [{"name": "rg"}],
            }
        )
    )

    result = _run_cli("run", str(cfg_path))
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 2
    assert payload["error"]["code"] == "E_CONFIG_VALIDATION"
    details = payload["error"]["details"]
    assert isinstance(details, list)
    assert any("field" in item and item["field"].startswith("analyses.") for item in details)


def test_subprocess_run_validation_error_ndjson(tmp_path) -> None:
    cfg_path = tmp_path / "bad-stream.json"
    cfg_path.write_text(
        json.dumps(
            {
                "system": "topology.pdb",
                "trajectory": "traj.xtc",
                "analyses": [{"name": "rg"}],
            }
        )
    )

    result = _run_cli("run", str(cfg_path), "--stream", "ndjson")
    assert result.returncode == 2
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines
    event = json.loads(lines[-1])
    assert event["event"] == "run_failed"
    assert event["final_envelope"]["exit_code"] == 2


def test_subprocess_schema_result_and_event_kinds() -> None:
    result_schema = _run_cli("schema", "--kind", "result")
    assert result_schema.returncode == 0
    result_payload = json.loads(result_schema.stdout)
    assert "anyOf" in result_payload or "oneOf" in result_payload

    event_schema = _run_cli("schema", "--kind", "event")
    assert event_schema.returncode == 0
    event_payload = json.loads(event_schema.stdout)
    assert "anyOf" in event_payload or "oneOf" in event_payload


def test_subprocess_list_plans_json_details() -> None:
    result = _run_cli("list-plans", "--format", "json", "--details")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "plans" in payload
    rg_plan = next(item for item in payload["plans"] if item["name"] == "rg")
    assert "arguments" in rg_plan
    assert any("--selection" in arg["flags"] for arg in rg_plan["arguments"])


def test_subprocess_list_plans_json_alias_details() -> None:
    result = _run_cli("list-plans", "--json", "--details")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "plans" in payload
    rg_plan = next(item for item in payload["plans"] if item["name"] == "rg")
    assert "arguments" in rg_plan
    assert any("--selection" in arg["flags"] for arg in rg_plan["arguments"])


def test_subprocess_water_models_json() -> None:
    result = _run_cli("water-models", "--format", "json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "water_models" in payload
    names = {item["name"] for item in payload["water_models"]}
    assert "tip3p" in names
