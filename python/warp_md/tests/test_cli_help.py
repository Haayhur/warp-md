import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYTHON_SRC = str(ROOT / "python")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = PYTHON_SRC + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "warp_md.cli", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_list_plans_help() -> None:
    result = _run("list-plans")
    assert result.returncode == 0


def test_list_plans_json() -> None:
    result = _run("list-plans", "--format", "json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "plans" in payload
    assert "rg" in payload["plans"]


def test_list_plans_json_alias() -> None:
    result = _run("list-plans", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "plans" in payload
    assert "rg" in payload["plans"]


def test_list_plans_json_details() -> None:
    result = _run("list-plans", "--format", "json", "--details")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "plans" in payload
    rg_plan = next(item for item in payload["plans"] if item["name"] == "rg")
    assert rg_plan["plan"] == "rg"
    assert any("--selection" in arg["flags"] for arg in rg_plan["arguments"])


def test_water_models_json() -> None:
    result = _run("water-models", "--format", "json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "water_models" in payload
    names = {item["name"] for item in payload["water_models"]}
    assert "tip3p" in names


def test_water_models_json_alias() -> None:
    result = _run("water-models", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "water_models" in payload
    names = {item["name"] for item in payload["water_models"]}
    assert "tip3p" in names


def test_atlas_fetch_help() -> None:
    result = _run("atlas-fetch", "--help")
    assert result.returncode == 0
    assert "--pdb-chain" in result.stdout


def test_rg_help() -> None:
    result = _run("rg", "--help")
    assert result.returncode == 0


def test_version_flag() -> None:
    result = _run("--version")
    assert result.returncode == 0
    assert "warp-md" in result.stdout


def test_example() -> None:
    result = _run("example")
    assert result.returncode == 0


def test_schema() -> None:
    result = _run("schema")
    assert result.returncode == 0


def test_schema_json_alias() -> None:
    result = _run("schema", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "properties" in payload


def test_schema_result_kind() -> None:
    result = _run("schema", "--kind", "result")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "anyOf" in payload or "oneOf" in payload


def test_schema_event_kind() -> None:
    result = _run("schema", "--kind", "event")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "anyOf" in payload or "oneOf" in payload


def test_run_help_lists_agent_flags() -> None:
    result = _run("run", "--help")
    assert result.returncode == 0
    assert "--stream" in result.stdout
    assert "--debug-errors" in result.stdout
