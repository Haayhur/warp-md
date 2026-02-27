import json

import pytest
from pydantic import ValidationError

from warp_md.agent_schema import (
    AGENT_REQUEST_SCHEMA_VERSION,
    run_event_json_schema,
    render_run_request_schema,
    run_request_json_schema,
    validate_run_request,
)


def test_validate_run_request_accepts_string_specs():
    req = validate_run_request(
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg", "selection": "protein"}],
        }
    )
    assert req.version == AGENT_REQUEST_SCHEMA_VERSION
    assert req.system_spec == {"path": "topology.pdb"}
    assert req.trajectory_spec == {"path": "traj.xtc"}
    assert req.analyses[0].name == "rg"


def test_validate_run_request_rejects_conflicting_specs():
    with pytest.raises(ValidationError):
        validate_run_request(
            {
                "system": "a.pdb",
                "topology": "b.pdb",
                "trajectory": "traj.xtc",
                "analyses": [{"name": "rg", "selection": "protein"}],
            }
        )


def test_validate_run_request_requires_analyses():
    with pytest.raises(ValidationError):
        validate_run_request(
            {
                "system": "topology.pdb",
                "trajectory": "traj.xtc",
                "analyses": [],
            }
        )


def test_validate_run_request_rejects_unknown_top_level_keys():
    with pytest.raises(ValidationError):
        validate_run_request(
            {
                "system": "topology.pdb",
                "trajectory": "traj.xtc",
                "analyses": [{"name": "rg", "selection": "protein"}],
                "unexpected": True,
            }
        )


def test_validate_run_request_rejects_unknown_analysis_name():
    with pytest.raises(ValidationError):
        validate_run_request(
            {
                "system": "topology.pdb",
                "trajectory": "traj.xtc",
                "analyses": [{"name": "unknown-plan", "selection": "protein"}],
            }
        )


def test_validate_run_request_enforces_required_fields():
    with pytest.raises(ValidationError):
        validate_run_request(
            {
                "system": "topology.pdb",
                "trajectory": "traj.xtc",
                "analyses": [{"name": "rg"}],
            }
        )


def test_render_schema_json():
    text = render_run_request_schema("json")
    payload = json.loads(text)
    assert payload["title"] == "RunRequest"
    assert "analyses" in payload.get("properties", {})


def test_schema_declares_analysis_name_enum():
    payload = run_request_json_schema()
    analysis_def = payload["$defs"]["AnalysisRequest"]["properties"]["name"]
    assert "enum" in analysis_def
    assert "rg" in analysis_def["enum"]
    assert "docking" in analysis_def["enum"]


def test_validate_run_request_enforces_docking_required_fields():
    with pytest.raises(ValidationError):
        validate_run_request(
            {
                "system": "topology.pdb",
                "trajectory": "traj.xtc",
                "analyses": [{"name": "docking", "receptor_mask": "protein"}],
            }
        )


def test_event_schema_includes_progress_fields():
    payload = run_event_json_schema()
    defs = payload.get("$defs", {})
    started = defs.get("RunStartedEvent", {})
    props = started.get("properties", {})
    assert "progress_pct" in props
    assert "completed" in props
    assert "total" in props


def test_render_schema_yaml():
    text = render_run_request_schema("yaml")
    assert "RunRequest" in text
    assert "analyses" in text
