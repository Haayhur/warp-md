import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from warp_md.agent_schema import (
    AGENT_REQUEST_SCHEMA_VERSION,
    ArtifactMetadata,
    ErrorCode,
    RunRequest,
    plot_manifest_json_schema,
    run_event_json_schema,
    render_run_request_schema,
    render_agent_schema,
    run_request_json_schema,
    validate_run_event_payload,
    validate_run_result_payload,
    validate_run_request,
)
from warp_md import contract


def _sample_run_success_envelope() -> dict[str, object]:
    return {
        "schema_version": AGENT_REQUEST_SCHEMA_VERSION,
        "status": "ok",
        "exit_code": 0,
        "output_dir": ".",
        "system": {"path": "topology.pdb"},
        "trajectory": {"path": "traj.xtc"},
        "analysis_count": 1,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "elapsed_ms": 1000,
        "warnings": [],
        "results": [],
    }


def _write_selection_topology(path) -> None:
    path.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "END\n",
        encoding="utf-8",
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
    assert "inputs" in payload.get("properties", {})


def test_plot_manifest_schema_available():
    payload = plot_manifest_json_schema()
    assert payload["title"] == "PlotManifest"
    assert "artifacts" in payload.get("properties", {})
    rendered = json.loads(render_agent_schema("plot-manifest"))
    assert rendered["title"] == "PlotManifest"


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


def test_validate_run_request_accepts_external_inputs():
    req = validate_run_request(
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "inputs": {"energy_table": {"path": "energy.xvg", "format": "xvg"}},
            "analyses": [{"name": "rg", "selection": "protein"}],
        }
    )
    assert req.inputs == {"energy_table": {"path": "energy.xvg", "format": "xvg"}}


def test_validate_run_request_accepts_string_external_inputs():
    req = validate_run_request(
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "inputs": {"energy_table": "energy.csv"},
            "analyses": [{"name": "rg", "selection": "protein"}],
        }
    )
    assert req.inputs == {"energy_table": {"path": "energy.csv"}}


def test_contract_capabilities_expose_bundles():
    bundles = contract.capabilities()["analysis_bundles"]
    names = {bundle["name"] for bundle in bundles}
    assert {"standard_md_report", "protein_md_report", "solvent_ion_report", "polymer_report"} <= names


def test_contract_capabilities_error_codes_match_schema_literal():
    assert tuple(contract.capabilities()["error_codes"]) == contract.ERROR_CODES
    assert tuple(ErrorCode.__args__) == contract.ERROR_CODES


def test_agent_contract_snapshot_is_current():
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "scripts/generate_agent_contract_snapshot.py", "--check"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_python_fallback_contract_metadata_comes_from_snapshot():
    from warp_md import _agent_contract_snapshot

    assert contract.ERROR_CODES == tuple(_agent_contract_snapshot.ERROR_CODES)
    assert contract.ANALYSIS_BUNDLES == _agent_contract_snapshot.ANALYSIS_BUNDLES


def test_python_request_schema_fields_match_run_request_model():
    schema_fields = set(run_request_json_schema()["properties"])
    model_fields = set(RunRequest.model_fields)
    assert schema_fields == model_fields


def test_native_contract_parity_when_bindings_are_available():
    native = contract._native()
    if native is None:
        pytest.skip("native traj_py bindings are unavailable")

    native_request_schema = native.warp_md_agent_schema("request")
    python_request_schema = run_request_json_schema()
    assert set(native_request_schema["properties"]) == set(python_request_schema["properties"])

    native_caps = native.warp_md_agent_capabilities()
    python_caps = contract.capabilities()
    assert native_caps["error_codes"] == python_caps["error_codes"]
    assert {
        item["name"]: item
        for item in native_caps["analysis_bundles"]
    } == {
        item["name"]: item
        for item in python_caps["analysis_bundles"]
    }


def test_plan_schema_includes_input_requirements():
    schema = contract.get_plan_schema("rg")
    requirements = schema["input_requirements"]
    assert requirements["required"] == ["topology", "trajectory"]
    assert requirements["requires_selections"] is True
    assert requirements["selection_fields"] == ["selection"]


def test_parse_external_tables_normalizes_columns(tmp_path):
    csv_path = tmp_path / "energy.csv"
    csv_path.write_text("Time (ps),Potential Energy\n0,-1.0\n", encoding="utf-8")
    tsv_path = tmp_path / "state.tsv"
    tsv_path.write_text("time\ttemperature K\n0\t300\n", encoding="utf-8")
    xvg_path = tmp_path / "state.xvg"
    xvg_path.write_text("@ title \"state\"\n# comment\n0 300 0.99\n", encoding="utf-8")

    assert contract.parse_external_table({"path": str(csv_path)})["columns"] == [
        "time_ps",
        "potential_energy",
    ]
    assert contract.parse_external_table({"path": str(tsv_path)})["columns"] == [
        "time",
        "temperature_k",
    ]
    assert contract.parse_external_table({"path": str(xvg_path)})["columns"] == [
        "column_1",
        "column_2",
        "column_3",
    ]


def test_bundle_plan_expands_valid_members_and_external_plots(monkeypatch, tmp_path):
    monkeypatch.setattr(contract, "_native", lambda: None)
    top = tmp_path / "topology.pdb"
    _write_selection_topology(top)
    traj = tmp_path / "traj.xtc"
    traj.write_bytes(b"placeholder")
    energy = tmp_path / "energy.csv"
    energy.write_text("time,potential_energy\n0,-1.0\n", encoding="utf-8")
    request = {
        "system": str(top),
        "trajectory": str(traj),
        "inputs": {"energy_table": {"path": str(energy)}},
        "analyses": [{"name": "rg", "selection": "all"}],
    }

    plan = contract.bundle_plan("standard_md_report", request)

    assert plan["status"] in {"ok", "partial"}
    assert any(item["name"] == "rg" for item in plan["analyses"])
    assert any(plot["source_input"] == "energy_table" for plot in plan["plot_recommendations"])


def test_bundle_plan_uses_actual_energy_column(monkeypatch, tmp_path):
    monkeypatch.setattr(contract, "_native", lambda: None)
    top = tmp_path / "topology.pdb"
    _write_selection_topology(top)
    traj = tmp_path / "traj.xtc"
    traj.write_bytes(b"placeholder")
    energy = tmp_path / "energy.csv"
    energy.write_text("time,potential\n0,-1.0\n", encoding="utf-8")
    request = {
        "system": str(top),
        "trajectory": str(traj),
        "inputs": {"energy_table": str(energy)},
        "analyses": [{"name": "rg", "selection": "all"}],
    }

    plan = contract.bundle_plan("standard_md_report", request)

    energy_plot = next(
        plot
        for plot in plan["plot_recommendations"]
        if plot["source_input"] == "energy_table"
    )
    assert energy_plot["y"]["field"] == "potential"


def test_bundle_plan_reports_missing_external_table_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(contract, "_native", lambda: None)
    top = tmp_path / "topology.pdb"
    _write_selection_topology(top)
    traj = tmp_path / "traj.xtc"
    traj.write_bytes(b"placeholder")
    energy = tmp_path / "energy.csv"
    energy.write_text("time,volume\n0,10\n", encoding="utf-8")

    plan = contract.bundle_plan(
        "standard_md_report",
        {
            "system": str(top),
            "trajectory": str(traj),
            "inputs": {"energy_table": {"path": str(energy)}},
            "analyses": [{"name": "rg", "selection": "all"}],
        },
    )

    assert any(item["code"] == "E_EXTERNAL_TABLE_COLUMN" for item in plan["skipped"])


def test_external_table_errors_distinguish_format_from_load(monkeypatch, tmp_path):
    monkeypatch.setattr(contract, "_native", lambda: None)
    top = tmp_path / "topology.pdb"
    _write_selection_topology(top)
    traj = tmp_path / "traj.xtc"
    traj.write_bytes(b"placeholder")

    unsupported = tmp_path / "energy.edr"
    unsupported.write_text("binary-ish", encoding="utf-8")
    unsupported_plan = contract.bundle_plan(
        "standard_md_report",
        {
            "system": str(top),
            "trajectory": str(traj),
            "inputs": {"energy_table": {"path": str(unsupported)}},
            "analyses": [{"name": "rg", "selection": "all"}],
        },
    )

    empty_csv = tmp_path / "energy.csv"
    empty_csv.write_text("", encoding="utf-8")
    load_plan = contract.bundle_plan(
        "standard_md_report",
        {
            "system": str(top),
            "trajectory": str(traj),
            "inputs": {"energy_table": {"path": str(empty_csv)}},
            "analyses": [{"name": "rg", "selection": "all"}],
        },
    )

    assert any(item["code"] == "E_UNSUPPORTED_FORMAT" for item in unsupported_plan["skipped"])
    assert any(item["code"] == "E_EXTERNAL_TABLE_LOAD" for item in load_plan["skipped"])


def test_event_schema_includes_progress_fields():
    payload = run_event_json_schema()
    defs = payload.get("$defs", {})
    started = defs.get("RunStartedEvent", {})
    props = started.get("properties", {})
    assert "progress_pct" in props
    assert "completed" in props
    assert "total" in props
    assert "CheckpointEvent" in defs


def test_validate_run_result_payload_rejects_invalid_status():
    payload = _sample_run_success_envelope()
    payload["status"] = "finished"
    with pytest.raises(ValidationError) as exc_info:
        validate_run_result_payload(payload)
    assert "status" in str(exc_info.value)


def test_validate_run_event_payload_rejects_invalid_event():
    with pytest.raises(ValidationError) as exc_info:
        validate_run_event_payload({"event": "run_done"})
    assert "event" in str(exc_info.value)


def test_render_schema_yaml():
    text = render_run_request_schema("yaml")
    assert "RunRequest" in text
    assert "analyses" in text


# Contract validation tests

def test_contract_validate_valid_request():
    """Test that validate_request accepts valid requests."""
    req = {
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "analyses": [{"name": "rg", "selection": "protein"}],
    }
    result = contract.validate_request(req)
    assert result.valid
    assert result.status == "ok"
    assert len(result.errors) == 0
    assert result.normalized_request is not None


def test_contract_validate_request_uses_native_when_available(monkeypatch):
    class Native:
        @staticmethod
        def warp_md_agent_validate_request(
            payload_json: str,
            strict: bool,
            check_selections: bool = False,
        ):
            payload = json.loads(payload_json)
            assert payload["analyses"][0]["name"] == "rg"
            assert strict is True
            assert check_selections is False
            return {
                "schema_version": AGENT_REQUEST_SCHEMA_VERSION,
                "status": "ok",
                "valid": True,
                "normalized_request": {"version": AGENT_REQUEST_SCHEMA_VERSION},
                "errors": [],
                "warnings": [],
            }

    monkeypatch.setattr(contract, "_native", lambda: Native())
    result = contract.validate_request(
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg", "selection": "protein"}],
        },
        strict=True,
    )
    assert result.valid
    assert result.normalized_request == {"version": AGENT_REQUEST_SCHEMA_VERSION}


def test_contract_validate_request_passes_check_selections_to_native(monkeypatch):
    class Native:
        @staticmethod
        def warp_md_agent_validate_request(
            payload_json: str,
            strict: bool,
            check_selections: bool = False,
        ):
            payload = json.loads(payload_json)
            assert payload["analyses"][0]["name"] == "rg"
            assert strict is False
            assert check_selections is True
            return {
                "schema_version": AGENT_REQUEST_SCHEMA_VERSION,
                "status": "ok",
                "valid": True,
                "normalized_request": {"version": AGENT_REQUEST_SCHEMA_VERSION},
                "errors": [],
                "warnings": [],
            }

    monkeypatch.setattr(contract, "_native", lambda: Native())
    result = contract.validate_request(
        {
            "system": "topology.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg", "selection": "protein"}],
        },
        check_selections=True,
    )
    assert result.valid


def test_contract_validate_missing_required_field():
    """Test that validate_request catches missing required fields."""
    req = {
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "analyses": [{"name": "rg"}],  # missing 'selection'
    }
    result = contract.validate_request(req)
    assert not result.valid
    assert result.status == "error"
    assert len(result.errors) > 0
    # Note: Pydantic validation catches this first with E_SCHEMA_VALIDATION
    assert result.errors[0].code in ("E_REQUIRED_FIELD", "E_SCHEMA_VALIDATION")


def test_contract_validate_rdf_requires_two_selections():
    """Test that RDF requires sel_a and sel_b."""
    req = {
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "analyses": [{"name": "rdf", "sel_a": "resname SOL"}],  # missing sel_b
    }
    result = contract.validate_request(req)
    assert not result.valid
    assert any("sel_b" in e.message for e in result.errors)


def test_contract_validate_field_type_errors():
    """Test that validate_request catches type errors."""
    # Use a request with all required fields but wrong type for bins
    # Pydantic will catch this as a schema validation error
    req = {
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "analyses": [{"name": "rdf", "sel_a": "resname SOL", "sel_b": "resname SOL", "bins": "not-an-int", "r_max": 1.0}],
    }
    result = contract.validate_request(req)
    # The string bins will fail schema validation since Pydantic doesn't know this field
    assert not result.valid


def test_contract_validate_value_range_errors():
    """Test that validate_request catches range violations."""
    req = {
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "analyses": [{"name": "rdf", "sel_a": "resname SOL", "sel_b": "resname SOL", "bins": -10, "r_max": 1.0}],
    }
    result = contract.validate_request(req)
    # Contract validation catches range violations for known fields
    assert not result.valid
    assert any(e.code == "E_VALUE_RANGE" for e in result.errors)


def test_contract_validate_rejects_invalid_stream_mode():
    req = {
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "stream": "stdout",
        "analyses": [{"name": "rg", "selection": "protein"}],
    }
    result = contract.validate_request(req)
    assert not result.valid
    assert any(e.path == "stream" for e in result.errors)


def test_contract_validate_rejects_zero_checkpoint_interval():
    req = {
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "checkpoint": {"enabled": True, "interval_frames": 0},
        "analyses": [{"name": "rg", "selection": "protein"}],
    }
    result = contract.validate_request(req)
    assert not result.valid
    assert any(e.path == "checkpoint.interval_frames" for e in result.errors)


def test_contract_validate_check_selections_disabled_keeps_current_behavior():
    req = {
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "analyses": [{"name": "rg", "selection": "("}],
    }
    result = contract.validate_request(req, check_selections=False)
    assert result.valid
    assert not any(e.code == "E_SELECTION_INVALID" for e in result.errors)


def test_contract_validate_check_selections_invalid_selection_error():
    req = {
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "analyses": [{"name": "rg", "selection": "("}],
    }
    result = contract.validate_request(req, check_selections=True)
    assert not result.valid
    assert any(
        e.code == "E_SELECTION_INVALID" and e.path == "analyses[0].selection"
        for e in result.errors
    )


def test_contract_validate_check_selections_zero_match_warning(tmp_path):
    topology = tmp_path / "topology.pdb"
    _write_selection_topology(topology)
    req = {
        "system": {"path": str(topology)},
        "trajectory": "traj.xtc",
        "analyses": [{"name": "rg", "selection": "resname SOL"}],
    }
    result = contract.validate_request(req, check_selections=True)
    assert result.valid
    assert any(
        warning == "analyses[0].selection: Selection matched zero atoms"
        for warning in result.warnings
    )


def test_contract_validate_check_selections_topology_load_warning():
    req = {
        "system": {"path": "missing-topology.pdb"},
        "trajectory": "traj.xtc",
        "analyses": [{"name": "rg", "selection": "protein"}],
    }
    result = contract.validate_request(req, check_selections=True)
    assert result.valid
    assert any(
        warning.startswith(
            "analyses[0].selection: Could not load topology for atom count:"
        )
        for warning in result.warnings
    )


def test_contract_validate_check_selections_mask_field():
    req = {
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "analyses": [{"name": "density", "mask": "("}],
    }
    result = contract.validate_request(req, check_selections=True)
    assert not result.valid
    assert any(
        e.code == "E_SELECTION_INVALID" and e.path == "analyses[0].mask"
        for e in result.errors
    )


def test_contract_validate_invalid_choice_errors():
    """Test that validate_request rejects unsupported enum values."""
    req = {
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "analyses": [
            {
                "name": "conductivity",
                "selection": "resname SOL",
                "charges": "not-a-mode",
                "temperature": 300.0,
            }
        ],
    }
    result = contract.validate_request(req)
    assert not result.valid
    assert any(e.code == "E_INVALID_CHOICE" for e in result.errors)


def test_contract_validate_strict_allows_shared_analysis_fields():
    """Strict mode should allow built-in per-analysis transport fields."""
    req = {
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "analyses": [
            {
                "name": "rg",
                "selection": "protein",
                "out": "rg.npz",
                "device": "cpu",
                "chunk_frames": 64,
            }
        ],
    }
    result = contract.validate_request(req, strict=True)
    assert result.valid
    assert result.status == "ok"


def test_contract_normalize_resolves_aliases():
    """Test that normalize resolves field and analysis name aliases."""
    req = {
        "topology": "top.pdb",  # alias for system
        "traj": "traj.xtc",      # alias for trajectory
        "analyses": [{"name": "dipole-alignment", "selection": "resname SOL", "charges": "by_resname"}],
    }
    normalized = contract.normalize_request(req)
    assert "system" in normalized
    assert "trajectory" in normalized
    assert "topology" not in normalized
    assert "traj" not in normalized
    assert normalized["analyses"][0]["name"] == "dipole_alignment"


def test_contract_normalize_fills_defaults():
    """Test that normalize fills default values."""
    req = {
        "system": "top.pdb",
        "trajectory": "traj.xtc",
        "analyses": [{"name": "rdf", "sel_a": "resname SOL", "sel_b": "resname SOL", "r_max": 1.0}],
    }
    normalized = contract.normalize_request(req)
    # bins has default 200 in the contract
    # Note: normalize only fills defaults, doesn't validate required fields
    assert "bins" in normalized["analyses"][0]
    assert normalized["analyses"][0]["bins"] == 200


def test_contract_normalize_strip_unknown_removes_extra_fields():
    """Strip mode should drop unknown top-level and analysis fields."""
    req = {
        "system": "top.pdb",
        "trajectory": "traj.xtc",
        "unexpected": True,
        "analyses": [
            {
                "name": "rdf",
                "sel_a": "resname SOL",
                "sel_b": "resname SOL",
                "r_max": 1.0,
                "mystery": "drop-me",
                "out": "rdf.npz",
            }
        ],
    }
    normalized = contract.normalize_request(req, strip_unknown=True)
    assert "unexpected" not in normalized
    analysis = normalized["analyses"][0]
    assert "mystery" not in analysis
    assert analysis["out"] == "rdf.npz"
    assert analysis["bins"] == 200


def test_contract_get_plan_schema():
    """Test getting plan schema."""
    schema = contract.get_plan_schema("rdf")
    assert schema["name"] == "rdf"
    assert "aliases" in schema
    assert "required_fields" in schema
    assert "sel_a" in schema["required_fields"]
    assert "sel_b" in schema["required_fields"]
    assert "fields" in schema
    assert "sel_a" in schema["fields"]
    assert schema["fields"]["sel_a"]["semantic_type"] == "selection"


def test_contract_get_plan_schema_unknown_analysis():
    """Test that get_plan_schema raises for unknown analysis."""
    with pytest.raises(ValueError, match="unknown plan"):
        contract.get_plan_schema("not-a-real-analysis")


def test_contract_field_spec_from_dict_accepts_null_choices():
    spec = contract._field_spec_from_dict(
        {
            "type": "string",
            "semantic_type": "string",
            "choices": None,
        }
    )
    assert spec.choices is None


def test_contract_generate_template():
    """Test generating analysis templates."""
    template = contract.generate_template("rdf")
    assert template["version"] == AGENT_REQUEST_SCHEMA_VERSION
    assert "system" in template
    assert "trajectory" in template
    assert template["analyses"][0]["name"] == "rdf"
    assert "sel_a" in template["analyses"][0]
    assert "sel_b" in template["analyses"][0]


def test_contract_generate_template_with_defaults():
    """Test template generation includes defaults when requested."""
    template = contract.generate_template("rdf", fill_defaults=True)
    analysis = template["analyses"][0]
    # bins has default 200
    assert analysis.get("bins") == 200


def test_contract_list_all_plans():
    """Test listing all available plans."""
    result = contract.list_all_plans(details=False)
    assert "plans" in result
    assert isinstance(result["plans"], list)
    assert "rg" in result["plans"]
    assert "rdf" in result["plans"]
    assert "docking" in result["plans"]


def test_contract_list_all_plans_with_details():
    """Test listing plans with full details."""
    result = contract.list_all_plans(details=True)
    assert "plans" in result
    assert len(result["plans"]) > 0
    # Each plan should have rich metadata
    plan = next(p for p in result["plans"] if p["name"] == "rg")
    assert "aliases" in plan
    assert "description" in plan
    assert "tags" in plan
    assert "outputs" in plan
    assert "examples" in plan


def test_contract_capabilities():
    """Test capabilities fingerprint."""
    caps = contract.capabilities()
    assert "schema_version" in caps
    assert "cli_version" in caps
    assert "available_plans" in caps
    assert "plan_catalog_hash" in caps
    assert caps["supports_streaming"] is True
    assert caps["supports_selection_linting"] is True


def test_contract_catalog_hash_tracks_metadata_changes(monkeypatch):
    before = contract._compute_catalog_hash()
    rg_contract = contract.ANALYSIS_METADATA["rg"]
    monkeypatch.setitem(
        contract.ANALYSIS_METADATA,
        "rg",
        replace(rg_contract, aliases=[*rg_contract.aliases, "rg-alt-contract"]),
    )
    after = contract._compute_catalog_hash()
    assert after != before


def test_contract_resolve_analysis_name():
    """Test analysis name resolution."""
    assert contract._resolve_analysis_name("rg") == "rg"
    assert contract._resolve_analysis_name("dipole-alignment") == "dipole_alignment"
    assert contract._resolve_analysis_name("dipole_alignment") == "dipole_alignment"


def test_contract_resolve_analysis_name_unknown():
    """Test that unknown analysis names raise."""
    with pytest.raises(ValueError, match="unknown analysis"):
        contract._resolve_analysis_name("not-a-real-analysis")


def test_contract_semantic_field_types():
    """Test that field specs include semantic types."""
    schema = contract.get_plan_schema("rg")
    assert schema["fields"]["selection"]["semantic_type"] == "selection"
    assert schema["fields"]["selection"]["type"] == "string"

    # RDF selections
    schema = contract.get_plan_schema("rdf")
    assert schema["fields"]["sel_a"]["semantic_type"] == "selection"
    assert schema["fields"]["sel_b"]["semantic_type"] == "selection"

    # Docking masks
    schema = contract.get_plan_schema("docking")
    assert schema["fields"]["receptor_mask"]["semantic_type"] == "mask"
    assert schema["fields"]["ligand_mask"]["semantic_type"] == "mask"


def test_contract_artifact_metadata():
    """Test that analysis contracts include artifact metadata."""
    schema = contract.get_plan_schema("rg")
    assert len(schema["outputs"]) > 0
    output = schema["outputs"][0]
    assert "kind" in output
    assert "format" in output
    assert "description" in output
    assert output["kind"] == "timeseries"
    assert output["format"] == "npz"
    assert output["description"] == "Time series of radius of gyration values"
    plot = output["plot_recommendations"][0]
    assert plot["plot_type"] == "line"
    assert plot["x"] == {"field": "time_ps", "units": "ps"}
    assert plot["y"] == {"field": "rg_nm", "units": "nm"}
    assert output["companions"][0]["role"] == "npz_companion_manifest"
    assert output["companions"][1]["role"] == "array_table"


def test_artifact_metadata_preserves_contract_description():
    metadata = ArtifactMetadata.model_validate(
        {
            "path": "rg.npz",
            "format": "npz",
            "bytes": 16,
            "sha256": "a" * 64,
            "kind": "timeseries",
            "fields": ["time_ps", "rg_nm"],
            "description": "Time series of radius of gyration values",
        }
    ).model_dump(mode="python", exclude_none=True)
    assert metadata["description"] == "Time series of radius of gyration values"


def test_contract_tags():
    """Test that analyses have tags for categorization."""
    schema = contract.get_plan_schema("rg")
    assert "structural" in schema["tags"]

    schema = contract.get_plan_schema("conductivity")
    assert "electrostatics" in schema["tags"]
    assert "transport" in schema["tags"]


# Selection linting tests

def test_lint_selection_valid_expression():
    """Test that valid selection expressions pass."""
    result = contract.lint_selection("protein")
    assert result.valid
    assert result.expression == "protein"
    assert result.field_type == "selection"
    assert result.error is None


def test_lint_selection_uses_native_when_available(monkeypatch):
    class Native:
        @staticmethod
        def warp_md_agent_lint_selection(expr: str, field_type: str, system_path: str | None):
            assert expr == "protein"
            assert field_type == "mask"
            assert system_path is None
            return {
                "valid": True,
                "expression": expr,
                "field_type": field_type,
                "matched_atoms": 10,
                "total_atoms": 20,
                "error": None,
                "warnings": [],
            }

    monkeypatch.setattr(contract, "_native", lambda: Native())
    result = contract.lint_selection("protein", field_type="mask")
    assert result.valid
    assert result.matched_atoms == 10
    assert result.total_atoms == 20


def test_lint_selection_empty_expression():
    """Test that empty expressions are rejected."""
    result = contract.lint_selection("")
    assert not result.valid
    assert "empty" in result.error.lower()


def test_lint_selection_whitespace_only():
    """Test that whitespace-only expressions are rejected."""
    result = contract.lint_selection("   ")
    assert not result.valid
    assert "empty" in result.error.lower()


def test_lint_selection_unbalanced_quotes():
    """Test that unbalanced quotes are caught."""
    result = contract.lint_selection("resname 'SOL")
    assert not result.valid
    assert "quote" in result.error.lower()


def test_lint_selection_unbalanced_parentheses():
    """Test that unbalanced parentheses are caught."""
    result = contract.lint_selection("(resname SOL")
    assert not result.valid
    assert "parentheses" in result.error.lower()


def test_lint_selection_field_type_mask():
    """Test that field_type can be set to mask."""
    result = contract.lint_selection("protein", field_type="mask")
    assert result.valid
    assert result.field_type == "mask"


def test_lint_selection_with_system():
    """Test selection with atom counting when system is provided."""
    # This test validates syntax only - actual atom count requires a real file
    result = contract.lint_selection("protein", system_path=None)
    assert result.valid
    assert result.matched_atoms is None
    assert result.total_atoms is None


# Analysis suggestion tests

def test_suggest_analyses_radius_gyration():
    """Test suggesting radius of gyration analysis."""
    result = contract.suggest_analyses("I want to compute the radius of gyration")
    assert len(result.candidates) > 0
    # rg should be the top suggestion
    assert result.candidates[0].name == "rg"
    # Should have matched on keyword "radius" or "gyration"
    reason_lower = result.candidates[0].reason.lower()
    assert "radius" in reason_lower or "gyration" in reason_lower


def test_suggest_analyses_uses_native_when_available(monkeypatch):
    class Native:
        @staticmethod
        def warp_md_agent_suggest_analyses(goal: str, provided_fields, top_n: int):
            assert goal == "radius of gyration"
            assert provided_fields == ["selection"]
            assert top_n == 2
            return {
                "goal": goal,
                "total_analyses": 37,
                "candidates": [
                    {
                        "name": "rg",
                        "reason": "name match: rg",
                        "missing_fields": [],
                        "score": 10.0,
                    }
                ],
            }

    monkeypatch.setattr(contract, "_native", lambda: Native())
    result = contract.suggest_analyses(
        "radius of gyration",
        provided_fields=["selection"],
        top_n=2,
    )
    assert result.goal == "radius of gyration"
    assert result.candidates[0].name == "rg"
    assert result.candidates[0].missing_fields == []


def test_suggest_analyses_diffusion():
    """Test suggesting diffusion/MSD analysis."""
    result = contract.suggest_analyses("I want to measure diffusion coefficient")
    assert len(result.candidates) > 0
    # msd should be in the suggestions
    names = [c.name for c in result.candidates]
    assert "msd" in names


def test_suggest_analyses_pair_distribution():
    """Test suggesting RDF for pair distribution."""
    result = contract.suggest_analyses("pair distribution function")
    assert len(result.candidates) > 0
    # rdf should be the top suggestion
    assert result.candidates[0].name == "rdf"


def test_suggest_analyses_with_provided_fields():
    """Test that missing_fields excludes provided fields."""
    result = contract.suggest_analyses(
        "radius of gyration",
        provided_fields=["selection"],
    )
    assert result.candidates[0].name == "rg"
    # selection is required but provided, so not in missing
    assert "selection" not in result.candidates[0].missing_fields


def test_suggest_analyses_hydrogen_bonds():
    """Test suggesting hydrogen bond analysis."""
    result = contract.suggest_analyses("I want to count hydrogen bonds")
    assert len(result.candidates) > 0
    names = [c.name for c in result.candidates]
    assert "hbond" in names
    assert result.candidates[0].name == "hbond"


def test_suggest_analyses_docking():
    """Test suggesting docking analysis."""
    result = contract.suggest_analyses("I want to do molecular docking")
    assert len(result.candidates) > 0
    assert result.candidates[0].name == "docking"


def test_suggest_analyses_secondary_structure():
    """Test suggesting secondary structure analysis."""
    result = contract.suggest_analyses("I want to analyze secondary structure")
    assert len(result.candidates) > 0
    names = [c.name for c in result.candidates]
    assert "dssp" in names


def test_suggest_analyses_water_count_avoids_docking():
    result = contract.suggest_analyses("count waters around a protein")
    assert len(result.candidates) > 0
    assert result.candidates[0].name == "water_count"


def test_suggest_analyses_interface_hbonds_avoid_docking():
    result = contract.suggest_analyses("hydrogen bonds in protein water interface")
    assert len(result.candidates) > 0
    assert result.candidates[0].name == "hbond"


def test_suggest_analyses_free_volume_prefers_generic_plan():
    result = contract.suggest_analyses("measure free volume in polymer")
    assert len(result.candidates) > 0
    assert result.candidates[0].name == "free_volume"


def test_suggest_analyses_top_n():
    """Test that top_n limits results."""
    result = contract.suggest_analyses("structural analysis", top_n=2)
    assert len(result.candidates) <= 2


def test_suggest_analyses_scores_are_descending():
    """Test that candidates are sorted by score descending."""
    result = contract.suggest_analyses("protein structure")
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_suggest_analyses_includes_total():
    """Test that result includes total analysis count."""
    result = contract.suggest_analyses("anything")
    assert result.total_analyses == len(contract.ANALYSIS_METADATA)
    assert result.total_analyses > 30


def test_suggest_analyses_returns_reason():
    """Test that candidates include a reason."""
    result = contract.suggest_analyses("radius of gyration")
    assert len(result.candidates[0].reason) > 0
    assert result.candidates[0].name == "rg"
