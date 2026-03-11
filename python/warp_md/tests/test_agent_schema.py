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
from warp_md import contract


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
    assert "CheckpointEvent" in defs


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
    assert output["kind"] == "timeseries"
    assert output["format"] == "npz"


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
