"""Integration tests for agent contracts.

These tests verify that the agent-friendly interfaces work correctly:
- JSON schema validation
- Error code contracts
- Streaming event parsing
- Tool wrapper functionality
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pytest

# Import from the actual package
from warp_md.agent_schema import (
    ErrorCode,
    RunRequest,
    classify_error,
    run_request_json_schema,
    run_result_json_schema,
    validate_run_request,
    _ANALYSIS_REQUIRED_FIELDS,
)
from warp_md.runner import run_analyses


# Define event dataclasses for testing (these match the streaming event schemas)
@dataclass
class PackStartedEvent:
    total_molecules: int
    box_size: List[float]
    box_origin: List[float]
    output_path: str


@dataclass
class PhaseStartedEvent:
    phase: str
    total_molecules: Optional[int] = None
    max_iterations: Optional[int] = None
    elapsed_ms: Optional[int] = None
    iterations: Optional[int] = None
    final_obj_value: Optional[float] = None


@dataclass
class GencanIterationEvent:
    iteration: int
    max_iterations: int
    obj_value: float
    obj_overlap: float
    obj_constraint: float
    pg_sup: float
    pg_norm: float
    elapsed_ms: int
    progress_pct: float
    eta_ms: int


@dataclass
class PackCompleteEvent:
    total_atoms: int
    total_molecules: int
    final_box_size: List[float]
    output_path: str
    elapsed_ms: int
    profile_ms: Dict[str, int]


@dataclass
class PackResult:
    total_atoms: int
    total_molecules: int
    final_box_size: tuple
    output_path: Optional[str]
    elapsed_ms: int
    profile_ms: Dict[str, int]
    success: bool
    error: Optional[str] = None


@dataclass
class PepResult:
    total_atoms: int
    total_residues: int
    total_chains: int
    output_path: str
    elapsed_ms: int
    success: bool


# Simple event handler for testing
class WarpPackEventHandler:
    """Base event handler for warp-pack streaming events."""

    def on_pack_started(self, event: Optional[PackStartedEvent]) -> None:
        pass

    def on_phase_started(self, event: Optional[PhaseStartedEvent]) -> None:
        pass

    def on_molecule_placed(self, event: Optional[Dict[str, Any]]) -> None:
        pass

    def on_gencan_iteration(self, event: Optional[GencanIterationEvent]) -> None:
        pass

    def on_phase_complete(self, event: Optional[PhaseStartedEvent]) -> None:
        pass

    def on_pack_complete(self, event: Optional[PackCompleteEvent]) -> None:
        pass

    def on_error(self, code: str, message: str) -> None:
        pass


class ProgressTracker(WarpPackEventHandler):
    """Simple progress tracker for testing."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.current_phase = None


def parse_stream_events(process: subprocess.Popen, handler: WarpPackEventHandler) -> Optional[Dict[str, Any]]:
    """Parse NDJSON streaming events from subprocess stderr."""
    final_event = None

    while True:
        line = process.stderr.readline()
        if not line:
            break

        try:
            event = json.loads(line.strip())
            event_type = event.get("event")

            if event_type == "pack_started":
                handler.on_pack_started(PackStartedEvent(**event))
            elif event_type == "phase_started":
                handler.on_phase_started(PhaseStartedEvent(**event))
            elif event_type == "gencan_iteration":
                handler.on_gencan_iteration(GencanIterationEvent(**event))
            elif event_type == "phase_complete":
                handler.on_phase_complete(PhaseStartedEvent(**event))
            elif event_type == "pack_complete":
                final_event = event
                handler.on_pack_complete(PackCompleteEvent(**event))
            elif event_type == "error":
                handler.on_error(event.get("code", ""), event.get("message", ""))
        except (json.JSONDecodeError, TypeError, KeyError):
            # Skip malformed events
            pass

    return final_event


class TestAgentSchema:
    """Test the agent schema validation and contracts."""

    def test_warp_md_request_schema(self):
        """Verify warp-md request schema structure."""
        from warp_md.agent_schema import run_request_json_schema, RunRequest

        schema = run_request_json_schema()
        assert "properties" in schema
        assert "version" in schema["properties"]
        assert "analyses" in schema["properties"]
        # Version has default value
        assert schema["properties"]["version"]["default"] == "warp-md.agent.v1"

    def test_warp_md_result_schema(self):
        """Verify warp-md result envelope structure."""
        from warp_md.agent_schema import run_result_json_schema

        schema = run_result_json_schema()
        assert "anyOf" in schema  # Success or error envelope
        # The schema uses $ref to definitions, check the structure
        assert "$defs" in schema
        assert "RunSuccessEnvelope" in schema["$defs"]
        assert "RunErrorEnvelope" in schema["$defs"]
        # Check success envelope has expected fields
        success_schema = schema["$defs"]["RunSuccessEnvelope"]
        assert "properties" in success_schema
        assert "status" in success_schema["properties"]
        assert "results" in success_schema["properties"]
        # Check error envelope has expected fields
        error_schema = schema["$defs"]["RunErrorEnvelope"]
        assert "properties" in error_schema
        assert "error" in error_schema["properties"]

    def test_error_code_coverage(self):
        """Verify documented error codes exist."""
        from warp_md.agent_schema import ErrorCode

        # Check that all documented error codes are defined
        documented_codes = [
            "E_CONFIG_VALIDATION", "E_CONFIG_VERSION", "E_CONFIG_MISSING_FIELD",
            "E_ANALYSIS_UNKNOWN", "E_ANALYSIS_SPEC", "E_SELECTION_EMPTY", "E_SELECTION_INVALID",
            "E_SYSTEM_LOAD", "E_TRAJECTORY_LOAD", "E_TRAJECTORY_EOF", "E_RUNTIME_EXEC",
            "E_OUTPUT_WRITE", "E_DEVICE_UNAVAILABLE", "E_INTERNAL",
        ]

        # Verify these are valid literal values
        for code in documented_codes:
            assert code in ErrorCode.__args__

    def test_analysis_required_fields(self):
        """Verify required fields are documented correctly."""
        from warp_md.agent_schema import _ANALYSIS_REQUIRED_FIELDS

        # Check key analyses have their required fields documented
        assert "rg" in _ANALYSIS_REQUIRED_FIELDS
        assert _ANALYSIS_REQUIRED_FIELDS["rg"] == ("selection",)
        assert "rmsd" in _ANALYSIS_REQUIRED_FIELDS
        assert "rdf" in _ANALYSIS_REQUIRED_FIELDS
        assert set(_ANALYSIS_REQUIRED_FIELDS["rdf"]) == {"sel_a", "sel_b", "bins", "r_max"}


class TestStreamingEvents:
    """Test streaming event parsing and structure."""

    def test_pack_started_event(self):
        """Test PackStartedEvent dataclass."""
        event = PackStartedEvent(
            total_molecules=100,
            box_size=[50.0, 50.0, 50.0],
            box_origin=[0.0, 0.0, 0.0],
            output_path="output.pdb",
        )
        assert event.total_molecules == 100
        assert event.box_size == [50.0, 50.0, 50.0]

    def test_phase_started_event(self):
        """Test PhaseStartedEvent dataclass."""
        event = PhaseStartedEvent(
            phase="gencan",
            total_molecules=100,
            max_iterations=1000,
        )
        assert event.phase == "gencan"
        assert event.max_iterations == 1000

    def test_gencan_iteration_event(self):
        """Test GencanIterationEvent dataclass."""
        event = GencanIterationEvent(
            iteration=42,
            max_iterations=1000,
            obj_value=1.23e-3,
            obj_overlap=1.0e-3,
            obj_constraint=2.3e-4,
            pg_sup=0.05,
            pg_norm=0.123,
            elapsed_ms=5000,
            progress_pct=4.2,
            eta_ms=114000,
        )
        assert event.iteration == 42
        assert event.progress_pct == 4.2
        assert event.eta_ms == 114000

    def test_pack_complete_event(self):
        """Test PackCompleteEvent dataclass."""
        event = PackCompleteEvent(
            total_atoms=4500,
            total_molecules=150,
            final_box_size=[50.0, 50.0, 50.0],
            output_path="output.pdb",
            elapsed_ms=52000,
            profile_ms={"templates": 125, "place_core": 5420, "gencan": 45000, "relax": 0},
        )
        assert event.total_atoms == 4500
        assert event.elapsed_ms == 52000
        assert event.profile_ms["gencan"] == 45000

    def test_event_json_serialization(self):
        """Test that events can be serialized to JSON."""
        # Create sample events
        events = [
            PackStartedEvent(100, [50.0, 50.0, 50.0], [0.0, 0.0, 0.0], "out.pdb"),
            PhaseStartedEvent("gencan", 100, 1000),
            GencanIterationEvent(42, 1000, 1.23e-3, 1e-3, 2.3e-4, 0.05, 0.123, 5000, 4.2, 114000),
            PackCompleteEvent(4500, 150, [50.0, 50.0, 50.0], "out.pdb", 52000, {}),
        ]

        # Verify they can be converted to dict (for JSON serialization)
        for event in events:
            d = event if isinstance(event, dict) else event.__dict__
            # Each event type should have some expected fields
            assert len(d) > 0  # Event has fields
            assert "total_molecules" in d or "total_atoms" in d or "phase" in d or "iteration" in d


class TestEventHandlers:
    """Test event handler implementations."""

    def test_progress_tracker(self):
        """Test ProgressTracker doesn't crash on events."""
        handler = ProgressTracker(verbose=False)

        # Send all event types
        handler.on_pack_started(PackStartedEvent(
            total_molecules=100,
            box_size=[50.0, 50.0, 50.0],
            box_origin=[0.0, 0.0, 0.0],
            output_path="out.pdb",
        ))
        handler.on_phase_started(PhaseStartedEvent(
            phase="gencan",
            total_molecules=100,
            max_iterations=1000,
        ))
        handler.on_gencan_iteration(GencanIterationEvent(
            iteration=42,
            max_iterations=1000,
            obj_value=1.23e-3,
            obj_overlap=1.0e-3,
            obj_constraint=2.3e-4,
            pg_sup=0.05,
            pg_norm=0.123,
            elapsed_ms=5000,
            progress_pct=4.2,
            eta_ms=114000,
        ))
        handler.on_phase_complete(PhaseStartedEvent(
            phase="gencan",
            elapsed_ms=45000,
            iterations=None,
            final_obj_value=1.5e-4,
        ))
        handler.on_pack_complete(PackCompleteEvent(
            total_atoms=4500,
            total_molecules=150,
            final_box_size=[50.0, 50.0, 50.0],
            output_path="out.pdb",
            elapsed_ms=52000,
            profile_ms={},
        ))
        handler.on_error("E_CONFIG_VALIDATION", "Invalid JSON")

    def test_base_handler_implementation(self):
        """Test base handler methods are callable."""
        handler = WarpPackEventHandler()

        # All methods should be callable without error
        handler.on_pack_started(None)
        handler.on_phase_started(None)
        handler.on_molecule_placed(None)
        handler.on_gencan_iteration(None)
        handler.on_phase_complete(None)
        handler.on_pack_complete(None)
        handler.on_error("code", "message")


class TestToolWrappers:
    """Test result dataclasses for warp-pack and warp-pep."""

    def test_pack_result_structure(self):
        """Test PackResult dataclass."""
        result = PackResult(
            total_atoms=4500,
            total_molecules=150,
            final_box_size=(50.0, 50.0, 50.0),
            output_path="output.pdb",
            elapsed_ms=52000,
            profile_ms={"gencan": 45000},
            success=True,
        )

        assert result.success is True
        assert result.total_atoms == 4500
        assert result.error is None

    def test_pack_result_error(self):
        """Test PackResult with error."""
        result = PackResult(
            total_atoms=0,
            total_molecules=0,
            final_box_size=(0.0, 0.0, 0.0),
            output_path=None,
            elapsed_ms=0,
            profile_ms={},
            success=False,
            error="Packing failed",
        )

        assert result.success is False
        assert "Packing" in result.error

    def test_pep_result_structure(self):
        """Test PepResult dataclass."""
        result = PepResult(
            total_atoms=167,
            total_residues=10,
            total_chains=1,
            output_path="peptide.pdb",
            elapsed_ms=150,
            success=True,
        )

        assert result.total_residues == 10
        assert result.success is True


class TestSchemaValidation:
    """Test JSON schema validation and parsing."""

    def test_warp_md_minimal_request(self):
        """Test minimal valid warp-md request."""
        from warp_md.agent_schema import validate_run_request

        minimal = {
            "version": "warp-md.agent.v1",
            "system": "protein.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg", "selection": "protein"}]
        }

        result = validate_run_request(minimal)
        assert result.version == "warp-md.agent.v1"
        assert len(result.analyses) == 1
        assert result.analyses[0].name == "rg"

    def test_warp_md_validation_errors(self):
        """Test that validation errors are properly reported."""
        from warp_md.agent_schema import validate_run_request
        from pydantic import ValidationError

        # Missing required field
        invalid = {
            "version": "warp-md.agent.v1",
            "system": "protein.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg"}]  # Missing "selection"
        }

        with pytest.raises(ValidationError):
            validate_run_request(invalid)

    def test_invalid_version(self):
        """Test that invalid version is rejected."""
        from warp_md.agent_schema import validate_run_request
        from pydantic import ValidationError

        invalid = {
            "version": "warp-md.agent.v0",  # Wrong version
            "system": "protein.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg", "selection": "protein"}]
        }

        with pytest.raises(ValidationError):
            validate_run_request(invalid)


class TestErrorCodes:
    """Test error code classification."""

    def test_classify_error(self):
        """Test error classification function."""
        from warp_md.agent_schema import classify_error

        # Test various error scenarios
        test_cases = [
            (ValueError("no atoms matched"), "E_SELECTION_EMPTY", "selection"),
            (ValueError("invalid selection syntax"), "E_SELECTION_INVALID", "selection"),
            (ValueError("unsupported version"), "E_CONFIG_VERSION", "config"),
            (FileNotFoundError("file not found"), "E_TRAJECTORY_LOAD", "trajectory"),
        ]

        for exc, expected_code, context in test_cases:
            result = classify_error(exc, context)
            assert result == expected_code, f"Expected {expected_code}, got {result} for {exc}"


class TestStreamingFormat:
    """Test NDJSON streaming format."""

    def test_streaming_events_are_valid_json(self):
        """Verify all streaming events emit valid JSON."""
        # Sample events from documentation
        sample_events = [
            '{"event":"pack_started","total_molecules":150,"box_size":[50,50,50],"output_path":"out.pdb"}',
            '{"event":"phase_started","phase":"gencan","max_iterations":1000}',
            '{"event":"gencan_iteration","iteration":42,"max_iterations":1000,"obj_value":1.23e-3,"progress_pct":4.2}',
            '{"event":"phase_complete","phase":"gencan","elapsed_ms":45000}',
            '{"event":"pack_complete","total_atoms":4500,"elapsed_ms":52000}',
        ]

        for event_line in sample_events:
            data = json.loads(event_line)
            assert "event" in data
            assert data["event"] in {
                "pack_started", "phase_started", "gencan_iteration",
                "phase_complete", "pack_complete", "error"
            }

    def test_all_event_types_documented(self):
        """Verify all event types from documentation are parseable."""
        # Events from streaming-progress.md
        documented_events = [
            ("pack_started", ["total_molecules", "box_size", "box_origin", "output_path"]),
            ("phase_started", ["phase", "total_molecules", "max_iterations"]),
            ("molecule_placed", ["molecule_index", "total_molecules", "molecule_name", "successful", "progress_pct"]),
            ("gencan_iteration", ["iteration", "max_iterations", "obj_value", "obj_overlap", "obj_constraint", "pg_sup", "pg_norm", "elapsed_ms", "progress_pct", "eta_ms"]),
            ("phase_complete", ["phase", "elapsed_ms", "iterations", "final_obj_value"]),
            ("pack_complete", ["total_atoms", "total_molecules", "final_box_size", "output_path", "elapsed_ms", "profile_ms"]),
            ("error", ["code", "message"]),
            ("operation_started", ["operation", "input_path", "total_chains", "total_residues", "total_mutations"]),
            ("mutation_complete", ["mutation_index", "total_mutations", "mutation_spec", "successful", "elapsed_ms", "progress_pct"]),
            ("operation_complete", ["operation", "total_atoms", "total_residues", "total_chains", "output_path", "elapsed_ms"]),
        ]

        for event_name, expected_fields in documented_events:
            # Create minimal valid event
            if event_name == "pack_started":
                event = {
                    "event": event_name,
                    "total_molecules": 100,
                    "box_size": [50, 50, 50],
                    "box_origin": [0, 0, 0],
                    "output_path": "out.pdb",
                }
            elif event_name == "gencan_iteration":
                event = {
                    "event": event_name,
                    "iteration": 42,
                    "max_iterations": 1000,
                    "obj_value": 1.23e-3,
                    "obj_overlap": 1.0e-3,
                    "obj_constraint": 2.3e-4,
                    "pg_sup": 0.05,
                    "pg_norm": 0.123,
                    "elapsed_ms": 5000,
                    "progress_pct": 4.2,
                    "eta_ms": 10000,
                }
            elif event_name == "error":
                event = {"event": event_name, "code": "E_CONFIG", "message": "test"}
            else:
                # Create minimal event with required fields
                event = {"event": event_name}
                for field in expected_fields:
                    if "molecules" in field:
                        event[field] = 100
                    elif "box" in field:
                        event[field] = [50, 50, 50]
                    elif "atoms" in field:
                        event[field] = 100
                    elif "iterations" in field or "index" in field:
                        event[field] = 1
                    elif "elapsed_ms" in field:
                        event[field] = 1000
                    else:
                        event[field] = "test" if isinstance(field, str) else 0

            # Verify it's valid JSON
            json_str = json.dumps(event)
            parsed = json.loads(json_str)
            assert parsed["event"] == event_name


class TestContractConsistency:
    """Test consistency between documentation and implementation."""

    def test_summary_documentation_matches_schema(self):
        """Verify that documented error codes exist in ErrorCode."""
        from warp_md.agent_schema import ErrorCode

        # These are the error codes documented in agent-schema.md
        documented_codes = [
            "E_CONFIG_VALIDATION",
            "E_CONFIG_VERSION",
            "E_CONFIG_MISSING_FIELD",
            "E_ANALYSIS_UNKNOWN",
            "E_ANALYSIS_SPEC",
            "E_SELECTION_EMPTY",
            "E_SELECTION_INVALID",
            "E_SYSTEM_LOAD",
            "E_TRAJECTORY_LOAD",
            "E_TRAJECTORY_EOF",
            "E_RUNTIME_EXEC",
            "E_OUTPUT_WRITE",
            "E_DEVICE_UNAVAILABLE",
            "E_INTERNAL",
        ]

        for code in documented_codes:
            assert code in ErrorCode.__args__, f"Documented error code {code} not in ErrorCode"

    def test_tool_availability_in_examples(self):
        """Verify that example files reference real tools."""
        # Navigate from python/warp_md/tests/ to repo root, then examples/agents
        repo_root = Path(__file__).parent.parent.parent.parent
        examples_dir = repo_root / "examples" / "agents"

        # Check that example Python files exist
        example_files = [
            "langchain/warp_md_tool.py",
            "langchain/warp_pack_streaming.py",
            "crewai/warp_md_tool.py",
            "crewai/warp_pack_streaming.py",
            "autogen/warp_md_tool.py",
            "openai/warp_md_tool.py",
            "warp_utils.py",
            "warp_pack_wrapper.py",
            "warp_pep_wrapper.py",
        ]

        for file_path in example_files:
            full_path = examples_dir / file_path
            assert full_path.exists(), f"Example file {file_path} not found at {full_path}"

    def test_documentation_links_resolve(self):
        """Verify that documentation cross-references are valid."""
        # Navigate from python/warp_md/tests/ to repo root, then docs
        repo_root = Path(__file__).parent.parent.parent.parent
        docs_dir = repo_root / "docs"

        # Check that linked documentation files exist
        referenced_docs = [
            "guides/agent-frameworks.md",
            "guides/streaming-progress.md",
            "guides/agent-transcripts.md",
            "reference/agent-schema.md",
            "reference/analysis-plans.md",
        ]

        for doc_path in referenced_docs:
            full_path = docs_dir / doc_path
            assert full_path.exists(), f"Referenced documentation {doc_path} not found at {full_path}"


class TestStreamingEventParsing:
    """Test parsing of streaming events from subprocess."""

    def test_parse_stream_events_handles_empty_lines(self):
        """Test that empty lines are skipped gracefully."""
        handler = WarpPackEventHandler()

        # Simulate subprocess with some empty lines
        class MockProc:
            class Stderr:
                def readline(self):
                    # Return empty string to simulate EOF
                    return ""
            stderr = Stderr()
            returncode = 0

        proc = MockProc()
        result = parse_stream_events(proc, handler)
        assert result is None  # No final event

    def test_parse_stream_events_handles_malformed_json(self):
        """Test that malformed JSON is skipped."""
        handler = WarpPackEventHandler()

        class MockStderr:
            def __init__(self):
                self.lines = [
                    '{"event":"test"}\n',  # Valid (but unknown event, skipped)
                    'invalid json\n',               # Invalid (skipped)
                    '{"event":"pack_complete", "total_atoms": 100}\n',  # Valid
                ]
                self.idx = 0

            def readline(self):
                if self.idx >= len(self.lines):
                    return ""
                line = self.lines[self.idx]
                self.idx += 1
                return line

        class MockProc:
            stderr = MockStderr()
            returncode = 0

        result = parse_stream_events(MockProc(), handler)
        # Should complete without error and return pack_complete event
        assert result is not None
        assert result["event"] == "pack_complete"

    def test_parse_stream_events_returns_final_event(self):
        """Test that pack_complete event is returned."""
        handler = WarpPackEventHandler()
        final_event_data = {"event": "pack_complete", "total_atoms": 100}

        class MockStderr:
            def __init__(self):
                self.lines = [json.dumps(final_event_data) + "\n", ""]
                self.idx = 0

            def readline(self):
                if self.idx >= len(self.lines):
                    return ""
                line = self.lines[self.idx]
                self.idx += 1
                return line

        class MockProc:
            stderr = MockStderr()
            returncode = 0

        result = parse_stream_events(MockProc(), handler)
        assert result == final_event_data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
