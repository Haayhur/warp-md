from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator

from . import contract as _contract
from ._json_types import JsonObject, JsonValue
from .contract_constants import AGENT_REQUEST_SCHEMA_VERSION, AGENT_RESULT_SCHEMA_VERSION

# Structured error codes for agent consumption
ErrorCode = Literal[
    # Validation errors (exit code 2)
    "E_CONFIG_LOAD",            # Failed to load config file
    "E_CONFIG_VALIDATION",      # Schema validation failed
    "E_CONFIG_VERSION",         # Unsupported schema version
    "E_CONFIG_MISSING_FIELD",   # Required field missing
    
    # Analysis specification errors (exit code 3)
    "E_ANALYSIS_UNKNOWN",       # Unknown analysis name
    "E_ANALYSIS_SPEC",          # Invalid analysis parameters
    "E_SELECTION_EMPTY",        # Mask matched no atoms
    "E_SELECTION_INVALID",      # Mask syntax error
    
    # Runtime errors (exit code 4)
    "E_SYSTEM_LOAD",            # Failed to load topology
    "E_TRAJECTORY_LOAD",        # Failed to load trajectory
    "E_TRAJECTORY_EOF",         # Unexpected end of trajectory
    "E_RUNTIME_EXEC",           # Analysis execution failed
    "E_OUTPUT_DIR",             # Failed to create output directory
    "E_OUTPUT_WRITE",           # Failed to write output file
    "E_DEVICE_UNAVAILABLE",     # Requested device not available
    "E_ATLAS_FETCH",            # Failed to fetch ATLAS asset
    
    # Internal errors (exit code 5)
    "E_INTERNAL",               # Unexpected internal error
]

_ANALYSIS_NAMES = tuple(sorted(_contract.ANALYSIS_METADATA))
AnalysisName = Literal.__getitem__(_ANALYSIS_NAMES)
_ANALYSIS_REQUIRED_FIELDS: Dict[str, tuple[str, ...]] = {
    name: tuple(_contract.ANALYSIS_METADATA[name].required_fields)
    for name in _ANALYSIS_NAMES
}


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint event streaming."""
    model_config = ConfigDict(extra="forbid")
    
    enabled: bool = False
    interval_frames: int = Field(default=1000, ge=1)


class CheckpointEvent(BaseModel):
    """Progress event emitted during analysis execution."""
    event: Literal["checkpoint"] = "checkpoint"
    analysis_index: int
    analysis_name: str
    frames_processed: int
    frames_total: Optional[int] = None
    progress_pct: Optional[float] = None
    eta_ms: Optional[int] = None


def classify_error(exc: Exception, context: str = "") -> str:
    """Classify an exception into a structured error code.
    
    Args:
        exc: The exception to classify
        context: Optional context hint (e.g., "selection", "config")
    
    Returns:
        One of the ErrorCode literals
    """
    msg = str(exc).lower()
    exc_type = type(exc).__name__.lower()
    
    # Selection errors
    if "selection" in context or "mask" in context:
        if "no atoms" in msg or "empty" in msg or "0 atoms" in msg:
            return "E_SELECTION_EMPTY"
        if "syntax" in msg or "parse" in msg or "invalid" in msg:
            return "E_SELECTION_INVALID"
    
    # Config errors
    if "version" in msg and ("unsupported" in msg or "expected" in msg):
        return "E_CONFIG_VERSION"
    if "missing" in msg and ("field" in msg or "required" in msg):
        return "E_CONFIG_MISSING_FIELD"
    if "validation" in exc_type or "pydantic" in exc_type:
        return "E_CONFIG_VALIDATION"
    
    # I/O errors
    if "trajectory" in context:
        if "eof" in msg or "end of file" in msg or "truncated" in msg:
            return "E_TRAJECTORY_EOF"
        return "E_TRAJECTORY_LOAD"
    if "system" in context or "topology" in context:
        return "E_SYSTEM_LOAD"
    if "write" in msg or "permission" in msg or "output" in context:
        return "E_OUTPUT_WRITE"
    
    # Device errors
    if "cuda" in msg or "device" in msg or "gpu" in msg:
        return "E_DEVICE_UNAVAILABLE"
    
    # Analysis errors
    if "unknown analysis" in msg:
        return "E_ANALYSIS_UNKNOWN"
    if context == "analysis_spec":
        return "E_ANALYSIS_SPEC"
    
    # Default to runtime or internal
    if context in ("runtime", "execution"):
        return "E_RUNTIME_EXEC"
    
    return "E_INTERNAL"


def _native_agent_schema(target: str) -> Optional[JsonObject]:
    native = _contract._native()
    if native is None or not hasattr(native, "warp_md_agent_schema"):
        return None
    payload = native.warp_md_agent_schema(target)
    return payload if isinstance(payload, dict) else None


def _path_to_loc(path: str) -> tuple[object, ...]:
    if not path or path == "root":
        return ("root",)
    loc: list[object] = []
    token = ""
    idx = 0
    while idx < len(path):
        ch = path[idx]
        if ch == ".":
            if token:
                loc.append(token)
                token = ""
            idx += 1
            continue
        if ch == "[":
            if token:
                loc.append(token)
                token = ""
            end = path.find("]", idx)
            if end < 0:
                break
            item = path[idx + 1 : end]
            loc.append(int(item) if item.isdigit() else item)
            idx = end + 1
            continue
        token += ch
        idx += 1
    if token:
        loc.append(token)
    return tuple(loc) or ("root",)


def _raise_native_validation_error(
    title: str,
    errors: list[dict[str, Any]],
    payload: Dict[str, Any],
) -> None:
    line_errors = [
        {
            "type": "value_error",
            "loc": _path_to_loc(str(item.get("path", "root"))),
            "input": payload,
            "ctx": {"error": ValueError(str(item.get("message", "validation error")))},
        }
        for item in errors
    ] or [
        {
            "type": "value_error",
            "loc": ("root",),
            "input": payload,
            "ctx": {"error": ValueError("validation error")},
        }
    ]
    raise ValidationError.from_exception_data(title, line_errors)


def _consume_native_validation_payload(
    result: Any,
    title: str,
    payload: JsonObject,
    normalized_key: str,
) -> Optional[JsonObject]:
    if not isinstance(result, dict):
        return None
    if "valid" not in result:
        return result
    if not result.get("valid", False):
        _raise_native_validation_error(
            title,
            [item for item in result.get("errors", []) if isinstance(item, dict)],
            payload,
        )
    normalized = result.get(normalized_key)
    if isinstance(normalized, dict):
        return normalized
    return None


def _coerce_io_spec(value: Any, label: str) -> Optional[JsonObject]:
    if value is None:
        return None
    if isinstance(value, str):
        path = value.strip()
        if not path:
            raise ValueError(f"{label} path cannot be empty")
        return {"path": path}
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"{label} spec cannot be empty")
        return {str(key): value for key, value in value.items()}
    raise TypeError(f"{label} must be a path string or object")


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: AnalysisName
    out: Optional[str] = None
    device: Optional[str] = None
    chunk_frames: Optional[int] = Field(default=None, ge=1)

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("analysis name must be a string")
        name = value.strip()
        if not name:
            raise ValueError("analysis name cannot be empty")
        return _contract._resolve_analysis_name(name)

    @model_validator(mode="after")
    def _validate_required_fields(self) -> "AnalysisRequest":
        required = _ANALYSIS_REQUIRED_FIELDS.get(self.name, ())
        if not required:
            return self
        extra = self.__pydantic_extra__ or {}
        missing = []
        for key in required:
            value = extra.get(key)
            if value is None:
                missing.append(key)
                continue
            if isinstance(value, str) and not value.strip():
                missing.append(key)
        if missing:
            raise ValueError(
                f"{self.name} missing required fields: {', '.join(missing)}"
            )
        return self


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = AGENT_REQUEST_SCHEMA_VERSION
    run_id: Optional[str] = None
    system: Optional[Dict[str, Any]] = None
    topology: Optional[Dict[str, Any]] = None
    trajectory: Optional[Dict[str, Any]] = None
    traj: Optional[Dict[str, Any]] = None
    device: str = "auto"
    stream: Literal["none", "ndjson"] = "none"
    chunk_frames: Optional[int] = Field(default=None, ge=1)
    output_dir: str = "."
    checkpoint: Optional[CheckpointConfig] = None
    fail_fast: bool = True
    analyses: list[AnalysisRequest] = Field(min_length=1)

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        if value != AGENT_REQUEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported run config version: {value}; "
                f"expected {AGENT_REQUEST_SCHEMA_VERSION}"
            )
        return value

    @field_validator("system", mode="before")
    @classmethod
    def _coerce_system(cls, value: Any) -> Optional[JsonObject]:
        return _coerce_io_spec(value, "system")

    @field_validator("topology", mode="before")
    @classmethod
    def _coerce_topology(cls, value: Any) -> Optional[JsonObject]:
        return _coerce_io_spec(value, "topology")

    @field_validator("trajectory", mode="before")
    @classmethod
    def _coerce_trajectory(cls, value: Any) -> Optional[JsonObject]:
        return _coerce_io_spec(value, "trajectory")

    @field_validator("traj", mode="before")
    @classmethod
    def _coerce_traj(cls, value: Any) -> Optional[JsonObject]:
        return _coerce_io_spec(value, "traj")

    @model_validator(mode="after")
    def _cross_validate(self) -> "RunRequest":
        if self.system is not None and self.topology is not None:
            raise ValueError("specify only one of `system` or `topology`")
        if self.trajectory is not None and self.traj is not None:
            raise ValueError("specify only one of `trajectory` or `traj`")
        if self.system is None and self.topology is None:
            raise ValueError("one of `system` or `topology` is required")
        if self.trajectory is None and self.traj is None:
            raise ValueError("one of `trajectory` or `traj` is required")
        return self

    @property
    def system_spec(self) -> JsonObject:
        return dict(self.system or self.topology or {})

    @property
    def trajectory_spec(self) -> JsonObject:
        return dict(self.trajectory or self.traj or {})


def validate_run_request(payload: Dict[str, Any]) -> RunRequest:
    native = _contract._native()
    if native is not None and hasattr(native, "warp_md_agent_validate_request"):
        normalized = _consume_native_validation_payload(
            native.warp_md_agent_validate_request(json.dumps(payload), False),
            "RunRequest",
            payload,
            "normalized_request",
        )
        if isinstance(normalized, dict):
            return RunRequest.model_validate(normalized)
    return RunRequest.model_validate(payload)


def run_request_json_schema() -> Dict[str, Any]:
    native_schema = _native_agent_schema("request")
    if native_schema is not None:
        return native_schema
    return RunRequest.model_json_schema()


class ArtifactMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    format: str
    bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)
    kind: Optional[str] = None  # timeseries, histogram, grid, profile, table, artifact
    fields: Optional[List[str]] = None  # Named arrays/columns
    description: Optional[str] = None  # Human-readable artifact meaning from the contract
    units: Optional[Dict[str, str]] = None  # Units for each field
    preview_stats: Optional[Dict[str, object]] = None  # n_frames, n_bins, min, max, etc.


class RunResultEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    analysis: str
    out: str
    status: Literal["ok", "dry_run"]
    artifact: Optional[ArtifactMetadata] = None


class RunErrorDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    field: str
    message: str
    context: Optional[Dict[str, object]] = None


class RunErrorPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: ErrorCode
    message: str
    context: Dict[str, object] = Field(default_factory=dict)
    details: Optional[object | list[RunErrorDetail]] = None
    traceback: Optional[str] = None


class RunSuccessEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = AGENT_RESULT_SCHEMA_VERSION
    status: Literal["ok", "dry_run"]
    exit_code: Literal[0]
    run_id: Optional[str] = None
    output_dir: Optional[str] = None
    system: Optional[Dict[str, Any]] = None
    trajectory: Optional[Dict[str, Any]] = None
    analysis_count: int = Field(ge=0)
    started_at: str
    finished_at: str
    elapsed_ms: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    results: list[RunResultEntry] = Field(default_factory=list)


class RunErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = AGENT_RESULT_SCHEMA_VERSION
    status: Literal["error"]
    exit_code: int = Field(ge=1)
    run_id: Optional[str] = None
    output_dir: Optional[str] = None
    system: Optional[Dict[str, Any]] = None
    trajectory: Optional[Dict[str, Any]] = None
    analysis_count: int = Field(ge=0)
    started_at: str
    finished_at: str
    elapsed_ms: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    results: list[RunResultEntry] = Field(default_factory=list)
    error: RunErrorPayload


RunEnvelope = Union[RunSuccessEnvelope, RunErrorEnvelope]


class RunStartedEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event: Literal["run_started"]
    run_id: Optional[str] = None
    config_path: str
    dry_run: bool
    analysis_count: int = Field(ge=0)
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    progress_pct: float = Field(ge=0.0, le=100.0)
    eta_ms: Optional[int] = Field(default=None, ge=0)


class AnalysisStartedEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event: Literal["analysis_started"]
    index: int = Field(ge=0)
    analysis: str
    out: str
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    progress_pct: float = Field(ge=0.0, le=100.0)
    eta_ms: Optional[int] = Field(default=None, ge=0)


class AnalysisCompletedEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event: Literal["analysis_completed"]
    index: int = Field(ge=0)
    analysis: str
    status: Literal["ok", "dry_run"]
    out: str
    timing_ms: Optional[int] = Field(default=None, ge=0)
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    progress_pct: float = Field(ge=0.0, le=100.0)
    eta_ms: Optional[int] = Field(default=None, ge=0)


class AnalysisFailedEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event: Literal["analysis_failed"]
    index: Optional[int] = Field(default=None, ge=0)
    analysis: Optional[str] = None
    error: RunErrorPayload
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    progress_pct: float = Field(ge=0.0, le=100.0)
    eta_ms: Optional[int] = Field(default=None, ge=0)


class RunCompletedEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event: Literal["run_completed"]
    final_envelope: RunSuccessEnvelope


class RunFailedEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event: Literal["run_failed"]
    final_envelope: RunErrorEnvelope


RunEvent = Union[
    RunStartedEvent,
    AnalysisStartedEvent,
    CheckpointEvent,
    AnalysisCompletedEvent,
    AnalysisFailedEvent,
    RunCompletedEvent,
    RunFailedEvent,
]


def run_result_json_schema() -> Dict[str, Any]:
    native_schema = _native_agent_schema("result")
    if native_schema is not None:
        return native_schema
    adapter = TypeAdapter(RunEnvelope)
    adapter.rebuild(force=True)
    return adapter.json_schema()


def run_event_json_schema() -> Dict[str, Any]:
    native_schema = _native_agent_schema("event")
    if native_schema is not None:
        return native_schema
    adapter = TypeAdapter(RunEvent)
    adapter.rebuild(force=True)
    return adapter.json_schema()


def validate_run_result_payload(payload: JsonObject) -> JsonObject:
    native = _contract._native()
    if native is not None and hasattr(native, "warp_md_agent_validate_result"):
        validated = _consume_native_validation_payload(
            native.warp_md_agent_validate_result(json.dumps(payload)),
            "RunEnvelope",
            payload,
            "normalized_result",
        )
        if isinstance(validated, dict):
            return validated
    adapter = TypeAdapter(RunEnvelope)
    adapter.rebuild(force=True)
    validated = adapter.validate_python(payload)
    return validated.model_dump(mode="python", exclude_none=True)


def validate_run_event_payload(payload: JsonObject) -> JsonObject:
    native = _contract._native()
    if native is not None and hasattr(native, "warp_md_agent_validate_event"):
        validated = _consume_native_validation_payload(
            native.warp_md_agent_validate_event(json.dumps(payload)),
            "RunEvent",
            payload,
            "normalized_event",
        )
        if isinstance(validated, dict):
            return validated
    adapter = TypeAdapter(RunEvent)
    adapter.rebuild(force=True)
    validated = adapter.validate_python(payload)
    return validated.model_dump(mode="python", exclude_none=True)


def render_agent_schema(target: str = "request", fmt: str = "json") -> str:
    target_l = target.lower()
    if target_l == "request":
        schema = run_request_json_schema()
    elif target_l == "result":
        schema = run_result_json_schema()
    elif target_l == "event":
        schema = run_event_json_schema()
    else:
        raise ValueError("schema target must be 'request', 'result', or 'event'")

    fmt_l = fmt.lower()
    if fmt_l == "json":
        return json.dumps(schema, indent=2)
    if fmt_l in ("yaml", "yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YAML schema output requires PyYAML installed") from exc
        return yaml.safe_dump(schema, sort_keys=False)
    raise ValueError("schema format must be 'json' or 'yaml'")


def render_run_request_schema(fmt: str = "json") -> str:
    return render_agent_schema(target="request", fmt=fmt)
