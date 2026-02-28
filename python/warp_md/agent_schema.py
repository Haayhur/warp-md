from __future__ import annotations

import json
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


AGENT_REQUEST_SCHEMA_VERSION = "warp-md.agent.v1"
AGENT_RESULT_SCHEMA_VERSION = AGENT_REQUEST_SCHEMA_VERSION

# Structured error codes for agent consumption
ErrorCode = Literal[
    # Validation errors (exit code 2)
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
    "E_OUTPUT_WRITE",           # Failed to write output file
    "E_DEVICE_UNAVAILABLE",     # Requested device not available
    
    # Internal errors (exit code 5)
    "E_INTERNAL",               # Unexpected internal error
]

AnalysisName = Literal[
    "rg",
    "rmsd",
    "msd",
    "rotacf",
    "conductivity",
    "dielectric",
    "dipole_alignment",
    "ion_pair_correlation",
    "structure_factor",
    "water_count",
    "equipartition",
    "hbond",
    "rdf",
    "end_to_end",
    "contour_length",
    "chain_rg",
    "bond_length_distribution",
    "bond_angle_distribution",
    "persistence_length",
    "docking",
    # New analyses
    "dssp",
    "diffusion",
    "pca",
    "rmsf",
    "density",
    "native_contacts",
    # Additional analyses
    "volmap",
    "surf",
    "molsurf",
    "watershell",
    "tordiff",
    "projection",
    # High priority analyses
    "gist",
    "nmr",
    "jcoupling",
]

_ANALYSIS_REQUIRED_FIELDS: Dict[str, tuple[str, ...]] = {
    "rg": ("selection",),
    "rmsd": ("selection",),
    "msd": ("selection",),
    "rotacf": ("selection",),
    "conductivity": ("selection", "charges", "temperature"),
    "dielectric": ("selection", "charges"),
    "dipole_alignment": ("selection", "charges"),
    "ion_pair_correlation": ("selection", "rclust_cat", "rclust_ani"),
    "structure_factor": ("selection", "bins", "r_max", "q_bins", "q_max"),
    "water_count": ("water_selection", "center_selection", "box_unit", "region_size"),
    "equipartition": ("selection",),
    "hbond": ("donors", "acceptors", "dist_cutoff"),
    "rdf": ("sel_a", "sel_b", "bins", "r_max"),
    "end_to_end": ("selection",),
    "contour_length": ("selection",),
    "chain_rg": ("selection",),
    "bond_length_distribution": ("selection", "bins", "r_max"),
    "bond_angle_distribution": ("selection", "bins"),
    "persistence_length": ("selection",),
    "docking": ("receptor_mask", "ligand_mask"),
    # New analyses (optional params have no required fields)
    "dssp": (),
    "diffusion": (),
    "pca": ("mask",),
    "rmsf": (),
    "density": (),
    "native_contacts": (),
    # Additional analyses
    "volmap": (),
    "surf": (),
    "molsurf": (),
    "watershell": ("solute_mask",),
    "tordiff": ("mask",),
    "projection": ("mask",),
    # High priority analyses
    "gist": ("solute", "solvent"),
    "nmr": ("selection",),
    "jcoupling": ("dihedrals",),
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


def _coerce_io_spec(value: Any, label: str) -> Optional[Dict[str, Any]]:
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
        return dict(value)
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
        name = value.strip().replace("-", "_")
        if not name:
            raise ValueError("analysis name cannot be empty")
        return name

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
    def _coerce_system(cls, value: Any) -> Optional[Dict[str, Any]]:
        return _coerce_io_spec(value, "system")

    @field_validator("topology", mode="before")
    @classmethod
    def _coerce_topology(cls, value: Any) -> Optional[Dict[str, Any]]:
        return _coerce_io_spec(value, "topology")

    @field_validator("trajectory", mode="before")
    @classmethod
    def _coerce_trajectory(cls, value: Any) -> Optional[Dict[str, Any]]:
        return _coerce_io_spec(value, "trajectory")

    @field_validator("traj", mode="before")
    @classmethod
    def _coerce_traj(cls, value: Any) -> Optional[Dict[str, Any]]:
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
    def system_spec(self) -> Dict[str, Any]:
        return dict(self.system or self.topology or {})

    @property
    def trajectory_spec(self) -> Dict[str, Any]:
        return dict(self.trajectory or self.traj or {})


def validate_run_request(payload: Dict[str, Any]) -> RunRequest:
    return RunRequest.model_validate(payload)


def run_request_json_schema() -> Dict[str, Any]:
    return RunRequest.model_json_schema()


class ArtifactMetadata(BaseModel):
    path: str
    format: str
    bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)


class RunResultEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    analysis: str
    out: str
    status: Literal["ok", "dry_run"]
    artifact: Optional[ArtifactMetadata] = None


class RunErrorPayload(BaseModel):
    code: str
    message: str
    context: Dict[str, Any]
    details: Optional[Any] = None
    traceback: Optional[str] = None


class RunSuccessEnvelope(BaseModel):
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
    event: Literal["analysis_started"]
    index: int = Field(ge=0)
    analysis: str
    out: str
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    progress_pct: float = Field(ge=0.0, le=100.0)
    eta_ms: Optional[int] = Field(default=None, ge=0)


class AnalysisCompletedEvent(BaseModel):
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
    event: Literal["analysis_failed"]
    index: Optional[int] = Field(default=None, ge=0)
    analysis: Optional[str] = None
    error: RunErrorPayload
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    progress_pct: float = Field(ge=0.0, le=100.0)
    eta_ms: Optional[int] = Field(default=None, ge=0)


class RunCompletedEvent(BaseModel):
    event: Literal["run_completed"]
    final_envelope: RunSuccessEnvelope


class RunFailedEvent(BaseModel):
    event: Literal["run_failed"]
    final_envelope: RunErrorEnvelope


RunEvent = Union[
    RunStartedEvent,
    AnalysisStartedEvent,
    AnalysisCompletedEvent,
    AnalysisFailedEvent,
    RunCompletedEvent,
    RunFailedEvent,
]


def run_result_json_schema() -> Dict[str, Any]:
    return TypeAdapter(RunEnvelope).json_schema()


def run_event_json_schema() -> Dict[str, Any]:
    return TypeAdapter(RunEvent).json_schema()


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
