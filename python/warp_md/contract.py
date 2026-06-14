"""Analysis contract metadata and validation for agent consumers.

This module provides:
- Complete analysis metadata registry
- Request validation with structured errors
- Plan schema discovery
- Request normalization
- Selection linting
- Capabilities fingerprint
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import traj_py
from . import _agent_contract_snapshot as _AGENT_CONTRACT_SNAPSHOT
from ._json_types import JsonObject
from .contract_constants import AGENT_REQUEST_SCHEMA_VERSION


# Semantic field types for machine-readable contracts
SemanticType = Literal[
    "selection",    # Atom selection string
    "mask",         # Atom mask string (same as selection but semantic distinction)
    "path",         # File path
    "integer",      # Integer value
    "float",        # Floating point value
    "boolean",      # True/False flag
    "charges",      # Charge specification (by_atom, by_resname, by_name)
    "vector",       # Vector/tuple of numbers
    "string",       # Generic string
]


# Artifact semantic kinds
ArtifactKind = Literal[
    "timeseries",   # Time-indexed data
    "histogram",    # Binned distribution
    "grid",         # 3D grid data
    "profile",      # 1D profile (density, etc.)
    "table",        # Tabular data
    "artifact",     # Generic artifact
]


@dataclass
class FieldSpec:
    """Metadata for a single analysis field."""
    type: str  # "string", "integer", "float", "boolean", "array"
    semantic_type: SemanticType
    description: str = ""
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: Optional[str] = None
    choices: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "type": self.type,
            "semantic_type": self.semantic_type,
        }
        if self.description:
            d["description"] = self.description
        if self.default is not None:
            d["default"] = self.default
        if self.minimum is not None:
            d["minimum"] = self.minimum
        if self.maximum is not None:
            d["maximum"] = self.maximum
        if self.unit:
            d["unit"] = self.unit
        if self.choices:
            d["choices"] = self.choices
        return d


@dataclass
class ArtifactSpec:
    """Output artifact metadata."""
    kind: ArtifactKind
    format: str  # npz, json, csv, etc.
    fields: List[str] = field(default_factory=list)
    description: str = ""
    plot_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    companions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {"kind": self.kind, "format": self.format}
        if self.fields:
            d["fields"] = self.fields
        if self.description:
            d["description"] = self.description
        if self.plot_recommendations:
            d["plot_recommendations"] = self.plot_recommendations
        if self.companions:
            d["companions"] = self.companions
        return d


@dataclass
class InputRequirements:
    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    requires_box: bool = False
    requires_velocities: bool = False
    requires_charges: bool = False
    requires_selections: bool = False
    supports_no_trajectory: bool = False
    selection_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": list(self.required),
            "optional": list(self.optional),
            "requires_box": self.requires_box,
            "requires_velocities": self.requires_velocities,
            "requires_charges": self.requires_charges,
            "requires_selections": self.requires_selections,
            "supports_no_trajectory": self.supports_no_trajectory,
            "selection_fields": list(self.selection_fields),
        }


@dataclass
class AnalysisContract:
    """Complete contract metadata for a single analysis."""
    name: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, FieldSpec] = field(default_factory=dict)
    outputs: List[ArtifactSpec] = field(default_factory=list)
    input_requirements: InputRequirements = field(default_factory=InputRequirements)
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "aliases": self.aliases,
            "description": self.description,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "fields": {k: v.to_dict() for k, v in self.field_types.items()},
            "outputs": [o.to_dict() for o in self.outputs],
            "input_requirements": self.input_requirements.to_dict(),
            "tags": self.tags,
            "examples": self.examples,
        }


# Rust-native catalog is canonical. Python only reconstructs helper views from the
# native payload; the Rust catalog is the single source of truth.


class _CatalogFieldPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = "string"
    semantic_type: SemanticType = "string"
    description: str = ""
    default: object = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: Optional[str] = None
    choices: Optional[List[str]] = None


class _CatalogArtifactPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: ArtifactKind = "artifact"
    format: str = ""
    fields: List[str] = Field(default_factory=list)
    description: str = ""
    plot_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    companions: List[Dict[str, Any]] = Field(default_factory=list)


class _CatalogAnalysisPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    aliases: List[str] = Field(default_factory=list)
    description: str = ""
    required_fields: List[str] = Field(default_factory=list)
    optional_fields: List[str] = Field(default_factory=list)
    fields: Dict[str, _CatalogFieldPayload] = Field(default_factory=dict)
    outputs: List[_CatalogArtifactPayload] = Field(default_factory=list)
    input_requirements: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    examples: List[Dict[str, object]] = Field(default_factory=list)


class _ContractCatalogPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = AGENT_REQUEST_SCHEMA_VERSION
    cli_to_analysis: Dict[str, str] = Field(default_factory=dict)
    analysis_shared_fields: List[str] = Field(default_factory=list)
    analyses: List[_CatalogAnalysisPayload] = Field(default_factory=list)


def _native() -> Any:
    if traj_py is None:
        return None
    required = (
        "warp_md_agent_contract_catalog",
        "warp_md_agent_plan_schema",
        "warp_md_agent_capabilities",
        "warp_md_agent_generate_template",
        "warp_md_agent_normalize_request",
        "warp_md_agent_validate_request",
        "warp_md_agent_lint_selection",
        "warp_md_agent_suggest_analyses",
    )
    if any(not hasattr(traj_py, name) for name in required):
        return None
    return traj_py


def _field_spec_from_payload(payload: _CatalogFieldPayload) -> FieldSpec:
    return FieldSpec(
        type=payload.type,
        semantic_type=payload.semantic_type,
        description=payload.description,
        default=payload.default,
        minimum=payload.minimum,
        maximum=payload.maximum,
        unit=payload.unit,
        choices=payload.choices,
    )


def _field_spec_from_dict(payload: Dict[str, Any]) -> FieldSpec:
    return _field_spec_from_payload(_CatalogFieldPayload.model_validate(payload))


def _artifact_spec_from_payload(payload: _CatalogArtifactPayload) -> ArtifactSpec:
    plot_recommendations = payload.plot_recommendations or _default_plot_recommendations(
        payload.kind,
        payload.fields,
        payload.description,
    )
    companions = payload.companions or _default_companion_specs(payload.format, payload.fields)
    return ArtifactSpec(
        kind=payload.kind,
        format=payload.format,
        fields=payload.fields,
        description=payload.description,
        plot_recommendations=plot_recommendations,
        companions=companions,
    )


def _default_plot_recommendations(
    kind: ArtifactKind,
    fields: List[str],
    description: str,
) -> List[Dict[str, Any]]:
    title = description or ""
    if kind in ("timeseries", "histogram", "profile") and len(fields) >= 2:
        return [
            {
                "plot_type": "line",
                "x": _plot_axis(fields[0]),
                "y": _plot_axis(field),
                "title": title or field.replace("_", " ").title(),
            }
            for field in fields[1:]
            if field != "..."
        ]
    if kind == "table" and len(fields) >= 2:
        return [
            {
                "plot_type": "bar",
                "x": _plot_axis(fields[0]),
                "y": _plot_axis(fields[1]),
                "title": title or fields[1].replace("_", " ").title(),
            }
        ]
    if kind == "grid" and fields:
        return [
            {
                "plot_type": "volume_grid",
                "z": _plot_axis(fields[0]),
                "title": title or fields[0].replace("_", " ").title(),
            }
        ]
    if kind == "artifact" and len(fields) >= 2:
        return [
            {
                "plot_type": "line",
                "x": _plot_axis(fields[0]),
                "y": _plot_axis(fields[1]),
                "title": title or fields[1].replace("_", " ").title(),
            }
        ]
    return []


def _default_companion_specs(fmt: str, fields: List[str]) -> List[Dict[str, Any]]:
    if fmt != "npz":
        return []
    return [
        {"format": "json", "role": "npz_companion_manifest", "fields": list(fields)},
        {"format": "csv", "role": "array_table", "fields": list(fields)},
    ]


def _plot_axis(field: str) -> Dict[str, str]:
    axis = {"field": field}
    unit = _field_units(field)
    if unit:
        axis["units"] = unit
    return axis


def _field_units(field: str) -> Optional[str]:
    if field.endswith("_ps"):
        return "ps"
    if field.endswith("_nm2") or field.endswith("_nm_2"):
        return "nm^2"
    if field.endswith("_nm") or field in {"position", "distance_nm"}:
        return "nm"
    if field.endswith("_a3"):
        return "angstrom^3"
    if field.endswith("_hz"):
        return "Hz"
    if field.endswith("_kJ_per_mol"):
        return "kJ/mol"
    if field.endswith("_S_per_cm"):
        return "S/cm"
    if field.endswith("_g_cm3"):
        return "g/cm^3"
    if field in {
        "probability",
        "gr",
        "acf",
        "correlation",
        "q_value",
        "structure_factor",
        "cos_theta",
    }:
        return "dimensionless"
    return None


_BOX_REQUIRED_ANALYSES = set(_AGENT_CONTRACT_SNAPSHOT.BOX_REQUIRED_ANALYSES)
_VELOCITY_REQUIRED_ANALYSES = set(_AGENT_CONTRACT_SNAPSHOT.VELOCITY_REQUIRED_ANALYSES)
_NO_TRAJECTORY_ANALYSES: set[str] = set()
_EXTERNAL_TABLE_ANALYSES = {"energy_table", "state_table"}
_ENERGY_TABLE_COLUMNS = ("energy", "potential", "potential_energy")
_STATE_TABLE_COLUMNS = ("temperature", "density")
_SOLVENT_SELECTION_CANDIDATES = ("resname SOL", "resname WAT", "resname HOH")
_POLYMER_SELECTION_CANDIDATES = ("polymer", "not resname SOL", "all")
ERROR_CODES = tuple(_AGENT_CONTRACT_SNAPSHOT.ERROR_CODES)

ANALYSIS_BUNDLES: Dict[str, Dict[str, Any]] = dict(_AGENT_CONTRACT_SNAPSHOT.ANALYSIS_BUNDLES)


def _analysis_contract_from_payload(payload: _CatalogAnalysisPayload) -> AnalysisContract:
    fields = {
        name: _field_spec_from_payload(spec)
        for name, spec in payload.fields.items()
    }
    return AnalysisContract(
        name=payload.name,
        aliases=payload.aliases,
        description=payload.description,
        required_fields=payload.required_fields,
        optional_fields=payload.optional_fields,
        field_types=fields,
        outputs=[_artifact_spec_from_payload(spec) for spec in payload.outputs],
        input_requirements=_input_requirements_from_payload(payload, fields),
        tags=payload.tags,
        examples=payload.examples,
    )


def _input_requirements_from_payload(
    payload: _CatalogAnalysisPayload,
    fields: Dict[str, FieldSpec],
) -> InputRequirements:
    raw = payload.input_requirements or {}
    selection_fields = [
        name
        for name, spec in fields.items()
        if spec.semantic_type in ("selection", "mask")
    ]
    required = list(raw.get("required", ["topology", "trajectory"]))
    optional = list(raw.get("optional", []))
    name = payload.name
    if name in _EXTERNAL_TABLE_ANALYSES:
        required = ["energy_table" if name == "energy_table" else "state_table"]
    return InputRequirements(
        required=required,
        optional=optional,
        requires_box=bool(raw.get("requires_box", name in _BOX_REQUIRED_ANALYSES)),
        requires_velocities=bool(
            raw.get("requires_velocities", name in _VELOCITY_REQUIRED_ANALYSES)
        ),
        requires_charges=bool(
            raw.get("requires_charges", "charges" in payload.required_fields)
        ),
        requires_selections=bool(raw.get("requires_selections", bool(selection_fields))),
        supports_no_trajectory=bool(
            raw.get("supports_no_trajectory", name in _NO_TRAJECTORY_ANALYSES)
        ),
        selection_fields=list(raw.get("selection_fields", selection_fields)),
    )


def _parse_contract_catalog(payload: object) -> _ContractCatalogPayload:
    return _ContractCatalogPayload.model_validate(payload)


def _load_contract_catalog() -> _ContractCatalogPayload:
    native = _native()
    if native is None:
        raise RuntimeError(
            "warp-md agent contract catalog requires native traj_py bindings; "
            "run `maturin develop` or install a wheel with bindings"
        )
    payload = native.warp_md_agent_contract_catalog()
    if not isinstance(payload, dict) or payload.get("schema_version") != AGENT_REQUEST_SCHEMA_VERSION:
        raise RuntimeError("native warp-md agent contract catalog is missing or has an invalid schema version")
    return _parse_contract_catalog(payload)


_CONTRACT_CATALOG = _load_contract_catalog()
CLI_TO_ANALYSIS: Dict[str, str] = dict(_CONTRACT_CATALOG.cli_to_analysis)
ANALYSIS_METADATA: Dict[str, AnalysisContract] = {
    item.name: _analysis_contract_from_payload(item)
    for item in _CONTRACT_CATALOG.analyses
}
if not ANALYSIS_METADATA:
    raise RuntimeError("agent contract catalog is empty")

_ANALYSIS_SHARED_FIELDS = frozenset(
    str(field_name) for field_name in _CONTRACT_CATALOG.analysis_shared_fields
)


def _analysis_name_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for canonical, analysis_contract in ANALYSIS_METADATA.items():
        for alias in (canonical, canonical.replace("_", "-"), *analysis_contract.aliases):
            lookup[alias] = canonical
            lookup[alias.replace("-", "_")] = canonical
    for alias, canonical in CLI_TO_ANALYSIS.items():
        lookup[alias] = canonical
        lookup[alias.replace("-", "_")] = canonical
    return lookup


_ANALYSIS_NAME_LOOKUP = _analysis_name_lookup()


def _run_request_top_level_fields() -> frozenset[str]:
    from .agent_schema import RunRequest

    return frozenset(RunRequest.model_fields.keys())


def _resolve_analysis_name(name: str) -> str:
    """Resolve CLI name to canonical analysis name."""
    lookup_name = name.strip()
    canonical = _ANALYSIS_NAME_LOOKUP.get(lookup_name)
    if canonical:
        return canonical
    raise ValueError(f"unknown analysis: {name}")


def get_plan_schema(plan_name: str) -> Dict[str, Any]:
    """Get full contract metadata for a single analysis plan.

    Args:
        plan_name: Analysis name (CLI or canonical form)

    Returns:
        Dictionary with complete contract metadata

    Raises:
        ValueError: If analysis name is unknown
    """
    native = _native()
    if native is not None:
        try:
            payload = native.warp_md_agent_plan_schema(plan_name)
        except Exception as exc:
            raise ValueError(f"unknown plan: {plan_name}") from exc
        if isinstance(payload, dict):
            return _analysis_contract_from_payload(
                _CatalogAnalysisPayload.model_validate(payload)
            ).to_dict()
    try:
        canonical = _resolve_analysis_name(plan_name)
    except ValueError:
        raise ValueError(f"unknown plan: {plan_name}")
    contract = ANALYSIS_METADATA.get(canonical)
    if not contract:
        raise ValueError(f"unknown plan: {plan_name}")
    return contract.to_dict()


def list_all_plans(details: bool = False) -> Dict[str, Any]:
    """List all available analysis plans.

    Args:
        details: Include full metadata for each plan

    Returns:
        Dictionary with plan names or detailed contracts
    """
    if not details:
        return {"plans": sorted(ANALYSIS_METADATA.keys())}
    return {
        "plans": [contract.to_dict() for contract in ANALYSIS_METADATA.values()],
    }


# Validation result structures
class ValidationErrorDetail(BaseModel):
    """Single validation error detail."""
    code: str
    path: str
    message: str
    context: Dict[str, object] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Result of request validation."""
    schema_version: str = AGENT_REQUEST_SCHEMA_VERSION
    status: Literal["ok", "error"]
    valid: bool
    normalized_request: Optional[Dict[str, Any]] = None
    errors: List[ValidationErrorDetail] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


def _request_system_path(payload: Dict[str, Any]) -> Optional[str]:
    for field_name in ("system", "topology"):
        value = payload.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            path = value.get("path")
            if isinstance(path, str) and path.strip():
                return path.strip()
    return None


def _apply_selection_validation(
    normalized: Dict[str, Any],
    errors: List[ValidationErrorDetail],
    warnings: List[str],
) -> None:
    system_path = _request_system_path(normalized)
    for idx, analysis in enumerate(normalized.get("analyses", [])):
        if not isinstance(analysis, dict):
            continue
        analysis_name = analysis.get("name")
        if not isinstance(analysis_name, str):
            continue
        try:
            canonical = _resolve_analysis_name(analysis_name)
        except ValueError:
            continue
        contract = ANALYSIS_METADATA.get(canonical)
        if not contract:
            continue
        for field_name, field_spec in contract.field_types.items():
            if field_spec.semantic_type not in ("selection", "mask"):
                continue
            field_value = analysis.get(field_name)
            if not isinstance(field_value, str):
                continue
            path = f"analyses[{idx}].{field_name}"
            lint = lint_selection(
                field_value,
                field_type=field_spec.semantic_type,
                system_path=system_path,
            )
            if not lint.valid:
                errors.append(ValidationErrorDetail(
                    code="E_SELECTION_INVALID",
                    path=path,
                    message=lint.error or "Selection syntax error",
                ))
                continue
            warnings.extend(f"{path}: {warning}" for warning in lint.warnings)


def validate_request(
    payload: JsonObject,
    *,
    strict: bool = False,
    check_selections: bool = False,
) -> ValidationResult:
    """Validate a RunRequest against full contract.

    Args:
        payload: Raw request dictionary
        strict: If True, reject unknown fields
        check_selections: If True, validate selection syntax (requires system load)

    Returns:
        ValidationResult with status, errors, warnings, and normalized request
    """
    native = _native()
    if native is not None:
        payload_json = json.dumps(payload)
        try:
            result = native.warp_md_agent_validate_request(
                payload_json,
                strict,
                check_selections,
            )
        except TypeError:
            if not check_selections:
                result = native.warp_md_agent_validate_request(payload_json, strict)
                if isinstance(result, dict):
                    return ValidationResult.model_validate(result)
        else:
            if isinstance(result, dict):
                return ValidationResult.model_validate(result)

    from .agent_schema import validate_run_request

    errors: List[ValidationErrorDetail] = []
    warnings: List[str] = []
    normalized: Optional[JsonObject] = None

    # Step 1: Pydantic schema validation
    try:
        request = validate_run_request(payload)
        normalized = request.model_dump(mode="python")
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) if err["loc"] else "root"
            errors.append(ValidationErrorDetail(
                code="E_SCHEMA_VALIDATION",
                path=loc,
                message=err.get("msg", "validation error"),
            ))
        return ValidationResult(
            status="error",
            valid=False,
            errors=errors,
            warnings=warnings,
        )

    # Step 2: Analysis-specific validation
    for idx, analysis in enumerate(normalized.get("analyses", [])):
        analysis_name = analysis.get("name", "")
        path_prefix = f"analyses[{idx}]"

        try:
            canonical = _resolve_analysis_name(analysis_name)
        except ValueError as e:
            errors.append(ValidationErrorDetail(
                code="E_UNKNOWN_ANALYSIS",
                path=f"{path_prefix}.name",
                message=str(e),
            ))
            continue

        contract = ANALYSIS_METADATA.get(canonical)
        if not contract:
            errors.append(ValidationErrorDetail(
                code="E_MISSING_CONTRACT",
                path=f"{path_prefix}",
                message=f"No contract found for analysis: {canonical}",
            ))
            continue

        # Check required fields
        provided_fields = set(k for k in analysis.keys() if k != "name")
        required = set(contract.required_fields)
        missing = required - provided_fields

        for field in missing:
            errors.append(ValidationErrorDetail(
                code="E_REQUIRED_FIELD",
                path=f"{path_prefix}.{field}",
                message=f"{canonical} requires field '{field}'",
            ))

        # Check field types where we can
        for field_name, field_value in analysis.items():
            if field_name == "name":
                continue
            if field_name in _ANALYSIS_SHARED_FIELDS:
                continue
            field_spec = contract.field_types.get(field_name)
            if not field_spec:
                if strict:
                    errors.append(ValidationErrorDetail(
                        code="E_UNKNOWN_FIELD",
                        path=f"{path_prefix}.{field_name}",
                        message=f"Unknown field for {canonical}: {field_name}",
                    ))
                continue

            # Type checks for known field types
            if field_spec.type == "boolean" and not isinstance(field_value, bool):
                errors.append(ValidationErrorDetail(
                    code="E_FIELD_TYPE",
                    path=f"{path_prefix}.{field_name}",
                    message=f"Expected boolean, got {type(field_value).__name__}",
                ))
            elif field_spec.type == "integer" and not isinstance(field_value, int):
                errors.append(ValidationErrorDetail(
                    code="E_FIELD_TYPE",
                    path=f"{path_prefix}.{field_name}",
                    message=f"Expected integer, got {type(field_value).__name__}",
                ))
            elif field_spec.type == "float" and not isinstance(field_value, (int, float)):
                errors.append(ValidationErrorDetail(
                    code="E_FIELD_TYPE",
                    path=f"{path_prefix}.{field_name}",
                    message=f"Expected float, got {type(field_value).__name__}",
                ))
            elif field_spec.type == "array" and not isinstance(field_value, (list, tuple)):
                errors.append(ValidationErrorDetail(
                    code="E_FIELD_TYPE",
                    path=f"{path_prefix}.{field_name}",
                    message=f"Expected array, got {type(field_value).__name__}",
                ))

            # Range checks
            if isinstance(field_value, (int, float)):
                if field_spec.minimum is not None and field_value < field_spec.minimum:
                    errors.append(ValidationErrorDetail(
                        code="E_VALUE_RANGE",
                        path=f"{path_prefix}.{field_name}",
                        message=f"Value {field_value} below minimum {field_spec.minimum}",
                    ))
                if field_spec.maximum is not None and field_value > field_spec.maximum:
                    errors.append(ValidationErrorDetail(
                        code="E_VALUE_RANGE",
                        path=f"{path_prefix}.{field_name}",
                        message=f"Value {field_value} above maximum {field_spec.maximum}",
                    ))

            # Choice checks
            if field_spec.choices and field_value not in field_spec.choices:
                errors.append(ValidationErrorDetail(
                    code="E_INVALID_CHOICE",
                    path=f"{path_prefix}.{field_name}",
                    message=f"Invalid choice: {field_value}. Must be one of {field_spec.choices}",
                ))

    # Step 3: Selection syntax validation (if requested)
    if check_selections and normalized is not None:
        _apply_selection_validation(normalized, errors, warnings)

    return ValidationResult(
        status="ok" if not errors else "error",
        valid=len(errors) == 0,
        normalized_request=normalized if not errors else None,
        errors=errors,
        warnings=warnings,
    )


def normalize_request(
    payload: JsonObject,
    *,
    strip_unknown: bool = False,
) -> JsonObject:
    """Canonicalize a request by resolving aliases and filling defaults.

    Args:
        payload: Raw request dictionary
        strip_unknown: Remove fields not in contract definitions

    Returns:
        Canonicalized request dictionary
    """
    native = _native()
    if native is not None:
        payload_json = json.dumps(payload)
        normalized = native.warp_md_agent_normalize_request(payload_json, strip_unknown)
        if isinstance(normalized, dict):
            return normalized

    # Start with a copy of the payload
    import copy
    normalized = copy.deepcopy(payload)

    # Normalize field aliases (topology -> system, traj -> trajectory)
    if "topology" in normalized:
        if "system" not in normalized:
            normalized["system"] = normalized.pop("topology")
        else:
            normalized.pop("topology")
    if "traj" in normalized:
        if "trajectory" not in normalized:
            normalized["trajectory"] = normalized.pop("traj")
        else:
            normalized.pop("traj")

    if strip_unknown:
        normalized = {
            key: value
            for key, value in normalized.items()
            if key in _run_request_top_level_fields()
        }

    # Normalize analysis names and fill defaults
    normalized_analyses = []
    for analysis in normalized.get("analyses", []):
        if not isinstance(analysis, dict):
            normalized_analyses.append(analysis)
            continue

        analysis_payload = dict(analysis)
        name = analysis_payload.get("name", "")
        try:
            canonical = _resolve_analysis_name(name)
            analysis_payload["name"] = canonical
        except ValueError:
            normalized_analyses.append(analysis_payload)
            continue  # Keep original name if unknown

        # Fill default values for optional fields
        contract = ANALYSIS_METADATA.get(canonical)
        if contract:
            if strip_unknown:
                allowed_fields = {"name", *contract.field_types.keys(), *_ANALYSIS_SHARED_FIELDS}
                analysis_payload = {
                    key: value
                    for key, value in analysis_payload.items()
                    if key in allowed_fields
                }
            for field_name, field_spec in contract.field_types.items():
                if field_name not in analysis_payload and field_spec.default is not None:
                    analysis_payload[field_name] = field_spec.default

        normalized_analyses.append(analysis_payload)

    if "analyses" in normalized:
        normalized["analyses"] = normalized_analyses

    return normalized


def generate_template(
    analysis_name: str,
    *,
    fill_defaults: bool = False,
) -> JsonObject:
    """Generate a request template for a single analysis.

    Args:
        analysis_name: Analysis name
        fill_defaults: Include default values for optional fields

    Returns:
        Template request dictionary
    """
    native = _native()
    if native is not None:
        payload = native.warp_md_agent_generate_template(analysis_name, fill_defaults)
        if isinstance(payload, dict):
            return payload

    canonical = _resolve_analysis_name(analysis_name)
    contract = ANALYSIS_METADATA.get(canonical)
    if not contract:
        raise ValueError(f"unknown analysis: {analysis_name}")

    analysis_spec: JsonObject = {"name": canonical}

    # Add required fields with placeholder values
    for field in contract.required_fields:
        field_spec = contract.field_types.get(field)
        if field_spec:
            if field_spec.semantic_type in ("selection", "mask"):
                analysis_spec[field] = f"<{field}_expression>"
            elif field_spec.type == "array":
                analysis_spec[field] = []
            elif field_spec.type == "integer":
                analysis_spec[field] = field_spec.default or 0
            elif field_spec.type == "float":
                analysis_spec[field] = field_spec.default or 0.0
            elif field_spec.type == "boolean":
                analysis_spec[field] = field_spec.default or False
            elif field_spec.semantic_type == "charges":
                analysis_spec[field] = "by_resname"
            else:
                analysis_spec[field] = f"<{field}>"

    # Add optional fields with defaults if requested
    if fill_defaults:
        for field, field_spec in contract.field_types.items():
            if field in contract.optional_fields and field not in analysis_spec:
                if field_spec.default is not None:
                    analysis_spec[field] = field_spec.default

    return {
        "version": AGENT_REQUEST_SCHEMA_VERSION,
        "system": {"path": "<topology-path>"},
        "trajectory": {"path": "<trajectory-path>"},
        "analyses": [analysis_spec],
    }


def _compute_catalog_hash() -> str:
    """Compute hash of analysis catalog for versioning."""
    payload = {
        "analyses": {
            name: ANALYSIS_METADATA[name].to_dict()
            for name in sorted(ANALYSIS_METADATA.keys())
        },
        "cli_to_analysis": {
            name: CLI_TO_ANALYSIS[name]
            for name in sorted(CLI_TO_ANALYSIS.keys())
        },
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def capabilities() -> Dict[str, Any]:
    """Return capabilities/version fingerprint."""
    native = _native()
    if native is not None:
        payload = native.warp_md_agent_capabilities()
        if isinstance(payload, dict):
            payload.setdefault("cli_version", _resolve_cli_version())
            payload.setdefault("analysis_bundles", _analysis_bundles_payload())
            payload.setdefault("error_codes", list(ERROR_CODES))
            return payload
    return {
        "schema_version": AGENT_REQUEST_SCHEMA_VERSION,
        "cli_version": _resolve_cli_version(),
        "available_plans": sorted(ANALYSIS_METADATA.keys()),
        "plan_catalog_hash": _compute_catalog_hash(),
        "analysis_bundles": _analysis_bundles_payload(),
        "error_codes": list(ERROR_CODES),
        "supports_streaming": True,
        "supports_selection_linting": True,
    }


def _analysis_bundles_payload() -> List[Dict[str, Any]]:
    return [
        {"name": name, **_analysis_bundle_payload(spec)}
        for name, spec in ANALYSIS_BUNDLES.items()
    ]


def _analysis_bundle_payload(spec: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "description": str(spec.get("description", "")),
        "analyses": list(spec.get("analyses", [])),
    }
    if "external_tables" in spec:
        payload["external_tables"] = list(spec["external_tables"])
    return payload


def _analysis_bundle_specs() -> Dict[str, Dict[str, Any]]:
    native = _native()
    if native is not None:
        payload = native.warp_md_agent_capabilities()
        if isinstance(payload, dict) and isinstance(payload.get("analysis_bundles"), list):
            bundles: Dict[str, Dict[str, Any]] = {}
            for item in payload["analysis_bundles"]:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    bundles[str(item["name"])] = _analysis_bundle_payload(item)
            if bundles:
                return bundles
    return ANALYSIS_BUNDLES


def _external_input_spec(payload: JsonObject, kind: str) -> Optional[JsonObject]:
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        return None
    spec = inputs.get(kind)
    if spec is None:
        return None
    if isinstance(spec, str):
        return {"path": spec}
    if isinstance(spec, dict):
        return dict(spec)
    return None


def _normalize_column_name(name: str, index: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return base or f"column_{index + 1}"


def _dedupe_column_names(names: Iterable[str]) -> List[str]:
    seen: Dict[str, int] = {}
    output: List[str] = []
    for idx, name in enumerate(names):
        base = _normalize_column_name(name, idx)
        count = seen.get(base, 0) + 1
        seen[base] = count
        output.append(base if count == 1 else f"{base}_{count}")
    return output


def _looks_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _split_table_line(line: str, delimiter: Optional[str]) -> List[str]:
    if delimiter is None:
        return [part for part in line.strip().split() if part]
    return [part.strip() for part in next(csv.reader([line], delimiter=delimiter))]


class UnsupportedExternalTableFormat(ValueError):
    pass


def _infer_external_table_format(path: Path, explicit: Optional[str]) -> str:
    if explicit:
        fmt = explicit.lower().lstrip(".")
    else:
        fmt = path.suffix.lower().lstrip(".")
    if fmt not in {"csv", "tsv", "xvg"}:
        raise UnsupportedExternalTableFormat(
            f"unsupported external table format: {fmt or path.suffix}"
        )
    return fmt


def parse_external_table(spec: Union[str, JsonObject]) -> JsonObject:
    """Lightly inspect a CSV, TSV, or XVG table and return normalized columns."""
    table_spec = {"path": spec} if isinstance(spec, str) else dict(spec)
    raw_path = table_spec.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("external table path is required")
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"external table not found: {path}")
    fmt = _infer_external_table_format(path, table_spec.get("format"))
    text_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if fmt == "xvg":
        lines = [
            line
            for line in text_lines
            if line.strip() and not line.lstrip().startswith(("#", "@"))
        ]
        delimiter: Optional[str] = None
    else:
        lines = [line for line in text_lines if line.strip()]
        delimiter = "\t" if fmt == "tsv" else ","
        if fmt == "csv" and lines:
            sample = lines[0]
            delimiter = "\t" if "\t" in sample and "," not in sample else ","
    if not lines:
        raise ValueError(f"external table has no data rows: {path}")

    first = _split_table_line(lines[0], delimiter)
    header = any(not _looks_number(part) for part in first)
    if header:
        columns = _dedupe_column_names(first)
        row_count = max(0, len(lines) - 1)
    else:
        columns = _dedupe_column_names(f"column_{idx + 1}" for idx in range(len(first)))
        row_count = len(lines)
    return {
        "path": str(path),
        "format": fmt,
        "columns": columns,
        "row_count": row_count,
    }


def _table_has_any_column(table: JsonObject, candidates: Iterable[str]) -> bool:
    columns = set(str(column) for column in table.get("columns", []))
    return any(column in columns for column in candidates)


def _first_table_column(table: JsonObject, candidates: Iterable[str]) -> Optional[str]:
    columns = set(str(column) for column in table.get("columns", []))
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _external_table_column_errors(tables: Dict[str, JsonObject]) -> List[JsonObject]:
    errors: List[JsonObject] = []
    energy = tables.get("energy_table")
    if energy is not None and not _table_has_any_column(energy, _ENERGY_TABLE_COLUMNS):
        errors.append({
            "input": "energy_table",
            "code": "E_EXTERNAL_TABLE_COLUMN",
            "message": f"energy_table requires one of: {', '.join(_ENERGY_TABLE_COLUMNS)}",
            "columns": list(energy.get("columns", [])),
        })
    state = tables.get("state_table")
    if state is not None and not _table_has_any_column(state, _STATE_TABLE_COLUMNS):
        errors.append({
            "input": "state_table",
            "code": "E_EXTERNAL_TABLE_COLUMN",
            "message": f"state_table requires one of: {', '.join(_STATE_TABLE_COLUMNS)}",
            "columns": list(state.get("columns", [])),
        })
    return errors


def _analysis_skip(analysis: str, code: str, reason: str) -> JsonObject:
    return {"analysis": analysis, "code": code, "reason": reason}


def _request_system_spec(payload: JsonObject) -> Optional[JsonObject]:
    spec = payload.get("system", payload.get("topology"))
    if isinstance(spec, str):
        return {"path": spec}
    if isinstance(spec, dict):
        return dict(spec)
    return None


def _request_trajectory_spec(payload: JsonObject) -> Optional[JsonObject]:
    spec = payload.get("trajectory", payload.get("traj"))
    if isinstance(spec, str):
        return {"path": spec}
    if isinstance(spec, dict):
        return dict(spec)
    return None


def _input_path_exists(spec: Optional[JsonObject]) -> bool:
    if not spec:
        return False
    path = spec.get("path")
    return isinstance(path, str) and Path(path).exists()


def _required_input_missing(
    name: str,
    system_spec: Optional[JsonObject],
    trajectory_spec: Optional[JsonObject],
    tables: Dict[str, JsonObject],
) -> bool:
    if name == "topology":
        return not _input_path_exists(system_spec)
    if name == "trajectory":
        return not _input_path_exists(trajectory_spec)
    if name in {"energy_table", "state_table"}:
        return name not in tables
    return False


def _selection_is_valid(expr: str, system_path: Optional[str]) -> bool:
    if not expr:
        return False
    lint = lint_selection(expr, system_path=system_path)
    if not lint.valid:
        return False
    if lint.matched_atoms is not None and lint.matched_atoms <= 0:
        return False
    return True


def _first_valid_selection(candidates: Iterable[str], system_path: Optional[str]) -> Optional[str]:
    if not system_path:
        return None
    for expr in candidates:
        if _selection_is_valid(expr, system_path):
            return expr
    return None


def _default_analysis_spec(
    name: str,
    bundle: str,
    payload: JsonObject,
) -> tuple[Optional[JsonObject], Optional[JsonObject]]:
    system_spec = _request_system_spec(payload)
    system_path = str(system_spec.get("path")) if system_spec and system_spec.get("path") else None

    def need_valid_selection(expr: str) -> Optional[str]:
        return expr if _selection_is_valid(expr, system_path) else None

    if name in {"rg", "rmsd", "msd"}:
        selection = "protein" if bundle == "protein_md_report" else "all"
        if not need_valid_selection(selection):
            return None, _analysis_skip(name, "E_SELECTION_EMPTY", f"default selection `{selection}` did not match")
        spec: JsonObject = {"name": name, "selection": selection}
        if name == "rmsd":
            spec.update({"reference": 0, "align": True})
        return spec, None

    if name == "rdf":
        if bundle == "solvent_ion_report":
            solvent = _first_valid_selection(_SOLVENT_SELECTION_CANDIDATES, system_path)
            if not solvent:
                return None, _analysis_skip(name, "E_SELECTION_EMPTY", "no supported solvent selection matched")
            return {"name": "rdf", "sel_a": solvent, "sel_b": solvent, "bins": 200, "r_max": 1.0}, None
        if not need_valid_selection("all"):
            return None, _analysis_skip(name, "E_SELECTION_EMPTY", "default selection `all` did not match")
        return {"name": "rdf", "sel_a": "all", "sel_b": "all", "bins": 200, "r_max": 1.0}, None

    if name == "density":
        if not need_valid_selection("all"):
            return None, _analysis_skip(name, "E_SELECTION_EMPTY", "default selection `all` did not match")
        return {"name": "density", "mask": "all"}, None

    if name == "dssp":
        selection = need_valid_selection("protein")
        if not selection:
            return None, _analysis_skip(name, "E_SELECTION_EMPTY", "default selection `protein` did not match")
        return {"name": "dssp", "mask": selection}, None

    if name in {"water_count", "watershell"}:
        solvent = _first_valid_selection(_SOLVENT_SELECTION_CANDIDATES, system_path)
        if not solvent:
            return None, _analysis_skip(name, "E_SELECTION_EMPTY", "no supported solvent selection matched")
        spec = {"name": name}
        if name == "water_count":
            spec.update({
                "water_selection": solvent,
                "center_selection": "all",
                "box_unit": 0.1,
                "region_size": [1.0, 1.0, 1.0],
            })
        if name == "watershell":
            solute = need_valid_selection("protein")
            if not solute:
                return None, _analysis_skip(name, "E_SELECTION_EMPTY", "default solute selection `protein` did not match")
            spec.update({"solute_mask": solute, "solvent_mask": solvent})
        return spec, None

    if name in {"conductivity", "dielectric"}:
        return None, _analysis_skip(name, "E_BUNDLE_PARTIAL", f"{name} requires charge metadata or explicit parameters")

    if name in {"hbond", "native_contacts"}:
        return None, _analysis_skip(name, "E_BUNDLE_PARTIAL", f"{name} needs analysis-specific selections/reference data")

    if name in {"chain_rg", "end_to_end", "contour_length", "persistence_length", "bondi_ffv"}:
        selection = _first_valid_selection(_POLYMER_SELECTION_CANDIDATES, system_path)
        if not selection:
            return None, _analysis_skip(name, "E_SELECTION_EMPTY", "no conservative polymer selection matched")
        return {"name": name, "selection": selection}, None

    return None, _analysis_skip(name, "E_BUNDLE_PARTIAL", "no conservative default available")


def _external_plot_recommendations(tables: Dict[str, JsonObject]) -> List[JsonObject]:
    plots: List[JsonObject] = []
    energy = tables.get("energy_table")
    energy_column = _first_table_column(energy, _ENERGY_TABLE_COLUMNS) if energy else None
    if energy and energy_column:
        plots.append({
            "source_input": "energy_table",
            "plot_type": "line",
            "x": {"field": "time" if "time" in energy.get("columns", []) else energy["columns"][0]},
            "y": {"field": energy_column},
            "title": "Potential Energy",
        })
    state = tables.get("state_table")
    for field, title in (("temperature", "Temperature"), ("density", "Density")):
        if state and field in state.get("columns", []):
            plots.append({
                "source_input": "state_table",
                "plot_type": "line",
                "x": {"field": "time" if "time" in state.get("columns", []) else state["columns"][0]},
                "y": {"field": field},
                "title": title,
            })
    return plots


def _inspect_external_tables(payload: JsonObject) -> tuple[Dict[str, JsonObject], List[JsonObject]]:
    tables: Dict[str, JsonObject] = {}
    errors: List[JsonObject] = []
    for kind in ("energy_table", "state_table"):
        spec = _external_input_spec(payload, kind)
        if spec is None:
            continue
        try:
            tables[kind] = parse_external_table(spec)
        except UnsupportedExternalTableFormat as exc:
            errors.append({"input": kind, "code": "E_UNSUPPORTED_FORMAT", "message": str(exc)})
        except Exception as exc:
            errors.append({"input": kind, "code": "E_EXTERNAL_TABLE_LOAD", "message": str(exc)})
    return tables, errors


def bundle_plan(bundle: str, payload: JsonObject) -> JsonObject:
    """Expand an advertised analysis bundle into a validated request plus typed skips."""
    bundles = _analysis_bundle_specs()
    if bundle not in bundles:
        raise ValueError(f"unknown bundle: {bundle}")
    normalized = normalize_request(payload, strip_unknown=False)
    tables, table_errors = _inspect_external_tables(normalized)
    analyses: List[JsonObject] = []
    skipped: List[JsonObject] = [*table_errors, *_external_table_column_errors(tables)]
    for name in bundles[bundle].get("analyses", []):
        if name not in ANALYSIS_METADATA:
            skipped.append(_analysis_skip(name, "E_ANALYSIS_UNKNOWN", "analysis is not available"))
            continue
        spec, skip = _default_analysis_spec(name, bundle, normalized)
        if spec is not None:
            analyses.append(spec)
        elif skip is not None:
            skipped.append(skip)
    expanded = dict(normalized)
    expanded["analyses"] = analyses
    status = "ok" if not skipped else ("partial" if analyses else "error")
    return {
        "schema_version": AGENT_REQUEST_SCHEMA_VERSION,
        "bundle": bundle,
        "status": status,
        "request": expanded,
        "analyses": analyses,
        "skipped": skipped,
        "external_tables": tables,
        "plot_recommendations": _external_plot_recommendations(tables),
    }


def _frame_count(traj: Any) -> Optional[int]:
    for attr in ("n_frames", "frames"):
        value = getattr(traj, attr, None)
        if isinstance(value, int):
            return value
    try:
        return len(traj)
    except Exception:
        return None


def inspect_inputs(payload: JsonObject) -> JsonObject:
    """Lightly validate input availability and report valid/skipped analyses."""
    normalized = normalize_request(payload, strip_unknown=False)
    system_spec = _request_system_spec(normalized)
    trajectory_spec = _request_trajectory_spec(normalized)
    errors: List[JsonObject] = []
    warnings: List[str] = []
    system = None
    system_path = (
        str(system_spec.get("path"))
        if system_spec and system_spec.get("path")
        else None
    )
    if not _input_path_exists(system_spec):
        errors.append({
            "code": "E_INPUT_MISSING",
            "path": "system",
            "message": "topology/system input is missing",
        })
    else:
        try:
            from .cli_api import _load_system
            system = _load_system(system_spec or {})
        except Exception as exc:
            errors.append({"code": "E_SYSTEM_LOAD", "path": "system", "message": str(exc)})
    if not _input_path_exists(trajectory_spec):
        errors.append({
            "code": "E_INPUT_MISSING",
            "path": "trajectory",
            "message": "trajectory input is missing",
        })
    elif system is not None:
        try:
            from .cli_api import _load_trajectory
            traj = _load_trajectory(trajectory_spec or {}, system)
            n_frames = _frame_count(traj)
            if n_frames == 0:
                errors.append({
                    "code": "E_NO_FRAMES",
                    "path": "trajectory",
                    "message": "trajectory has no frames",
                })
        except Exception as exc:
            errors.append({"code": "E_TRAJECTORY_LOAD", "path": "trajectory", "message": str(exc)})
    tables, table_errors = _inspect_external_tables(normalized)
    errors.extend(table_errors)
    errors.extend(_external_table_column_errors(tables))

    valid_analyses: List[JsonObject] = []
    skipped: List[JsonObject] = []
    for idx, analysis in enumerate(normalized.get("analyses", [])):
        if not isinstance(analysis, dict):
            skipped.append({
                "analysis": None,
                "code": "E_ANALYSIS_SPEC",
                "reason": "analysis entry is not an object",
            })
            continue
        name = str(analysis.get("name", ""))
        try:
            canonical = _resolve_analysis_name(name)
        except ValueError:
            skipped.append(_analysis_skip(name, "E_ANALYSIS_UNKNOWN", "unknown analysis"))
            continue
        contract = ANALYSIS_METADATA[canonical]
        missing_inputs = [
            item
            for item in contract.input_requirements.required
            if _required_input_missing(item, system_spec, trajectory_spec, tables)
        ]
        if missing_inputs:
            skipped.append(
                _analysis_skip(
                    canonical,
                    "E_INPUT_MISSING",
                    f"missing inputs: {', '.join(missing_inputs)}",
                )
            )
            continue
        selection_errors = []
        for field_name in contract.input_requirements.selection_fields:
            value = analysis.get(field_name)
            if isinstance(value, str) and system_path and not _selection_is_valid(value, system_path):
                selection_errors.append(field_name)
        if selection_errors:
            skipped.append(
                _analysis_skip(
                    canonical,
                    "E_SELECTION_EMPTY",
                    f"empty/invalid selections: {', '.join(selection_errors)}",
                )
            )
            continue
        valid_analyses.append({"index": idx, "name": canonical})

    bundle_summaries = []
    for name in _analysis_bundle_specs():
        planned = bundle_plan(name, normalized)
        bundle_summaries.append({
            "name": name,
            "status": planned["status"],
            "valid_count": len(planned["analyses"]),
            "skipped": planned["skipped"],
        })

    return {
        "schema_version": AGENT_REQUEST_SCHEMA_VERSION,
        "status": "ok" if not errors else "error",
        "system": {"present": _input_path_exists(system_spec), "path": system_path},
        "trajectory": {
            "present": _input_path_exists(trajectory_spec),
            "path": trajectory_spec.get("path") if trajectory_spec else None,
        },
        "external_tables": tables,
        "valid_analyses": valid_analyses,
        "skipped_analyses": skipped,
        "bundles": bundle_summaries,
        "errors": errors,
        "warnings": warnings,
    }


def _resolve_cli_version() -> str:
    """Get CLI version string."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        for name in ("warp-md", "warp_md"):
            try:
                return version(name)
            except PackageNotFoundError:
                continue
    except Exception:
        pass
    return "dev"


# Selection linting

class SelectionLintResult(BaseModel):
    """Result of selection/mask linting."""
    model_config = ConfigDict(extra="forbid")

    valid: bool
    expression: str
    field_type: Literal["selection", "mask"]
    matched_atoms: Optional[int] = None
    total_atoms: Optional[int] = None
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


def lint_selection(
    expr: str,
    field_type: Literal["selection", "mask"] = "selection",
    system_path: Optional[str] = None,
) -> SelectionLintResult:
    """Validate a selection expression without running analysis.

    Args:
        expr: Selection expression to validate
        field_type: Type of field (selection or mask)
        system_path: Optional path to topology file for atom count

    Returns:
        SelectionLintResult with validation status
    """
    native = _native()
    if native is not None:
        result = native.warp_md_agent_lint_selection(expr, field_type, system_path)
        if isinstance(result, dict):
            return SelectionLintResult.model_validate(result)

    # Basic syntax validation
    if not expr or not expr.strip():
        return SelectionLintResult(
            valid=False,
            expression=expr,
            field_type=field_type,
            error="Selection expression cannot be empty",
        )

    # Check for obviously malformed expressions
    expr_stripped = expr.strip()

    # Check for unbalanced quotes
    single_quotes = expr_stripped.count("'")
    double_quotes = expr_stripped.count('"')
    if single_quotes % 2 != 0:
        return SelectionLintResult(
            valid=False,
            expression=expr,
            field_type=field_type,
            error="Unbalanced single quotes in selection",
        )
    if double_quotes % 2 != 0:
        return SelectionLintResult(
            valid=False,
            expression=expr,
            field_type=field_type,
            error="Unbalanced double quotes in selection",
        )

    # Check for unbalanced parentheses
    paren_depth = 0
    for i, char in enumerate(expr_stripped):
        if char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
        if paren_depth < 0:
            return SelectionLintResult(
                valid=False,
                expression=expr,
                field_type=field_type,
                error=f"Unbalanced parentheses at position {i}",
            )
    if paren_depth != 0:
        return SelectionLintResult(
            valid=False,
            expression=expr,
            field_type=field_type,
            error="Unbalanced parentheses in selection",
        )

    # If system path provided, try to load and count atoms
    matched_atoms = None
    total_atoms = None
    warnings_list = []

    if system_path:
        try:
            from .cli_api import _load_system
            system = _load_system({"path": system_path})
            total_atoms = len(system.atoms)

            try:
                selection = system.select(expr_stripped)
                matched_atoms = len(selection.indices)

                if matched_atoms == 0:
                    warnings_list.append("Selection matched zero atoms")

            except Exception as sel_exc:
                # Selection compilation failed
                return SelectionLintResult(
                    valid=False,
                    expression=expr,
                    field_type=field_type,
                    error=f"Selection syntax error: {sel_exc}",
                )
        except Exception as load_exc:
            fallback_count = _fallback_pdb_selection_count(system_path, expr_stripped)
            if fallback_count is None:
                warnings_list.append(f"Could not load topology for atom count: {load_exc}")
            else:
                matched_atoms, total_atoms = fallback_count
                if matched_atoms == 0:
                    warnings_list.append("Selection matched zero atoms")

    return SelectionLintResult(
        valid=True,
        expression=expr,
        field_type=field_type,
        matched_atoms=matched_atoms,
        total_atoms=total_atoms,
        warnings=warnings_list,
    )


def _fallback_pdb_selection_count(
    system_path: str,
    expr: str,
) -> Optional[tuple[int, int]]:
    path = Path(system_path)
    if path.suffix.lower() != ".pdb" or not path.exists():
        return None
    atoms: List[Dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atoms.append(
            {
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "record": line[:6].strip(),
            }
        )
    if not atoms:
        return None

    selected = set(range(len(atoms)))
    for clause in expr.split(" and "):
        parts = clause.strip().split()
        if clause.strip() in ("all", "*"):
            clause_indices = set(range(len(atoms)))
        elif clause.strip() == "protein":
            clause_indices = {
                idx for idx, atom in enumerate(atoms)
                if atom["record"] == "ATOM" and atom["resname"] not in {"HOH", "SOL", "WAT"}
            }
        elif len(parts) == 2 and parts[0] in {"name", "resname"}:
            field, value = parts
            clause_indices = {
                idx for idx, atom in enumerate(atoms)
                if atom[field] == value
            }
        else:
            return None
        selected &= clause_indices
    return len(selected), len(atoms)


# Keyword mapping for goal-to-analysis suggestions
# Maps goal keywords to analysis tags, descriptions, or direct analysis names
_GOAL_KEYWORDS: Dict[str, List[str]] = {
    # Structural analysis
    "radius": ["rg"],
    "gyration": ["rg"],
    "size": ["rg"],
    "compactness": ["rg"],
    "rmsd": ["rmsd"],
    "alignment": ["rmsd", "dipole_alignment"],
    "structure": ["rmsd", "dssp", "helix", "native_contacts"],
    "secondary": ["dssp", "helix"],
    "backbone": ["dssp", "helix", "rmsd"],
    "helix": ["helix", "helixorient", "dssp"],
    "dihedral": ["jcoupling", "tordiff"],
    "torsion": ["tordiff", "jcoupling"],

    # Dynamics
    "motion": ["msd", "diffusion"],
    "displacement": ["msd"],
    "diffusion": ["diffusion", "msd"],
    "transport": ["diffusion", "msd", "conductivity"],

    # Electrostatics
    "charge": ["dipole_alignment", "conductivity"],
    "dipole": ["dipole_alignment"],
    "polarization": ["dipole_alignment"],
    "conductivity": ["conductivity"],
    "permittivity": ["dielectric"],
    "dielectric": ["dielectric"],
    "electrostatic": ["dielectric", "dipole_alignment", "conductivity"],

    # Distribution/Correlation
    "distribution": ["rdf", "bond_length_distribution", "bond_angle_distribution"],
    "pair": ["rdf"],
    "radial": ["rdf"],
    "correlation": ["rdf", "ion_pair_correlation", "rotacf"],
    "neighbor": ["rdf", "native_contacts"],
    "contact": ["native_contacts", "hbond"],
    "hydrogen": ["hbond"],
    "water": ["water_count", "watershell", "rdf"],
    "waters": ["water_count", "watershell", "rdf"],
    "shell": ["watershell"],
    "solvation": ["watershell", "free_volume", "gist"],
    "hydration": ["watershell"],

    # Density/Spatial
    "density": ["density", "volmap"],
    "profile": ["density", "rdf"],
    "spatial": ["water_count", "free_volume", "volmap"],
    "map": ["volmap", "free_volume", "gist"],
    "grid": ["volmap", "free_volume", "gist"],
    "free": ["free_volume", "bondi_ffv"],
    "volume": ["free_volume", "bondi_ffv", "volmap"],
    "void": ["free_volume", "bondi_ffv"],
    "ffv": ["bondi_ffv"],
    "bondi": ["bondi_ffv"],

    # Energy
    "energy": ["equipartition", "gist"],
    "kinetic": ["equipartition"],

    # Time series/properties
    "temperature": ["conductivity", "dielectric", "equipartition"],

    # Docking/Binding
    "docking": ["docking"],
    "dock": ["docking"],
    "binding": ["docking"],
    "pose": ["docking"],
    "ligand": ["docking"],
    "receptor": ["docking"],

    # Misc
    "cluster": ["ion_pair_correlation"],
    "protein": ["dssp", "helix", "rmsd", "rg", "native_contacts"],
    "polymer": ["free_volume", "bondi_ffv", "rg", "chain_rg", "contour_length", "end_to_end", "persistence_length"],
}

_GOAL_PHRASES: Dict[str, List[tuple[str, float]]] = {
    "radius of gyration": [("rg", 12.0)],
    "secondary structure": [("dssp", 12.0), ("helix", 8.0)],
    "alpha helix": [("helix", 13.0), ("helixorient", 10.0), ("dssp", 7.0)],
    "helix geometry": [("helix", 13.0), ("helixorient", 10.0)],
    "mean square displacement": [("msd", 12.0)],
    "diffusion coefficient": [("diffusion", 11.0), ("msd", 8.0)],
    "pair distribution": [("rdf", 11.0)],
    "radial distribution": [("rdf", 11.0)],
    "hydrogen bond": [("hbond", 12.0)],
    "hydrogen bonds": [("hbond", 12.0)],
    "free volume": [("free_volume", 12.0), ("bondi_ffv", 9.0)],
    "fractional free volume": [("bondi_ffv", 13.0)],
    "water shell": [("watershell", 12.0)],
    "solvation shell": [("watershell", 12.0)],
    "count water": [("water_count", 12.0)],
    "count waters": [("water_count", 12.0)],
    "water count": [("water_count", 12.0)],
    "molecular docking": [("docking", 12.0)],
    "binding pose": [("docking", 12.0)],
    "ligand contacts": [("docking", 12.0)],
}

_GENERIC_GOAL_WORDS = {
    "analysis",
    "analyze",
    "around",
    "calculate",
    "compute",
    "count",
    "determine",
    "find",
    "function",
    "interface",
    "measure",
    "over",
    "series",
    "show",
    "time",
    "track",
    "want",
}

_GENERIC_TAGS = {
    "dynamic",
    "polymer",
    "protein",
    "solvent",
    "spatial",
    "structural",
    "transport",
}

_DOCKING_TRIGGER_WORDS = {"binding", "dock", "docking", "ligand", "pose", "receptor"}


def _expand_goal_words(text: str) -> set[str]:
    import re

    words = set(re.findall(r"\b\w+\b", text))
    expanded = set(words)
    for word in words:
        if len(word) > 4 and word.endswith("ies"):
            expanded.add(f"{word[:-3]}y")
        elif len(word) > 4 and word.endswith("s"):
            expanded.add(word[:-1])
    return expanded


def _tokenize_contract_text(text: str) -> set[str]:
    return _expand_goal_words(text.lower())


@dataclass
class SuggestionCandidate:
    """A single analysis suggestion."""
    name: str
    reason: str  # Why this analysis was suggested
    missing_fields: List[str]  # Required fields not yet provided
    score: float = 0.0  # Match score for ranking


@dataclass
class SuggestionResult:
    """Result of an analysis suggestion query."""
    candidates: List[SuggestionCandidate]
    goal: str
    total_analyses: int


def suggest_analyses(
    goal: str,
    *,
    provided_fields: Optional[List[str]] = None,
    top_n: int = 5,
) -> SuggestionResult:
    """Suggest analyses based on a natural language goal.

    Uses deterministic keyword matching against analysis tags, descriptions,
    and field names. No LLM or fuzzy matching - simple scoring rules.

    Args:
        goal: Natural language description of what the user wants to compute
        provided_fields: Optional list of field names already provided
        top_n: Maximum number of suggestions to return

    Returns:
        SuggestionResult with ranked candidates
    """
    native = _native()
    if native is not None:
        payload = native.warp_md_agent_suggest_analyses(goal, provided_fields, top_n)
        if isinstance(payload, dict):
            return SuggestionResult(
                candidates=[
                    SuggestionCandidate(
                        name=str(candidate["name"]),
                        reason=str(candidate.get("reason", "")),
                        missing_fields=list(candidate.get("missing_fields", [])),
                        score=float(candidate.get("score", 0.0)),
                    )
                    for candidate in payload.get("candidates", [])
                    if isinstance(candidate, dict)
                ],
                goal=str(payload.get("goal", goal)),
                total_analyses=int(payload.get("total_analyses", len(ANALYSIS_METADATA))),
            )

    provided_fields = provided_fields or []
    goal_lower = goal.lower()
    words = _expand_goal_words(goal_lower)

    scored: List[tuple[str, float, str]] = []

    for name, contract in ANALYSIS_METADATA.items():
        if name == "docking" and not any(
            trigger in goal_lower or trigger in words
            for trigger in _DOCKING_TRIGGER_WORDS
        ):
            continue

        score = 0.0
        reasons = []

        for phrase, candidates in _GOAL_PHRASES.items():
            if phrase in goal_lower:
                for candidate, weight in candidates:
                    if candidate == name or candidate in contract.aliases:
                        score += weight
                        reasons.append(f"phrase match: {phrase}")

        # Check direct name/alias matches
        if name in goal_lower:
            score += 10.0
            reasons.append(f"name match: {name}")
        for alias in contract.aliases:
            if alias in goal_lower or alias.replace("-", "_") in goal_lower:
                score += 8.0
                reasons.append(f"alias match: {alias}")

        # Check tag matches
        for tag in contract.tags:
            if tag in goal_lower:
                score += 1.0 if tag in _GENERIC_TAGS else 3.0
                reasons.append(f"tag match: {tag}")

        # Check description token matches, but ignore generic goal words.
        desc_terms = _tokenize_contract_text(contract.description)
        desc_matches = [
            word
            for word in words
            if len(word) > 3
            and word not in _GENERIC_GOAL_WORDS
            and word in desc_terms
        ]
        score += 1.5 * len(desc_matches)
        if desc_matches:
            reasons.append("description match")

        # Check keyword mapping
        for word in words:
            if word in _GOAL_KEYWORDS:
                for candidate in _GOAL_KEYWORDS[word]:
                    if candidate == name or candidate in contract.aliases:
                        score += 4.0
                        reasons.append(f"keyword match: {word}")

        # Check field name matches
        for field_name in contract.field_types.keys():
            if field_name in goal_lower:
                score += 2.0
                reasons.append(f"field match: {field_name}")

        # Check output kind matches
        for output in contract.outputs:
            if output.kind in goal_lower:
                score += 2.0
                reasons.append(f"output kind: {output.kind}")

        if score > 0:
            # Deduplicate reasons
            unique_reasons = list(dict.fromkeys(reasons))
            scored.append((name, score, ", ".join(unique_reasons)))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Build candidates
    candidates = []
    for name, score, reason in scored[:top_n]:
        contract = ANALYSIS_METADATA[name]

        # Determine missing required fields
        missing = []
        for field in contract.required_fields:
            if field not in provided_fields:
                missing.append(field)

        candidates.append(SuggestionCandidate(
            name=name,
            reason=reason,
            missing_fields=missing,
            score=score,
        ))

    return SuggestionResult(
        candidates=candidates,
        goal=goal,
        total_analyses=len(ANALYSIS_METADATA),
    )


def render_plots(payload: JsonObject, out_dir: str = "plots") -> JsonObject:
    """Render deterministic SVG plots from a result envelope."""
    native = _native()
    if native is not None and hasattr(native, "warp_md_agent_render_plots"):
        result = native.warp_md_agent_render_plots(json.dumps(payload), out_dir)
        if isinstance(result, dict):
            return result
    raise RuntimeError(
        "warp-md plot requires native traj_py bindings with "
        "warp_md_agent_render_plots; rebuild with maturin develop"
    )


__all__ = [
    "ANALYSIS_METADATA",
    "get_plan_schema",
    "list_all_plans",
    "validate_request",
    "normalize_request",
    "generate_template",
    "capabilities",
    "ANALYSIS_BUNDLES",
    "ERROR_CODES",
    "bundle_plan",
    "inspect_inputs",
    "parse_external_table",
    "lint_selection",
    "suggest_analyses",
    "render_plots",
    "ValidationResult",
    "ValidationErrorDetail",
    "SelectionLintResult",
    "SuggestionResult",
    "SuggestionCandidate",
    "FieldSpec",
    "ArtifactSpec",
    "AnalysisContract",
]
