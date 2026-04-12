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

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import traj_py
from ._contract_catalog_snapshot import CATALOG as _FALLBACK_CONTRACT_CATALOG
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

    def to_dict(self) -> Dict[str, Any]:
        d = {"kind": self.kind, "format": self.format}
        if self.fields:
            d["fields"] = self.fields
        if self.description:
            d["description"] = self.description
        return d


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
            "tags": self.tags,
            "examples": self.examples,
        }


# Rust-native catalog is canonical. Python only reconstructs helper views from the
# native payload, with a generated snapshot fallback for no-bindings environments.


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


def _field_spec_from_dict(payload: Dict[str, Any]) -> FieldSpec:
    raw_choices = payload.get("choices")
    return FieldSpec(
        type=str(payload.get("type", "string")),
        semantic_type=str(payload.get("semantic_type", "string")),  # type: ignore[arg-type]
        description=str(payload.get("description", "")),
        default=payload.get("default"),
        minimum=payload.get("minimum"),
        maximum=payload.get("maximum"),
        unit=payload.get("unit"),
        choices=(
            [str(choice) for choice in raw_choices]
            if isinstance(raw_choices, (list, tuple))
            else None
        ),
    )


def _artifact_spec_from_dict(payload: Dict[str, Any]) -> ArtifactSpec:
    return ArtifactSpec(
        kind=str(payload.get("kind", "artifact")),  # type: ignore[arg-type]
        format=str(payload.get("format", "")),
        fields=list(payload.get("fields", [])),
        description=str(payload.get("description", "")),
    )


def _analysis_contract_from_dict(payload: Dict[str, Any]) -> AnalysisContract:
    fields = payload.get("fields", {})
    outputs = payload.get("outputs", [])
    return AnalysisContract(
        name=str(payload.get("name", "")),
        aliases=list(payload.get("aliases", [])),
        description=str(payload.get("description", "")),
        required_fields=list(payload.get("required_fields", [])),
        optional_fields=list(payload.get("optional_fields", [])),
        field_types={
            str(name): _field_spec_from_dict(spec)
            for name, spec in fields.items()
            if isinstance(spec, dict)
        },
        outputs=[
            _artifact_spec_from_dict(spec)
            for spec in outputs
            if isinstance(spec, dict)
        ],
        tags=list(payload.get("tags", [])),
        examples=[example for example in payload.get("examples", []) if isinstance(example, dict)],
    )


def _load_contract_catalog() -> Dict[str, Any]:
    native = _native()
    if native is not None:
        payload = native.warp_md_agent_contract_catalog()
        if isinstance(payload, dict) and payload.get("schema_version") == AGENT_REQUEST_SCHEMA_VERSION:
            return payload
    return _FALLBACK_CONTRACT_CATALOG


_CONTRACT_CATALOG = _load_contract_catalog()
CLI_TO_ANALYSIS: Dict[str, str] = {
    str(key): str(value)
    for key, value in (_CONTRACT_CATALOG.get("cli_to_analysis") or {}).items()
}
ANALYSIS_METADATA: Dict[str, AnalysisContract] = {
    item["name"]: _analysis_contract_from_dict(item)
    for item in _CONTRACT_CATALOG.get("analyses", [])
    if isinstance(item, dict) and isinstance(item.get("name"), str)
}
if not ANALYSIS_METADATA:
    raise RuntimeError("agent contract catalog is empty")

_ANALYSIS_SHARED_FIELDS = frozenset(
    str(field_name) for field_name in _CONTRACT_CATALOG.get("analysis_shared_fields", [])
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
            return payload
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
    context: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Result of request validation."""
    schema_version: str = AGENT_REQUEST_SCHEMA_VERSION
    status: Literal["ok", "error"]
    valid: bool
    normalized_request: Optional[Dict[str, Any]] = None
    errors: List[ValidationErrorDetail] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


def validate_request(
    payload: Dict[str, Any],
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
        result = native.warp_md_agent_validate_request(payload_json, strict)
        if isinstance(result, dict):
            return ValidationResult.model_validate(result)

    from .agent_schema import validate_run_request

    errors: List[ValidationErrorDetail] = []
    warnings: List[str] = []
    normalized: Optional[Dict[str, Any]] = None

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
    # This would require loading the system - defer to lint-selection command

    return ValidationResult(
        status="ok" if not errors else "error",
        valid=len(errors) == 0,
        normalized_request=normalized if not errors else None,
        errors=errors,
        warnings=warnings,
    )


def normalize_request(
    payload: Dict[str, Any],
    *,
    strip_unknown: bool = False,
) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
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

    analysis_spec: Dict[str, Any] = {"name": canonical}

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
            return payload
    return {
        "schema_version": AGENT_REQUEST_SCHEMA_VERSION,
        "cli_version": _resolve_cli_version(),
        "available_plans": sorted(ANALYSIS_METADATA.keys()),
        "plan_catalog_hash": _compute_catalog_hash(),
        "supports_streaming": True,
        "supports_selection_linting": True,
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
            warnings_list.append(f"Could not load topology for atom count: {load_exc}")

    return SelectionLintResult(
        valid=True,
        expression=expr,
        field_type=field_type,
        matched_atoms=matched_atoms,
        total_atoms=total_atoms,
        warnings=warnings_list,
    )


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


__all__ = [
    "ANALYSIS_METADATA",
    "get_plan_schema",
    "list_all_plans",
    "validate_request",
    "normalize_request",
    "generate_template",
    "capabilities",
    "lint_selection",
    "suggest_analyses",
    "ValidationResult",
    "ValidationErrorDetail",
    "SelectionLintResult",
    "SuggestionResult",
    "SuggestionCandidate",
    "FieldSpec",
    "ArtifactSpec",
    "AnalysisContract",
]
