from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from . import traj_py


CG_AGENT_SCHEMA_VERSION = "warp-cg.agent.v1"
CG_AGENT_RESULT_VERSION = CG_AGENT_SCHEMA_VERSION


def _native() -> Any:
    if traj_py is None:
        raise RuntimeError(
            "warp-md Python bindings are unavailable. Run `maturin develop` or install warp-md."
        )
    required = (
        "cg_agent_schema",
        "cg_agent_example",
        "cg_agent_capabilities",
        "cg_agent_validate",
        "cg_agent_run",
    )
    missing = [name for name in required if not hasattr(traj_py, name)]
    if missing:
        raise RuntimeError(
            "warp-md coarse-graining bindings unavailable in this build. Missing: "
            + ", ".join(missing)
        )
    return traj_py


def _native_build() -> Any:
    native = _native()
    required = (
        "cg_build_schema",
        "cg_build_example",
        "cg_build_capabilities",
        "cg_build_validate",
        "cg_build_run",
    )
    missing = [name for name in required if not hasattr(native, name)]
    if missing:
        raise RuntimeError(
            "warp-md coarse-graining build bindings unavailable in this build. Missing: "
            + ", ".join(missing)
        )
    return native


def _render_payload(payload: Dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(payload, indent=2)
    if fmt == "yaml":
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError("YAML output requires PyYAML installed") from exc
        return yaml.safe_dump(payload, sort_keys=False)
    raise ValueError("format must be json or yaml")


def render_cg_schema(target: str = "request", fmt: str = "json") -> str:
    payload = _native().cg_agent_schema(target)
    return _render_payload(payload, fmt)


def example_request() -> Dict[str, Any]:
    payload = _native().cg_agent_example()
    if not isinstance(payload, dict):
        raise RuntimeError("native example request must decode to a dict")
    return payload


def cg_capabilities() -> Dict[str, Any]:
    payload = _native().cg_agent_capabilities()
    if not isinstance(payload, dict):
        raise RuntimeError("native capabilities payload must decode to a dict")
    return payload


def validate_request_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    exit_code, result = _native().cg_agent_validate(json.dumps(payload))
    if not isinstance(result, dict):
        raise RuntimeError("native validate payload must decode to a dict")
    if exit_code and result.get("valid") is True:
        raise RuntimeError("native validate returned inconsistent result")
    return result


def run_cg_request(
    payload: Dict[str, Any],
    *,
    stream: str = "none",
) -> Tuple[int, Dict[str, Any]]:
    stream_ndjson = stream == "ndjson"
    exit_code, result = _native().cg_agent_run(json.dumps(payload), stream_ndjson)
    if not isinstance(result, dict):
        raise RuntimeError("native run payload must decode to a dict")
    return int(exit_code), result


def render_cg_build_schema(target: str = "request", fmt: str = "json") -> str:
    payload = _native_build().cg_build_schema(target)
    return _render_payload(payload, fmt)


def build_example_request() -> Dict[str, Any]:
    payload = _native_build().cg_build_example()
    if not isinstance(payload, dict):
        raise RuntimeError("native build example request must decode to a dict")
    return payload


def cg_build_capabilities() -> Dict[str, Any]:
    payload = _native_build().cg_build_capabilities()
    if not isinstance(payload, dict):
        raise RuntimeError("native build capabilities payload must decode to a dict")
    return payload


def validate_build_request_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    exit_code, result = _native_build().cg_build_validate(json.dumps(payload))
    if not isinstance(result, dict):
        raise RuntimeError("native build validate payload must decode to a dict")
    if exit_code and result.get("valid") is True:
        raise RuntimeError("native build validate returned inconsistent result")
    return result


def run_cg_build_request(
    payload: Dict[str, Any],
    *,
    stream: str = "none",
) -> Tuple[int, Dict[str, Any]]:
    stream_ndjson = stream == "ndjson"
    exit_code, result = _native_build().cg_build_run(json.dumps(payload), stream_ndjson)
    if not isinstance(result, dict):
        raise RuntimeError("native build run payload must decode to a dict")
    return int(exit_code), result


__all__ = [
    "CG_AGENT_SCHEMA_VERSION",
    "CG_AGENT_RESULT_VERSION",
    "render_cg_schema",
    "example_request",
    "cg_capabilities",
    "validate_request_payload",
    "run_cg_request",
    "render_cg_build_schema",
    "build_example_request",
    "cg_build_capabilities",
    "validate_build_request_payload",
    "run_cg_build_request",
]
