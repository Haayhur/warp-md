from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from . import traj_py


PACK_AGENT_SCHEMA_VERSION = "warp-pack.agent.v1"
PACK_AGENT_RESULT_VERSION = PACK_AGENT_SCHEMA_VERSION


def _native() -> Any:
    if traj_py is None:
        raise RuntimeError(
            "warp-md Python bindings are unavailable. Run `maturin develop` or install warp-md."
        )
    required = (
        "pack_agent_schema",
        "pack_agent_example",
        "pack_agent_capabilities",
        "pack_agent_validate",
        "pack_agent_run",
    )
    missing = [name for name in required if not hasattr(traj_py, name)]
    if missing:
        raise RuntimeError(
            "warp-md pack agent bindings unavailable in this build. Missing: "
            + ", ".join(missing)
        )
    return traj_py


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


def render_pack_schema(target: str = "request", fmt: str = "json") -> str:
    payload = _native().pack_agent_schema(target)
    return _render_payload(payload, fmt)


def example_request(mode: str = "solute_solvate") -> Dict[str, Any]:
    payload = _native().pack_agent_example(mode)
    if not isinstance(payload, dict):
        raise RuntimeError("native example request must decode to a dict")
    return payload


def pack_capabilities() -> Dict[str, Any]:
    payload = _native().pack_agent_capabilities()
    if not isinstance(payload, dict):
        raise RuntimeError("native capabilities payload must decode to a dict")
    return payload


def validate_request_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    exit_code, result = _native().pack_agent_validate(json.dumps(payload))
    if not isinstance(result, dict):
        raise RuntimeError("native validate payload must decode to a dict")
    if exit_code and result.get("valid") is True:
        raise RuntimeError("native validate returned inconsistent result")
    return result


def run_build_request(
    payload: Dict[str, Any],
    *,
    stream: str = "none",
) -> Tuple[int, Dict[str, Any]]:
    stream_ndjson = stream == "ndjson"
    exit_code, result = _native().pack_agent_run(json.dumps(payload), stream_ndjson)
    if not isinstance(result, dict):
        raise RuntimeError("native run payload must decode to a dict")
    return int(exit_code), result


__all__ = [
    "PACK_AGENT_SCHEMA_VERSION",
    "PACK_AGENT_RESULT_VERSION",
    "render_pack_schema",
    "example_request",
    "pack_capabilities",
    "validate_request_payload",
    "run_build_request",
]
