from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

from . import traj_py


def _native() -> Any:
    if traj_py is None:
        return None
    required = (
        "build_agent_schema",
        "build_agent_example",
        "build_agent_example_bundle",
        "build_agent_capabilities",
        "build_agent_inspect_source",
        "build_agent_validate",
        "build_agent_run",
    )
    if any(not hasattr(traj_py, name) for name in required):
        return None
    return traj_py


def _binary() -> str:
    command = os.environ.get("WARP_BUILD_BINARY") or os.environ.get(
        "POLYMER_BUILD_BINARY", "warp-build"
    )
    if os.path.isabs(command) or os.path.dirname(command):
        return command
    return shutil.which(command) or command


def _loads_json(raw: str, *, context: str) -> Dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{context}: output was not valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{context}: expected JSON object")
    return value


def _run_cli(
    args: list[str],
    payload: Dict[str, Any] | None = None,
    *,
    stream: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    cli = _binary()
    command = [cli]
    command.extend(args)

    if payload is not None:
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.write("\n")
            command.append(path)
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            os.remove(path)
    else:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

    stdout = result.stdout.strip()
    if not stdout:
        if stream:
            return result.returncode, {}
        raise RuntimeError("warp-build returned empty output")

    envelope = _loads_json(stdout, context="warp-build output")

    if result.returncode != 0:
        return result.returncode, envelope

    return 0, envelope


def schema_json(kind: str = "request") -> Dict[str, Any]:
    native = _native()
    if native is not None:
        payload = native.build_agent_schema(kind)
        if not isinstance(payload, dict):
            raise RuntimeError("native build schema payload must decode to a dict")
        return payload
    _, payload = _run_cli(["schema", "--kind", kind])
    return payload


def schema(kind: str = "request") -> Dict[str, Any]:
    return schema_json(kind)


def example_request(mode: str = "random_walk") -> Dict[str, Any]:
    native = _native()
    if native is not None:
        payload = native.build_agent_example(mode)
        if not isinstance(payload, dict):
            raise RuntimeError("native build example payload must decode to a dict")
        return payload
    _, payload = _run_cli(["example", "--mode", mode])
    return payload


def example_bundle() -> Dict[str, Any]:
    native = _native()
    if native is not None:
        payload = native.build_agent_example_bundle()
        if not isinstance(payload, dict):
            raise RuntimeError("native build example bundle payload must decode to a dict")
        return payload
    _, payload = _run_cli(["example-bundle"])
    return payload


def capabilities() -> Dict[str, Any]:
    native = _native()
    if native is not None:
        payload = native.build_agent_capabilities()
        if not isinstance(payload, dict):
            raise RuntimeError("native build capabilities payload must decode to a dict")
        return payload
    _, payload = _run_cli(["capabilities"])
    return payload


def inspect_source(source: str | Path) -> Dict[str, Any]:
    native = _native()
    if native is not None:
        _, payload = native.build_agent_inspect_source(str(source))
        if not isinstance(payload, dict):
            raise RuntimeError("native inspect-source payload must decode to a dict")
        return payload
    _, payload = _run_cli(["inspect-source", str(source)])
    return payload


def validate_request_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    native = _native()
    if native is not None:
        _, response = native.build_agent_validate(json.dumps(payload))
        if not isinstance(response, dict):
            raise RuntimeError("native validate payload must decode to a dict")
        return response
    _, response = _run_cli(["validate"], payload=payload)
    return response


def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    return validate_request_payload(payload)


def run_build_request(
    payload: Dict[str, Any],
    *,
    stream: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    native = _native()
    if native is not None:
        exit_code, result = native.build_agent_run(json.dumps(payload), stream)
        if not isinstance(result, dict):
            raise RuntimeError("native run payload must decode to a dict")
        return int(exit_code), result
    args = ["run"]
    if stream:
        args.append("--stream")
    return _run_cli(args, payload=payload, stream=stream)


def run(payload: Dict[str, Any], *, stream: bool = False) -> Tuple[int, Dict[str, Any]]:
    return run_build_request(payload, stream=stream)


__all__ = [
    "_binary",
    "_loads_json",
    "_native",
    "_run_cli",
    "schema_json",
    "schema",
    "example_request",
    "example_bundle",
    "capabilities",
    "inspect_source",
    "validate_request_payload",
    "validate",
    "run_build_request",
    "run",
]
