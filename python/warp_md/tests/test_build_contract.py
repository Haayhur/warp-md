from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from warp_md import build as build_contract


def test_schema_json_forwards_subcommand(monkeypatch) -> None:
    calls = {"cmd": None}

    def fake_run(cmd, capture_output, text, check):  # type: ignore[override]
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd,
            0,
            json.dumps({"schema_version": "polymer-build.agent.v1"}),
            "",
        )

    monkeypatch.setattr(build_contract, "_native", lambda: None)
    monkeypatch.setattr(build_contract, "_binary", lambda: "warp-build")
    monkeypatch.setattr(subprocess, "run", fake_run)

    payload = build_contract.schema_json("request")
    assert payload["schema_version"] == "polymer-build.agent.v1"
    assert calls["cmd"] == ["warp-build", "schema", "--kind", "request"]


def test_example_and_bundle_endpoints(monkeypatch) -> None:
    def fake_run(cmd, capture_output, text, check):  # type: ignore[override]
        command = Path(cmd[1]).name if len(cmd) > 1 else ""
        if command == "example":
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"mode": "extended"}), "")
        if command == "example-bundle":
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"version": "bundle"}), "")
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"result": True}), "")

    monkeypatch.setattr(build_contract, "_native", lambda: None)
    monkeypatch.setattr(build_contract, "_binary", lambda: "warp-build")
    monkeypatch.setattr(subprocess, "run", fake_run)

    example = build_contract.example_request("linear_homopolymer")
    bundle = build_contract.example_bundle()
    caps = build_contract.capabilities()

    assert example["mode"] == "extended"
    assert bundle["version"] == "bundle"
    assert isinstance(caps, dict)


def test_capabilities_endpoints(monkeypatch) -> None:
    captured = {"cmd": None}

    def fake_run(cmd, capture_output, text, check):  # type: ignore[override]
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"supports_named_termini_tokens": True}), "")

    monkeypatch.setattr(build_contract, "_native", lambda: None)
    monkeypatch.setattr(build_contract, "_binary", lambda: "warp-build")
    monkeypatch.setattr(subprocess, "run", fake_run)

    payload = build_contract.capabilities()
    assert payload["supports_named_termini_tokens"] is True
    assert captured["cmd"] == ["warp-build", "capabilities"]


def test_inspect_source_passes_source_path(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "bundle.json"
    source.write_text("{}", encoding="utf-8")
    source_path: Path | None = None

    def fake_run(cmd, capture_output, text, check):  # type: ignore[override]
        nonlocal source_path
        source_path = Path(cmd[-1])
        return subprocess.CompletedProcess(
            cmd,
            0,
            json.dumps({"unit_library_size": 0}),
            "",
        )

    monkeypatch.setattr(build_contract, "_native", lambda: None)
    monkeypatch.setattr(build_contract, "_binary", lambda: "warp-build")
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = build_contract.inspect_source(source)

    assert result["unit_library_size"] == 0
    assert source_path is not None
    assert source_path == source


def test_validate_request_writes_payload_file(monkeypatch) -> None:
    payload = {"version": "polymer-build.agent.v1"}
    request_payload: dict[str, Any] = {}

    def fake_run(cmd, capture_output, text, check):  # type: ignore[override]
        request_payload_path = Path(cmd[-1])
        request_payload.update(json.loads(request_payload_path.read_text(encoding="utf-8")))
        return subprocess.CompletedProcess(
            cmd,
            0,
            json.dumps({"valid": True, "errors": []}),
            "",
        )

    monkeypatch.setattr(build_contract, "_native", lambda: None)
    monkeypatch.setattr(build_contract, "_binary", lambda: "warp-build")
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = build_contract.validate(payload)
    assert result["valid"] is True
    assert request_payload == payload


def test_run_build_request_supports_stream_flag(monkeypatch) -> None:
    calls = {"cmd": None}
    payload = {"version": "polymer-build.agent.v1"}

    def fake_run(cmd, capture_output, text, check):  # type: ignore[override]
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(build_contract, "_native", lambda: None)
    monkeypatch.setattr(build_contract, "_binary", lambda: "warp-build")
    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code, envelope = build_contract.run(payload, stream=True)
    assert exit_code == 0
    assert envelope == {}
    assert "--stream" in calls["cmd"]


def test_capabilities_prefers_native_binding(monkeypatch) -> None:
    class Native:
        @staticmethod
        def build_agent_capabilities():
            return {"native": True}

    monkeypatch.setattr(build_contract, "_native", lambda: Native())

    payload = build_contract.capabilities()
    assert payload == {"native": True}
