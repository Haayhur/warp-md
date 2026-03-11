from __future__ import annotations

from typing import Any, Dict, List

import pytest

from warp_md import mcp_server


class _LineReader:
    def __init__(self, lines: List[str]):
        self._lines = iter(lines)

    def readline(self) -> str:
        return next(self._lines, "")


class _FakeProc:
    def __init__(self, stderr_lines: List[str], returncode: int):
        self.stderr = _LineReader(stderr_lines)
        self.returncode = returncode

    def wait(self) -> int:
        return self.returncode


class _DummyServer:
    def __init__(self):
        self.tools: Dict[str, Any] = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


@pytest.fixture(autouse=True)
def _reset_mcp():
    mcp_server.mcp = None
    yield
    mcp_server.mcp = None


def test_consume_stderr_events_collects_json_and_plaintext():
    proc = _FakeProc(
        [
            "warning: plain stderr message\n",
            '{"event":"operation_complete","total_atoms":12}\n',
            '{"event":"error","message":"bad"}\n',
        ],
        returncode=0,
    )

    final_event, error_event, diagnostics = mcp_server._consume_stderr_events(
        proc, "operation_complete"
    )

    assert final_event is not None
    assert final_event["total_atoms"] == 12
    assert error_event is not None
    assert error_event["message"] == "bad"
    assert diagnostics == "warning: plain stderr message"


def test_run_pep_command_appends_stream_and_preserves_diagnostics(monkeypatch):
    captured_cmd: List[str] = []

    def fake_popen(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return _FakeProc(["fatal: invalid mutation\n"], returncode=1)

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    result = mcp_server._run_pep_command(
        ["warp-pep", "mutate", "--input", "in.pdb", "--mutations", "A5G"],
        output="out.pdb",
    )

    assert "--stream" in captured_cmd
    assert result["success"] is False
    assert result["error"] == "fatal: invalid mutation"


def test_build_and_mutate_wrappers_pass_stream(monkeypatch):
    server = _DummyServer()
    mcp_server.mcp = server
    captured: List[List[str]] = []

    def fake_run(cmd: List[str], output: str):
        captured.append(list(cmd))
        return {"success": True, "output_path": output}

    monkeypatch.setattr(mcp_server, "_run_pep_command", fake_run)
    mcp_server.register_tools()

    build = server.tools["build_peptide"]
    mutate = server.tools["mutate_peptide"]

    build(sequence="ACD", output="build_out.pdb")
    mutate(input="in.pdb", mutations=["A5G"], output="mut_out.pdb")

    assert "--stream" in captured[0]
    assert "--stream" in captured[1]


def test_pack_molecules_surfaces_stderr_diagnostics(monkeypatch):
    server = _DummyServer()
    mcp_server.mcp = server
    mcp_server.register_tools()
    pack = server.tools["pack_molecules"]

    def fake_popen(cmd, **kwargs):
        return _FakeProc(["cli failed: bad config\n"], returncode=1)

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    result = pack(config_path="missing.yaml", output="out.pdb", stream=False)

    assert result["success"] is False
    assert result["error"] == "cli failed: bad config"


def test_run_analysis_passes_fail_fast_override(monkeypatch):
    server = _DummyServer()
    mcp_server.mcp = server
    captured = {}

    class _Envelope:
        def model_dump(self, mode="json"):
            return {"status": "ok"}

    def fake_run_analyses(request, **kwargs):
        captured["request"] = request
        captured["kwargs"] = kwargs
        return _Envelope()

    monkeypatch.setattr("warp_md.runner.run_analyses", fake_run_analyses)
    mcp_server.register_tools()

    run_analysis = server.tools["run_analysis"]
    result = run_analysis(
        system_path="topology.pdb",
        trajectory_path="traj.xtc",
        analyses=[{"name": "rg", "selection": "protein"}],
        fail_fast=False,
    )

    assert result["status"] == "ok"
    assert captured["request"]["fail_fast"] is False
    assert captured["kwargs"]["fail_fast"] is False
