from __future__ import annotations

import subprocess

from warp_md import build_cli, polymer_build


def test_build_cli_forwards_args(monkeypatch) -> None:
    calls = {"cmd": None}

    def fake_run(cmd, check):  # type: ignore[override]
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(polymer_build, "_binary", lambda: "polymer-build")
    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = build_cli.run_cli(["schema", "--kind", "request"])

    assert exit_code == 0
    assert calls["cmd"] == ["polymer-build", "schema", "--kind", "request"]


def test_polymer_build_binary_prefers_warp_build_env(monkeypatch) -> None:
    monkeypatch.setenv("WARP_BUILD_BINARY", "/tmp/warp-build-bin")
    monkeypatch.setenv("POLYMER_BUILD_BINARY", "/tmp/polymer-build-bin")

    assert polymer_build._binary() == "/tmp/warp-build-bin"
