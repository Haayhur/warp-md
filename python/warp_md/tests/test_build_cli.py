from __future__ import annotations

from warp_md import build as build_contract
from warp_md import build_cli


def test_build_cli_schema_uses_python_wrapper(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        build_contract,
        "schema_json",
        lambda kind="request": {"schema_version": "polymer-build.agent.v1", "kind": kind},
    )

    exit_code = build_cli.run_cli(["schema", "--kind", "request"])

    assert exit_code == 0
    assert "polymer-build.agent.v1" in capsys.readouterr().out


def test_build_cli_help_renders_without_native_bindings(capsys) -> None:
    try:
        build_cli.run_cli(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    assert "warp-build" in capsys.readouterr().out


def test_polymer_build_binary_prefers_warp_build_env(monkeypatch) -> None:
    monkeypatch.setenv("WARP_BUILD_BINARY", "/tmp/warp-build-bin")
    monkeypatch.setenv("POLYMER_BUILD_BINARY", "/tmp/polymer-build-bin")

    assert build_contract._binary() == "/tmp/warp-build-bin"
