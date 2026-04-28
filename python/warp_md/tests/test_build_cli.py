from __future__ import annotations

from warp_md import cli_run
from warp_md import build as build_contract
from warp_md import build_cli


def test_build_cli_schema_uses_python_wrapper(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        build_contract,
        "schema_json",
        lambda kind="request": {"schema_version": "warp-build.agent.v1", "kind": kind},
    )

    exit_code = build_cli.run_cli(["schema", "--kind", "request"])

    assert exit_code == 0
    assert "warp-build.agent.v1" in capsys.readouterr().out


def test_build_cli_help_renders_without_native_bindings(capsys) -> None:
    try:
        build_cli.run_cli(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    assert "warp-build" in capsys.readouterr().out


def test_build_binary_uses_warp_build_env(monkeypatch) -> None:
    monkeypatch.setenv("WARP_BUILD_BINARY", "/tmp/warp-build-bin")

    assert build_contract._binary() == "/tmp/warp-build-bin"


def test_build_binary_falls_back_to_legacy_env(monkeypatch) -> None:
    monkeypatch.delenv("WARP_BUILD_BINARY", raising=False)
    monkeypatch.setenv("POLYMER_BUILD_BINARY", "/tmp/polymer-build-bin")

    assert build_contract._binary() == "/tmp/polymer-build-bin"


def test_build_cli_example_bundle_out_uses_writer(monkeypatch, capsys, tmp_path) -> None:
    out = tmp_path / "source.bundle.json"
    called = {"path": None}

    def fake_write_example_bundle(path):
        called["path"] = str(path)
        return {"bundle_id": "example_polymer_bundle_v1"}

    monkeypatch.setattr(build_contract, "write_example_bundle", fake_write_example_bundle)

    exit_code = build_cli.run_cli(["example-bundle", "--out", str(out)])

    assert exit_code == 0
    assert called["path"] == str(out)
    assert capsys.readouterr().out.strip() == str(out)


def test_build_cli_validate_forwards_deep_flag(monkeypatch, capsys, tmp_path) -> None:
    request = tmp_path / "request.json"
    request.write_text('{"schema_version":"warp-build.agent.v1"}', encoding="utf-8")
    captured = {}

    def fake_validate(payload, *, deep=False):
        captured["payload"] = payload
        captured["deep"] = deep
        return {"valid": True, "status": "ok"}

    monkeypatch.setattr(build_contract, "validate", fake_validate)

    exit_code = build_cli.run_cli(["validate", str(request), "--deep"])

    assert exit_code == 0
    assert captured["deep"] is True
    assert captured["payload"]["schema_version"] == "warp-build.agent.v1"
    assert '"valid": true' in capsys.readouterr().out.lower()


def test_warp_md_build_forwards_to_build_cli(monkeypatch) -> None:
    captured = {}

    def fake_main(argv=None):
        captured["argv"] = list(argv or [])
        return 7

    monkeypatch.setattr(build_cli, "main", fake_main)

    exit_code = cli_run.main(["build", "schema", "--kind", "request"])

    assert exit_code == 7
    assert captured["argv"] == ["schema", "--kind", "request"]
