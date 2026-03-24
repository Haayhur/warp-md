from __future__ import annotations

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


def test_build_cli_example_bundle_out_uses_writer(monkeypatch, capsys, tmp_path) -> None:
    out = tmp_path / "source.bundle.json"
    called = {"path": None}

    def fake_write_example_bundle(path):
        called["path"] = str(path)
        return {"bundle_id": "pmma_param_bundle_v1"}

    monkeypatch.setattr(build_contract, "write_example_bundle", fake_write_example_bundle)

    exit_code = build_cli.run_cli(["example-bundle", "--out", str(out)])

    assert exit_code == 0
    assert called["path"] == str(out)
    assert capsys.readouterr().out.strip() == str(out)
