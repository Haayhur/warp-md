from __future__ import annotations

import pytest

from warp_md import cli_api


def test_load_system_calls_from_file(monkeypatch) -> None:
    class FakeSystem:
        calls: list[tuple[str, str, str | None]] = []

        @staticmethod
        def from_file(path: str, format: str | None = None):
            FakeSystem.calls.append(("from_file", path, format))
            return {"path": path, "format": format}

    monkeypatch.setattr(cli_api, "_API_IMPORT_ERROR", None)
    monkeypatch.setattr(cli_api, "System", FakeSystem)

    loaded = cli_api._load_system({"path": "ligand.pdbqt", "format": "pdbqt"})

    assert loaded == {"path": "ligand.pdbqt", "format": "pdbqt"}
    assert FakeSystem.calls == [("from_file", "ligand.pdbqt", "pdbqt")]


def test_load_system_preserves_from_file_errors(monkeypatch) -> None:
    class FakeSystem:
        @staticmethod
        def from_file(path: str, format: str | None = None):
            raise RuntimeError("parse error: invalid resid 'A000'")

    monkeypatch.setattr(cli_api, "_API_IMPORT_ERROR", None)
    monkeypatch.setattr(cli_api, "System", FakeSystem)

    with pytest.raises(RuntimeError, match="invalid resid"):
        cli_api._load_system({"path": "min.pdb", "format": "pdb"})
