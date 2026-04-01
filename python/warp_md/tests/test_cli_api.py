from __future__ import annotations

import pytest

from warp_md import cli_api


def test_load_system_retries_pdb_with_permissive_loader(monkeypatch) -> None:
    class FakeSystem:
        calls: list[tuple[str, str]] = []

        @staticmethod
        def from_pdb(path: str):
            FakeSystem.calls.append(("strict", path))
            raise RuntimeError("parse error: invalid resid 'A000'")

        @staticmethod
        def from_pdb_permissive(path: str):
            FakeSystem.calls.append(("permissive", path))
            return {"path": path, "mode": "permissive"}

    monkeypatch.setattr(cli_api, "_API_IMPORT_ERROR", None)
    monkeypatch.setattr(cli_api, "System", FakeSystem)

    loaded = cli_api._load_system({"path": "min.pdb", "format": "pdb"})

    assert loaded == {"path": "min.pdb", "mode": "permissive"}
    assert FakeSystem.calls == [("strict", "min.pdb"), ("permissive", "min.pdb")]


def test_load_system_keeps_original_pdb_error_for_other_failures(monkeypatch) -> None:
    class FakeSystem:
        @staticmethod
        def from_pdb(path: str):
            raise RuntimeError("parse error: no atoms found in PDB")

        @staticmethod
        def from_pdb_permissive(path: str):
            raise AssertionError("unexpected permissive retry")

    monkeypatch.setattr(cli_api, "_API_IMPORT_ERROR", None)
    monkeypatch.setattr(cli_api, "System", FakeSystem)

    with pytest.raises(RuntimeError, match="no atoms found in PDB"):
        cli_api._load_system({"path": "min.pdb", "format": "pdb"})
