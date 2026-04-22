from __future__ import annotations

import pytest

from warp_md import io as io_mod


def test_open_trajectory_auto_delegates_to_native_binding(monkeypatch) -> None:
    calls: list[tuple[str, object, str | None, float | None]] = []

    class FakeTrajPy:
        @staticmethod
        def open_trajectory_auto(path, system, format=None, length_scale=None):
            calls.append((path, system, format, length_scale))
            return ("native", path, system, format, length_scale)

    monkeypatch.setattr(io_mod, "_traj_py", FakeTrajPy)
    monkeypatch.setattr(io_mod, "_TRAJ_PY_IMPORT_ERROR", None)

    out = io_mod.open_trajectory_auto(
        "traj.xtc",
        system="sys",
        format="xtc",
        length_scale=10.0,
    )

    assert out == ("native", "traj.xtc", "sys", "xtc", 10.0)
    assert calls == [("traj.xtc", "sys", "xtc", 10.0)]


def test_open_trajectory_alias_delegates_to_native_binding(monkeypatch) -> None:
    class FakeTrajPy:
        @staticmethod
        def open_trajectory_auto(path, system, format=None, length_scale=None):
            return ("native", path, system, format, length_scale)

    monkeypatch.setattr(io_mod, "_traj_py", FakeTrajPy)
    monkeypatch.setattr(io_mod, "_TRAJ_PY_IMPORT_ERROR", None)

    out = io_mod.open_trajectory("traj.dcd", system="sys", length_scale=10.0)

    assert out == ("native", "traj.dcd", "sys", None, 10.0)


def test_open_trajectory_auto_requires_native_binding(monkeypatch) -> None:
    monkeypatch.setattr(io_mod, "_traj_py", None)
    monkeypatch.setattr(io_mod, "_TRAJ_PY_IMPORT_ERROR", ImportError("no traj_py"))

    with pytest.raises(RuntimeError, match="Python bindings are unavailable"):
        io_mod.open_trajectory_auto("traj.xtc", system="sys")
