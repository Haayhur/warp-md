from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from warp_md import frame_edit


class FakeSystem:
    def __init__(self, n_atoms: int) -> None:
        self._n_atoms = n_atoms

    def n_atoms(self) -> int:
        return self._n_atoms


class FakeTrajectory:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = [np.asarray(frame, dtype=np.float32) for frame in frames]
        self._cursor = 0

    def read_chunk(
        self,
        max_frames=None,
        include_box=True,
        include_box_matrix=True,
        include_time=False,
    ):
        if self._cursor >= len(self._frames):
            return None
        size = len(self._frames) - self._cursor if max_frames is None else max(1, int(max_frames))
        chunk = self._frames[self._cursor : self._cursor + size]
        self._cursor += len(chunk)
        payload = {
            "coords": np.stack(chunk, axis=0),
            "frames": len(chunk),
        }
        payload["box"] = None
        payload["box_matrix"] = None
        return payload

    def reset(self) -> None:
        self._cursor = 0


class FakeRustTrajectory(FakeTrajectory):
    def __init__(self, frames: list[np.ndarray]) -> None:
        super().__init__(frames)
        self.read_chunk_calls = 0
        self.count_frames_calls = 0
        self.read_frames_calls = 0

    def read_chunk(self, *args, **kwargs):
        self.read_chunk_calls += 1
        return super().read_chunk(*args, **kwargs)

    def count_frames(self, chunk_frames=None) -> int:
        self.count_frames_calls += 1
        return len(self._frames)

    def read_frames(
        self,
        frame_indices,
        chunk_frames=None,
        include_box=True,
        include_box_matrix=True,
        include_time=False,
    ):
        self.read_frames_calls += 1
        coords = np.stack([self._frames[int(index)] for index in frame_indices], axis=0)
        return {
            "coords": coords,
            "box": None,
            "box_matrix": None,
            "source_indices": list(frame_indices),
            "frames": len(frame_indices),
        }


class FakeTrajectorySink:
    def __init__(self, written: list[int]) -> None:
        self._written = written

    def write_frame(self, frame) -> None:
        self._written.append(frame.index)


class FakeWriter:
    def __init__(self, topology_path: str, *, topology_format: str | None, n_atoms: int) -> None:
        self.topology_path = topology_path
        self.topology_format = topology_format
        self.n_atoms = n_atoms
        self.structure_writes: list[tuple[str, int]] = []
        self.trajectory_writes: dict[str, list[int]] = {}

    def write_structure(self, path: str, frame) -> None:
        self.structure_writes.append((path, frame.index))

    @contextmanager
    def open_trajectory(self, path: str):
        written: list[int] = []
        self.trajectory_writes[path] = written
        yield FakeTrajectorySink(written)


def _args(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "topology": str(tmp_path / "min.pdb"),
        "traj": str(tmp_path / "eq_npt.dcd"),
        "out": str(tmp_path / "md_new.dcd"),
        "begin": 0,
        "end": None,
        "step": 1,
        "index": None,
        "topology_format": "pdb",
        "traj_format": "dcd",
        "traj_length_scale": None,
        "chunk_frames": 2,
    }
    values.update(overrides)
    return Namespace(**values)


def test_resolve_frame_indices_matches_script_behavior() -> None:
    frame_indices, selection = frame_edit._resolve_frame_indices(
        total_frames=10,
        begin=2,
        end=9,
        step=3,
        index=None,
    )

    assert frame_indices == [2, 5, 8]
    assert selection == {"mode": "range", "begin": 2, "end": 9, "step": 3}


def test_resolve_frame_indices_rejects_out_of_range_single_index() -> None:
    with pytest.raises(ValueError, match="out of range"):
        frame_edit._resolve_frame_indices(
            total_frames=4,
            begin=0,
            end=None,
            step=1,
            index=4,
        )


def test_run_frame_edit_writes_structure_series(monkeypatch, tmp_path: Path) -> None:
    system = FakeSystem(n_atoms=2)
    traj = FakeTrajectory(
        [
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
            np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]], dtype=np.float32),
            np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]], dtype=np.float32),
        ]
    )
    created: list[FakeWriter] = []

    monkeypatch.setattr(frame_edit, "_load_system", lambda spec: system)
    monkeypatch.setattr(frame_edit, "_load_trajectory", lambda spec, loaded_system: traj)
    monkeypatch.setattr(
        frame_edit,
        "_make_writer",
        lambda topology_path, *, topology_format, n_atoms: created.append(
            FakeWriter(topology_path, topology_format=topology_format, n_atoms=n_atoms)
        )
        or created[-1],
    )

    exit_code, payload = frame_edit.run_frame_edit(
        _args(
            tmp_path,
            out=str(tmp_path / "frames.pdb"),
            begin=1,
            end=4,
            step=2,
        )
    )

    assert exit_code == 0
    assert payload["written_frames"] == 2
    assert payload["output_mode"] == "structure_series"
    assert payload["outputs"] == [
        str(tmp_path / "frames_1.pdb"),
        str(tmp_path / "frames_3.pdb"),
    ]
    assert created[0].structure_writes == [
        (str(tmp_path / "frames_1.pdb"), 1),
        (str(tmp_path / "frames_3.pdb"), 3),
    ]


def test_run_frame_edit_writes_single_frame_to_trajectory(monkeypatch, tmp_path: Path) -> None:
    system = FakeSystem(n_atoms=2)
    traj = FakeTrajectory(
        [
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
            np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]], dtype=np.float32),
        ]
    )
    created: list[FakeWriter] = []

    monkeypatch.setattr(frame_edit, "_load_system", lambda spec: system)
    monkeypatch.setattr(frame_edit, "_load_trajectory", lambda spec, loaded_system: traj)
    monkeypatch.setattr(
        frame_edit,
        "_make_writer",
        lambda topology_path, *, topology_format, n_atoms: created.append(
            FakeWriter(topology_path, topology_format=topology_format, n_atoms=n_atoms)
        )
        or created[-1],
    )

    out_path = tmp_path / "single.xtc"
    exit_code, payload = frame_edit.run_frame_edit(
        _args(
            tmp_path,
            out=str(out_path),
            index=2,
            begin=0,
            end=3,
            step=1,
        )
    )

    assert exit_code == 0
    assert payload["selection"] == {"mode": "single", "index": 2}
    assert payload["outputs"] == [str(out_path)]
    assert created[0].trajectory_writes[str(out_path)] == [2]


def test_run_frame_edit_prefers_rust_count_and_read_methods(monkeypatch, tmp_path: Path) -> None:
    system = FakeSystem(n_atoms=2)
    traj = FakeRustTrajectory(
        [
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
            np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]], dtype=np.float32),
        ]
    )
    created: list[FakeWriter] = []

    monkeypatch.setattr(frame_edit, "_load_system", lambda spec: system)
    monkeypatch.setattr(frame_edit, "_load_trajectory", lambda spec, loaded_system: traj)
    monkeypatch.setattr(
        frame_edit,
        "_make_writer",
        lambda topology_path, *, topology_format, n_atoms: created.append(
            FakeWriter(topology_path, topology_format=topology_format, n_atoms=n_atoms)
        )
        or created[-1],
    )

    exit_code, payload = frame_edit.run_frame_edit(
        _args(
            tmp_path,
            out=str(tmp_path / "single.xtc"),
            begin=0,
            end=3,
            step=2,
        )
    )

    assert exit_code == 0
    assert payload["written_frames"] == 2
    assert traj.count_frames_calls == 1
    assert traj.read_frames_calls == 1
    assert traj.read_chunk_calls == 0
