import numpy as np
import pytest

import warp_md
from warp_md.analysis.randomize_ions import randomize_ions


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._n_atoms = n_atoms

    def select(self, _mask):
        return _DummySelection(range(self._n_atoms))

    def n_atoms(self):
        return self._n_atoms


class _DummyTraj:
    def __init__(self, coords, box=None):
        self._coords = coords
        self._box = box
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        out = {"coords": self._coords}
        if self._box is not None:
            out["box"] = self._box
        return out


class _DummyPlan:
    def __init__(self, _sel, _seed, _around_sel, _by, _overlap, _noimage):
        pass

    def run(self, traj, _system, chunk_frames=None, device="auto", frame_indices=None):
        del chunk_frames, device
        chunk = traj.read_chunk()
        coords = np.asarray(chunk["coords"], dtype=np.float32).copy()
        if frame_indices is not None:
            coords = coords[np.asarray(frame_indices, dtype=np.int64)]
        coords[:, :, 0] += 1.0
        return coords.reshape(-1)


def test_randomize_ions_uses_plan(monkeypatch):
    monkeypatch.setattr(warp_md, "RandomizeIonsPlan", _DummyPlan, raising=False)
    coords = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out = randomize_ions(traj, system, mask="all")
    out_coords = out.read_chunk()["coords"]
    np.testing.assert_allclose(out_coords[0, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_randomize_ions_frame_indices(monkeypatch):
    monkeypatch.setattr(warp_md, "RandomizeIonsPlan", _DummyPlan, raising=False)
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out = randomize_ions(traj, system, mask="all", frame_indices=[1])
    out_coords = out.read_chunk()["coords"]
    assert out_coords.shape[0] == 1
    np.testing.assert_allclose(out_coords[0, 0], np.array([11.0, 0.0, 0.0], dtype=np.float32))


def test_randomize_ions_missing_plan(monkeypatch):
    class _Missing:
        __name__ = "_Missing"

    monkeypatch.setattr(warp_md, "RandomizeIonsPlan", _Missing, raising=False)
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    with pytest.raises(RuntimeError):
        randomize_ions(traj, system, mask="all")
