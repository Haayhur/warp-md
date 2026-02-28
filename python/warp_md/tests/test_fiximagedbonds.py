import numpy as np
import pytest
import warp_md

from warp_md.analysis.fiximagedbonds import fiximagedbonds


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._n_atoms = n_atoms
        self._atoms = {"resid": [1] * n_atoms, "chain_id": [0] * n_atoms}

    def select(self, _mask):
        return _DummySelection(list(range(self._n_atoms)))

    def select_indices(self, indices):
        return _DummySelection(list(indices))

    def n_atoms(self):
        return self._n_atoms

    def atom_table(self):
        return self._atoms


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_fiximagedbonds_uses_plan_and_slices(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, sel):
            called["n_sel"] = len(sel.indices)

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            if frame_indices is None:
                return coords.reshape(-1)
            selected = coords[np.asarray(frame_indices, dtype=np.int64)]
            return selected.reshape(-1)

    monkeypatch.setattr(warp_md, "FixImageBondsPlan", _DummyPlan, raising=False)
    out = fiximagedbonds(traj, system, mask=[0, 1], frame_indices=[1])
    got = out.read_chunk()["coords"]
    assert called["n_sel"] == 2
    assert called["frame_indices"] == [1]
    assert got.shape == (1, 2, 3)
    np.testing.assert_allclose(got[0], coords[1], atol=1e-6)


def test_fiximagedbonds_no_python_fallback_when_plan_missing(monkeypatch):
    traj = _DummyTraj(np.zeros((1, 2, 3), dtype=np.float32))
    system = _DummySystem(2)

    class _MissingPlan:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("FixImageBondsPlan binding unavailable in this build.")

    monkeypatch.setattr(warp_md, "FixImageBondsPlan", _MissingPlan, raising=False)
    with pytest.raises(RuntimeError, match="FixImageBondsPlan binding unavailable"):
        fiximagedbonds(traj, system, mask=[0, 1])
