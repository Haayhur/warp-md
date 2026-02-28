import numpy as np

from warp_md.analysis.structure import make_structure


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._n_atoms = n_atoms

    def select(self, _mask):
        return _DummySelection(range(self._n_atoms))

    def select_indices(self, indices):
        return _DummySelection(indices)

    def atom_table(self):
        return {"resid": list(range(self._n_atoms))}


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_make_structure_mean():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out = make_structure(traj, system, mask="all")
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(out[1], np.array([3.0, 0.0, 0.0], dtype=np.float32))
