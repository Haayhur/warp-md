import numpy as np

from warp_md.analysis.atomiccorr import atomiccorr


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._n_atoms = n_atoms

    def select(self, _mask):
        return _DummySelection(range(self._n_atoms))

    def positions0(self):
        return None


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_atomiccorr_basic():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    time, data = atomiccorr(traj, system, mask="all", reference="frame0")
    np.testing.assert_allclose(time, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(data, np.array([1.0, 0.0], dtype=np.float32), atol=1e-5)
