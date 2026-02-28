import numpy as np

from warp_md.analysis.matrix import correl, covar, dist, mwcovar


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._atoms = {
            "name": ["CA"] * n_atoms,
            "resname": ["ALA"] * n_atoms,
            "resid": [1] * n_atoms,
            "chain_id": [0] * n_atoms,
            "mass": [1.0] * n_atoms,
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(range(len(self._atoms["name"])))


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_matrix_shapes():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    mat = dist(traj, system, mask="all")
    assert mat.shape == (2, 2)

    traj = _DummyTraj(coords)
    cov = covar(traj, system, mask="all")
    assert cov.shape == (6, 6)

    traj = _DummyTraj(coords)
    mw = mwcovar(traj, system, mask="all")
    assert mw.shape == (6, 6)

    traj = _DummyTraj(coords)
    cc = correl(traj, system, mask="all")
    assert cc.shape == (2, 2)
    assert np.allclose(np.diag(cc), 1.0, atol=1e-6)
