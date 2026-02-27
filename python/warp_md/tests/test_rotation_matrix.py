import numpy as np

from warp_md.analysis.rotation import rotation_matrix


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


def test_rotation_matrix_basic():
    ref = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    rot = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    frame1 = ref @ rot.T
    coords = np.stack([ref, frame1], axis=0).astype(np.float32)

    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    mats, rmsd = rotation_matrix(traj, system, mask="all", ref=0, with_rmsd=True)

    assert mats.shape == (2, 3, 3)
    np.testing.assert_allclose(mats[0], np.eye(3), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(mats[1], rot.T, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rmsd[0], 0.0, rtol=1e-6, atol=1e-6)
