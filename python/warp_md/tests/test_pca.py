import numpy as np

from warp_md.analysis.pca import pca, projection


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


def test_pca_projection_shapes():
    coords = np.zeros((4, 2, 3), dtype=np.float32)
    coords[:, 0, 0] = np.arange(4, dtype=np.float32)
    coords[:, 1, 0] = np.arange(4, dtype=np.float32) * 2.0
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    proj, (evals, evecs) = pca(traj, system, mask="all", n_vecs=2, fit=False)
    assert proj.shape == (2, 4)
    assert evals.shape == (2,)
    assert evecs.shape == (2, 6)
    assert evals[0] >= evals[1]
    assert evals[1] >= 0.0


def test_projection_matches_pca():
    coords = np.zeros((4, 2, 3), dtype=np.float32)
    coords[:, 0, 0] = np.arange(4, dtype=np.float32)
    coords[:, 1, 0] = np.arange(4, dtype=np.float32) * 2.0
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    proj, (evals, evecs) = pca(traj, system, mask="all", n_vecs=2, fit=False)

    traj2 = _DummyTraj(coords)
    proj2 = projection(traj2, system, mask="all", eigenvectors=evecs, eigenvalues=evals)
    np.testing.assert_allclose(proj, proj2, rtol=1e-5)
