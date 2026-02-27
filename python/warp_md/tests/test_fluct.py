import numpy as np

from warp_md.analysis.fluct import atomicfluct, bfactors, rmsf
from warp_md.analysis.trajectory import ArrayTrajectory


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "name": ["CA", "CB", "CA", "CB"],
            "resname": ["ALA", "ALA", "GLY", "GLY"],
            "resid": [1, 1, 2, 2],
            "chain_id": [0, 0, 0, 0],
            "mass": [1.0, 1.0, 1.0, 1.0],
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(range(len(self._atoms["name"])))


def _make_traj():
    coords = np.zeros((3, 4, 3), dtype=np.float32)
    coords[:, 1, 0] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    coords[:, 3, 0] = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    return ArrayTrajectory(coords)


def test_rmsf_byatom_byres():
    system = _DummySystem()
    traj = _make_traj()

    data = rmsf(traj, system, mask="resid 1:2", byres=False)
    assert data.shape == (4, 2)
    r = np.sqrt(2.0 / 3.0)
    np.testing.assert_allclose(data[:, 1], np.array([0.0, r, 0.0, r]), rtol=1e-6)

    traj.reset()
    byres = rmsf(traj, system, mask="resid 1:2", byres=True)
    assert byres.shape == (2, 2)
    np.testing.assert_allclose(byres[:, 0], np.array([1.0, 2.0]), rtol=1e-6)
    np.testing.assert_allclose(byres[:, 1], np.array([r / 2.0, r / 2.0]), rtol=1e-6)


def test_bfactors_bymask():
    system = _DummySystem()
    traj = _make_traj()

    data = bfactors(traj, system, mask="resid 1:2", byres=False, bymask=True)
    assert data.shape == (1, 2)
    r = np.sqrt(2.0 / 3.0)
    factor = 8.0 * np.pi * np.pi / 3.0
    b_expected = factor * r * r * 0.5
    np.testing.assert_allclose(data[0, 1], b_expected, rtol=1e-6)


def test_atomicfluct_calcadp():
    system = _DummySystem()
    traj = _make_traj()

    data = atomicfluct(traj, system, mask="resid 1:2", calcadp=True)
    assert data.shape == (4, 7)
    # Only x varies for atoms 1 and 3; uxx > 0, others 0
    np.testing.assert_allclose(data[[0, 2], 1], 0.0, rtol=1e-6)
    np.testing.assert_allclose(data[[1, 3], 1], np.array([2.0 / 3.0, 2.0 / 3.0]), rtol=1e-6)
    np.testing.assert_allclose(data[:, 2:], 0.0, rtol=1e-6)
