import numpy as np

from warp_md.analysis.geometry import angle, dihedral


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._atoms = {
            "name": ["A"] * n_atoms,
            "resname": ["RES"] * n_atoms,
            "resid": list(range(1, n_atoms + 1)),
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


def test_angle_basic():
    coords = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    out = angle(traj, system, "@1 @2 @3")
    np.testing.assert_allclose(out, np.array([90.0, 90.0], dtype=np.float32), rtol=1e-6)

    traj2 = _DummyTraj(coords)
    out2 = angle(traj2, system, np.array([[0, 1, 2]], dtype=np.int64))
    np.testing.assert_allclose(out2[0], np.array([90.0, 90.0], dtype=np.float32), rtol=1e-6)


def test_dihedral_basic():
    coords = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    out = dihedral(traj, system, "@1 @2 @3 @4")
    np.testing.assert_allclose(out, np.array([180.0], dtype=np.float32), rtol=1e-6, atol=1e-6)

    traj2 = _DummyTraj(coords)
    out2 = dihedral(traj2, system, np.array([[0, 1, 2, 3]], dtype=np.int64), range360=True)
    np.testing.assert_allclose(out2[0], np.array([180.0], dtype=np.float32), rtol=1e-6, atol=1e-6)
