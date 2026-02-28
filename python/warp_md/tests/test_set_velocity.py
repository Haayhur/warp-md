import numpy as np

from warp_md.analysis.set_velocity import set_velocity


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


def test_set_velocity_shape_and_seed():
    coords = np.zeros((3, 2, 3), dtype=np.float32)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    vel1 = set_velocity(traj, system, temperature=100.0, ig=7, mask="all")
    traj2 = _DummyTraj(coords)
    vel2 = set_velocity(traj2, system, temperature=100.0, ig=7, mask="all")
    assert vel1.shape == (3, 2, 3)
    np.testing.assert_allclose(vel1, vel2)
