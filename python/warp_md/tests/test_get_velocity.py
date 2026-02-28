import numpy as np

from warp_md.analysis.velocity import get_velocity


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
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(range(len(self._atoms["name"])))


class _DummyTraj:
    def __init__(self, coords, time=None):
        self._coords = coords
        self._time = time
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        out = {"coords": self._coords}
        if self._time is not None:
            out["time"] = self._time
        return out


def test_get_velocity_basic():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    vel = get_velocity(traj, system, mask="all")
    assert vel.shape == coords.shape
    assert np.allclose(vel[1, 0], [1.0, 0.0, 0.0])


def test_get_velocity_time_scale():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    time = np.array([0.0, 2.0], dtype=np.float64)
    traj = _DummyTraj(coords, time=time)
    system = _DummySystem(coords.shape[1])
    vel = get_velocity(traj, system, mask="all", time_scale=1.0)
    assert np.allclose(vel[1, 0], [1.0, 0.0, 0.0])
