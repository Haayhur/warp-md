import numpy as np

from warp_md.analysis.closest import closest, closest_atom


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


def test_closest_atom():
    coords = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    system = _DummySystem(coords.shape[0])
    idx = closest_atom(system, coords, point=(1.0, 0.0, 0.0))
    assert idx == 0


def test_closest_basic():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    out = closest(traj, system, mask="all", n_solvents=2)
    chunk = out.read_chunk()
    assert chunk["coords"].shape == (1, 2, 3)
