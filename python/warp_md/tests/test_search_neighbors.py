import numpy as np

from warp_md.analysis.neighbors import search_neighbors


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


def test_search_neighbors_basic():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out = search_neighbors(traj, system, mask="all", distance=2.5)
    assert isinstance(out, list)
    assert list(out[0].keys()) == ["0"]
    # all atoms within 2.5 of at least one selected atom (all selected)
    assert out[0]["0"].shape[0] == 3

    traj2 = _DummyTraj(coords)
    out2 = search_neighbors(traj2, system, mask="all", distance=1.0)
    # self-counting keeps all atoms within cutoff of themselves
    assert out2[0]["0"].shape[0] == 3
