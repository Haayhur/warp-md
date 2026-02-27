import numpy as np

from warp_md.analysis.voxel import count_in_voxel


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


def test_count_in_voxel():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.4, 0.4, 0.4]],
            [[1.1, 0.0, 0.0], [2.0, 0.0, 0.0], [0.4, 0.4, 0.4]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out = count_in_voxel(traj, system, mask="all", voxel_cntr=(0.0, 0.0, 0.0), voxel_size=1.0)
    assert out[0] == [0, 2]
    assert out[1] == [2]
