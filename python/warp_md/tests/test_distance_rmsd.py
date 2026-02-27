import numpy as np

from warp_md.analysis.rmsd import distance_rmsd


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


def test_distance_rmsd_translation_invariant():
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    trans = np.array([2.0, -1.0, 3.0], dtype=np.float64)
    frame1 = ref + trans
    coords = np.stack([ref, frame1], axis=0)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    vals = distance_rmsd(traj, system, mask="protein", ref=0)
    assert np.allclose(vals, 0.0, atol=1e-6)


def test_distance_rmsd_ref_last():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    vals = distance_rmsd(traj, system, mask="protein", ref=-1)
    assert vals[1] == 0.0
    assert vals[0] > 0.0
