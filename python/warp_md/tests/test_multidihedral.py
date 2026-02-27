import numpy as np

from warp_md.analysis.multidihedral import multidihedral


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        # two residues with basic backbone atoms
        self._atoms = {
            "name": ["N", "CA", "C", "N", "CA", "C"],
            "resid": [1, 1, 1, 2, 2, 2],
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


def test_multidihedral_phi_psi():
    coords = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [3.0, 2.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem()

    out = multidihedral(traj, system, dihedral_types="phi psi", dtype="dict")
    assert isinstance(out, dict)
    assert out
    for key, val in out.items():
        assert val.shape == (1,)
