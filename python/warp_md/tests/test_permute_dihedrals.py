import numpy as np
from pathlib import Path

from warp_md.analysis.permute_dihedrals import permute_dihedrals


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
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


def test_permute_dihedrals_writes(tmp_path: Path):
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
    outfile = tmp_path / "permuted.npz"

    permute_dihedrals(traj, system, str(outfile), dihedral_types="phi psi")
    assert outfile.exists()
    data = np.load(outfile)
    assert data.files
