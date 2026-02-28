import numpy as np

from warp_md.analysis.watershell import watershell


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "name": ["S", "W1", "W2", "W3"],
            "resid": [1, 2, 2, 2],
        }

    def atom_table(self):
        return self._atoms

    def select(self, mask):
        if mask == "SOLUTE":
            return _DummySelection([0])
        if mask == "SOLVENT":
            return _DummySelection([1, 2, 3])
        return _DummySelection([])


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_watershell_basic():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [4.5, 0.0, 0.0], [7.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem()

    counts = watershell(traj, system, solute_mask="SOLUTE", solvent_mask="SOLVENT", lower=3.0, upper=5.0)
    assert counts.shape == (2,)
    assert counts[0] == 1
    assert counts[1] == 2
