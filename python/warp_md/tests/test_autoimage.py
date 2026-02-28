import numpy as np

from warp_md.analysis.autoimage import autoimage


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
    def __init__(self, coords, box):
        self._coords = coords
        self._box = box
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords, "box": self._box}


def test_autoimage_basic():
    coords = np.array(
        [
            [[5.0, -1.0, 2.0], [12.0, 7.0, -3.0]],
        ],
        dtype=np.float32,
    )
    box = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    traj = _DummyTraj(coords, box)
    system = _DummySystem(coords.shape[1])

    out = autoimage(traj, system, mask="all")
    chunk = out.read_chunk()
    xyz = chunk["coords"]
    assert np.all((xyz >= 0.0) & (xyz <= 10.0))
