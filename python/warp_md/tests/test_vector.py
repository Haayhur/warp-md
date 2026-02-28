import numpy as np

from warp_md.analysis.vector import vector, vector_mask


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
    def __init__(self, coords, box=None):
        self._coords = coords
        self._box = box
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        chunk = {"coords": self._coords}
        if self._box is not None:
            chunk["box"] = self._box
        return chunk


def test_vector_mask_basic():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out = vector(traj, system, "@1 @2")
    np.testing.assert_allclose(out[:, 0], np.array([1.0, 2.0], dtype=np.float32), rtol=1e-6)

    traj2 = _DummyTraj(coords)
    out2 = vector_mask(traj2, system, np.array([[0, 1]], dtype=np.int64))
    np.testing.assert_allclose(out2[0, :, 0], np.array([1.0, 2.0], dtype=np.float32), rtol=1e-6)


def test_vector_boxcenter():
    coords = np.zeros((1, 1, 3), dtype=np.float32)
    box = np.array([[4.0, 6.0, 8.0]], dtype=np.float32)
    traj = _DummyTraj(coords, box=box)
    system = _DummySystem(coords.shape[1])

    out = vector(traj, system, "boxcenter")
    np.testing.assert_allclose(out[0], np.array([2.0, 3.0, 4.0], dtype=np.float32), rtol=1e-6)
