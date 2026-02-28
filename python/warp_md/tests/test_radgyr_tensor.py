import numpy as np

from warp_md.analysis.structure import radgyr_tensor
from warp_md.analysis.trajectory import ArrayTrajectory


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


def test_radgyr_tensor_basic():
    coords = np.array(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = ArrayTrajectory(coords)
    system = _DummySystem(coords.shape[1])

    rg, tensor = radgyr_tensor(traj, system, mask="all")
    np.testing.assert_allclose(rg, np.array([1.0, 2.0], dtype=np.float32), rtol=1e-6)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(tensor, expected, rtol=1e-6)

    traj.reset()
    out = radgyr_tensor(traj, system, mask="all", frame_indices=[1], dtype="dict")
    assert set(out.keys()) == {"rg", "tensor"}
    np.testing.assert_allclose(out["rg"], np.array([2.0], dtype=np.float32), rtol=1e-6)
