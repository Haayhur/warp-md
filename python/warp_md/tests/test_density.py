import numpy as np

from warp_md.analysis.density import density


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


def test_density_basic():
    coords = np.array(
        [
            [[0.0, 0.0, 0.1], [0.0, 0.0, 0.9], [0.0, 0.0, 1.9]],
            [[0.0, 0.0, 0.2], [0.0, 0.0, 1.1], [0.0, 0.0, 1.8]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out = density(traj, system, mask="all", density_type="number", delta=1.0, direction="z")
    assert "density" in out
    assert "z" in out
    assert out["density"].shape == out["z"].shape
    assert np.isclose(out["density"].sum() * 1.0, 3.0, atol=1e-6)
