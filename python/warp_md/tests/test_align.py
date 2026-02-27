import numpy as np

from warp_md.analysis.align import align, align_principal_axis


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


def test_align_basic():
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    rot = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    trans = np.array([2.0, 3.0, 0.0], dtype=np.float64)
    frame1 = ref @ rot.T + trans
    coords = np.stack([ref, frame1], axis=0)

    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    aligned = align(traj, system, mask="protein", ref=0)
    chunk = aligned.read_chunk()
    out = chunk["coords"]
    assert np.allclose(out[0], ref, atol=1e-5)
    assert np.allclose(out[1], ref, atol=1e-5)


def test_align_principal_axis_centered():
    coords = np.array(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    aligned = align_principal_axis(traj, system, mask="protein", mass=False)
    chunk = aligned.read_chunk()
    out = chunk["coords"]
    assert out.shape == coords.shape
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-5)
