import numpy as np

from warp_md.analysis.transform import center, rotate, scale, transform, translate


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
    def __init__(self, coords, box=None):
        self._coords = coords
        self._box = box
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        out = {"coords": self._coords}
        if self._box is not None:
            out["box"] = self._box
        return out


def test_translate_and_scale():
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    out = translate(traj, [1.0, 2.0, 3.0]).read_chunk()["coords"]
    assert np.allclose(out[0, 0], [1.0, 2.0, 3.0])

    traj = _DummyTraj(coords)
    out = scale(traj, 2.0).read_chunk()["coords"]
    assert np.allclose(out[0, 1], [2.0, 0.0, 0.0])


def test_rotate_and_transform():
    coords = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    out = rotate(traj, rot).read_chunk()["coords"]
    assert np.allclose(out[0, 0], [0.0, 1.0, 0.0])

    traj = _DummyTraj(coords)
    out = transform(traj, rotation=rot, translation=[1.0, 0.0, 0.0]).read_chunk()["coords"]
    assert np.allclose(out[0, 0], [1.0, 1.0, 0.0])


def test_center_origin():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    out = center(traj, system, mask="all").read_chunk()["coords"]
    assert np.allclose(out.mean(axis=1), 0.0)
