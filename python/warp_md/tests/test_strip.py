import importlib

import numpy as np

import warp_md

from warp_md.analysis.structure import strip


structure_mod = importlib.import_module("warp_md.analysis.structure")


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


def test_strip_mask_indices():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    out = strip(traj, system, mask=[1]).read_chunk()["coords"]
    assert out.shape[1] == 2


def test_strip_native_without_python_read(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    box = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype=np.float32)
    time = np.array([0.5, 1.5], dtype=np.float32)
    atoms = {
        "name": ["CA", "CB", "CG"],
        "resname": ["ALA", "ALA", "ALA"],
        "resid": [1, 1, 1],
        "chain": ["A", "A", "A"],
        "element": ["C", "C", "C"],
        "mass": [1.0, 1.0, 1.0],
    }
    traj = warp_md.Trajectory.from_numpy(coords, box=box, time_ps=time)
    system = warp_md.System.from_arrays(atoms, positions0=coords[0])

    def _fail_read_all(*_args, **_kwargs):
        raise AssertionError("strip should not materialize native trajectory frames")

    monkeypatch.setattr(structure_mod, "read_all_frames", _fail_read_all)
    out = structure_mod.strip(traj, system, mask=[1]).read_chunk()
    np.testing.assert_allclose(out["coords"], coords[:, [0, 2], :])
    np.testing.assert_allclose(out["box"], box)
    np.testing.assert_allclose(out["time"], time)
