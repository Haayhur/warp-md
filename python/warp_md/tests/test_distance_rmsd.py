import importlib

import numpy as np
import warp_md

from warp_md.analysis import _runtime as runtime_mod
from warp_md.analysis.rmsd import _pair_distances, distance_rmsd

rmsd_mod = importlib.import_module("warp_md.analysis.rmsd")


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


def test_distance_rmsd_translation_invariant():
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    trans = np.array([2.0, -1.0, 3.0], dtype=np.float64)
    frame1 = ref + trans
    coords = np.stack([ref, frame1], axis=0)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    vals = distance_rmsd(traj, system, mask="protein", ref=0)
    assert np.allclose(vals, 0.0, atol=1e-6)


def test_pair_distances_no_pbc_and_orthorhombic():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(_pair_distances(coords, "none", None), [3.0, 4.0, 5.0])
    wrapped = _pair_distances(
        np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=np.float64),
        "orthorhombic",
        np.array([10.0, 10.0, 10.0], dtype=np.float64),
    )
    np.testing.assert_allclose(wrapped, [1.0])


def test_pair_distances_uses_native_array_kernel_when_available(monkeypatch):
    called = {}

    def fake_native(coords, pbc, box):
        called["coords_shape"] = coords.shape
        called["coords_dtype"] = coords.dtype
        called["pbc"] = pbc
        called["box"] = None if box is None else box.tolist()
        return np.array([8.0], dtype=np.float64)

    monkeypatch.setattr(warp_md, "pair_distances_array", fake_native, raising=False)
    out = _pair_distances(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        "orthorhombic",
        np.array([10.0, 10.0, 10.0], dtype=np.float32),
    )
    np.testing.assert_allclose(out, [8.0])
    assert called == {
        "coords_shape": (2, 3),
        "coords_dtype": np.dtype("float64"),
        "pbc": "orthorhombic",
        "box": [10.0, 10.0, 10.0],
    }


def test_distance_rmsd_ref_last():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    vals = distance_rmsd(traj, system, mask="protein", ref=-1)
    assert vals[1] == 0.0
    assert vals[0] > 0.0


def test_distance_rmsd_ref0_native_does_not_materialize_in_python(monkeypatch):
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    coords = np.stack([ref, ref + np.array([2.0, -1.0, 3.0], dtype=np.float32)], axis=0)
    system = warp_md.System.from_arrays(_DummySystem(coords.shape[1]).atom_table(), positions0=ref)

    def _fail_read_all(*args, **kwargs):
        raise AssertionError("native distance_rmsd ref=0 path should stream through Rust")

    monkeypatch.setattr(rmsd_mod, "read_all_frames", _fail_read_all)
    monkeypatch.setattr(runtime_mod, "read_all_frames", _fail_read_all)
    vals = rmsd_mod.distance_rmsd(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="all",
        ref=0,
        chunk_frames=1,
    )
    np.testing.assert_allclose(vals, np.zeros(2, dtype=np.float32), atol=1e-6)


def test_distance_rmsd_nonzero_ref_native_does_not_materialize_in_python(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_DummySystem(coords.shape[1]).atom_table(), positions0=coords[0])

    def _fail_read_all(*args, **kwargs):
        raise AssertionError("native distance_rmsd nonzero reference should avoid full-frame reads")

    monkeypatch.setattr(rmsd_mod, "read_all_frames", _fail_read_all)
    monkeypatch.setattr(runtime_mod, "read_all_frames", _fail_read_all)
    vals = rmsd_mod.distance_rmsd(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="all",
        ref=-1,
        chunk_frames=1,
    )
    assert vals[1] == 0.0
    assert vals[0] > 0.0
