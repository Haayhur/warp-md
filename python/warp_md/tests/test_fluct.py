import importlib

import numpy as np
import warp_md

from warp_md.analysis.fluct import _aggregate, atomicfluct, bfactors, rmsf
from warp_md.analysis import _runtime as runtime_mod
from warp_md.analysis.trajectory import ArrayTrajectory

fluct_mod = importlib.import_module("warp_md.analysis.fluct")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "name": ["CA", "CB", "CA", "CB"],
            "resname": ["ALA", "ALA", "GLY", "GLY"],
            "resid": [1, 1, 2, 2],
            "chain_id": [0, 0, 0, 0],
            "mass": [1.0, 1.0, 1.0, 1.0],
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(range(len(self._atoms["name"])))


def _make_traj():
    coords = np.zeros((3, 4, 3), dtype=np.float32)
    coords[:, 1, 0] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    coords[:, 3, 0] = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    return ArrayTrajectory(coords)


def _native_inputs():
    coords = np.zeros((3, 4, 3), dtype=np.float32)
    coords[:, 1, 0] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    coords[:, 3, 0] = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    system = warp_md.System.from_arrays(_DummySystem().atom_table(), positions0=coords[0])
    return coords, system


def test_rmsf_byatom_byres():
    system = _DummySystem()
    traj = _make_traj()

    data = rmsf(traj, system, mask="resid 1:2", byres=False)
    assert data.shape == (4, 2)
    r = np.sqrt(2.0 / 3.0)
    np.testing.assert_allclose(data[:, 1], np.array([0.0, r, 0.0, r]), rtol=1e-6)

    traj.reset()
    byres = rmsf(traj, system, mask="resid 1:2", byres=True)
    assert byres.shape == (2, 2)
    np.testing.assert_allclose(byres[:, 0], np.array([1.0, 2.0]), rtol=1e-6)
    np.testing.assert_allclose(byres[:, 1], np.array([r / 2.0, r / 2.0]), rtol=1e-6)


def test_fluct_aggregate_fallback_modes():
    system = _DummySystem()
    values = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int64)

    byatom = _aggregate(values, indices, system, byres=False, bymask=False)
    np.testing.assert_allclose(
        byatom,
        np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]], dtype=np.float32),
    )
    byres = _aggregate(values, indices, system, byres=True, bymask=False)
    np.testing.assert_allclose(byres, np.array([[1.0, 2.0], [2.0, 6.0]], dtype=np.float32))
    bymask = _aggregate(values, indices, system, byres=False, bymask=True)
    np.testing.assert_allclose(bymask, np.array([[0.0, 4.0]], dtype=np.float32))


def test_fluct_aggregate_uses_native_array_kernel_when_available(monkeypatch):
    system = _DummySystem()
    called = {}

    def fake_native(values, indices, resids, mode):
        called["values_shape"] = values.shape
        called["indices_shape"] = indices.shape
        called["resids"] = None if resids is None else resids.tolist()
        called["mode"] = mode
        return np.array([[9.0, 2.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "fluct_aggregate_array", fake_native, raising=False)
    out = _aggregate(
        np.array([1.0, 3.0], dtype=np.float64),
        np.array([0, 1], dtype=np.int64),
        system,
        byres=True,
        bymask=False,
    )
    np.testing.assert_allclose(out, np.array([[9.0, 2.0]], dtype=np.float32))
    assert called == {
        "values_shape": (2,),
        "indices_shape": (2,),
        "resids": [1, 1],
        "mode": "byres",
    }


def test_bfactors_bymask():
    system = _DummySystem()
    traj = _make_traj()

    data = bfactors(traj, system, mask="resid 1:2", byres=False, bymask=True)
    assert data.shape == (1, 2)
    r = np.sqrt(2.0 / 3.0)
    factor = 8.0 * np.pi * np.pi / 3.0
    b_expected = factor * r * r * 0.5
    np.testing.assert_allclose(data[0, 1], b_expected, rtol=1e-6)


def test_atomicfluct_calcadp():
    system = _DummySystem()
    traj = _make_traj()

    data = atomicfluct(traj, system, mask="resid 1:2", calcadp=True)
    assert data.shape == (4, 7)
    # Only x varies for atoms 1 and 3; uxx > 0, others 0
    np.testing.assert_allclose(data[[0, 2], 1], 0.0, rtol=1e-6)
    np.testing.assert_allclose(data[[1, 3], 1], np.array([2.0 / 3.0, 2.0 / 3.0]), rtol=1e-6)
    np.testing.assert_allclose(data[:, 2:], 0.0, rtol=1e-6)


def test_rmsf_native_frame_indices_do_not_materialize_in_python(monkeypatch):
    coords, system = _native_inputs()

    def _fail_read_all(*args, **kwargs):
        raise AssertionError("native rmsf frame subset should stream through Rust")

    monkeypatch.setattr(runtime_mod, "read_all_frames", _fail_read_all)
    data = fluct_mod.rmsf(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="resid 1:2",
        byres=False,
        frame_indices=[2, 0],
        chunk_frames=1,
    )
    assert data.shape == (4, 2)
    np.testing.assert_allclose(data[:, 1], np.array([0.0, 1.0, 0.0, 1.0]), rtol=1e-6)


def test_atomicfluct_adp_native_frame_indices_do_not_materialize_in_python(monkeypatch):
    coords, system = _native_inputs()

    def _fail_read_all(*args, **kwargs):
        raise AssertionError("native atomicfluct ADP frame subset should stream through Rust")

    monkeypatch.setattr(runtime_mod, "read_all_frames", _fail_read_all)
    data = fluct_mod.atomicfluct(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="resid 1:2",
        calcadp=True,
        frame_indices=[2, 0],
        chunk_frames=1,
    )
    assert data.shape == (4, 7)
    np.testing.assert_allclose(data[[1, 3], 1], np.array([1.0, 1.0]), rtol=1e-6)
    np.testing.assert_allclose(data[[0, 2], 1:], 0.0, atol=1e-6)
