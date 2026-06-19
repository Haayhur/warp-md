import numpy as np
import pytest

import warp_md
from warp_md.analysis.infraredspec import _velocity_differences, infraredspec


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
    def __init__(self, coords, time_ps=None):
        self._coords = coords
        self._time = time_ps
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        chunk = {"coords": self._coords}
        if self._time is not None:
            chunk["time_ps"] = self._time
        return chunk


def test_infraredspec_infer_timestep():
    coords = np.zeros((5, 2, 3), dtype=np.float32)
    coords[:, 0, 0] = np.arange(5, dtype=np.float32)
    coords[:, 1, 0] = np.arange(5, dtype=np.float32) * 2.0
    time_ps = np.array([0.0, 0.002, 0.004, 0.006, 0.008], dtype=np.float64)
    traj = _DummyTraj(coords, time_ps=time_ps)
    system = _DummySystem(coords.shape[1])

    freq, spec = infraredspec(traj, system, "all", timestep_fs=None, timestep_ps=None, freq_unit="hz")
    assert freq.shape[0] == spec.shape[0]
    dt_s = 0.002e-12
    expected = np.fft.rfftfreq(4, dt_s)
    np.testing.assert_allclose(freq, expected.astype(np.float32), rtol=1e-6)


def test_velocity_differences_native_route(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[5.0, 0.0, 3.0]]], dtype=np.float64)
    out = _velocity_differences(coords, np.array([2.0], dtype=np.float64), 1.0)
    np.testing.assert_allclose(out[:, 0, :], np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 3.0]]))

    called = {}

    def fake_native(sel_coords, dt_steps, dt_ps):
        called["shape"] = sel_coords.shape
        called["dt"] = dt_steps.tolist()
        called["fallback"] = dt_ps
        return np.ones((1, 1, 3), dtype=np.float64)

    monkeypatch.setattr(warp_md, "velocity_differences_array", fake_native, raising=False)
    fake_native.__name__ = "velocity_differences_array"
    native_out = _velocity_differences(coords[:2], np.array([2.0], dtype=np.float64), 1.5)
    np.testing.assert_allclose(native_out, np.ones((1, 1, 3), dtype=np.float64))
    assert called == {"shape": (2, 1, 3), "dt": [2.0], "fallback": 1.5}


def test_infraredspec_requires_timestep_without_time():
    coords = np.zeros((4, 1, 3), dtype=np.float32)
    traj = _DummyTraj(coords, time_ps=None)
    system = _DummySystem(coords.shape[1])
    with pytest.raises(ValueError):
        infraredspec(traj, system, "all", timestep_fs=None, timestep_ps=None)
