import importlib

import numpy as np
import pytest
import warp_md
rotdif_module = importlib.import_module("warp_md.analysis.rotdif")

from warp_md.analysis.rotdif import rotdif


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._n_atoms = n_atoms
        self._atoms = {"resid": list(range(n_atoms))}

    def select(self, _mask):
        return _DummySelection(list(range(self._n_atoms)))

    def select_indices(self, indices):
        return _DummySelection(list(indices))

    def atom_table(self):
        return self._atoms


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_rotdif_uses_plan(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, **kwargs):
            called["kwargs"] = kwargs

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            time = np.array([0.0, 1.0, 2.0], dtype=np.float32)
            data = np.array([[1.0, 1.0], [0.5, 0.2], [0.1, 0.01]], dtype=np.float32)
            return time, data

    monkeypatch.setattr(warp_md, "RotAcfPlan", _DummyPlan, raising=False)
    time, data = rotdif(traj, system, mask="all", orientation=[1, 2], group_by="all", max_lag=2)
    assert time.shape[0] == 3
    assert data.shape == (3, 2)
    assert called["kwargs"]["group_by"] == "all"
    assert called["kwargs"]["orientation"] == [1, 2]
    assert called["kwargs"]["max_lag"] == 2
    assert called["frame_indices"] is None


def test_rotdif_fit_output(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(self, _sel, **kwargs):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            time = np.array([0.0, 1.0, 2.0], dtype=np.float32)
            data = np.array([[1.0, 1.0], [0.8, 0.6], [0.6, 0.4]], dtype=np.float32)
            return time, data

    monkeypatch.setattr(warp_md, "RotAcfPlan", _DummyPlan, raising=False)
    monkeypatch.setattr(
        rotdif_module,
        "_fit_diffusion_rust",
        lambda _t, _d, _fc, _fw: {
            "d_rot": 0.1,
            "tau": 2.0,
            "slope": -0.2,
            "intercept": -0.1,
            "n_fit": 2,
        },
        raising=False,
    )
    out = rotdif(
        traj,
        system,
        mask="all",
        orientation=[1, 2],
        group_by="all",
        max_lag=2,
        return_fit=True,
    )
    assert set(out.keys()) >= {"time", "data", "d_rot", "tau", "slope", "intercept", "n_fit"}
    assert out["d_rot"] >= 0.0


def test_rotdif_frame_indices_passed_to_plan(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(self, _sel, **kwargs):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            assert frame_indices == [0]
            return np.array([0.0], dtype=np.float32), np.array([[1.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "RotAcfPlan", _DummyPlan, raising=False)
    time, data = rotdif(
        traj,
        system,
        mask="all",
        orientation=[1, 2],
        group_by="all",
        frame_indices=[0],
    )
    assert time.shape == (1,)
    assert data.shape == (1, 2)


def test_rotdif_rejects_invalid_orientation(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(self, _sel, **kwargs):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([0.0], dtype=np.float32), np.array([[1.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "RotAcfPlan", _DummyPlan, raising=False)
    with pytest.raises(ValueError, match="orientation must have length 2 or 3"):
        rotdif(traj, system, mask="all", orientation=[1], group_by="all")


def test_rotdif_rejects_invalid_fit_contract(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(self, _sel, **kwargs):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([0.0], dtype=np.float32), np.array([[1.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "RotAcfPlan", _DummyPlan, raising=False)
    with pytest.raises(ValueError, match="fit_component must be 'p1' or 'p2'"):
        rotdif(traj, system, mask="all", orientation=[1, 2], return_fit=True, fit_component="bad")
    with pytest.raises(ValueError, match="fit_window must be a 2-item tuple/list"):
        rotdif(traj, system, mask="all", orientation=[1, 2], return_fit=True, fit_window=(1.0,))  # type: ignore[arg-type]


def test_rotdif_rejects_non_bool_p2_legendre(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(self, _sel, **kwargs):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([0.0], dtype=np.float32), np.array([[1.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "RotAcfPlan", _DummyPlan, raising=False)
    with pytest.raises(ValueError, match="p2_legendre must be bool"):
        rotdif(traj, system, mask="all", orientation=[1, 2], p2_legendre=1)  # type: ignore[arg-type]
