import numpy as np
import pytest

import warp_md
from warp_md.analysis.multipucker import multipucker


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._n_atoms = n_atoms
        self._atoms = {"resid": list(range(n_atoms))}

    def select(self, _mask):
        return _DummySelection(range(self._n_atoms))

    def select_indices(self, indices):
        return _DummySelection(indices)

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


def test_multipucker_bins(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(self, _sel, bins=10, mode="legacy", range_max=None, normalize=True):
            assert bins == 3
            assert mode == "histogram"
            assert range_max is None
            assert normalize is True

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            assert frame_indices is None
            return np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "MultiPuckerPlan", _DummyPlan, raising=False)
    out = multipucker(traj, system, mask="all", bins=3)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(out[1], np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_multipucker_histogram_uses_plan(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, bins=10, mode="legacy", range_max=None, normalize=True):
            called["bins"] = bins
            called["mode"] = mode
            called["range_max"] = range_max
            called["normalize"] = normalize

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            assert frame_indices == [1]
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "MultiPuckerPlan", _DummyPlan, raising=False)
    out = multipucker(
        traj,
        system,
        mask="all",
        bins=2,
        mode="histogram",
        range_max=2.0,
        normalize=False,
        frame_indices=[1],
    )
    np.testing.assert_allclose(out, np.array([[1.0, 0.0]], dtype=np.float32))
    assert called["bins"] == 2
    assert called["mode"] == "histogram"
    assert called["range_max"] == 2.0
    assert called["normalize"] is False
    assert called["frame_indices"] == [1]


def test_multipucker_histogram_auto_range_uses_plan(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, bins=10, mode="legacy", range_max=None, normalize=True):
            called["bins"] = bins
            called["mode"] = mode
            called["range_max"] = range_max
            called["normalize"] = normalize

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            assert frame_indices == [0]
            return np.array([[0.1, 0.9]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "MultiPuckerPlan", _DummyPlan, raising=False)
    out = multipucker(
        traj,
        system,
        mask="all",
        bins=2,
        mode="histogram",
        range_max=None,
        normalize=True,
        frame_indices=[0],
    )
    np.testing.assert_allclose(out, np.array([[0.1, 0.9]], dtype=np.float32))
    assert called["bins"] == 2
    assert called["mode"] == "histogram"
    assert called["range_max"] is None
    assert called["normalize"] is True
    assert called["frame_indices"] == [0]


def test_multipucker_mode_case_normalized(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, bins=10, mode="legacy", range_max=None, normalize=True):
            called["mode"] = mode

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "MultiPuckerPlan", _DummyPlan, raising=False)
    out = multipucker(traj, system, mask="all", bins=2, mode="HISTOGRAM")
    np.testing.assert_allclose(out, np.array([[1.0, 0.0]], dtype=np.float32))
    assert called["mode"] == "histogram"


def test_multipucker_rejects_invalid_contract_values(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(self, _sel, bins=10, mode="legacy", range_max=None, normalize=True):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "MultiPuckerPlan", _DummyPlan, raising=False)
    with pytest.raises(ValueError, match="bins must be a positive integer"):
        multipucker(traj, system, mask="all", bins=0)
    with pytest.raises(ValueError, match="mode must be 'histogram' or 'legacy'"):
        multipucker(traj, system, mask="all", mode="bad")
    with pytest.raises(ValueError, match="normalize must be bool"):
        multipucker(traj, system, mask="all", normalize=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="range_max must be a finite value > 0"):
        multipucker(traj, system, mask="all", mode="histogram", range_max=0.0)
