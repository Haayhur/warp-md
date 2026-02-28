import numpy as np
import pytest
import warp_md

from warp_md.analysis.pucker import pucker


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._n_atoms = n_atoms

    def select(self, _mask):
        return _DummySelection(list(range(self._n_atoms)))

    def select_indices(self, indices):
        return _DummySelection(list(indices))

    def atom_table(self):
        return {"resid": list(range(self._n_atoms))}


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_pucker_max_radius_uses_plan(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, metric="max_radius", return_phase=False):
            called["metric"] = metric
            called["return_phase"] = return_phase

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            return np.array([1.0, 1.5], dtype=np.float32)

    monkeypatch.setattr(warp_md, "PuckerPlan", _DummyPlan, raising=False)
    out = pucker(traj, system, mask="all", metric="max_radius")
    np.testing.assert_allclose(out, np.array([1.0, 1.5], dtype=np.float32), atol=1e-6)
    assert called["metric"] == "max_radius"
    assert called["return_phase"] is False
    assert called["frame_indices"] is None


def test_pucker_amplitude_uses_plan(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, metric="max_radius", return_phase=False):
            called["metric"] = metric
            called["return_phase"] = return_phase

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            return np.array([2.5], dtype=np.float32)

    monkeypatch.setattr(warp_md, "PuckerPlan", _DummyPlan, raising=False)
    out = pucker(traj, system, mask="all", metric="amplitude", return_phase=False)
    np.testing.assert_allclose(out, np.array([2.5], dtype=np.float32), atol=1e-6)
    assert called["metric"] == "amplitude"
    assert called["return_phase"] is False
    assert called["frame_indices"] is None


def test_pucker_return_phase_supported(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, metric="max_radius", return_phase=False):
            called["metric"] = metric
            called["return_phase"] = return_phase

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            return (
                np.array([2.5], dtype=np.float32),
                np.array([180.0], dtype=np.float32),
            )

    monkeypatch.setattr(warp_md, "PuckerPlan", _DummyPlan, raising=False)
    values, phase = pucker(
        traj,
        system,
        mask="all",
        metric="amplitude",
        return_phase=True,
        frame_indices=[0],
    )
    np.testing.assert_allclose(values, np.array([2.5], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(phase, np.array([180.0], dtype=np.float32), atol=1e-6)
    assert called["metric"] == "amplitude"
    assert called["return_phase"] is True
    assert called["frame_indices"] == [0]


def test_pucker_no_double_slice_after_rust_subset(monkeypatch):
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
        def __init__(self, _sel, metric="max_radius", return_phase=False):
            called["metric"] = metric
            called["return_phase"] = return_phase

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            assert frame_indices == [2]
            # Rust plan already applied frame subset and returns one frame.
            return np.array([9.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "PuckerPlan", _DummyPlan, raising=False)
    out = pucker(
        traj,
        system,
        mask="all",
        metric="max_radius",
        return_phase=False,
        frame_indices=[2],
    )
    np.testing.assert_allclose(out, np.array([9.0], dtype=np.float32), atol=1e-6)
    assert called["metric"] == "max_radius"
    assert called["return_phase"] is False
    assert called["frame_indices"] == [2]


def test_pucker_metric_case_normalized(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, metric="max_radius", return_phase=False):
            called["metric"] = metric

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([1.25], dtype=np.float32)

    monkeypatch.setattr(warp_md, "PuckerPlan", _DummyPlan, raising=False)
    out = pucker(traj, system, mask="all", metric="MAX_RADIUS")
    np.testing.assert_allclose(out, np.array([1.25], dtype=np.float32), atol=1e-6)
    assert called["metric"] == "max_radius"


def test_pucker_rejects_non_bool_return_phase(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(self, _sel, metric="max_radius", return_phase=False):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([1.25], dtype=np.float32)

    monkeypatch.setattr(warp_md, "PuckerPlan", _DummyPlan, raising=False)
    with pytest.raises(ValueError, match="return_phase must be bool"):
        pucker(traj, system, mask="all", return_phase=1)  # type: ignore[arg-type]


def test_pucker_rejects_legacy_constructor_signature(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _LegacyPlan:
        def __init__(self, _sel, metric="max_radius"):  # no return_phase arg
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "PuckerPlan", _LegacyPlan, raising=False)
    with pytest.raises(RuntimeError, match="requires updated Rust bindings"):
        pucker(traj, system, mask="all", metric="amplitude", return_phase=False)
