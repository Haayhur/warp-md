import numpy as np
import pytest

import warp_md
from warp_md.analysis.surf import molsurf, surf


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._n_atoms = n_atoms

    def select(self, _mask):
        return _DummySelection(range(self._n_atoms))

    def select_indices(self, indices):
        return _DummySelection(indices)


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_surf_bbox_area(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(self, _sel, algorithm="bbox", probe_radius=1.4, n_sphere_points=64, radii=None):
            assert algorithm == "bbox"

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            assert frame_indices is None
            return np.array([6.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    out = surf(traj, system, mask=[0, 1], algorithm="bbox")
    np.testing.assert_allclose(out, np.array([6.0, 0.0], dtype=np.float32), atol=1e-5)


def test_molsurf_alias(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(self, _sel, algorithm="bbox", probe_radius=1.4, n_sphere_points=64, radii=None):
            assert algorithm == "bbox"

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            assert frame_indices is None
            dx, dy, dz = 1.0, 2.0, 3.0
            expected = 2.0 * (dx * dy + dy * dz + dx * dz)
            return np.array([expected], dtype=np.float32)

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    out = molsurf(traj, system, mask=[0, 1], algorithm="bbox")
    dx, dy, dz = 1.0, 2.0, 3.0
    expected = 2.0 * (dx * dy + dy * dz + dx * dz)
    np.testing.assert_allclose(out, np.array([expected], dtype=np.float32), atol=1e-5)


def test_surf_sasa_positive(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(self, _sel, algorithm="bbox", probe_radius=1.4, n_sphere_points=64, radii=None):
            assert algorithm == "sasa"
            assert probe_radius == 0.0
            assert n_sphere_points == 32
            assert radii == [1.0]

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            assert frame_indices is None
            expected = 4.0 * np.pi * 1.0 * 1.0
            return np.array([expected], dtype=np.float32)

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    out = surf(traj, system, mask=[0], algorithm="sasa", probe_radius=0.0, n_sphere_points=32, radii=[1.0])
    expected = 4.0 * np.pi * 1.0 * 1.0
    np.testing.assert_allclose(out, np.array([expected], dtype=np.float32), rtol=0.15)


def test_surf_sasa_uses_plan(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, algorithm="bbox", probe_radius=1.4, n_sphere_points=64, radii=None):
            called["algorithm"] = algorithm
            called["probe_radius"] = probe_radius
            called["n_sphere_points"] = n_sphere_points
            called["radii"] = radii

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            values = np.array([10.0, 20.0], dtype=np.float32)
            if frame_indices is None:
                return values
            return values[np.asarray(frame_indices, dtype=np.int64)]

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    out = surf(
        traj,
        system,
        mask=[0],
        algorithm="sasa",
        probe_radius=0.0,
        n_sphere_points=16,
        radii=[1.0],
        frame_indices=[1],
    )
    np.testing.assert_allclose(out, np.array([20.0], dtype=np.float32))
    assert called["algorithm"] == "sasa"
    assert called["probe_radius"] == 0.0
    assert called["n_sphere_points"] == 16
    assert called["frame_indices"] == [1]


def test_surf_no_python_fallback_when_plan_missing(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _MissingPlan:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("SurfPlan binding unavailable in this build.")

    monkeypatch.setattr(warp_md, "SurfPlan", _MissingPlan, raising=False)
    try:
        surf(traj, system, mask=[0], algorithm="sasa")
    except RuntimeError as exc:
        assert "SurfPlan binding unavailable" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing SurfPlan binding")


def test_surf_algorithm_case_normalized(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, algorithm="bbox", probe_radius=1.4, n_sphere_points=64, radii=None):
            called["algorithm"] = algorithm

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    out = surf(traj, system, mask=[0], algorithm="SASA")
    np.testing.assert_allclose(out, np.array([1.0], dtype=np.float32))
    assert called["algorithm"] == "sasa"


def test_surf_rejects_invalid_numeric_contract(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(self, _sel, algorithm="bbox", probe_radius=1.4, n_sphere_points=64, radii=None):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    with pytest.raises(ValueError, match="probe_radius must be a finite value >= 0"):
        surf(traj, system, mask=[0], probe_radius=-0.1)
    with pytest.raises(ValueError, match="n_sphere_points must be a positive integer"):
        surf(traj, system, mask=[0], n_sphere_points=0)
    with pytest.raises(ValueError, match="radii values must be finite and > 0"):
        surf(traj, system, mask=[0], radii=[1.0, np.inf])
