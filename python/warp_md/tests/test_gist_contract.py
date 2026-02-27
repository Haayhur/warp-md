import numpy as np
import pytest
import warp_md

from ._gist_test_utils import (
    _DummySystem,
    _FakeTraj,
    _build_system_and_topology,
    _direct_payload,
)

def test_gist_frame_indices(monkeypatch):
    from warp_md.analysis.gist import GistConfig, gist

    system, top = _build_system_and_topology()
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.05, 0.0], [0.1, -0.05, 0.0]],
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.05, 0.0], [0.2, -0.05, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _FakeTraj(coords)
    dummy_system = _DummySystem()
    called = {}

    class _DummyGistDirectPlan:
        def __init__(self, *args, **kwargs):
            called["created"] = True
            called["kwargs"] = kwargs

        def run(self, _traj, _system, chunk_frames=None, device="auto"):
            called["run"] = True
            counts = np.zeros((2, 2, 2), dtype=np.float64)
            counts[0, 0, 0] = 1.0
            orient = np.zeros((2, 2, 2, 4), dtype=np.float64)
            orient[0, 0, 0, -1] = 1.0
            energy_sw = np.zeros((2, 2, 2), dtype=np.float64)
            energy_ww = np.zeros((2, 2, 2), dtype=np.float64)
            return _direct_payload(
                counts,
                orient,
                energy_sw,
                energy_ww,
                (0.0, 0.0, 0.0),
                1,
                0.0,
                0.0,
            )

    monkeypatch.setattr(warp_md, "GistDirectPlan", _DummyGistDirectPlan, raising=False)
    config = GistConfig(
        length_scale=1.0,
        frame_indices=[1],
        grid_spacing=0.2,
        padding=0.4,
        orientation_bins=4,
        energy_method="direct",
    )
    out = gist(traj, dummy_system, system, top, config=config)
    assert called["created"] is True
    assert called["run"] is True
    assert called["kwargs"]["frame_indices"] == [1]
    assert called["kwargs"]["orientation_bins"] == 4
    assert out.n_frames == 1
    assert int(out.counts.sum()) == 1


def test_gist_none_uses_rust_grid_plan(monkeypatch):
    from warp_md.analysis.gist import GistConfig, gist

    system, top = _build_system_and_topology()
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.05, 0.0], [0.2, -0.05, 0.0]],
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.05, 0.0], [0.3, -0.05, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _FakeTraj(coords)
    dummy_system = _DummySystem()
    called = {}

    class _DummyGistGridPlan:
        def __init__(self, *args, **kwargs):
            called["created"] = True
            called["kwargs"] = kwargs

        def run(self, _traj, _system, chunk_frames=None, device="auto"):
            called["run"] = True
            counts = np.ones((2, 2, 2), dtype=np.float32)
            orient = np.zeros((2, 2, 2, 4), dtype=np.float32)
            orient[..., -1] = 1.0
            return counts, orient, (0.0, 0.0, 0.0), 2

    monkeypatch.setattr(warp_md, "GistGridPlan", _DummyGistGridPlan, raising=False)
    config = GistConfig(length_scale=1.0, energy_method="none", orientation_bins=4)
    out = gist(traj, dummy_system, system, top, config=config)
    assert called["created"] is True
    assert called["run"] is True
    assert called["kwargs"]["spacing"] == 0.1
    assert called["kwargs"]["orientation_bins"] == 4
    assert out.n_frames == 2
    assert out.counts.shape == (2, 2, 2)
    assert np.allclose(out.energy_sw, 0.0)
    assert np.allclose(out.energy_ww, 0.0)


def test_gist_none_no_python_fallback_when_plan_missing(monkeypatch):
    from warp_md.analysis.gist import GistConfig, gist

    system, top = _build_system_and_topology()
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.05, 0.0], [0.2, -0.05, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _FakeTraj(coords)
    dummy_system = _DummySystem()

    class _MissingPlan:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("GistGridPlan binding unavailable in this build.")

    monkeypatch.setattr(warp_md, "GistGridPlan", _MissingPlan, raising=False)
    config = GistConfig(length_scale=1.0, energy_method="none")
    with pytest.raises(RuntimeError, match="GistGridPlan binding unavailable"):
        gist(traj, dummy_system, system, top, config=config)


def test_gist_none_requires_rust_system():
    from warp_md.analysis.gist import GistConfig, gist

    system, top = _build_system_and_topology()
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.05, 0.0], [0.2, -0.05, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _FakeTraj(coords)
    config = GistConfig(length_scale=1.0, energy_method="none")
    with pytest.raises(ValueError, match="energy_method='none' requires"):
        gist(traj, None, system, top, config=config)


def test_gist_direct_no_python_fallback_when_plan_missing(monkeypatch):
    from warp_md.analysis.gist import GistConfig, gist

    system, top = _build_system_and_topology()
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.05, 0.0], [0.2, -0.05, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _FakeTraj(coords)
    dummy_system = _DummySystem()

    class _MissingPlan:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("GistDirectPlan binding unavailable in this build.")

    monkeypatch.setattr(warp_md, "GistDirectPlan", _MissingPlan, raising=False)
    config = GistConfig(length_scale=1.0, energy_method="direct")
    with pytest.raises(RuntimeError, match="GistDirectPlan binding unavailable"):
        gist(traj, dummy_system, system, top, config=config)


def test_gist_rejects_invalid_pme_totals_source():
    from warp_md.analysis.gist import GistConfig, gist

    config = GistConfig(energy_method="pme", pme_totals_source="invalid")
    with pytest.raises(ValueError, match="pme_totals_source"):
        gist(None, None, None, None, config=config)


def test_gist_direct_rejects_noncanonical_output_arity(monkeypatch):
    from warp_md.analysis.gist import GistConfig, gist

    system, top = _build_system_and_topology()
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.05, 0.0], [0.2, -0.05, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _FakeTraj(coords)
    dummy_system = _DummySystem()

    class _DummyGistDirectPlan:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto"):
            counts = np.ones((1, 1, 1), dtype=np.float64)
            orient = np.ones((1, 1, 1, 4), dtype=np.float64)
            energy_sw = np.array([[[1.0]]], dtype=np.float64)
            energy_ww = np.array([[[1.0]]], dtype=np.float64)
            # Legacy shape intentionally returned to enforce hard-cut behavior.
            return counts, orient, energy_sw, energy_ww, (0.0, 0.0, 0.0), 1, 1.0, 1.0

    monkeypatch.setattr(warp_md, "GistDirectPlan", _DummyGistDirectPlan, raising=False)
    config = GistConfig(length_scale=1.0, energy_method="direct", orientation_bins=4)
    with pytest.raises(RuntimeError, match="expected 16"):
        gist(traj, dummy_system, system, top, config=config)

