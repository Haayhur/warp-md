import numpy as np
import pytest
import warp_md

from ._gist_test_utils import (
    _DummySystem,
    _FakeTraj,
    _build_system_and_topology,
    _direct_payload,
)

def test_gist_pme_native_uses_rust_frame_totals(monkeypatch):
    import importlib

    gist_mod = importlib.import_module("warp_md.analysis.gist")
    from warp_md.analysis.gist import GistConfig, gist

    system, top = _build_system_and_topology()
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.05, 0.0], [0.2, -0.05, 0.0]],
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.05, 0.0], [0.3, -0.05, 0.0]],
        ],
        dtype=np.float64,
    )
    box = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=np.float64)
    traj = _FakeTraj(coords, box=box)
    dummy_system = _DummySystem()
    called = {"pme": False, "scaler": False, "native_flag": False}

    class _DummyGistDirectPlan:
        def __init__(self, *args, **kwargs):
            assert kwargs.get("record_frame_energies") is True
            called["native_flag"] = bool(kwargs.get("record_pme_frame_totals"))

        def run(self, _traj, _system, chunk_frames=None, device="auto"):
            counts = np.ones((2, 1, 1), dtype=np.float64)
            orient = np.ones((2, 1, 1, 4), dtype=np.float64)
            energy_sw = np.array([[[1.0]], [[5.0]]], dtype=np.float64)
            energy_ww = np.array([[[2.0]], [[4.0]]], dtype=np.float64)
            frame_direct_sw = np.array([1.0, 5.0], dtype=np.float64)
            frame_direct_ww = np.array([2.0, 4.0], dtype=np.float64)
            frame_offsets = np.array([0, 1, 2], dtype=np.uint64)
            frame_cells = np.array([0, 1], dtype=np.uint32)
            frame_sw = np.array([1.0, 5.0], dtype=np.float64)
            frame_ww = np.array([2.0, 4.0], dtype=np.float64)
            frame_native_sw = np.array([3.0, 6.0], dtype=np.float64)
            frame_native_ww = np.array([2.0, 5.0], dtype=np.float64)
            return _direct_payload(
                counts,
                orient,
                energy_sw,
                energy_ww,
                (0.0, 0.0, 0.0),
                2,
                6.0,
                6.0,
                frame_direct_sw,
                frame_direct_ww,
                frame_offsets,
                frame_cells,
                frame_sw,
                frame_ww,
                frame_native_sw,
                frame_native_ww,
            )

    def _pme_totals_should_not_run(*_args, **_kwargs):
        called["pme"] = True
        raise AssertionError("_pme_totals should not run when pme_totals_source='native'")

    def _fake_scaler(
        energy_sw_direct,
        energy_ww_direct,
        direct_sw_total,
        direct_ww_total,
        direct_sw_frame,
        direct_ww_frame,
        pme_sw_frame,
        pme_ww_frame,
        frame_offsets,
        frame_cells,
        frame_sw,
        frame_ww,
    ):
        called["scaler"] = True
        assert np.allclose(pme_sw_frame, np.array([3.0, 6.0], dtype=np.float64))
        assert np.allclose(pme_ww_frame, np.array([2.0, 5.0], dtype=np.float64))
        return np.asarray(energy_sw_direct), np.asarray(energy_ww_direct)

    monkeypatch.setattr(warp_md, "GistDirectPlan", _DummyGistDirectPlan, raising=False)
    monkeypatch.setattr(gist_mod, "_pme_totals", _pme_totals_should_not_run, raising=True)
    monkeypatch.setattr(warp_md, "gist_apply_pme_scaling", _fake_scaler, raising=False)
    config = GistConfig(
        length_scale=1.0,
        energy_method="pme",
        pme_totals_source="native",
        orientation_bins=4,
    )
    out = gist(traj, dummy_system, system, top, config=config)
    assert called["native_flag"] is True
    assert called["pme"] is False
    assert called["scaler"] is True
    assert out.n_frames == 2
    assert np.isclose(float(out.energy_sw[0, 0, 0]), 1.0)
    assert np.isclose(float(out.energy_sw[1, 0, 0]), 5.0)
    assert np.isclose(float(out.energy_ww[0, 0, 0]), 2.0)
    assert np.isclose(float(out.energy_ww[1, 0, 0]), 4.0)


def test_gist_pme_native_requires_rust_totals_payload(monkeypatch):
    from warp_md.analysis.gist import GistConfig, gist

    system, top = _build_system_and_topology()
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.05, 0.0], [0.2, -0.05, 0.0]],
        ],
        dtype=np.float64,
    )
    box = np.array([[2.0, 2.0, 2.0]], dtype=np.float64)
    traj = _FakeTraj(coords, box=box)
    dummy_system = _DummySystem()

    class _DummyGistDirectPlan:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto"):
            counts = np.ones((1, 1, 1), dtype=np.float64)
            orient = np.ones((1, 1, 1, 4), dtype=np.float64)
            energy_sw = np.array([[[6.0]]], dtype=np.float64)
            energy_ww = np.array([[[4.0]]], dtype=np.float64)
            frame_direct_sw = np.array([6.0], dtype=np.float64)
            frame_direct_ww = np.array([4.0], dtype=np.float64)
            frame_offsets = np.array([0, 1], dtype=np.uint64)
            frame_cells = np.array([0], dtype=np.uint32)
            frame_sw = np.array([6.0], dtype=np.float64)
            frame_ww = np.array([4.0], dtype=np.float64)
            return _direct_payload(
                counts,
                orient,
                energy_sw,
                energy_ww,
                (0.0, 0.0, 0.0),
                1,
                6.0,
                4.0,
                frame_direct_sw,
                frame_direct_ww,
                frame_offsets,
                frame_cells,
                frame_sw,
                frame_ww,
            )

    monkeypatch.setattr(warp_md, "GistDirectPlan", _DummyGistDirectPlan, raising=False)
    config = GistConfig(
        length_scale=1.0,
        energy_method="pme",
        pme_totals_source="native",
        orientation_bins=4,
    )
    with pytest.raises(RuntimeError, match="native PME totals unavailable"):
        gist(traj, dummy_system, system, top, config=config)


def test_gist_pme_direct_approx_bypasses_openmm_totals(monkeypatch):
    import importlib

    gist_mod = importlib.import_module("warp_md.analysis.gist")
    from warp_md.analysis.gist import GistConfig, gist

    system, top = _build_system_and_topology()
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.05, 0.0], [0.2, -0.05, 0.0]],
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.05, 0.0], [0.3, -0.05, 0.0]],
        ],
        dtype=np.float64,
    )
    box = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=np.float64)
    traj = _FakeTraj(coords, box=box)
    dummy_system = _DummySystem()
    called = {"pme": False, "scaler": False}

    class _DummyGistDirectPlan:
        def __init__(self, *args, **kwargs):
            assert kwargs.get("record_frame_energies") is True

        def run(self, _traj, _system, chunk_frames=None, device="auto"):
            counts = np.ones((2, 1, 1), dtype=np.float64)
            orient = np.ones((2, 1, 1, 4), dtype=np.float64)
            energy_sw = np.array([[[1.0]], [[5.0]]], dtype=np.float64)
            energy_ww = np.array([[[2.0]], [[4.0]]], dtype=np.float64)
            frame_direct_sw = np.array([1.0, 5.0], dtype=np.float64)
            frame_direct_ww = np.array([2.0, 4.0], dtype=np.float64)
            frame_offsets = np.array([0, 1, 2], dtype=np.uint64)
            frame_cells = np.array([0, 1], dtype=np.uint32)
            frame_sw = np.array([1.0, 5.0], dtype=np.float64)
            frame_ww = np.array([2.0, 4.0], dtype=np.float64)
            return _direct_payload(
                counts,
                orient,
                energy_sw,
                energy_ww,
                (0.0, 0.0, 0.0),
                2,
                6.0,
                6.0,
                frame_direct_sw,
                frame_direct_ww,
                frame_offsets,
                frame_cells,
                frame_sw,
                frame_ww,
            )

    def _pme_totals_should_not_run(*_args, **_kwargs):
        called["pme"] = True
        raise AssertionError("_pme_totals should not run when pme_totals_source='direct_approx'")

    def _fake_scaler(
        energy_sw_direct,
        energy_ww_direct,
        direct_sw_total,
        direct_ww_total,
        direct_sw_frame,
        direct_ww_frame,
        pme_sw_frame,
        pme_ww_frame,
        frame_offsets,
        frame_cells,
        frame_sw,
        frame_ww,
    ):
        called["scaler"] = True
        assert np.allclose(pme_sw_frame, direct_sw_frame)
        assert np.allclose(pme_ww_frame, direct_ww_frame)
        return np.asarray(energy_sw_direct), np.asarray(energy_ww_direct)

    monkeypatch.setattr(warp_md, "GistDirectPlan", _DummyGistDirectPlan, raising=False)
    monkeypatch.setattr(gist_mod, "_pme_totals", _pme_totals_should_not_run, raising=True)
    monkeypatch.setattr(warp_md, "gist_apply_pme_scaling", _fake_scaler, raising=False)
    config = GistConfig(
        length_scale=1.0,
        energy_method="pme",
        pme_totals_source="direct_approx",
        orientation_bins=4,
    )
    out = gist(traj, dummy_system, system, top, config=config)
    assert called["pme"] is False
    assert called["scaler"] is True
    assert traj.reset_calls == 0
    assert out.n_frames == 2
    assert np.isclose(float(out.energy_sw[0, 0, 0]), 1.0)
    assert np.isclose(float(out.energy_sw[1, 0, 0]), 5.0)
    assert np.isclose(float(out.energy_ww[0, 0, 0]), 2.0)
    assert np.isclose(float(out.energy_ww[1, 0, 0]), 4.0)


def test_gist_pme_direct_approx_falls_back_to_global_totals(monkeypatch):
    from warp_md.analysis.gist import GistConfig, gist

    system, top = _build_system_and_topology()
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.05, 0.0], [0.2, -0.05, 0.0]],
        ],
        dtype=np.float64,
    )
    box = np.array([[2.0, 2.0, 2.0]], dtype=np.float64)
    traj = _FakeTraj(coords, box=box)
    dummy_system = _DummySystem()
    called = {"scaler": False}

    class _DummyGistDirectPlan:
        def __init__(self, *args, **kwargs):
            assert kwargs.get("record_frame_energies") is True

        def run(self, _traj, _system, chunk_frames=None, device="auto"):
            counts = np.ones((1, 1, 1), dtype=np.float64)
            orient = np.ones((1, 1, 1, 4), dtype=np.float64)
            energy_sw = np.array([[[6.0]]], dtype=np.float64)
            energy_ww = np.array([[[4.0]]], dtype=np.float64)
            return _direct_payload(
                counts,
                orient,
                energy_sw,
                energy_ww,
                (0.0, 0.0, 0.0),
                1,
                6.0,
                4.0,
            )

    def _scale_global(
        energy_sw_direct,
        energy_ww_direct,
        direct_sw_total,
        direct_ww_total,
        direct_sw_frame,
        direct_ww_frame,
        pme_sw_frame,
        pme_ww_frame,
        frame_offsets,
        frame_cells,
        frame_sw,
        frame_ww,
    ):
        called["scaler"] = True
        assert np.asarray(direct_sw_frame).size == 0
        assert np.asarray(direct_ww_frame).size == 0
        assert np.allclose(pme_sw_frame, np.array([direct_sw_total], dtype=np.float64))
        assert np.allclose(pme_ww_frame, np.array([direct_ww_total], dtype=np.float64))
        return np.asarray(energy_sw_direct), np.asarray(energy_ww_direct)

    monkeypatch.setattr(warp_md, "GistDirectPlan", _DummyGistDirectPlan, raising=False)
    monkeypatch.setattr(warp_md, "gist_apply_pme_scaling", _scale_global, raising=False)
    config = GistConfig(
        length_scale=1.0,
        energy_method="pme",
        pme_totals_source="direct_approx",
        orientation_bins=4,
    )
    out = gist(traj, dummy_system, system, top, config=config)
    assert called["scaler"] is True
    assert out.n_frames == 1
    assert np.isclose(float(out.energy_sw[0, 0, 0]), 6.0)
    assert np.isclose(float(out.energy_ww[0, 0, 0]), 4.0)


def test_gist_pme_accepts_triclinic_box_matrix(monkeypatch):
    import importlib

    gist_mod = importlib.import_module("warp_md.analysis.gist")
    from warp_md.analysis.gist import GistConfig, gist

    system, top = _build_system_and_topology()
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.05, 0.0], [0.2, -0.05, 0.0]],
        ],
        dtype=np.float64,
    )
    box_matrix = np.array(
        [
            [
                [2.0, 0.1, 0.0],
                [0.0, 2.1, 0.0],
                [0.0, 0.0, 2.2],
            ]
        ],
        dtype=np.float64,
    )
    traj = _FakeTraj(coords, box=None, box_matrix=box_matrix)
    dummy_system = _DummySystem()

    class _DummyGistDirectPlan:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto"):
            counts = np.ones((1, 1, 1), dtype=np.float64)
            orient = np.ones((1, 1, 1, 4), dtype=np.float64)
            energy_sw = np.array([[[1.0]]], dtype=np.float64)
            energy_ww = np.array([[[1.0]]], dtype=np.float64)
            return _direct_payload(
                counts,
                orient,
                energy_sw,
                energy_ww,
                (0.0, 0.0, 0.0),
                1,
                1.0,
                1.0,
            )

    class _DummyPmeEstimator:
        def __init__(self, *_args, **_kwargs):
            self._mask = None

        def set_positions(self, positions_nm, box_nm):
            assert positions_nm.shape == (4, 3)
            assert box_nm.shape == (3, 3)

        def set_active_mask(self, active_mask):
            self._mask = np.asarray(active_mask, dtype=bool)

        def energy(self):
            assert self._mask is not None
            if bool(np.all(self._mask)):
                return 8.0
            if bool(self._mask[0]) and int(np.count_nonzero(self._mask)) == 1:
                return 2.0
            return 3.0

    def _scale_global(
        energy_sw_direct,
        energy_ww_direct,
        direct_sw_total,
        direct_ww_total,
        direct_sw_frame,
        direct_ww_frame,
        pme_sw_frame,
        pme_ww_frame,
        frame_offsets,
        frame_cells,
        frame_sw,
        frame_ww,
    ):
        sw_scale = pme_sw_frame.sum() / direct_sw_total if direct_sw_total != 0.0 else 0.0
        ww_scale = pme_ww_frame.sum() / direct_ww_total if direct_ww_total != 0.0 else 0.0
        return energy_sw_direct * sw_scale, energy_ww_direct * ww_scale

    monkeypatch.setattr(warp_md, "GistDirectPlan", _DummyGistDirectPlan, raising=False)
    monkeypatch.setattr(gist_mod, "_PmeEnergyEstimator", _DummyPmeEstimator, raising=True)
    monkeypatch.setattr(warp_md, "gist_apply_pme_scaling", _scale_global, raising=False)
    config = GistConfig(length_scale=1.0, energy_method="pme", orientation_bins=4)
    out = gist(traj, dummy_system, system, top, config=config)
    assert out.n_frames == 1
    assert np.isclose(float(out.energy_sw[0, 0, 0]), 3.0)
    assert np.isclose(float(out.energy_ww[0, 0, 0]), 3.0)


def test_gist_rejects_invalid_sparse_frame_payload(monkeypatch):
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
            return _direct_payload(
                counts,
                orient,
                energy_sw,
                energy_ww,
                (0.0, 0.0, 0.0),
                1,
                1.0,
                1.0,
                frame_direct_sw=np.array([1.0], dtype=np.float64),
                frame_direct_ww=np.array([1.0], dtype=np.float64),
                frame_offsets=np.array([1, 1], dtype=np.uint64),
                frame_cells=np.array([0], dtype=np.uint32),
                frame_sw=np.array([1.0], dtype=np.float64),
                frame_ww=np.array([1.0], dtype=np.float64),
            )

    monkeypatch.setattr(warp_md, "GistDirectPlan", _DummyGistDirectPlan, raising=False)
    config = GistConfig(length_scale=1.0, energy_method="direct", orientation_bins=4)
    with pytest.raises(RuntimeError, match="frame_offsets must start at 0"):
        gist(traj, dummy_system, system, top, config=config)
