import numpy as np
import warp_md

from ._gist_test_utils import (
    _DummySystem,
    _FakeTraj,
    _build_system_and_topology,
    _direct_payload,
)

def test_gist_pme_uses_rust_direct_and_scales(monkeypatch):
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

    class _DummyPmeEstimator:
        def __init__(self, *_args, **_kwargs):
            self._mask = None

        def set_positions(self, positions_nm, box_nm):
            assert positions_nm.shape == (4, 3)
            assert box_nm.shape == (3,)

        def set_active_mask(self, active_mask):
            self._mask = np.asarray(active_mask, dtype=bool)

        def energy(self):
            assert self._mask is not None
            if bool(np.all(self._mask)):
                return 10.0
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
    assert traj.reset_calls >= 1
    assert out.n_frames == 1
    assert np.isclose(float(out.energy_sw[0, 0, 0]), 5.0)
    assert np.isclose(float(out.energy_ww[0, 0, 0]), 3.0)


def test_gist_pme_scales_per_frame_sparse_energy(monkeypatch):
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

    class _PerFramePmeEstimator:
        def __init__(self, *_args, **_kwargs):
            self._mask = None
            self._frame = -1

        def set_positions(self, positions_nm, box_nm):
            assert positions_nm.shape == (4, 3)
            assert box_nm.shape == (3,)
            self._frame += 1

        def set_active_mask(self, active_mask):
            self._mask = np.asarray(active_mask, dtype=bool)

        def energy(self):
            assert self._mask is not None
            if bool(np.all(self._mask)):
                return [7.0, 11.0][self._frame]
            if bool(self._mask[0]) and int(np.count_nonzero(self._mask)) == 1:
                return [2.0, 2.0][self._frame]
            return [3.0, 4.0][self._frame]

    def _scale_sparse(
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
        out_sw = np.zeros_like(energy_sw_direct, dtype=np.float64).reshape(-1)
        out_ww = np.zeros_like(energy_ww_direct, dtype=np.float64).reshape(-1)
        for frame_idx in range(direct_sw_frame.size):
            sw_scale = pme_sw_frame[frame_idx] / direct_sw_frame[frame_idx] if direct_sw_frame[frame_idx] != 0.0 else 0.0
            ww_scale = pme_ww_frame[frame_idx] / direct_ww_frame[frame_idx] if direct_ww_frame[frame_idx] != 0.0 else 0.0
            start = int(frame_offsets[frame_idx])
            end = int(frame_offsets[frame_idx + 1])
            if end <= start:
                continue
            cells = frame_cells[start:end].astype(np.int64, copy=False)
            out_sw[cells] += frame_sw[start:end] * sw_scale
            out_ww[cells] += frame_ww[start:end] * ww_scale
        return out_sw.reshape(energy_sw_direct.shape), out_ww.reshape(energy_ww_direct.shape)

    monkeypatch.setattr(warp_md, "GistDirectPlan", _DummyGistDirectPlan, raising=False)
    monkeypatch.setattr(gist_mod, "_PmeEnergyEstimator", _PerFramePmeEstimator, raising=True)
    monkeypatch.setattr(warp_md, "gist_apply_pme_scaling", _scale_sparse, raising=False)
    config = GistConfig(length_scale=1.0, energy_method="pme", orientation_bins=4)
    out = gist(traj, dummy_system, system, top, config=config)
    assert traj.reset_calls >= 1
    assert out.n_frames == 2
    assert np.isclose(float(out.energy_sw[0, 0, 0]), 2.0)
    assert np.isclose(float(out.energy_sw[1, 0, 0]), 5.0)
    assert np.isclose(float(out.energy_ww[0, 0, 0]), 3.0)
    assert np.isclose(float(out.energy_ww[1, 0, 0]), 4.0)


def test_gist_pme_uses_rust_sparse_scaler_binding(monkeypatch):
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
    called = {"scaler": False}

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

    class _PerFramePmeEstimator:
        def __init__(self, *_args, **_kwargs):
            self._mask = None
            self._frame = -1

        def set_positions(self, positions_nm, box_nm):
            assert positions_nm.shape == (4, 3)
            assert box_nm.shape == (3,)
            self._frame += 1

        def set_active_mask(self, active_mask):
            self._mask = np.asarray(active_mask, dtype=bool)

        def energy(self):
            assert self._mask is not None
            if bool(np.all(self._mask)):
                return [7.0, 11.0][self._frame]
            if bool(self._mask[0]) and int(np.count_nonzero(self._mask)) == 1:
                return [2.0, 2.0][self._frame]
            return [3.0, 4.0][self._frame]

    def _fake_scaler(*_args):
        called["scaler"] = True
        return (
            np.array([[[2.0]], [[5.0]]], dtype=np.float64),
            np.array([[[3.0]], [[4.0]]], dtype=np.float64),
        )

    monkeypatch.setattr(warp_md, "GistDirectPlan", _DummyGistDirectPlan, raising=False)
    monkeypatch.setattr(gist_mod, "_PmeEnergyEstimator", _PerFramePmeEstimator, raising=True)
    monkeypatch.setattr(warp_md, "gist_apply_pme_scaling", _fake_scaler, raising=False)
    config = GistConfig(length_scale=1.0, energy_method="pme", orientation_bins=4)
    out = gist(traj, dummy_system, system, top, config=config)
    assert called["scaler"] is True
    assert out.n_frames == 2
    assert np.isclose(float(out.energy_sw[0, 0, 0]), 2.0)
    assert np.isclose(float(out.energy_sw[1, 0, 0]), 5.0)
    assert np.isclose(float(out.energy_ww[0, 0, 0]), 3.0)
    assert np.isclose(float(out.energy_ww[1, 0, 0]), 4.0)


