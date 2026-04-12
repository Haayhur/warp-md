import importlib

import numpy as np
import pytest

current_mod = importlib.import_module("warp_md.analysis.current")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "resid": [1, 2],
            "charge": [1.0, -1.0],
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection([0, 1])

    def select_indices(self, indices):
        return _DummySelection(list(indices))


class _DummyTraj:
    pass


def test_current_uses_rust_plan_wrapper(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            sel,
            charges,
            temperature=300.0,
            group_by="resid",
            length_scale=None,
            group_types=None,
            make_whole=True,
            frame_decimation=None,
            dt_decimation=None,
            time_binning=None,
            lag_mode=None,
            max_lag=None,
            memory_budget_bytes=None,
            multi_tau_m=None,
            multi_tau_levels=None,
        ):
            called["indices"] = list(sel.indices)
            called["charges"] = list(charges)
            called["temperature"] = temperature
            called["group_by"] = group_by
            called["length_scale"] = length_scale
            called["group_types"] = group_types
            called["make_whole"] = make_whole
            called["frame_decimation"] = frame_decimation
            called["dt_decimation"] = dt_decimation
            called["time_binning"] = time_binning
            called["lag_mode"] = lag_mode
            called["max_lag"] = max_lag
            called["memory_budget_bytes"] = memory_budget_bytes
            called["multi_tau_m"] = multi_tau_m
            called["multi_tau_levels"] = multi_tau_levels

        def run(
            self,
            _traj,
            _system,
            chunk_frames=None,
            device="auto",
            frame_indices=None,
        ):
            called["chunk_frames"] = chunk_frames
            called["device"] = device
            called["frame_indices"] = frame_indices
            return {
                "conductivity_time": np.array([1.0], dtype=np.float32),
                "conductivity": np.array([[0.25]], dtype=np.float32),
                "time": np.array([0.0, 1.0], dtype=np.float32),
                "md_sq": np.array([0.0, 0.0], dtype=np.float32),
                "mj_sq": np.array([1.0, 2.0], dtype=np.float32),
                "md_mj": np.array([0.0, 0.0], dtype=np.float32),
                "dielectric_rot": 1.0,
                "dielectric_total": 1.2,
                "mu_avg": 3.4,
                "conductivity_static": 0.25,
            }

    monkeypatch.setattr(current_mod, "_CurrentPlan", _DummyPlan, raising=True)
    out = current_mod.current(
        _DummyTraj(),
        _DummySystem(),
        selection="all",
        temperature=310.0,
        group_by="atom",
        length_scale=0.1,
        group_types=[0, 1],
        make_whole=False,
        frame_decimation=(0, 2),
        dt_decimation=(8, 2, 16, 4),
        time_binning=(1e-6, 1e-8),
        lag_mode="ring",
        max_lag=8,
        memory_budget_bytes=4096,
        multi_tau_m=8,
        multi_tau_levels=6,
        chunk_frames=32,
        frame_indices=[0, 2],
    )
    assert called["indices"] == [0, 1]
    assert called["charges"] == [1.0, -1.0]
    assert called["temperature"] == 310.0
    assert called["group_by"] == "atom"
    assert called["length_scale"] == 0.1
    assert called["group_types"] == [0, 1]
    assert called["make_whole"] is False
    assert called["frame_decimation"] == (0, 2)
    assert called["dt_decimation"] == (8, 2, 16, 4)
    assert called["time_binning"] == (1e-6, 1e-8)
    assert called["lag_mode"] == "ring"
    assert called["max_lag"] == 8
    assert called["memory_budget_bytes"] == 4096
    assert called["multi_tau_m"] == 8
    assert called["multi_tau_levels"] == 6
    assert called["chunk_frames"] == 32
    assert called["device"] == "auto"
    assert called["frame_indices"] == [0, 2]
    np.testing.assert_allclose(out["conductivity"], np.array([[0.25]], dtype=np.float32))
    assert out["conductivity_static"] == pytest.approx(0.25)
    assert out["dielectric_total"] == pytest.approx(1.2)


def test_current_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(current_mod, "_CurrentPlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PyCurrentPlan binding unavailable"):
        current_mod.current(_DummyTraj(), _DummySystem(), selection="all")
