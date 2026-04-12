import numpy as np
import pytest

import warp_md
from warp_md.analysis.vanhove import vanhove


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._atoms = {
            "name": ["A"] * n_atoms,
            "resid": list(range(1, n_atoms + 1)),
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(range(len(self._atoms["name"])))


class _DummyTraj:
    pass


def test_vanhove_forwards_postprocessing_to_plan_run(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            r_bin=0.1,
            r_max=10.0,
            length_scale=None,
            max_lag=None,
            sqrt_time_bin=None,
            scale_to_average_box=True,
            remove_pbc_jumps=True,
            time_scale=None,
        ):
            called["r_bin"] = r_bin
            called["r_max"] = r_max
            called["length_scale"] = length_scale
            called["max_lag"] = max_lag
            called["sqrt_time_bin"] = sqrt_time_bin
            called["scale_to_average_box"] = scale_to_average_box
            called["remove_pbc_jumps"] = remove_pbc_jumps
            called["time_scale"] = time_scale

        def run(
            self,
            _traj,
            _system,
            chunk_frames=None,
            device="auto",
            frame_indices=None,
            integral_radius=None,
            curve_lags=None,
            curve_step=None,
        ):
            called["chunk_frames"] = chunk_frames
            called["device"] = device
            called["frame_indices"] = frame_indices
            called["integral_radius"] = integral_radius
            called["curve_lags"] = curve_lags
            called["curve_step"] = curve_step
            return {
                "time": np.array([0.0, 1.0, 2.0], dtype=np.float32),
                "time_sqrt": np.array([0.0, 1.0, np.sqrt(2.0)], dtype=np.float32),
                "r": np.array([0.0, 1.0, 2.0], dtype=np.float32),
                "matrix": np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.25, 0.5],
                    ],
                    dtype=np.float32,
                ),
                "counts": np.array([3, 2, 1], dtype=np.uint64),
                "r_bin": 1.0,
                "r_max": 3.0,
                "integral_radius": 1.0,
                "integral": np.array([0.5, 1.0, 0.25], dtype=np.float32),
                "curve_indices": np.array([1, 2], dtype=np.int64),
                "curve_time": np.array([1.0, 2.0], dtype=np.float32),
                "curve_matrix": np.array([[0.0, 1.0, 0.0], [0.0, 0.25, 0.5]], dtype=np.float32),
            }

    monkeypatch.setattr(warp_md, "VanHovePlan", _DummyPlan, raising=False)
    out = vanhove(
        _DummyTraj(),
        _DummySystem(1),
        selection="all",
        r_bin=1.0,
        r_max=3.0,
        max_lag=2,
        integral_radius=1.0,
        curve_lags=[1, -1],
        chunk_frames=32,
        frame_indices=[0, 2],
    )

    assert called["r_bin"] == 1.0
    assert called["r_max"] == 3.0
    assert called["max_lag"] == 2
    assert called["chunk_frames"] == 32
    assert called["device"] == "auto"
    assert called["frame_indices"] == [0, 2]
    assert called["integral_radius"] == 1.0
    assert called["curve_lags"] == [1, -1]
    assert called["curve_step"] is None
    np.testing.assert_allclose(out["integral"], np.array([0.5, 1.0, 0.25], dtype=np.float32))
    np.testing.assert_array_equal(out["curve_indices"], np.array([1, 2], dtype=np.int64))
    np.testing.assert_allclose(out["curve_time"], np.array([1.0, 2.0], dtype=np.float32))
    assert out["curve_matrix"].shape == (2, 3)


def test_vanhove_raises_when_binding_missing(monkeypatch):
    class _Missing:
        pass

    monkeypatch.setattr(warp_md, "VanHovePlan", _Missing, raising=False)
    with pytest.raises(RuntimeError, match="VanHovePlan binding unavailable"):
        vanhove(_DummyTraj(), _DummySystem(1), selection="all")
