import importlib

import numpy as np
import pytest

densmap_mod = importlib.import_module("warp_md.analysis.densmap")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "resid": [1, 2, 3],
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection([0, 1, 2])

    def select_indices(self, indices):
        return _DummySelection(list(indices))


class _DummyTraj:
    pass


def test_densmap_uses_rust_plan_wrapper(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            sel,
            average="z",
            bin=0.25,
            n1=None,
            n2=None,
            xmin=None,
            xmax=None,
            unit="nm-3",
            length_scale=None,
        ):
            called["indices"] = list(sel.indices)
            called["average"] = average
            called["bin"] = bin
            called["n1"] = n1
            called["n2"] = n2
            called["xmin"] = xmin
            called["xmax"] = xmax
            called["unit"] = unit
            called["length_scale"] = length_scale

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
                "axis1": np.array([0.5, 1.5], dtype=np.float32),
                "axis2": np.array([0.5, 1.5], dtype=np.float32),
                "matrix": np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32),
                "plane_axes": ["x", "y"],
                "average_axis": "z",
                "unit": "nm-3",
                "n_frames": 2,
                "bounds": np.array([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32),
                "bin_width": np.array([1.0, 1.0], dtype=np.float32),
                "used_box": True,
                "length_scale": 0.1,
            }

    monkeypatch.setattr(densmap_mod, "_DensityMapPlan", _DummyPlan, raising=True)
    out = densmap_mod.densmap(
        _DummyTraj(),
        _DummySystem(),
        selection="all",
        average="z",
        bin=0.2,
        n1=2,
        n2=2,
        xmin=0.5,
        xmax=1.5,
        unit="count",
        length_scale=0.1,
        chunk_frames=32,
        frame_indices=[0, 2],
    )
    assert called["indices"] == [0, 1, 2]
    assert called["average"] == "z"
    assert called["bin"] == 0.2
    assert called["n1"] == 2
    assert called["n2"] == 2
    assert called["xmin"] == 0.5
    assert called["xmax"] == 1.5
    assert called["unit"] == "count"
    assert called["length_scale"] == 0.1
    assert called["chunk_frames"] == 32
    assert called["device"] == "auto"
    assert called["frame_indices"] == [0, 2]
    assert out["plane_axes"] == ["x", "y"]
    assert out["average_axis"] == "z"
    assert out["unit"] == "nm-3"
    assert out["n_frames"] == 2
    assert out["used_box"] is True
    np.testing.assert_allclose(out["matrix"], np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32))


def test_densmap_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(densmap_mod, "_DensityMapPlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PyDensityMapPlan binding unavailable"):
        densmap_mod.densmap(_DummyTraj(), _DummySystem(), selection="all")
