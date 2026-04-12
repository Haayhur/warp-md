import importlib

import numpy as np
import pytest

hydorder_mod = importlib.import_module("warp_md.analysis.hydorder")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def select(self, _mask):
        return _DummySelection([1, 2, 3, 4, 5])

    def select_indices(self, indices):
        return _DummySelection(list(indices))


class _DummyTraj:
    pass


def test_hydorder_forwards_selection_and_options(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            selection,
            axis="z",
            bin=1.0,
            tblock=1,
            sgang1=None,
            sgang2=None,
            length_scale=None,
        ):
            called["selection"] = list(selection.indices)
            called["axis"] = axis
            called["bin"] = bin
            called["tblock"] = tblock
            called["sgang1"] = sgang1
            called["sgang2"] = sgang2
            called["length_scale"] = length_scale

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["chunk_frames"] = chunk_frames
            called["device"] = device
            called["frame_indices"] = frame_indices
            return {
                "sg_mean": 0.2,
                "sk_mean": 0.1,
                "sg_grid": np.zeros((2, 2, 2), dtype=np.float32),
                "sk_grid": np.ones((2, 2, 2), dtype=np.float32),
                "counts": np.ones((2, 2, 2), dtype=np.uint64),
                "x": np.array([0.5, 1.5], dtype=np.float32),
                "y": np.array([0.5, 1.5], dtype=np.float32),
                "z": np.array([0.5, 1.5], dtype=np.float32),
                "dims": [2, 2, 2],
                "bounds": np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]], dtype=np.float32),
                "bin_width": 1.0,
                "axis": "z",
                "plane_axes": ["x", "y"],
                "n_frames": 4,
                "used_box": True,
                "length_scale": 0.1,
                "interface_lower": np.zeros((1, 2, 2), dtype=np.float32),
                "interface_upper": np.ones((1, 2, 2), dtype=np.float32),
                "interface_blocks": 1,
                "interface_threshold": 0.5,
                "block_size": 2,
            }

    monkeypatch.setattr(hydorder_mod, "_HydOrderPlan", _DummyPlan, raising=True)
    out = hydorder_mod.hydorder(
        _DummyTraj(),
        _DummySystem(),
        selection="name OW",
        axis="z",
        bin=0.5,
        tblock=2,
        sgang1=0.1,
        sgang2=0.9,
        length_scale=0.1,
        chunk_frames=16,
        frame_indices=[0, 3],
    )
    assert called["selection"] == [1, 2, 3, 4, 5]
    assert called["axis"] == "z"
    assert called["bin"] == 0.5
    assert called["tblock"] == 2
    assert called["sgang1"] == pytest.approx(0.1)
    assert called["sgang2"] == pytest.approx(0.9)
    assert called["length_scale"] == pytest.approx(0.1)
    assert called["chunk_frames"] == 16
    assert called["device"] == "auto"
    assert called["frame_indices"] == [0, 3]
    assert out["dims"] == (2, 2, 2)
    assert out["plane_axes"] == ("x", "y")
    assert out["interface_threshold"] == pytest.approx(0.5)
    assert out["interface_upper"].shape == (1, 2, 2)


def test_hydorder_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(hydorder_mod, "_HydOrderPlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PyHydOrderPlan binding unavailable"):
        hydorder_mod.hydorder(_DummyTraj(), _DummySystem(), selection="name OW")
