import importlib

import numpy as np
import pytest

bundle_mod = importlib.import_module("warp_md.analysis.bundle")


class _DummySelection:
    def __init__(self, indices):
        self.indices = list(indices)


class _DummySystem:
    def select(self, mask):
        if mask == "top":
            return _DummySelection([0, 1])
        if mask == "bottom":
            return _DummySelection([2, 3])
        if mask == "kink":
            return _DummySelection([4, 5])
        raise AssertionError(f"unexpected mask: {mask}")

    def select_indices(self, indices):
        return _DummySelection(indices)


class _DummyTraj:
    pass


def test_bundle_forwards_selections_and_options(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            top_selection,
            bottom_selection,
            n_axes,
            kink_selection=None,
            use_z_reference=False,
            mass_weighted=True,
            length_scale=None,
        ):
            called["top_selection"] = list(top_selection.indices)
            called["bottom_selection"] = list(bottom_selection.indices)
            called["n_axes"] = n_axes
            called["kink_selection"] = None if kink_selection is None else list(kink_selection.indices)
            called["use_z_reference"] = use_z_reference
            called["mass_weighted"] = mass_weighted
            called["length_scale"] = length_scale

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["chunk_frames"] = chunk_frames
            called["device"] = device
            called["frame_indices"] = frame_indices
            return {
                "labels": ["axis_1", "axis_2"],
                "time": np.array([0.0], dtype=np.float32),
                "reference_axis": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                "top": np.array([[[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]], dtype=np.float32),
                "bottom": np.array([[[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]], dtype=np.float32),
                "mid": np.array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32),
                "direction": np.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=np.float32),
                "length": np.array([[2.0, 2.0]], dtype=np.float32),
                "distance": np.array([[1.0, 1.0]], dtype=np.float32),
                "z_shift": np.array([[0.0, 0.0]], dtype=np.float32),
                "tilt": np.array([[0.0, 0.0]], dtype=np.float32),
                "radial_tilt": np.array([[0.0, 0.0]], dtype=np.float32),
                "lateral_tilt": np.array([[0.0, 0.0]], dtype=np.float32),
                "kink": np.array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32),
                "kink_angle": np.array([[0.0, 0.0]], dtype=np.float32),
                "kink_radial": np.array([[0.0, 0.0]], dtype=np.float32),
                "kink_lateral": np.array([[0.0, 0.0]], dtype=np.float32),
                "frames": 1,
                "axes": 2,
                "has_kink": True,
                "use_z_reference": True,
                "mass_weighted": False,
                "used_box": True,
                "length_scale": 0.1,
            }

    monkeypatch.setattr(bundle_mod, "_BundlePlan", _DummyPlan, raising=True)
    out = bundle_mod.bundle(
        _DummyTraj(),
        _DummySystem(),
        top_selection="top",
        bottom_selection="bottom",
        n_axes=2,
        kink_selection="kink",
        use_z_reference=True,
        mass_weighted=False,
        length_scale=0.1,
        chunk_frames=8,
        frame_indices=[0, 2],
    )
    assert called["top_selection"] == [0, 1]
    assert called["bottom_selection"] == [2, 3]
    assert called["n_axes"] == 2
    assert called["kink_selection"] == [4, 5]
    assert called["use_z_reference"] is True
    assert called["mass_weighted"] is False
    assert called["length_scale"] == pytest.approx(0.1)
    assert called["chunk_frames"] == 8
    assert called["device"] == "auto"
    assert called["frame_indices"] == [0, 2]
    assert out["labels"] == ("axis_1", "axis_2")
    assert out["reference_axis"].shape == (1, 3)
    assert out["top"].shape == (1, 2, 3)
    assert out["length"].shape == (1, 2)
    assert out["kink"].shape == (1, 2, 3)
    assert out["has_kink"] is True
    assert out["use_z_reference"] is True
    assert out["mass_weighted"] is False
    assert out["used_box"] is True


def test_bundle_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(bundle_mod, "_BundlePlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PyBundlePlan binding unavailable"):
        bundle_mod.bundle(_DummyTraj(), _DummySystem(), top_selection="top", bottom_selection="bottom", n_axes=2)
