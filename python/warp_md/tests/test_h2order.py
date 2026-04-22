import importlib

import numpy as np
import pytest

h2order_mod = importlib.import_module("warp_md.analysis.h2order")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "name": ["OW", "HW1", "HW2", "CA"],
            "resname": ["SOL", "SOL", "SOL", "ALA"],
            "resid": [1, 1, 1, 2],
            "element": ["O", "H", "H", "C"],
            "chain_id": [0, 0, 0, 0],
            "charge": [-0.834, 0.417, 0.417, 0.0],
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection([0, 1, 2])

    def select_indices(self, indices):
        return _DummySelection(list(indices))


class _DummyTraj:
    pass


def test_h2order_detects_triplets_and_forwards(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            oxygen_indices,
            hydrogen1_indices,
            hydrogen2_indices,
            charges,
            axis="z",
            bin=0.25,
            n_slices=None,
            length_scale=None,
        ):
            called["oxygen_indices"] = list(oxygen_indices)
            called["hydrogen1_indices"] = list(hydrogen1_indices)
            called["hydrogen2_indices"] = list(hydrogen2_indices)
            called["charges"] = list(charges)
            called["axis"] = axis
            called["bin"] = bin
            called["n_slices"] = n_slices
            called["length_scale"] = length_scale

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["chunk_frames"] = chunk_frames
            called["device"] = device
            called["frame_indices"] = frame_indices
            return {
                "coordinate": np.array([0.5, 1.5], dtype=np.float32),
                "order": np.array([1.0, 0.0], dtype=np.float32),
                "dipole": np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]], dtype=np.float32),
                "counts": np.array([1, 0], dtype=np.uint64),
                "axis": "z",
                "bounds": np.array([0.0, 2.0], dtype=np.float32),
                "slice_width": 1.0,
                "n_frames": 2,
                "used_box": True,
                "length_scale": 0.1,
                "dipole_unit": "debye",
            }

    monkeypatch.setattr(h2order_mod, "_WaterOrderPlan", _DummyPlan, raising=True)
    out = h2order_mod.h2order(
        _DummyTraj(),
        _DummySystem(),
        selection="resname SOL",
        axis="z",
        bin=0.2,
        n_slices=8,
        length_scale=0.1,
        chunk_frames=32,
        frame_indices=[0, 2],
    )
    assert called["oxygen_indices"] == [0]
    assert called["hydrogen1_indices"] == [1]
    assert called["hydrogen2_indices"] == [2]
    assert called["charges"][:3] == [-0.834, 0.417, 0.417]
    assert called["axis"] == "z"
    assert called["bin"] == 0.2
    assert called["n_slices"] == 8
    assert called["length_scale"] == 0.1
    assert called["chunk_frames"] == 32
    assert called["device"] == "auto"
    assert called["frame_indices"] == [0, 2]
    assert out["axis"] == "z"
    assert out["dipole_unit"] == "debye"
    np.testing.assert_allclose(out["order"], np.array([1.0, 0.0], dtype=np.float32))


def test_h2order_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(h2order_mod, "_WaterOrderPlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PyWaterOrderPlan binding unavailable"):
        h2order_mod.h2order(_DummyTraj(), _DummySystem(), selection="resname SOL")
