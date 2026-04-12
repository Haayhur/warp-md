import importlib

import numpy as np
import pytest

helixorient_mod = importlib.import_module("warp_md.analysis.helixorient")


class _DummySelection:
    def __init__(self, indices):
        self.indices = list(indices)


class _DummySystem:
    def select(self, _mask):
        return _DummySelection([4, 5, 6, 7])

    def select_indices(self, indices):
        return _DummySelection(indices)


class _DummyTraj:
    pass


def test_helixorient_forwards_selections_and_options(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            ca_selection,
            sidechain_selection=None,
            incremental=False,
            length_scale=None,
        ):
            called["ca_selection"] = list(ca_selection.indices)
            called["sidechain_selection"] = (
                None if sidechain_selection is None else list(sidechain_selection.indices)
            )
            called["incremental"] = incremental
            called["length_scale"] = length_scale

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["chunk_frames"] = chunk_frames
            called["device"] = device
            called["frame_indices"] = frame_indices
            return {
                "labels": ["ALA:1", "ALA:2", "ALA:3", "ALA:4"],
                "time": np.array([0.0, 1.0], dtype=np.float32),
                "axis": np.zeros((2, 4, 3), dtype=np.float32),
                "center": np.ones((2, 4, 3), dtype=np.float32),
                "residue_vector": np.ones((2, 4, 3), dtype=np.float32) * 2.0,
                "normal": np.ones((2, 4, 3), dtype=np.float32) * 3.0,
                "rise": np.ones((2, 4), dtype=np.float32) * 0.15,
                "radius": np.ones((2, 4), dtype=np.float32) * 0.23,
                "twist": np.ones((2, 4), dtype=np.float32) * 100.0,
                "bending": np.zeros((2, 4), dtype=np.float32),
                "tilt": np.zeros((2, 4), dtype=np.float32),
                "rotation": np.ones((2, 4), dtype=np.float32) * 20.0,
                "theta1": np.zeros((2, 4), dtype=np.float32),
                "theta2": np.zeros((2, 4), dtype=np.float32),
                "theta3": np.ones((2, 4), dtype=np.float32) * 20.0,
                "frames": 2,
                "residues": 4,
                "use_sidechain": True,
                "incremental": False,
                "used_box": True,
                "length_scale": 0.1,
            }

    monkeypatch.setattr(helixorient_mod, "_HelixOrientPlan", _DummyPlan, raising=True)
    out = helixorient_mod.helixorient(
        _DummyTraj(),
        _DummySystem(),
        ca_selection="name CA",
        sidechain_selection=[20, 21, 22, 23],
        incremental=True,
        length_scale=0.1,
        chunk_frames=8,
        frame_indices=[0, 2],
    )
    assert called["ca_selection"] == [4, 5, 6, 7]
    assert called["sidechain_selection"] == [20, 21, 22, 23]
    assert called["incremental"] is True
    assert called["length_scale"] == pytest.approx(0.1)
    assert called["chunk_frames"] == 8
    assert called["device"] == "auto"
    assert called["frame_indices"] == [0, 2]
    assert out["labels"] == ("ALA:1", "ALA:2", "ALA:3", "ALA:4")
    assert out["axis"].shape == (2, 4, 3)
    assert out["rotation"].shape == (2, 4)
    assert out["use_sidechain"] is True


def test_helixorient_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(helixorient_mod, "_HelixOrientPlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PyHelixOrientPlan binding unavailable"):
        helixorient_mod.helixorient(_DummyTraj(), _DummySystem(), ca_selection="name CA")
