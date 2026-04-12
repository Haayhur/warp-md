import importlib

import numpy as np
import pytest

saltbr_mod = importlib.import_module("warp_md.analysis.saltbr")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "resid": [1, 2, 3, 4],
            "charge": [1.0, -1.0, 1.0, -1.0],
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection([0, 1, 2, 3])

    def select_indices(self, indices):
        return _DummySelection(list(indices))


class _DummyTraj:
    pass


def test_saltbr_uses_rust_plan_wrapper(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            sel,
            charges,
            group_by="atom",
            truncate=None,
            contact_cutoff=None,
            length_scale=None,
        ):
            called["indices"] = list(sel.indices)
            called["charges"] = list(charges)
            called["group_by"] = group_by
            called["truncate"] = truncate
            called["contact_cutoff"] = contact_cutoff
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
                "time": np.array([0.0, 1.0], dtype=np.float32),
                "distance": np.array([[1.0], [3.0]], dtype=np.float32),
                "group_labels": ["LYS1-1", "GLU2-2"],
                "group_charge": np.array([1.0, -1.0], dtype=np.float64),
                "pair_group_index": np.array([[0, 1]], dtype=np.uint32),
                "pair_labels": ["LYS1-1:GLU2-2"],
                "pair_class": ["plus_min"],
                "min_distance": np.array([1.0], dtype=np.float32),
                "contact_cutoff": 1.1,
                "contact_count": np.array([1], dtype=np.uint64),
                "contact_fraction": np.array([0.5], dtype=np.float32),
            }

    monkeypatch.setattr(saltbr_mod, "_SaltBridgePlan", _DummyPlan, raising=True)
    out = saltbr_mod.saltbr(
        _DummyTraj(),
        _DummySystem(),
        selection="all",
        group_by="atom",
        truncate=1.5,
        contact_cutoff=1.1,
        chunk_frames=32,
        frame_indices=[0, 2],
    )
    assert called["indices"] == [0, 1, 2, 3]
    assert called["charges"] == [1.0, -1.0, 1.0, -1.0]
    assert called["group_by"] == "atom"
    assert called["truncate"] == 1.5
    assert called["contact_cutoff"] == 1.1
    assert called["chunk_frames"] == 32
    assert called["device"] == "auto"
    assert called["frame_indices"] == [0, 2]
    np.testing.assert_allclose(out["time"], np.array([0.0, 1.0], dtype=np.float32))
    assert out["pair_labels"] == ["LYS1-1:GLU2-2"]
    assert out["pair_class"] == ["plus_min"]


def test_saltbr_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(saltbr_mod, "_SaltBridgePlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PySaltBridgePlan binding unavailable"):
        saltbr_mod.saltbr(_DummyTraj(), _DummySystem(), selection="all")
