import importlib

import numpy as np
import pytest

rama_mod = importlib.import_module("warp_md.analysis.rama")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "resid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(list(range(9)))

    def select_indices(self, indices):
        return _DummySelection(list(indices))


class _DummyTraj:
    pass


def test_rama_uses_rust_plan_wrapper(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(self, sel, range360=False):
            called["indices"] = list(sel.indices)
            called["range360"] = range360

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
                "labels": ["ALA:1", "GLY:2", "SER:3"],
                "phi": np.array([[np.nan, -29.0, 45.0]], dtype=np.float32),
                "psi": np.array([[10.0, 1.0, np.nan]], dtype=np.float32),
            }

    monkeypatch.setattr(rama_mod, "_RamaPlan", _DummyPlan, raising=True)
    out = rama_mod.rama(
        _DummyTraj(),
        _DummySystem(),
        selection="protein",
        range360=True,
        frame_indices=[0, 2],
        chunk_frames=64,
    )
    assert called["indices"] == list(range(9))
    assert called["range360"] is True
    assert called["chunk_frames"] == 64
    assert called["device"] == "auto"
    assert called["frame_indices"] == [0, 2]
    assert out["labels"].shape == (3,)
    assert out["phi"].shape == (1, 3)
    assert out["psi"].shape == (1, 3)
    assert np.isnan(out["phi"][0, 0])
    assert out["phi"][0, 1] == pytest.approx(-29.0)
    assert np.isnan(out["psi"][0, 2])


def test_rama_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(rama_mod, "_RamaPlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PyRamaPlan binding unavailable"):
        rama_mod.rama(_DummyTraj(), _DummySystem(), selection="protein")
