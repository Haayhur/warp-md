import importlib

import numpy as np
import pytest

potential_mod = importlib.import_module("warp_md.analysis.potential")


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


def test_potential_uses_rust_plan_wrapper(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            sel,
            charges,
            axis="z",
            bin=0.25,
            n_slices=None,
            center_selection=None,
            symmetrize=False,
            correct=False,
            discard_start=0,
            discard_end=0,
            length_scale=None,
        ):
            called["indices"] = list(sel.indices)
            called["charges"] = list(charges)
            called["axis"] = axis
            called["bin"] = bin
            called["n_slices"] = n_slices
            called["center_indices"] = None if center_selection is None else list(center_selection.indices)
            called["symmetrize"] = symmetrize
            called["correct"] = correct
            called["discard_start"] = discard_start
            called["discard_end"] = discard_end
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
                "coordinate": np.array([0.5, 1.5], dtype=np.float32),
                "charge_density": np.array([1.0, -1.0], dtype=np.float32),
                "field": np.array([0.0, 2.0], dtype=np.float32),
                "potential": np.array([0.0, -1.0], dtype=np.float32),
                "axis": "z",
                "bounds": np.array([0.0, 2.0], dtype=np.float32),
                "slice_width": 1.0,
                "n_frames": 2,
                "used_box": True,
                "centered": True,
                "symmetrized": False,
                "corrected": True,
                "length_scale": 0.1,
                "discard_start": 1,
                "discard_end": 0,
            }

    monkeypatch.setattr(potential_mod, "_PotentialPlan", _DummyPlan, raising=True)
    out = potential_mod.potential(
        _DummyTraj(),
        _DummySystem(),
        selection="all",
        axis="z",
        bin=0.2,
        n_slices=8,
        center="all",
        symmetrize=True,
        correct=True,
        discard_start=1,
        discard_end=2,
        length_scale=0.1,
        chunk_frames=32,
        frame_indices=[0, 2],
    )
    assert called["indices"] == [0, 1]
    assert called["charges"] == [1.0, -1.0]
    assert called["axis"] == "z"
    assert called["bin"] == 0.2
    assert called["n_slices"] == 8
    assert called["center_indices"] == [0, 1]
    assert called["symmetrize"] is True
    assert called["correct"] is True
    assert called["discard_start"] == 1
    assert called["discard_end"] == 2
    assert called["length_scale"] == 0.1
    assert called["chunk_frames"] == 32
    assert called["device"] == "auto"
    assert called["frame_indices"] == [0, 2]
    assert out["axis"] == "z"
    assert out["used_box"] is True
    assert out["centered"] is True
    assert out["corrected"] is True
    np.testing.assert_allclose(out["potential"], np.array([0.0, -1.0], dtype=np.float32))


def test_potential_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(potential_mod, "_PotentialPlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PyPotentialPlan binding unavailable"):
        potential_mod.potential(_DummyTraj(), _DummySystem(), selection="all")
