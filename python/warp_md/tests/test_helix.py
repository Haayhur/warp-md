import importlib

import numpy as np
import pytest

helix_mod = importlib.import_module("warp_md.analysis.helix")


class _DummySelection:
    def __init__(self, indices):
        self.indices = list(indices)


class _DummySystem:
    def select(self, _mask):
        return _DummySelection([0, 1, 2, 3, 4])

    def select_indices(self, indices):
        return _DummySelection(indices)


class _DummyTraj:
    pass


def test_helix_forwards_selection_and_options(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            selection,
            fit=True,
            check_each_frame=False,
            residue_start=None,
            residue_end=None,
            length_scale=None,
        ):
            called["selection"] = list(selection.indices)
            called["fit"] = fit
            called["check_each_frame"] = check_each_frame
            called["residue_start"] = residue_start
            called["residue_end"] = residue_end
            called["length_scale"] = length_scale

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["chunk_frames"] = chunk_frames
            called["device"] = device
            called["frame_indices"] = frame_indices
            return {
                "labels": ["ALA:1", "ALA:2", "ALA:3", "ALA:4", "ALA:5"],
                "time": np.array([0.0, 1.0], dtype=np.float32),
                "fragment_start": np.array([1, 2], dtype=np.int32),
                "fragment_end": np.array([5, 5], dtype=np.int32),
                "radius": np.array([2.3, 2.2], dtype=np.float32),
                "twist": np.array([100.0, 101.0], dtype=np.float32),
                "rise": np.array([1.5, 1.4], dtype=np.float32),
                "length": np.array([6.0, 5.6], dtype=np.float32),
                "dipole": np.array([7.0, 7.1], dtype=np.float32),
                "rmsd": np.array([0.1, 0.2], dtype=np.float32),
                "ca_phi": np.array([52.0, 53.0], dtype=np.float32),
                "phi": np.array([-57.0, -58.0], dtype=np.float32),
                "psi": np.array([-47.0, -48.0], dtype=np.float32),
                "hb3": np.array([3.1, 3.2], dtype=np.float32),
                "hb4": np.array([2.9, 3.0], dtype=np.float32),
                "hb5": np.array([4.6, 4.7], dtype=np.float32),
                "ellipticity": np.array([2.0, 2.1], dtype=np.float32),
                "fragment_mask": np.array(
                    [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]],
                    dtype=bool,
                ),
                "residue_rmsd": np.ones((2, 5), dtype=np.float32) * 0.2,
                "helicity_fraction": np.array([0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                "jca_ha": np.array([np.nan, 8.1, 8.2, 8.3, np.nan], dtype=np.float32),
                "frames": 2,
                "residues": 5,
                "fit": True,
                "check_each_frame": True,
                "length_scale": 0.1,
                "used_box": True,
            }

    monkeypatch.setattr(helix_mod, "_HelixPlan", _DummyPlan, raising=True)
    out = helix_mod.helix(
        _DummyTraj(),
        _DummySystem(),
        selection="name CA",
        fit=False,
        check_each_frame=True,
        residue_start=2,
        residue_end=5,
        length_scale=0.1,
        chunk_frames=16,
        frame_indices=[0, 3],
    )
    assert called["selection"] == [0, 1, 2, 3, 4]
    assert called["fit"] is False
    assert called["check_each_frame"] is True
    assert called["residue_start"] == 2
    assert called["residue_end"] == 5
    assert called["length_scale"] == pytest.approx(0.1)
    assert called["chunk_frames"] == 16
    assert called["device"] == "auto"
    assert called["frame_indices"] == [0, 3]
    assert out["labels"] == ("ALA:1", "ALA:2", "ALA:3", "ALA:4", "ALA:5")
    assert out["fragment_mask"].dtype == np.bool_
    assert out["fragment_mask"].shape == (2, 5)
    assert out["rmsd"].shape == (2,)
    assert out["check_each_frame"] is True


def test_helix_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(helix_mod, "_HelixPlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PyHelixPlan binding unavailable"):
        helix_mod.helix(_DummyTraj(), _DummySystem(), selection="name CA")
