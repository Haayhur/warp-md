import importlib

import numpy as np
import pytest

mdmat_mod = importlib.import_module("warp_md.analysis.mdmat")


class _DummySelection:
    def __init__(self, indices):
        self.indices = list(indices)


class _DummySystem:
    def select(self, mask):
        if mask in ("", "all", "resid 1:2"):
            return _DummySelection([0, 1, 2, 3])
        raise AssertionError(f"unexpected mask: {mask}")

    def select_indices(self, indices):
        return _DummySelection(indices)

    def atom_table(self):
        return {
            "name": ["A1", "A2", "A1", "A2"],
            "resname": ["ALA", "ALA", "ALA", "ALA"],
            "resid": [1, 1, 2, 2],
            "chain_id": [0, 0, 0, 0],
        }


class _DummyTraj:
    def count_frames(self, _chunk_frames=None):
        return 3


def test_mdmat_forwards_and_writes_artifact(monkeypatch, tmp_path):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            selection,
            truncate=1.5,
            include_contacts=False,
            include_frames=False,
            length_scale=None,
        ):
            called["selection"] = list(selection.indices)
            called["truncate"] = truncate
            called["include_contacts"] = include_contacts
            called["include_frames"] = include_frames
            called["length_scale"] = length_scale

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["chunk_frames"] = chunk_frames
            called["device"] = device
            called["frame_indices"] = frame_indices
            return {
                "labels": ["ALA:1", "ALA:2"],
                "mean_matrix": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32),
                "time": np.array([0.0, 1.0, 2.0], dtype=np.float32),
                "frame_matrices": np.array(
                    [
                        [[0.0, 0.4], [0.4, 0.0]],
                        [[0.0, 0.5], [0.5, 0.0]],
                        [[0.0, 0.6], [0.6, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                "frames": 3,
                "residues": 2,
                "truncate": 1.5,
                "used_box": True,
                "length_scale": 0.1,
                "distinct_contact_atoms": np.array([1, 2], dtype=np.uint32),
                "mean_contact_atoms": np.array([0.5, 1.0], dtype=np.float32),
                "contact_ratio": np.array([2.0, 2.0], dtype=np.float32),
                "residue_atom_counts": np.array([2, 2], dtype=np.uint32),
                "mean_contact_atoms_per_residue_atom": np.array([0.25, 0.5], dtype=np.float32),
            }

    monkeypatch.setattr(mdmat_mod, "_MdmatPlan", _DummyPlan, raising=True)
    out_path = tmp_path / "mdmat_frames.npz"
    out = mdmat_mod.mdmat(
        _DummyTraj(),
        _DummySystem(),
        selection="all",
        truncate=1.5,
        include_contacts=True,
        include_frames=True,
        frames_out=out_path,
        length_scale=0.1,
        frame_indices=[0, 2],
        chunk_frames=8,
    )
    assert called["selection"] == [0, 1, 2, 3]
    assert called["truncate"] == pytest.approx(1.5)
    assert called["include_contacts"] is True
    assert called["include_frames"] is True
    assert called["length_scale"] == pytest.approx(0.1)
    assert called["chunk_frames"] == 8
    assert called["device"] == "auto"
    assert called["frame_indices"] == [0, 2]
    assert out["labels"] == ("ALA:1", "ALA:2")
    assert out["mean_matrix"].shape == (2, 2)
    assert np.array_equal(out["distinct_contact_atoms"], np.array([1, 2], dtype=np.uint32))
    assert "time" not in out
    assert "frame_matrices" not in out
    assert out["frames_artifact"]["path"] == str(out_path)
    saved = np.load(out_path)
    assert saved["frame_matrices"].shape == (3, 2, 2)
    assert tuple(saved["labels"].tolist()) == ("ALA:1", "ALA:2")


def test_mdmat_auto_budget_raises_before_run(monkeypatch):
    class _DummyPlan:
        def __init__(self, *args, **kwargs):
            raise AssertionError("plan should not be constructed when auto budget fails")

    monkeypatch.setattr(mdmat_mod, "_MdmatPlan", _DummyPlan, raising=True)
    with pytest.raises(RuntimeError, match="memory_budget_bytes"):
        mdmat_mod.mdmat(
            _DummyTraj(),
            _DummySystem(),
            selection="all",
            include_frames=True,
            memory_budget_bytes=32,
        )


def test_mdmat_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(mdmat_mod, "_MdmatPlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PyMdmatPlan binding unavailable"):
        mdmat_mod.mdmat(_DummyTraj(), _DummySystem(), selection="all")
