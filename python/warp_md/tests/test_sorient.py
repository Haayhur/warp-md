import importlib

import numpy as np
import pytest

sorient_mod = importlib.import_module("warp_md.analysis.sorient")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "name": ["CA", "OW", "HW1", "HW2"],
            "resname": ["ALA", "SOL", "SOL", "SOL"],
            "resid": [1, 2, 2, 2],
            "element": ["C", "O", "H", "H"],
            "chain_id": [0, 0, 0, 0],
        }

    def atom_table(self):
        return self._atoms

    def select(self, mask):
        if mask == "resid 1":
            return _DummySelection([0])
        return _DummySelection([1, 2, 3])

    def select_indices(self, indices):
        return _DummySelection(list(indices))


class _DummyTraj:
    pass


def test_sorient_detects_triplets_and_forwards(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            solute_selection,
            atom1_indices,
            atom2_indices,
            atom3_indices,
            r_min=0.0,
            r_max=0.5,
            cbin=0.02,
            rbin=0.02,
            use_com=False,
            use_vector23=False,
            r_profile_max=None,
            length_scale=None,
        ):
            called["solute_indices"] = list(solute_selection.indices)
            called["atom1_indices"] = list(atom1_indices)
            called["atom2_indices"] = list(atom2_indices)
            called["atom3_indices"] = list(atom3_indices)
            called["r_min"] = r_min
            called["r_max"] = r_max
            called["cbin"] = cbin
            called["rbin"] = rbin
            called["use_com"] = use_com
            called["use_vector23"] = use_vector23
            called["r_profile_max"] = r_profile_max
            called["length_scale"] = length_scale

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["chunk_frames"] = chunk_frames
            called["device"] = device
            called["frame_indices"] = frame_indices
            return {
                "cos_theta1": np.array([-0.5, 0.5], dtype=np.float32),
                "cos_theta1_distribution": np.array([0.0, 2.0], dtype=np.float32),
                "abs_cos_theta2": np.array([0.25, 0.75], dtype=np.float32),
                "abs_cos_theta2_distribution": np.array([2.0, 0.0], dtype=np.float32),
                "r": np.array([0.5, 1.5], dtype=np.float32),
                "mean_cos_theta1": np.array([0.0, 0.7], dtype=np.float32),
                "mean_p2_theta2": np.array([0.0, -1.0], dtype=np.float32),
                "cumulative_r": np.array([1.0, 2.0], dtype=np.float32),
                "cumulative_cos_theta1": np.array([0.0, 0.7], dtype=np.float32),
                "cumulative_p2_theta2": np.array([0.0, -1.0], dtype=np.float32),
                "count_density": np.array([0.0, 0.5], dtype=np.float32),
                "counts": np.array([0, 1], dtype=np.uint64),
                "window_count": 1,
                "average_shell_size": 1.0,
                "window_mean_cos_theta1": 0.7,
                "window_mean_p2_theta2": -1.0,
                "r_window": np.array([0.0, 1.5], dtype=np.float32),
                "cbin": 0.5,
                "rbin": 1.0,
                "r_profile_max": 2.0,
                "use_vector23": True,
                "use_com": True,
                "n_frames": 2,
                "n_reference_positions": 1,
                "used_box": False,
                "length_scale": 0.1,
            }

    monkeypatch.setattr(sorient_mod, "_SolventOrientationPlan", _DummyPlan, raising=True)
    out = sorient_mod.sorient(
        _DummyTraj(),
        _DummySystem(),
        solute_selection="resid 1",
        solvent_selection="resname SOL",
        r_min=0.1,
        r_max=1.5,
        cbin=0.5,
        rbin=1.0,
        use_com=True,
        use_vector23=True,
        r_profile_max=2.0,
        length_scale=0.1,
        chunk_frames=32,
        frame_indices=[0, 2],
    )
    assert called["solute_indices"] == [0]
    assert called["atom1_indices"] == [1]
    assert called["atom2_indices"] == [2]
    assert called["atom3_indices"] == [3]
    assert called["r_min"] == pytest.approx(0.1)
    assert called["r_max"] == pytest.approx(1.5)
    assert called["use_com"] is True
    assert called["use_vector23"] is True
    assert called["r_profile_max"] == pytest.approx(2.0)
    assert called["length_scale"] == pytest.approx(0.1)
    assert called["chunk_frames"] == 32
    assert called["frame_indices"] == [0, 2]
    assert out["use_com"] is True
    assert out["use_vector23"] is True
    np.testing.assert_allclose(out["mean_p2_theta2"], np.array([0.0, -1.0], dtype=np.float32))


def test_sorient_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(sorient_mod, "_SolventOrientationPlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PySolventOrientationPlan binding unavailable"):
        sorient_mod.sorient(_DummyTraj(), _DummySystem(), solute_selection="resid 1")
