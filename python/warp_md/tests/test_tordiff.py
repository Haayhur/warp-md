import numpy as np
import pytest

import warp_md
from warp_md.analysis.diffusion import tordiff


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._n_atoms = n_atoms

    def select(self, _mask):
        return _DummySelection(range(self._n_atoms))

    def select_indices(self, indices):
        return _DummySelection(indices)


class _DummyTraj:
    def __init__(self, n_frames=4, n_atoms=4):
        self._coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_tordiff_uses_torsion_plan(monkeypatch):
    traj = _DummyTraj(n_frames=2, n_atoms=4)
    system = _DummySystem(4)
    called = {}

    class _DummyTorsionPlan:
        def __init__(self, _sel):
            called["created"] = True

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "TorsionDiffusionPlan", _DummyTorsionPlan, raising=False)
    out = tordiff(traj, system, mask="all")
    assert called["created"] is True
    assert called["frame_indices"] is None
    np.testing.assert_allclose(out["trans"], np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(out["cis"], np.array([0.0, 1.0], dtype=np.float32))


def test_tordiff_output_files(tmp_path, monkeypatch):
    traj = _DummyTraj(n_frames=2, n_atoms=4)
    system = _DummySystem(4)

    class _DummyTorsionPlan:
        def __init__(self, _sel):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([[0.5, 0.5, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "TorsionDiffusionPlan", _DummyTorsionPlan, raising=False)
    out_path = tmp_path / "tordiff_out.txt"
    diff_path = tmp_path / "tordiff_diff.txt"
    out = tordiff(traj, system, mask="all", out=str(out_path), diffout=str(diff_path), time=2.0)
    assert out["time"].shape[0] == 2
    assert out_path.exists()
    assert diff_path.exists()
    data = np.loadtxt(out_path, comments="#")
    assert data.shape[0] == 2


def test_tordiff_plan_with_frame_indices(monkeypatch):
    traj = _DummyTraj(n_frames=2, n_atoms=4)
    system = _DummySystem(4)
    called = {}

    class _DummyTorsionPlan:
        def __init__(self, _sel):
            called["created"] = True

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            if frame_indices is None:
                return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
            return np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "TorsionDiffusionPlan", _DummyTorsionPlan, raising=False)
    out = tordiff(traj, system, mask="all", frame_indices=[1])
    assert called["created"] is True
    assert called["frame_indices"] == [1]
    np.testing.assert_allclose(out["trans"], np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(out["cis"], np.array([1.0], dtype=np.float32))


def test_tordiff_mass_uses_toroidal_plan(monkeypatch):
    traj = _DummyTraj(n_frames=2, n_atoms=4)
    system = _DummySystem(4)
    called = {}

    class _DummyToroidalPlan:
        def __init__(self, _sel, mass_weighted=False, transition_lag=1, emit_transitions=False):
            called["mass_weighted"] = mass_weighted
            called["transition_lag"] = transition_lag
            called["emit_transitions"] = emit_transitions

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["run"] = True
            called["frame_indices"] = frame_indices
            return np.array([[0.5, 0.5, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "ToroidalDiffusionPlan", _DummyToroidalPlan, raising=False)
    out = tordiff(traj, system, mask="all", mass=True)
    assert called["mass_weighted"] is True
    assert called["emit_transitions"] is False
    assert called["run"] is True
    assert called["frame_indices"] is None
    np.testing.assert_allclose(out["trans"], np.array([0.5, 0.2], dtype=np.float32))


def test_tordiff_transitions_uses_toroidal_run_full(monkeypatch):
    traj = _DummyTraj(n_frames=2, n_atoms=4)
    system = _DummySystem(4)
    called = {}

    class _DummyToroidalPlan:
        def __init__(self, _sel, mass_weighted=False, transition_lag=1, emit_transitions=False):
            called["mass_weighted"] = mass_weighted
            called["transition_lag"] = transition_lag
            called["emit_transitions"] = emit_transitions

        def run_full(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["run_full"] = True
            called["frame_indices"] = frame_indices
            data = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
            counts = np.array(
                [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            )
            probs = counts.copy()
            return data, counts, probs, 0.0

    monkeypatch.setattr(warp_md, "ToroidalDiffusionPlan", _DummyToroidalPlan, raising=False)
    out = tordiff(traj, system, mask="all", return_transitions=True, transition_lag=1)
    assert called["mass_weighted"] is False
    assert called["emit_transitions"] is True
    assert called["transition_lag"] == 1
    assert called["run_full"] is True
    assert called["frame_indices"] is None
    assert out["transition_counts"].shape == (4, 4)
    assert out["transition_matrix"].shape == (4, 4)


def test_tordiff_transitions_with_frame_indices_use_toroidal_states(monkeypatch):
    traj = _DummyTraj(n_frames=4, n_atoms=4)
    system = _DummySystem(4)
    called = {}

    class _DummyToroidalPlan:
        def __init__(
            self,
            _sel,
            mass_weighted=False,
            transition_lag=1,
            emit_transitions=False,
            store_transition_states=False,
        ):
            called["mass_weighted"] = mass_weighted
            called["transition_lag"] = transition_lag
            called["emit_transitions"] = emit_transitions
            called["store_transition_states"] = store_transition_states

        def run_full_with_states(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["run_full_with_states"] = True
            called["frame_indices"] = frame_indices
            data = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            counts = np.zeros((4, 4), dtype=np.float32)
            probs = np.zeros((4, 4), dtype=np.float32)
            states = np.array([[0], [1], [3]], dtype=np.uint8)
            return data, counts, probs, 0.0, states

    monkeypatch.setattr(warp_md, "ToroidalDiffusionPlan", _DummyToroidalPlan, raising=False)
    out = tordiff(
        traj,
        system,
        mask="all",
        return_transitions=True,
        transition_lag=1,
        frame_indices=[0, 2, 3],
    )
    assert called["emit_transitions"] is True
    assert called["store_transition_states"] is True
    assert called["run_full_with_states"] is True
    assert called["frame_indices"] == [0, 2, 3]
    np.testing.assert_allclose(out["trans"], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(out["cis"], np.array([0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(out["g_minus"], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    counts = out["transition_counts"]
    assert counts.shape == (4, 4)
    assert counts[0, 1] == 1.0
    assert counts[1, 3] == 1.0
    assert float(np.sum(counts)) == 2.0


def test_tordiff_missing_plan_binding_raises(monkeypatch):
    traj = _DummyTraj(n_frames=1, n_atoms=4)
    system = _DummySystem(4)

    class _Missing:
        __name__ = "_Missing"

    monkeypatch.setattr(warp_md, "TorsionDiffusionPlan", _Missing, raising=False)
    monkeypatch.setattr(warp_md, "ToroidalDiffusionPlan", _Missing, raising=False)
    with pytest.raises(RuntimeError, match="binding unavailable"):
        tordiff(traj, system, mask="all")
