import importlib

import numpy as np
import pytest

from warp_md.analysis.pairdist import pairdist

pairdist_mod = importlib.import_module("warp_md.analysis.pairdist")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._atoms = {
            "name": ["A"] * n_atoms,
            "resid": list(range(1, n_atoms + 1)),
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(range(len(self._atoms["name"])))

    def select_indices(self, indices):
        return _DummySelection(indices)


class _DummyNativeTraj:
    def __init__(self):
        self.count_calls = []

    def count_frames(self, chunk_frames=None):
        self.count_calls.append(chunk_frames)
        return 4


def test_pairdist_default_passes_native_trajectory_to_dynamic_rust(monkeypatch):
    seen = {}

    class _DummyPlan:
        def __init__(
            self,
            sel_a,
            sel_b,
            delta,
            pbc,
            output_distribution=False,
            unique_pairs=False,
            compact_output=False,
        ):
            seen["sel_a"] = list(sel_a.indices)
            seen["sel_b"] = list(sel_b.indices)
            seen["delta"] = delta
            seen["pbc"] = pbc
            seen["output_distribution"] = output_distribution
            seen["unique_pairs"] = unique_pairs
            seen["compact_output"] = compact_output

        def run(self, traj, system, chunk_frames=None, device="auto", frame_indices=None):
            seen["traj"] = traj
            seen["system"] = system
            seen["chunk_frames"] = chunk_frames
            seen["device"] = device
            seen["frame_indices"] = frame_indices
            return (
                np.array([1.5], dtype=np.float32),
                np.array([3.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
                np.array([3], dtype=np.uint64),
                4,
            )

    traj = _DummyNativeTraj()
    system = _DummySystem(3)

    def _load(name):
        seen["plan_name"] = name
        return _DummyPlan

    monkeypatch.setattr(pairdist_mod, "load_native_symbol", _load)
    monkeypatch.setattr(pairdist_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(pairdist_mod, "coerce_native_system", lambda got: got)

    out = pairdist(traj, system, mask="all", delta=1.0, chunk_frames=5)

    assert seen["plan_name"] == "PairDistDynamicPlan"
    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["chunk_frames"] == 5
    assert seen["device"] == "auto"
    assert seen["frame_indices"] is None
    assert seen["delta"] == np.float32(1.0)
    assert seen["pbc"] == "none"
    assert seen["output_distribution"] is True
    assert seen["unique_pairs"] is True
    assert seen["compact_output"] is True
    assert traj.count_calls == []
    assert out["n_frames"] == 4
    assert out["bin_centers"].tolist() == [1.5]
    assert out["hist"].tolist() == [3.0]
    assert out["std"].tolist() == [1.0]
    assert out["counts"].tolist() == [3]


def test_pairdist_maxdist_passes_native_trajectory_to_rust(monkeypatch):
    seen = {}

    class _DummyPlan:
        def __init__(
            self,
            sel_a,
            sel_b,
            n_bins,
            r_max,
            pbc,
            output_distribution=False,
            unique_pairs=False,
            compact_output=False,
        ):
            seen["sel_a"] = list(sel_a.indices)
            seen["sel_b"] = list(sel_b.indices)
            seen["n_bins"] = n_bins
            seen["r_max"] = r_max
            seen["pbc"] = pbc
            seen["output_distribution"] = output_distribution
            seen["unique_pairs"] = unique_pairs
            seen["compact_output"] = compact_output

        def run(self, traj, system, chunk_frames=None, device="auto", frame_indices=None):
            seen["traj"] = traj
            seen["system"] = system
            seen["chunk_frames"] = chunk_frames
            seen["device"] = device
            seen["frame_indices"] = frame_indices
            return (
                np.array([1.5], dtype=np.float32),
                np.array([3.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
                np.array([3], dtype=np.uint64),
                2,
            )

    traj = _DummyNativeTraj()
    system = _DummySystem(3)
    monkeypatch.setattr(pairdist_mod, "load_native_symbol", lambda name: _DummyPlan)
    monkeypatch.setattr(pairdist_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(pairdist_mod, "coerce_native_system", lambda got: got)

    out = pairdist_mod.pairdist(
        traj,
        system,
        mask="all",
        delta=1.0,
        maxdist=3.0,
        frame_indices=[0, -1, 99],
        chunk_frames=11,
    )

    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["chunk_frames"] == 11
    assert seen["device"] == "auto"
    assert seen["frame_indices"] == [0, -1, 99]
    assert seen["n_bins"] == 3
    assert seen["pbc"] == "none"
    assert seen["output_distribution"] is True
    assert seen["unique_pairs"] is True
    assert seen["compact_output"] is True
    assert traj.count_calls == []
    assert out["n_frames"] == 2
    assert out["bin_centers"].tolist() == [1.5]
    assert out["hist"].tolist() == [3.0]
    assert out["std"].tolist() == [1.0]
    assert out["counts"].tolist() == [3]


def test_pairdist_extrema_passes_native_trajectory_to_rust(monkeypatch):
    seen = {}

    class _DummyPlan:
        def __init__(self, sel_a, sel_b, mode="min", pbc="none", unique_pairs=False):
            seen["sel_a"] = list(sel_a.indices)
            seen["sel_b"] = list(sel_b.indices)
            seen["mode"] = mode
            seen["pbc"] = pbc
            seen["unique_pairs"] = unique_pairs

        def run(self, traj, system, chunk_frames=None, device="auto", frame_indices=None):
            seen["traj"] = traj
            seen["system"] = system
            seen["chunk_frames"] = chunk_frames
            seen["device"] = device
            seen["frame_indices"] = frame_indices
            return np.array([4.0, 5.0], dtype=np.float32)

    traj = _DummyNativeTraj()
    system = _DummySystem(3)

    def _load(name):
        seen["plan_name"] = name
        return _DummyPlan

    monkeypatch.setattr(pairdist_mod, "load_native_symbol", _load)
    monkeypatch.setattr(pairdist_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(pairdist_mod, "coerce_native_system", lambda got: got)

    out = pairdist_mod.pairdist(
        traj,
        system,
        mask="all",
        mode="max",
        frame_indices=[0, 1],
        chunk_frames=3,
    )

    assert seen["plan_name"] == "PairDistanceExtremaPlan"
    assert seen["sel_a"] == [0, 1, 2]
    assert seen["sel_b"] == [0, 1, 2]
    assert seen["mode"] == "max"
    assert seen["pbc"] == "none"
    assert seen["unique_pairs"] is True
    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["chunk_frames"] == 3
    assert seen["device"] == "auto"
    assert seen["frame_indices"] == [0, 1]
    assert out["mode"] == "max"
    assert out["n_frames"] == 2
    assert out["pairdist"].tolist() == [4.0, 5.0]


def test_pairdist_rejects_non_native_trajectory(monkeypatch):
    monkeypatch.setattr(pairdist_mod, "is_native_traj", lambda _traj: False)
    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        pairdist(_DummyNativeTraj(), _DummySystem(3), mask="all")
