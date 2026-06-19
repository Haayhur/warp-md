import importlib

import numpy as np
import pytest

import warp_md
from warp_md.analysis.pairdist import maxdist, mindist, pairdist

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


def test_pairdist_forwards_device_to_native_plan(monkeypatch):
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
            pass

        def run(self, traj, system, chunk_frames=None, device="auto", frame_indices=None):
            seen["device"] = device
            return (
                np.array([0.5], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([1], dtype=np.uint64),
                1,
            )

    traj = _DummyNativeTraj()
    system = _DummySystem(2)
    monkeypatch.setattr(pairdist_mod, "load_native_symbol", lambda name: _DummyPlan)
    monkeypatch.setattr(pairdist_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(pairdist_mod, "coerce_native_system", lambda got: got)

    pairdist(traj, system, mask="all", delta=1.0, device="cpu")

    assert seen["device"] == "cpu"


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
        image=True,
        frame_indices=[0, -1, 99],
        chunk_frames=11,
    )

    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["chunk_frames"] == 11
    assert seen["device"] == "auto"
    assert seen["frame_indices"] == [0, -1, 99]
    assert seen["n_bins"] == 3
    assert seen["pbc"] == "orthorhombic"
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
        def __init__(
            self,
            sel_a,
            sel_b,
            mode="min",
            pbc="none",
            unique_pairs=False,
            cutoff=None,
            empty_value=None,
        ):
            seen["sel_a"] = list(sel_a.indices)
            seen["sel_b"] = list(sel_b.indices)
            seen["mode"] = mode
            seen["pbc"] = pbc
            seen["unique_pairs"] = unique_pairs
            seen["cutoff"] = cutoff
            seen["empty_value"] = empty_value

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
        maxdist=6.0,
        image=True,
        frame_indices=[0, 1],
        chunk_frames=3,
    )

    assert seen["plan_name"] == "PairDistanceExtremaPlan"
    assert seen["sel_a"] == [0, 1, 2]
    assert seen["sel_b"] == [0, 1, 2]
    assert seen["mode"] == "max"
    assert seen["pbc"] == "orthorhombic"
    assert seen["unique_pairs"] is True
    assert seen["cutoff"] == 6.0
    assert seen["empty_value"] == 6.0
    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["chunk_frames"] == 3
    assert seen["device"] == "auto"
    assert seen["frame_indices"] == [0, 1]
    assert out["mode"] == "max"
    assert out["n_frames"] == 2
    assert out["pairdist"].tolist() == [4.0, 5.0]


def test_mindist_and_maxdist_call_native_extrema_plan(monkeypatch):
    seen = []

    class _DummyPlan:
        def __init__(
            self,
            sel_a,
            sel_b,
            mode="min",
            pbc="none",
            unique_pairs=False,
            cutoff=None,
            empty_value=None,
        ):
            seen.append(
                {
                    "sel_a": list(sel_a.indices),
                    "sel_b": list(sel_b.indices),
                    "mode": mode,
                    "pbc": pbc,
                    "unique_pairs": unique_pairs,
                    "cutoff": cutoff,
                    "empty_value": empty_value,
                }
            )

        def run(self, traj, system, chunk_frames=None, device="auto", frame_indices=None):
            seen[-1]["traj"] = traj
            seen[-1]["system"] = system
            seen[-1]["chunk_frames"] = chunk_frames
            seen[-1]["device"] = device
            seen[-1]["frame_indices"] = frame_indices
            return np.array([1.0, 2.0], dtype=np.float32)

    traj = _DummyNativeTraj()
    system = _DummySystem(3)
    monkeypatch.setattr(pairdist_mod, "load_native_symbol", lambda name: _DummyPlan)
    monkeypatch.setattr(pairdist_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(pairdist_mod, "coerce_native_system", lambda got: got)

    min_out = mindist(
        traj,
        system,
        mask="all",
        frame_indices=[0, 2],
        chunk_frames=4,
        device="cpu",
    )
    max_out = maxdist(
        traj,
        system,
        mask="@1",
        mask2="@2",
        maxdist=6.0,
        image=True,
        dtype="dict",
    )

    np.testing.assert_allclose(min_out, np.array([1.0, 2.0], dtype=np.float32))
    assert seen[0] == {
        "sel_a": [0, 1, 2],
        "sel_b": [0, 1, 2],
        "mode": "min",
        "pbc": "none",
        "unique_pairs": True,
        "cutoff": None,
        "empty_value": None,
        "traj": traj,
        "system": system,
        "chunk_frames": 4,
        "device": "cpu",
        "frame_indices": [0, 2],
    }
    assert max_out["mode"] == "max"
    assert max_out["n_frames"] == 2
    assert seen[1]["sel_a"] == [0]
    assert seen[1]["sel_b"] == [1]
    assert seen[1]["mode"] == "max"
    assert seen[1]["pbc"] == "orthorhombic"
    assert seen[1]["unique_pairs"] is False
    assert seen[1]["cutoff"] == 6.0
    assert seen[1]["empty_value"] == 6.0


def test_pairdist_rejects_non_native_trajectory(monkeypatch):
    monkeypatch.setattr(pairdist_mod, "is_native_traj", lambda _traj: False)
    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        pairdist(_DummyNativeTraj(), _DummySystem(3), mask="all")


def test_pairdist_extrema_image_numeric():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    box = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    system = warp_md.System.from_arrays(
        {
            "name": ["A", "B"],
            "resname": ["RES", "RES"],
            "resid": [1, 2],
            "chain_id": [0, 0],
            "mass": [1.0, 1.0],
        },
        positions0=coords[0],
    )

    no_image = pairdist(
        warp_md.Trajectory.from_numpy(coords, box=box),
        system,
        mask="@1",
        mask2="@2",
        mode="min",
        image=False,
    )
    with_image = pairdist(
        warp_md.Trajectory.from_numpy(coords, box=box),
        system,
        mask="@1",
        mask2="@2",
        mode="min",
        image=True,
    )

    np.testing.assert_allclose(no_image["pairdist"], np.array([9.0], dtype=np.float32))
    np.testing.assert_allclose(with_image["pairdist"], np.array([1.0], dtype=np.float32))


def test_pairdist_extrema_maxdist_filters_and_fills_empty_frames():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(
        {
            "name": ["A", "B"],
            "resname": ["RES", "RES"],
            "resid": [1, 2],
            "chain_id": [0, 0],
            "mass": [1.0, 1.0],
        },
        positions0=coords[0],
    )

    out = pairdist(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="@1",
        mask2="@2",
        mode="min",
        maxdist=2.0,
        image=False,
    )

    np.testing.assert_allclose(out["pairdist"], np.array([2.0, 1.5], dtype=np.float32))
