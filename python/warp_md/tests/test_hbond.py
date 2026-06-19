import importlib

import numpy as np
import pytest
import warp_md

from warp_md.analysis.hbond import hbond
from warp_md.analysis.trajectory import ArrayTrajectory

hbond_mod = importlib.import_module("warp_md.analysis.hbond")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "name": ["D", "H", "A"],
            "resname": ["MOL"] * 3,
            "resid": [1, 1, 1],
            "chain_id": [0, 0, 0],
            "mass": [14.0, 1.0, 16.0],
        }

    def atom_table(self):
        return self._atoms

    def select(self, mask):
        if mask == "donor":
            return _DummySelection([0])
        if mask == "hydrogen":
            return _DummySelection([1])
        if mask == "acceptor":
            return _DummySelection([2])
        return _DummySelection([0, 1, 2])

    def select_indices(self, indices):
        return _DummySelection(indices)


def _system(coords):
    return warp_md.System.from_arrays(
        {
            "name": ["D", "H", "A"],
            "resname": ["MOL"] * 3,
            "resid": [1, 1, 1],
            "chain_id": [0, 0, 0],
            "mass": [14.0, 1.0, 16.0],
        },
        positions0=coords[0],
    )


def test_hbond_counts_and_angle_filter_run_native():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = warp_md.Trajectory.from_numpy(coords, time_ps=np.array([10.0, 20.0], dtype=np.float32))
    system = _system(coords)

    out = hbond(
        traj,
        system,
        donors="@1",
        hydrogens="@2",
        acceptors="@3",
        dist_cutoff=3.0,
        angle_cutoff=120.0,
        device="cpu",
    )

    np.testing.assert_allclose(out["time"], np.array([10.0, 20.0], dtype=np.float32))
    np.testing.assert_allclose(out["count"], np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(out["donors"], np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(out["hydrogens"], np.array([1], dtype=np.int64))
    np.testing.assert_array_equal(out["acceptors"], np.array([2], dtype=np.int64))
    assert out["dist_cutoff"] == 3.0
    assert out["angle_cutoff"] == 120.0


def test_hbond_dtype_routes_and_validation():
    coords = np.array(
        [[[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    traj = warp_md.Trajectory.from_numpy(coords)
    system = _system(coords)

    counts = hbond(traj, system, donors="@1", acceptors="@3", dist_cutoff=3.0, dtype="ndarray")
    time, tuple_counts = hbond(
        warp_md.Trajectory.from_numpy(coords),
        system,
        donors="@1",
        acceptors="@3",
        dist_cutoff=3.0,
        dtype="tuple",
    )

    np.testing.assert_allclose(counts, np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(time, np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(tuple_counts, counts)
    with pytest.raises(ValueError, match="angle_cutoff is required"):
        hbond(traj, system, donors="@1", hydrogens="@2", acceptors="@3", dist_cutoff=3.0)


def test_hbond_uses_live_native_path(monkeypatch):
    traj = object()
    system = _DummySystem()
    seen = {}

    class _DummyPlan:
        def __init__(self, donors, acceptors, dist_cutoff, hydrogens=None, angle_cutoff=None):
            seen["donors"] = list(donors.indices)
            seen["acceptors"] = list(acceptors.indices)
            seen["hydrogens"] = list(hydrogens.indices)
            seen["dist_cutoff"] = dist_cutoff
            seen["angle_cutoff"] = angle_cutoff

        def run(self, got_traj, got_system, chunk_frames=None, device="auto", frame_indices=None):
            seen["traj"] = got_traj
            seen["system"] = got_system
            seen["chunk_frames"] = chunk_frames
            seen["device"] = device
            seen["frame_indices"] = frame_indices
            return np.array([0.0, 1.0], dtype=np.float32), np.array([[2.0], [3.0]], dtype=np.float32)

    monkeypatch.setattr(hbond_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(hbond_mod, "coerce_native_system", lambda got: got)
    monkeypatch.setattr(hbond_mod, "load_native_symbol", lambda name: _DummyPlan)

    out = hbond(
        traj,
        system,
        donors="donor",
        hydrogens="hydrogen",
        acceptors="acceptor",
        dist_cutoff=3.0,
        angle_cutoff=120.0,
        chunk_frames=4,
        device="cpu",
        frame_indices=[1, 0],
    )

    np.testing.assert_allclose(out["count"], np.array([2.0, 3.0], dtype=np.float32))
    assert seen == {
        "donors": [0],
        "acceptors": [2],
        "hydrogens": [1],
        "dist_cutoff": 3.0,
        "angle_cutoff": 120.0,
        "traj": traj,
        "system": system,
        "chunk_frames": 4,
        "device": "cpu",
        "frame_indices": [1, 0],
    }


def test_hbond_requires_native_trajectory():
    coords = np.array([[[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        hbond(ArrayTrajectory(coords), _system(coords), donors="@1", acceptors="@3", dist_cutoff=3.0)
