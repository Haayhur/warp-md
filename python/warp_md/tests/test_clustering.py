import numpy as np
import pytest

import warp_md
from warp_md.analysis.clustering import cluster_trajectory


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._n_atoms = n_atoms
        self._atoms = {"resid": list(range(1, n_atoms + 1))}

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(range(self._n_atoms))

    def select_indices(self, indices):
        return _DummySelection(indices)


class _DummyTraj:
    pass


def test_cluster_trajectory_uses_plan(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            sel,
            method="dbscan",
            eps=2.0,
            min_samples=5,
            n_clusters=8,
            max_iter=100,
            tol=1.0e-4,
            seed=0,
            memory_budget_bytes=None,
        ):
            called["selection"] = list(sel.indices)
            called["method"] = method
            called["eps"] = eps
            called["min_samples"] = min_samples
            called["n_clusters"] = n_clusters
            called["max_iter"] = max_iter
            called["tol"] = tol
            called["seed"] = seed
            called["memory_budget_bytes"] = memory_budget_bytes

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
                "labels": [0, 0, -1],
                "centroids": [0],
                "sizes": [2],
                "method": "dbscan",
                "n_frames": 3,
            }

    monkeypatch.setattr(warp_md, "TrajectoryClusterPlan", _DummyPlan, raising=False)
    out = cluster_trajectory(
        _DummyTraj(),
        _DummySystem(4),
        mask=[0, 2, 3],
        method="dbscan",
        eps=1.5,
        min_samples=2,
        frame_indices=[1, -1],
        chunk_frames=64,
        device="cuda",
        memory_budget_bytes=1024,
    )
    assert called["selection"] == [0, 2, 3]
    assert called["method"] == "dbscan"
    assert called["eps"] == 1.5
    assert called["min_samples"] == 2
    assert called["n_clusters"] == 8
    assert called["max_iter"] == 100
    assert called["tol"] == 1.0e-4
    assert called["seed"] == 0
    assert called["memory_budget_bytes"] == 1024
    assert called["chunk_frames"] == 64
    assert called["device"] == "cuda"
    assert called["frame_indices"] == [1, -1]
    np.testing.assert_array_equal(out["labels"], np.array([0, 0, -1], dtype=np.int32))
    np.testing.assert_array_equal(out["centroids"], np.array([0], dtype=np.uint32))
    np.testing.assert_array_equal(out["sizes"], np.array([2], dtype=np.uint32))
    assert out["method"] == "dbscan"
    assert out["n_frames"] == 3


def test_cluster_trajectory_kmeans_method(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, method="dbscan", **_kwargs):
            called["method"] = method

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return {"labels": [1, 0], "centroids": [0, 1], "sizes": [1, 1], "method": "kmeans", "n_frames": 2}

    monkeypatch.setattr(warp_md, "TrajectoryClusterPlan", _DummyPlan, raising=False)
    out = cluster_trajectory(_DummyTraj(), _DummySystem(2), method="kmeans", n_clusters=2, seed=7)
    assert called["method"] == "kmeans"
    assert out["labels"].dtype == np.int32
    assert out["centroids"].dtype == np.uint32
    assert out["sizes"].dtype == np.uint32
    assert out["method"] == "kmeans"
    assert out["n_frames"] == 2


def test_cluster_trajectory_rejects_invalid_method():
    with pytest.raises(ValueError, match="method must be 'dbscan' or 'kmeans'"):
        cluster_trajectory(_DummyTraj(), _DummySystem(2), method="bogus")


def test_cluster_trajectory_raises_when_binding_missing(monkeypatch):
    class _Missing:
        pass

    monkeypatch.setattr(warp_md, "TrajectoryClusterPlan", _Missing, raising=False)
    with pytest.raises(RuntimeError, match="TrajectoryClusterPlan binding unavailable"):
        cluster_trajectory(_DummyTraj(), _DummySystem(2))
