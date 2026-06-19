import numpy as np
import pytest
import warp_md

from warp_md.analysis.check_structure import check_structure
from warp_md.analysis.trajectory import ArrayTrajectory


def _system(n_atoms):
    return warp_md.System.from_arrays(
        {
            "name": ["A"] * n_atoms,
            "resname": ["RES"] * n_atoms,
            "resid": list(range(1, n_atoms + 1)),
            "chain_id": [0] * n_atoms,
            "mass": [1.0] * n_atoms,
        }
    )


def test_check_structure_counts():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = warp_md.Trajectory.from_numpy(coords)
    counts, report = check_structure(traj, system=None)
    np.testing.assert_allclose(counts, np.array([1, 0], dtype=np.int64))
    assert "frame 0" in report


def test_check_structure_frame_subset_and_selection():
    coords = np.array(
        [
            [[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]],
            [[np.nan, 0.0, 0.0], [np.inf, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    counts, report = check_structure(
        warp_md.Trajectory.from_numpy(coords),
        _system(coords.shape[1]),
        mask="@2",
        frame_indices=[2, 0],
    )
    np.testing.assert_allclose(counts, np.array([1, 0], dtype=np.int64))
    assert "frame 0" in report
    assert "frame 1" not in report


def test_check_structure_requires_native_trajectory():
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        check_structure(ArrayTrajectory(coords), _system(1))
