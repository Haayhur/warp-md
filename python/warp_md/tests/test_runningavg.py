import numpy as np
import pytest
import warp_md

from warp_md.analysis.runningavg import runningavg
from warp_md.analysis.trajectory import ArrayTrajectory


def _atom_table(n_atoms):
    return {
        "name": ["CA"] * n_atoms,
        "resname": ["ALA"] * n_atoms,
        "resid": [1] * n_atoms,
        "chain_id": [0] * n_atoms,
        "mass": [1.0] * n_atoms,
    }


def test_runningavg_cumulative_and_windowed():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_atom_table(2), positions0=coords[0])

    cumulative = runningavg(
        warp_md.Trajectory.from_numpy(coords),
        system,
        selection="all",
        device="cpu",
    )
    expected_cumulative = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [14.0 / 3.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(cumulative, expected_cumulative, rtol=1e-6)

    windowed = runningavg(warp_md.Trajectory.from_numpy(coords), system, selection="all", window=2)
    expected_windowed = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(windowed, expected_windowed, rtol=1e-6)


def test_runningavg_atom_indices_frame_subset_and_dict():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [99.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [99.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [99.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_atom_table(3), positions0=coords[0])

    out = runningavg(
        warp_md.Trajectory.from_numpy(coords),
        system,
        selection="@1 @3",
        frame_indices=[1, 2],
        dtype="dict",
    )
    expected = np.array(
        [
            [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out["coords"], expected, rtol=1e-6)
    np.testing.assert_array_equal(out["atom_indices"], np.array([0, 2], dtype=np.int64))
    assert out["mode"] == "cumulative"


def test_runningavg_trajectory_preserves_metadata():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [99.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [99.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [99.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    box = np.array(
        [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        dtype=np.float32,
    )
    time = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    system = warp_md.System.from_arrays(_atom_table(3), positions0=coords[0])

    out = runningavg(
        warp_md.Trajectory.from_numpy(coords, box=box, time_ps=time),
        system,
        selection="@1 @2",
        frame_indices=[2, 0],
        chunk_frames=1,
        dtype="trajectory",
    )
    chunk = out.read_chunk(4)
    expected = np.array(
        [
            [[4.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(chunk["coords"], expected, rtol=1e-6)
    np.testing.assert_allclose(chunk["box"], box[[2, 0]], rtol=1e-6)
    np.testing.assert_allclose(chunk["time"], time[[2, 0]].astype(np.float64), rtol=1e-6)


def test_runningavg_requires_native_trajectory():
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
    system = warp_md.System.from_arrays(_atom_table(1), positions0=coords[0])

    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        runningavg(ArrayTrajectory(coords), system)
