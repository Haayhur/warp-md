import numpy as np
import pytest
import warp_md

from warp_md.analysis.dipole import dipole_moments
from warp_md.analysis.trajectory import ArrayTrajectory


def _atom_table():
    return {
        "name": ["A", "B", "A", "B"],
        "resname": ["MOL", "MOL", "MOL", "MOL"],
        "resid": [1, 1, 2, 2],
        "chain_id": [0, 0, 0, 0],
        "mass": [1.0, 1.0, 1.0, 1.0],
    }


def test_dipole_moments_grouped_residues():
    coords = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 3.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 5.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )
    time = np.array([10.0, 20.0], dtype=np.float32)
    system = warp_md.System.from_arrays(_atom_table(), positions0=coords[0])
    traj = warp_md.Trajectory.from_numpy(coords, time_ps=time)

    out = dipole_moments(
        traj,
        system,
        selection="all",
        charges=[1.0, -1.0, 1.0, -1.0],
        group_by="resid",
        length_scale=2.0,
        frame_indices=[1],
        device="cpu",
    )

    np.testing.assert_allclose(out["time"], np.array([20.0], dtype=np.float32))
    expected = np.array([[[-6.0, 0.0, 0.0], [0.0, -6.0, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(out["dipole"], expected, rtol=1e-6, atol=1e-6)
    assert out["group_by"] == "resid"


def test_dipole_moments_ndarray_and_atom_selection():
    coords = np.array(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [99.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(
        {
            "name": ["A", "B", "C"],
            "resname": ["MOL", "MOL", "MOL"],
            "resid": [1, 1, 1],
            "chain_id": [0, 0, 0],
            "mass": [1.0, 1.0, 1.0],
        },
        positions0=coords[0],
    )

    dip = dipole_moments(
        warp_md.Trajectory.from_numpy(coords),
        system,
        selection="@1 @2",
        charges=[1.0, -1.0, 100.0],
        group_by="resid",
        dtype="ndarray",
    )
    np.testing.assert_allclose(
        dip,
        np.array([[[-2.0, 0.0, 0.0]]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_dipole_moments_requires_native_trajectory():
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
    system = warp_md.System.from_arrays(
        {
            "name": ["A", "B"],
            "resname": ["MOL", "MOL"],
            "resid": [1, 1],
            "chain_id": [0, 0],
            "mass": [1.0, 1.0],
        },
        positions0=coords[0],
    )

    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        dipole_moments(ArrayTrajectory(coords), system, charges=[1.0, -1.0])
