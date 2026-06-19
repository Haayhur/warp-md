import numpy as np
import pytest
import warp_md

from warp_md.analysis.drid import drid
from warp_md.analysis.trajectory import ArrayTrajectory


def _atom_table(n_atoms, bonds=None):
    return {
        "name": ["CA"] * n_atoms,
        "resname": ["ALA"] * n_atoms,
        "resid": [1] * n_atoms,
        "chain_id": [0] * n_atoms,
        "mass": [1.0] * n_atoms,
        "bonds": [] if bonds is None else bonds,
    }


def test_drid_basic_and_bond_exclusion():
    coords = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_atom_table(3, bonds=[(0, 1)]), positions0=coords[0])

    values = drid(
        warp_md.Trajectory.from_numpy(coords),
        system,
        selection="all",
        exclude_bonds=False,
        device="cpu",
    )
    expected = np.array(
        [[2.0 / 3.0, 1.0 / 3.0, 0.0, 0.75, 0.25, 0.0, 5.0 / 12.0, 1.0 / 12.0, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(values, expected, rtol=1e-6, atol=1e-6)

    out = drid(
        warp_md.Trajectory.from_numpy(coords),
        system,
        selection="all",
        exclude_bonds=True,
        dtype="dict",
    )
    expected_excluded = np.array(
        [[1.0 / 3.0, 0.0, 0.0, 0.5, 0.0, 0.0, 5.0 / 12.0, 1.0 / 12.0, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out["values"], expected_excluded, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out["moments"].reshape(1, 9), expected_excluded, rtol=1e-6)
    np.testing.assert_array_equal(out["atom_indices"], np.array([0, 1, 2], dtype=np.int64))
    assert out["exclude_bonds"] is True


def test_drid_atom_indices_and_frame_subset():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [99.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [99.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_atom_table(3), positions0=coords[0])

    values = drid(
        warp_md.Trajectory.from_numpy(coords),
        system,
        atom_indices=[0, 2],
        frame_indices=[1],
    )
    expected = np.array([[1.0 / 6.0, 0.0, 0.0, 1.0 / 6.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


def test_drid_requires_native_trajectory():
    coords = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_atom_table(3), positions0=coords[0])

    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        drid(ArrayTrajectory(coords), system)
