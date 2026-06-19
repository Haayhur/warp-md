import numpy as np
import pytest

import warp_md
from warp_md.analysis.velocity import get_velocity


def _atom_table(n_atoms):
    return {
        "name": ["CA"] * n_atoms,
        "resname": ["ALA"] * n_atoms,
        "resid": [1] * n_atoms,
        "chain_id": [0] * n_atoms,
        "mass": [1.0] * n_atoms,
    }


def test_get_velocity_basic_native():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = warp_md.Trajectory.from_numpy(coords)
    system = warp_md.System.from_arrays(_atom_table(coords.shape[1]), positions0=coords[0])

    vel = get_velocity(traj, system, mask="all")

    assert vel.shape == coords.shape
    np.testing.assert_allclose(vel[0], np.zeros((2, 3), dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(vel[1, 0], [1.0, 0.0, 0.0], atol=1e-6)


def test_get_velocity_frame_indices_and_time_scale_native():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
            [[6.0, 0.0, 0.0], [16.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    time = np.array([0.0, 2.0, 6.0], dtype=np.float32)
    traj = warp_md.Trajectory.from_numpy(coords, time_ps=time)
    system = warp_md.System.from_arrays(_atom_table(coords.shape[1]), positions0=coords[0])

    vel = get_velocity(
        traj,
        system,
        mask=[0],
        frame_indices=[0, 2],
        length_scale=2.0,
        time_scale=4.0,
    )

    assert vel.shape == (2, 1, 3)
    np.testing.assert_allclose(vel[0, 0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(vel[1, 0], [0.5, 0.0, 0.0], atol=1e-6)


def test_get_velocity_requires_native_trajectory():
    class DummyTraj:
        pass

    coords0 = np.zeros((1, 3), dtype=np.float32)
    system = warp_md.System.from_arrays(_atom_table(1), positions0=coords0)

    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        get_velocity(DummyTraj(), system, mask="all")
