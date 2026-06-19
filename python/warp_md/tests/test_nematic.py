import numpy as np
import pytest
import warp_md

from warp_md.analysis.nematic import compute_directors, nematic_order
from warp_md.analysis.trajectory import ArrayTrajectory


def _atom_table(n_atoms):
    return {
        "name": ["C"] * n_atoms,
        "resname": ["MOL"] * n_atoms,
        "resid": list(range(1, n_atoms + 1)),
        "chain_id": [0] * n_atoms,
        "mass": [1.0] * n_atoms,
    }


def _coords():
    return np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )


def test_nematic_order_pairs_report_director_and_axis_projection():
    coords = _coords()
    system = warp_md.System.from_arrays(_atom_table(coords.shape[1]), positions0=coords[0])
    traj = warp_md.Trajectory.from_numpy(coords, time_ps=np.array([5.0, 9.0], dtype=np.float32))

    out = nematic_order(
        traj,
        system,
        pairs=[(0, 1), (2, 3)],
        reference_axis=[0.0, 0.0, 1.0],
        frame_indices=[0],
        device="cpu",
    )

    np.testing.assert_allclose(out["time"], np.array([5.0], dtype=np.float32))
    np.testing.assert_allclose(out["order"], np.array([1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out["director"], np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(out["axis_order"], np.array([1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        out["q_tensor"][0],
        np.array([[-0.5, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(out["valid_vectors"], np.array([2.0], dtype=np.float32))


def test_nematic_order_adjacent_selection_and_director_helper():
    coords = _coords()
    system = warp_md.System.from_arrays(_atom_table(coords.shape[1]), positions0=coords[0])

    directors = compute_directors(
        warp_md.Trajectory.from_numpy(coords),
        system,
        selection=[0, 1, 2, 3],
        frame_indices=[1],
    )

    np.testing.assert_allclose(directors, np.array([[1.0, 0.0, 0.0]], dtype=np.float32), atol=1e-6)


def test_nematic_order_requires_native_trajectory():
    coords = _coords()[:1]
    system = warp_md.System.from_arrays(_atom_table(coords.shape[1]), positions0=coords[0])

    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        nematic_order(ArrayTrajectory(coords), system, pairs=[(0, 1)])
