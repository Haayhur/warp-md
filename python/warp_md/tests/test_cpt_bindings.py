from __future__ import annotations

from pathlib import Path

import numpy as np

import warp_md as wmd


def _atom_table(n_atoms: int):
    return {
        "name": ["OW"] * n_atoms,
        "resname": ["SOL"] * n_atoms,
        "resid": list(range(1, n_atoms + 1)),
        "chain": ["A"] * n_atoms,
        "element": ["O"] * n_atoms,
        "mass": [16.0] * n_atoms,
    }


def test_cpt_writer_roundtrip_from_python(tmp_path: Path):
    coords = np.array(
        [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]],
        dtype=np.float32,
    )
    box_matrix = np.array(
        [[[20.0, 1.0, 2.0], [3.0, 21.0, 4.0], [5.0, 6.0, 22.0]]],
        dtype=np.float32,
    )
    time_ps = np.array([1.75], dtype=np.float32)
    velocities = np.array(
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
        dtype=np.float32,
    )
    lambda_value = np.array([0.25], dtype=np.float32)

    path = tmp_path / "roundtrip.cpt"
    writer = wmd.TrajectoryWriter.open(str(path), "cpt", coords.shape[1], None)
    try:
        writer.write_frame(
            coords[0],
            box_matrix=box_matrix[0],
            step=7,
            time_ps=float(time_ps[0]),
            velocities=velocities[0],
            lambda_value=float(lambda_value[0]),
        )
    finally:
        writer.flush()

    system = wmd.System.from_arrays(_atom_table(coords.shape[1]), positions0=coords[0])
    traj = wmd.Trajectory.open_cpt(str(path), system)

    assert traj.n_atoms() == 2
    assert traj.count_frames() == 1

    chunk = traj.read_chunk(
        include_time=True,
        include_box_matrix=True,
        include_velocities=True,
        include_lambda=True,
    )

    assert chunk["frames"] == 1
    np.testing.assert_allclose(chunk["coords"], coords, atol=1e-5)
    np.testing.assert_allclose(chunk["box_matrix"], box_matrix, atol=1e-5)
    np.testing.assert_allclose(chunk["time_ps"], time_ps, atol=1e-5)
    np.testing.assert_allclose(chunk["velocities"], velocities, atol=1e-5)
    np.testing.assert_allclose(chunk["lambda_value"], lambda_value, atol=1e-5)

    traj.reset()
    again = traj.read_chunk(include_time=True, include_lambda=True)
    np.testing.assert_allclose(again["coords"], coords, atol=1e-5)
    np.testing.assert_allclose(again["time_ps"], time_ps, atol=1e-5)
    np.testing.assert_allclose(again["lambda_value"], lambda_value, atol=1e-5)
