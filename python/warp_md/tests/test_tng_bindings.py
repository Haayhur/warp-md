from __future__ import annotations

from pathlib import Path

import numpy as np

import warp_md as wmd

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "tng"


def _fixture(name: str) -> Path:
    return FIXTURE_DIR / name


def _atom_table(n_atoms: int):
    return {
        "name": ["OW"] * n_atoms,
        "resname": ["SOL"] * n_atoms,
        "resid": list(range(1, n_atoms + 1)),
        "chain": ["A"] * n_atoms,
        "element": ["O"] * n_atoms,
        "mass": [16.0] * n_atoms,
    }


def test_open_tng_matches_gromacs_xtc_fixture():
    system = wmd.System.from_gro(str(_fixture("spc2.gro")))
    tng = wmd.Trajectory.open_tng(str(_fixture("spc2-traj.tng")), system)
    xtc = wmd.Trajectory.open_xtc(str(_fixture("spc2-traj.xtc")), system)

    assert tng.n_atoms() == 6
    assert tng.count_frames() == 2
    assert xtc.count_frames() == 2

    tng_chunk = tng.read_chunk(include_time=True)
    xtc_chunk = xtc.read_chunk(include_time=True)

    assert tng_chunk["frames"] == 2
    np.testing.assert_allclose(tng_chunk["coords"], xtc_chunk["coords"], atol=1e-4)
    np.testing.assert_allclose(tng_chunk["box"], xtc_chunk["box"], atol=1e-4)
    np.testing.assert_allclose(tng_chunk["time_ps"], xtc_chunk["time_ps"], atol=1e-4)

    tng.reset()
    again = tng.read_chunk(include_time=True)
    np.testing.assert_allclose(again["coords"], tng_chunk["coords"], atol=1e-4)
    np.testing.assert_allclose(again["box"], tng_chunk["box"], atol=1e-4)
    np.testing.assert_allclose(again["time_ps"], tng_chunk["time_ps"], atol=1e-4)


def test_tng_writer_roundtrip_from_python(tmp_path: Path):
    coords = np.array(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[5.5, 4.5, 3.5], [2.5, 1.5, 0.5]],
        ],
        dtype=np.float32,
    )
    box = np.array([[20.0, 21.0, 22.0], [23.0, 24.0, 25.0]], dtype=np.float32)
    time_ps = np.array([0.25, 1.75], dtype=np.float32)
    velocities = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ],
        dtype=np.float32,
    )
    forces = np.array(
        [
            [[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]],
            [[3.5, 3.0, 2.5], [2.0, 1.5, 1.0]],
        ],
        dtype=np.float32,
    )

    path = tmp_path / "roundtrip.tng"
    writer = wmd.TrajectoryWriter.open(str(path), "tng", coords.shape[1], None)
    try:
        for index in range(coords.shape[0]):
            writer.write_frame(
                coords[index],
                box_lengths=box[index],
                step=index,
                time_ps=float(time_ps[index]),
                velocities=velocities[index],
                forces=forces[index],
            )
    finally:
        writer.flush()

    system = wmd.System.from_arrays(_atom_table(coords.shape[1]), positions0=coords[0])
    traj = wmd.Trajectory.open_tng(str(path), system)

    assert traj.count_frames() == 2
    chunk = traj.read_chunk(include_time=True, include_velocities=True, include_forces=True)

    np.testing.assert_allclose(chunk["coords"], coords, atol=1e-5)
    np.testing.assert_allclose(chunk["box"], box, atol=1e-5)
    np.testing.assert_allclose(chunk["time_ps"], time_ps, atol=1e-5)
    np.testing.assert_allclose(chunk["velocities"], velocities, atol=1e-5)
    np.testing.assert_allclose(chunk["forces"], forces, atol=1e-5)
    assert chunk["lambda_value"] is None
