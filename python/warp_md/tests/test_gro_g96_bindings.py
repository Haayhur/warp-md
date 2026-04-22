from __future__ import annotations

from pathlib import Path

import numpy as np

import warp_md as wmd

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "gro_g96"


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


def test_open_gro_matches_gromacs_xtc_fixture():
    system = wmd.System.from_file(str(_fixture("spc2.gro")))
    gro = wmd.Trajectory.open_gro(str(_fixture("spc2-traj.gro")), system)
    xtc = wmd.Trajectory.open_xtc(str(_fixture("spc2-traj.xtc")), system)

    assert gro.n_atoms() == 6
    assert gro.count_frames() == 2
    assert xtc.count_frames() == 2

    gro_chunk = gro.read_chunk(include_time=True, include_velocities=True)
    xtc_chunk = xtc.read_chunk(include_time=True)

    np.testing.assert_allclose(gro_chunk["coords"], xtc_chunk["coords"], atol=1e-4)
    np.testing.assert_allclose(gro_chunk["box"], xtc_chunk["box"], atol=1e-4)
    np.testing.assert_allclose(gro_chunk["time_ps"], xtc_chunk["time_ps"], atol=1e-4)
    assert gro_chunk["velocities"] is not None
    assert gro_chunk["velocities"].shape == (2, 6, 3)


def test_open_g96_matches_gromacs_xtc_fixture():
    system = wmd.System.from_file(str(_fixture("spc2.gro")))
    g96 = wmd.Trajectory.open_g96(str(_fixture("spc2-traj.g96")), system)
    xtc = wmd.Trajectory.open_xtc(str(_fixture("spc2-traj.xtc")), system)

    assert g96.n_atoms() == 6
    assert g96.count_frames() == 2

    g96_chunk = g96.read_chunk(include_time=True, include_velocities=True)
    xtc_chunk = xtc.read_chunk(include_time=True)

    np.testing.assert_allclose(g96_chunk["coords"], xtc_chunk["coords"], atol=1e-4)
    np.testing.assert_allclose(g96_chunk["box"], xtc_chunk["box"], atol=1e-4)
    np.testing.assert_allclose(g96_chunk["time_ps"], xtc_chunk["time_ps"], atol=1e-4)
    assert g96_chunk["velocities"] is not None
    assert g96_chunk["velocities"].shape == (2, 6, 3)


def test_gro_writer_roundtrip_from_python(tmp_path: Path):
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

    path = tmp_path / "roundtrip.gro"
    writer = wmd.TrajectoryWriter.open(str(path), "gro", coords.shape[1], None)
    try:
        for index in range(coords.shape[0]):
            writer.write_frame(
                coords[index],
                box_lengths=box[index],
                step=index,
                time_ps=float(time_ps[index]),
                velocities=velocities[index],
            )
    finally:
        writer.flush()

    system = wmd.System.from_arrays(_atom_table(coords.shape[1]), positions0=coords[0])
    traj = wmd.Trajectory.open_gro(str(path), system)
    chunk = traj.read_chunk(include_time=True, include_velocities=True)

    np.testing.assert_allclose(chunk["coords"], coords, atol=1e-2)
    np.testing.assert_allclose(chunk["box"], box, atol=1e-3)
    np.testing.assert_allclose(chunk["time_ps"], time_ps, atol=1e-5)
    np.testing.assert_allclose(chunk["velocities"], velocities, atol=1e-3)


def test_g96_writer_roundtrip_from_python(tmp_path: Path):
    coords = np.array(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[5.5, 4.5, 3.5], [2.5, 1.5, 0.5]],
        ],
        dtype=np.float32,
    )
    box_matrix = np.array(
        [
            [[20.0, 1.0, 2.0], [3.0, 21.0, 4.0], [5.0, 6.0, 22.0]],
            [[23.0, 1.5, 2.5], [3.5, 24.0, 4.5], [5.5, 6.5, 25.0]],
        ],
        dtype=np.float32,
    )
    time_ps = np.array([0.25, 1.75], dtype=np.float32)
    velocities = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ],
        dtype=np.float32,
    )

    path = tmp_path / "roundtrip.g96"
    writer = wmd.TrajectoryWriter.open(str(path), "g96", coords.shape[1], None)
    try:
        for index in range(coords.shape[0]):
            writer.write_frame(
                coords[index],
                box_matrix=box_matrix[index],
                step=index,
                time_ps=float(time_ps[index]),
                velocities=velocities[index],
            )
    finally:
        writer.flush()

    system = wmd.System.from_arrays(_atom_table(coords.shape[1]), positions0=coords[0])
    traj = wmd.Trajectory.open_g96(str(path), system)
    chunk = traj.read_chunk(include_time=True, include_velocities=True, include_box_matrix=True)

    np.testing.assert_allclose(chunk["coords"], coords, atol=1e-5)
    np.testing.assert_allclose(chunk["box_matrix"], box_matrix, atol=1e-5)
    np.testing.assert_allclose(chunk["time_ps"], time_ps, atol=1e-5)
    np.testing.assert_allclose(chunk["velocities"], velocities, atol=1e-5)
