from __future__ import annotations

from pathlib import Path

import numpy as np

import warp_md as wmd

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "h5md"


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


def test_open_h5md_matches_gromacs_xtc_fixture():
    system = wmd.System.from_gro(str(_fixture("spc2.gro")))
    h5md = wmd.Trajectory.open_h5md(str(_fixture("spc2-traj.h5md")), system)
    xtc = wmd.Trajectory.open_xtc(str(_fixture("spc2-traj.xtc")), system)

    assert h5md.n_atoms() == 6
    assert h5md.count_frames() == 2
    assert xtc.count_frames() == 2

    h5md_chunk = h5md.read_chunk(include_time=True, include_velocities=True)
    xtc_chunk = xtc.read_chunk(include_time=True)

    assert h5md_chunk["frames"] == 2
    np.testing.assert_allclose(h5md_chunk["coords"], xtc_chunk["coords"], atol=1e-4)
    np.testing.assert_allclose(h5md_chunk["box"], xtc_chunk["box"], atol=1e-4)
    np.testing.assert_allclose(h5md_chunk["time_ps"], xtc_chunk["time_ps"], atol=1e-4)
    assert h5md_chunk["velocities"] is not None
    assert h5md_chunk["velocities"].shape == (2, 6, 3)

    h5md.reset()
    again = h5md.read_chunk(include_time=True)
    np.testing.assert_allclose(again["coords"], h5md_chunk["coords"], atol=1e-4)


def test_h5md_writer_roundtrip_from_python(tmp_path: Path):
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

    path = tmp_path / "roundtrip.h5md"
    writer = wmd.TrajectoryWriter.open(str(path), "h5md", coords.shape[1], None)
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
    traj = wmd.Trajectory.open_h5md(str(path), system)
    chunk = traj.read_chunk(include_time=True, include_velocities=True, include_forces=True)

    np.testing.assert_allclose(chunk["coords"], coords, atol=1e-5)
    np.testing.assert_allclose(chunk["box"], box, atol=1e-5)
    np.testing.assert_allclose(chunk["time_ps"], time_ps, atol=1e-5)
    np.testing.assert_allclose(chunk["velocities"], velocities, atol=1e-5)
    np.testing.assert_allclose(chunk["forces"], forces, atol=1e-5)
