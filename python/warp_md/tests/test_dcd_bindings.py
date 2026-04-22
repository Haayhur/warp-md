from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mdtraj = pytest.importorskip("mdtraj")
traj_py = pytest.importorskip("warp_md.traj_py")
if not hasattr(traj_py, "PyTrajectoryWriter"):
    pytest.skip("native trajectory writer binding unavailable", allow_module_level=True)


def _write_pdb(path: Path) -> None:
    path.write_text(
        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CB  GLY A   1       1.000   1.000   1.000  1.00  0.00           C\n"
        "END\n",
        encoding="ascii",
    )


def test_dcd_writer_round_trips_with_mdtraj(tmp_path: Path) -> None:
    topology = tmp_path / "top.pdb"
    output = tmp_path / "frames.dcd"
    _write_pdb(topology)

    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    writer = traj_py.PyTrajectoryWriter.open(str(output), "dcd", 2, 1)
    try:
        writer.write_frame(
            coords,
            box_lengths=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        )
    finally:
        writer.flush()

    traj = mdtraj.load(str(output), top=str(topology))
    assert traj.xyz.shape == (1, 2, 3)
    assert np.allclose(traj.xyz[0], coords / 10.0)
    assert np.allclose(traj.unitcell_lengths, [[1.0, 2.0, 3.0]])
    assert np.allclose(traj.unitcell_angles, [[90.0, 90.0, 90.0]])


def test_dcd_writer_without_box_round_trips_with_mdtraj(tmp_path: Path) -> None:
    topology = tmp_path / "top.pdb"
    output = tmp_path / "frames_no_box.dcd"
    _write_pdb(topology)

    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    writer = traj_py.PyTrajectoryWriter.open(str(output), "dcd", 2, 1)
    try:
        writer.write_frame(coords)
    finally:
        writer.flush()

    traj = mdtraj.load(str(output), top=str(topology))
    assert traj.xyz.shape == (1, 2, 3)
    assert np.allclose(traj.xyz[0], coords / 10.0)
    assert traj.unitcell_lengths is None
    assert traj.unitcell_angles is None
