from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mdtraj = pytest.importorskip("mdtraj")
traj_py = pytest.importorskip("warp_md.traj_py")

TRAJ_WRITE_FORMATS = ("dcd", "xtc", "trr", "gro")
TRAJ_READ_FORMATS = ("dcd", "xtc", "trr", "gro", "pdb")


def _write_topology(path: Path) -> None:
    path.write_text(
        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CB  GLY A   1       1.000   1.000   1.000  1.00  0.00           C\n"
        "END\n",
        encoding="ascii",
    )


@pytest.fixture
def compat_fixture(tmp_path: Path):
    topology_path = tmp_path / "topology.pdb"
    _write_topology(topology_path)

    coords_nm = np.array(
        [
            [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]],
            [[0.15, 0.25, 0.35], [0.45, 0.55, 0.65]],
        ],
        dtype=np.float32,
    )
    lengths_nm = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]], dtype=np.float32)
    angles_deg = np.array([[90.0, 90.0, 90.0], [90.0, 90.0, 90.0]], dtype=np.float32)
    time_ps = np.array([1.5, 2.5], dtype=np.float32)

    top = mdtraj.load(str(topology_path)).topology
    mdtraj_traj = mdtraj.Trajectory(
        xyz=coords_nm.copy(),
        topology=top,
        time=time_ps.copy(),
        unitcell_lengths=lengths_nm.copy(),
        unitcell_angles=angles_deg.copy(),
    )
    system = traj_py.PySystem.from_file(str(topology_path))

    coords_angstrom = coords_nm * 10.0
    box_matrix_angstrom = np.zeros((2, 3, 3), dtype=np.float32)
    box_matrix_angstrom[:, 0, 0] = lengths_nm[:, 0] * 10.0
    box_matrix_angstrom[:, 1, 1] = lengths_nm[:, 1] * 10.0
    box_matrix_angstrom[:, 2, 2] = lengths_nm[:, 2] * 10.0

    return {
        "tmp_path": tmp_path,
        "topology_path": topology_path,
        "coords_nm": coords_nm,
        "coords_angstrom": coords_angstrom,
        "lengths_nm": lengths_nm,
        "angles_deg": angles_deg,
        "time_ps": time_ps,
        "box_matrix_angstrom": box_matrix_angstrom,
        "mdtraj_traj": mdtraj_traj,
        "system": system,
    }


def _open_warp_traj(fmt: str, path: Path, system):
    return {
        "dcd": lambda: traj_py.PyTrajectory.open_dcd(str(path), system, None),
        "xtc": lambda: traj_py.PyTrajectory.open_xtc(str(path), system),
        "trr": lambda: traj_py.PyTrajectory.open_trr(str(path), system),
        "gro": lambda: traj_py.PyTrajectory.open_gro(str(path), system),
        "pdb": lambda: traj_py.PyTrajectory.open_pdb(str(path), system),
    }[fmt]()


def _load_mdtraj(fmt: str, path: Path, topology_path: Path):
    if fmt in {"dcd", "xtc", "trr"}:
        return mdtraj.load(str(path), top=str(topology_path))
    return mdtraj.load(str(path))


@pytest.mark.parametrize("fmt", TRAJ_WRITE_FORMATS)
def test_warp_traj_writer_round_trips_with_mdtraj(compat_fixture, fmt: str) -> None:
    path = compat_fixture["tmp_path"] / f"warp_out.{fmt}"
    writer = traj_py.PyTrajectoryWriter.open(str(path), fmt, 2, 2)
    try:
        for i in range(2):
            writer.write_frame(
                compat_fixture["coords_angstrom"][i],
                box_lengths=compat_fixture["box_matrix_angstrom"][i].diagonal(),
                step=i,
                time_ps=float(compat_fixture["time_ps"][i]),
            )
    finally:
        writer.flush()

    traj = _load_mdtraj(fmt, path, compat_fixture["topology_path"])
    assert traj.xyz.shape == (2, 2, 3)
    assert np.allclose(traj.xyz, compat_fixture["coords_nm"], atol=5e-4)
    assert np.allclose(traj.unitcell_lengths, compat_fixture["lengths_nm"], atol=5e-4)
    assert np.allclose(traj.unitcell_angles, compat_fixture["angles_deg"], atol=5e-3)
    expected_time = np.array([0.0, 1.0], dtype=np.float32) if fmt == "dcd" else compat_fixture["time_ps"]
    assert np.allclose(traj.time, expected_time, atol=5e-4)


@pytest.mark.parametrize("fmt", TRAJ_READ_FORMATS)
def test_mdtraj_writers_round_trip_into_warp_readers(compat_fixture, fmt: str) -> None:
    path = compat_fixture["tmp_path"] / f"mdtraj_out.{fmt}"
    getattr(compat_fixture["mdtraj_traj"], f"save_{fmt}")(str(path))

    traj = _open_warp_traj(fmt, path, compat_fixture["system"])
    chunk = traj.read_chunk(include_box=True, include_time=(fmt in {"xtc", "trr", "gro"}))

    assert chunk["frames"] == 2
    # warp-md's Python trajectory readers normalize all overlapping formats to Angstrom.
    assert np.allclose(chunk["coords"], compat_fixture["coords_angstrom"], atol=5e-4)

    expected_box = compat_fixture["box_matrix_angstrom"]
    if fmt == "pdb":
        # MDTraj writes one CRYST1 header for multi-model PDB output, so rereads repeat frame 0.
        expected_box = np.repeat(expected_box[:1], 2, axis=0)
    assert np.allclose(chunk["box_matrix"], expected_box, atol=5e-4)

    if fmt in {"xtc", "trr", "gro"}:
        assert np.allclose(chunk["time_ps"], compat_fixture["time_ps"], atol=5e-4)
    else:
        assert chunk["time_ps"] is None


def test_warp_pdb_structure_writer_round_trips_with_mdtraj(compat_fixture) -> None:
    path = compat_fixture["tmp_path"] / "warp_structure_out.pdb"
    writer = traj_py.PyStructureWriter.open(str(compat_fixture["topology_path"]), 2, "pdb")
    writer.write_structure(
        str(path),
        compat_fixture["coords_angstrom"][0],
        box_lengths=compat_fixture["box_matrix_angstrom"][0].diagonal(),
        frame_index=0,
        time_ps=float(compat_fixture["time_ps"][0]),
    )

    traj = mdtraj.load(str(path))
    assert traj.xyz.shape == (1, 2, 3)
    assert np.allclose(traj.xyz[0], compat_fixture["coords_nm"][0], atol=5e-4)
    assert np.allclose(traj.unitcell_lengths[0], compat_fixture["lengths_nm"][0], atol=5e-4)
    assert np.allclose(traj.unitcell_angles[0], compat_fixture["angles_deg"][0], atol=5e-3)
