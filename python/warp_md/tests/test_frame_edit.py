from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import warp_md as wmd
from warp_md import frame_edit


def _args(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "topology": str(tmp_path / "min.pdb"),
        "traj": str(tmp_path / "eq_npt.xtc"),
        "out": str(tmp_path / "md_new.xtc"),
        "begin": 0,
        "end": None,
        "step": 1,
        "index": None,
        "topology_format": "pdb",
        "traj_format": None,
        "traj_length_scale": None,
        "chunk_frames": 2,
    }
    values.update(overrides)
    return Namespace(**values)


def _write_pdb(path: Path) -> None:
    path.write_text(
        "HETATM    1 CA   GLY A   7      10.000  11.000  12.000  0.50 22.00      TEST C  \n"
        "ATOM      2 CB   GLY A   7      13.000  14.000  15.000  1.00 11.00      TEST C  \n"
        "END\n",
        encoding="ascii",
    )


def _write_xtc(path: Path, frames: list[np.ndarray], times: list[float] | None = None) -> None:
    writer = wmd.TrajectoryWriter.open(str(path), "xtc", 2, None)
    try:
        for index, coords in enumerate(frames):
            writer.write_frame(
                np.asarray(coords, dtype=np.float32),
                box_lengths=np.array([20.0, 20.0, 20.0], dtype=np.float32),
                step=index,
                time_ps=None if times is None else float(times[index]),
            )
    finally:
        writer.flush()


def _write_trr(
    path: Path,
    frames: list[np.ndarray],
    times: list[float],
    velocities: list[np.ndarray],
    forces: list[np.ndarray],
    lambdas: list[float],
) -> None:
    writer = wmd.TrajectoryWriter.open(str(path), "trr", 2, None)
    try:
        for index, coords in enumerate(frames):
            writer.write_frame(
                np.asarray(coords, dtype=np.float32),
                box_lengths=np.array([20.0, 20.0, 20.0], dtype=np.float32),
                step=index,
                time_ps=float(times[index]),
                velocities=np.asarray(velocities[index], dtype=np.float32),
                forces=np.asarray(forces[index], dtype=np.float32),
                lambda_value=float(lambdas[index]),
            )
    finally:
        writer.flush()


def _open_chunk(
    topology: Path,
    traj_path: Path,
    *,
    include_time: bool = False,
    include_velocities: bool = False,
    include_forces: bool = False,
    include_lambda: bool = False,
):
    system = wmd.System.from_pdb(str(topology))
    suffix = traj_path.suffix.lower()
    if suffix == ".xtc":
        traj = wmd.Trajectory.open_xtc(str(traj_path), system)
    elif suffix == ".trr":
        traj = wmd.Trajectory.open_trr(str(traj_path), system)
    else:
        raise AssertionError(f"unsupported test trajectory suffix {suffix}")
    return traj.read_chunk(
        include_time=include_time,
        include_velocities=include_velocities,
        include_forces=include_forces,
        include_lambda=include_lambda,
    )


def test_run_frame_edit_writes_structure_series(tmp_path: Path) -> None:
    topology = tmp_path / "min.pdb"
    traj = tmp_path / "eq_npt.xtc"
    _write_pdb(topology)
    frames = [
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
        np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]], dtype=np.float32),
        np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]], dtype=np.float32),
    ]
    _write_xtc(traj, frames)

    exit_code, payload = frame_edit.run_frame_edit(
        _args(
            tmp_path,
            out=str(tmp_path / "frames.pdb"),
            begin=1,
            end=4,
            step=2,
        )
    )

    assert exit_code == 0
    assert payload["written_frames"] == 2
    assert payload["output_mode"] == "structure_series"
    assert payload["outputs"] == [
        str(tmp_path / "frames_1.pdb"),
        str(tmp_path / "frames_3.pdb"),
    ]
    assert (tmp_path / "frames_1.pdb").exists()
    assert (tmp_path / "frames_3.pdb").exists()


def test_run_frame_edit_writes_single_frame_to_trajectory(tmp_path: Path) -> None:
    topology = tmp_path / "min.pdb"
    traj = tmp_path / "eq_npt.xtc"
    _write_pdb(topology)
    frames = [
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
        np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]], dtype=np.float32),
    ]
    _write_xtc(traj, frames, times=[0.5, 1.5, 2.5])

    out_path = tmp_path / "single.xtc"
    exit_code, payload = frame_edit.run_frame_edit(_args(tmp_path, out=str(out_path), index=2))

    assert exit_code == 0
    assert payload["selection"] == {"mode": "single", "index": 2}
    assert payload["outputs"] == [str(out_path)]
    assert payload["trajectory"]["total_frames"] is None
    chunk = _open_chunk(topology, out_path, include_time=True)
    assert chunk["frames"] == 1
    assert np.allclose(chunk["coords"][0], frames[2])
    assert np.allclose(chunk["time_ps"], [2.5])


def test_run_frame_edit_open_ended_range_counts(tmp_path: Path) -> None:
    topology = tmp_path / "min.pdb"
    traj = tmp_path / "eq_npt.xtc"
    _write_pdb(topology)
    frames = [
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
        np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]], dtype=np.float32),
    ]
    _write_xtc(traj, frames)

    out_path = tmp_path / "subset.xtc"
    exit_code, payload = frame_edit.run_frame_edit(
        _args(tmp_path, out=str(out_path), begin=0, end=None, step=2)
    )

    assert exit_code == 0
    assert payload["written_frames"] == 2
    assert payload["trajectory"]["total_frames"] == 3
    chunk = _open_chunk(topology, out_path)
    assert chunk["frames"] == 2
    assert np.allclose(chunk["coords"][0], frames[0])
    assert np.allclose(chunk["coords"][1], frames[2])


def test_run_frame_edit_single_index_reports_out_of_range(tmp_path: Path) -> None:
    topology = tmp_path / "min.pdb"
    traj = tmp_path / "eq_npt.xtc"
    _write_pdb(topology)
    _write_xtc(
        traj,
        [
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
        ],
    )

    with pytest.raises(ValueError, match="frame index 5 is out of range"):
        frame_edit.run_frame_edit(_args(tmp_path, index=5))


def test_run_frame_edit_reports_missing_topology_path(tmp_path: Path) -> None:
    traj = tmp_path / "eq_npt.xtc"
    _write_xtc(
        traj,
        [np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)],
    )

    with pytest.raises(FileNotFoundError, match="topology file not found"):
        frame_edit.run_frame_edit(
            _args(tmp_path, topology=str(tmp_path / "missing.pdb"), traj=str(traj))
        )


def test_run_frame_edit_reports_missing_trajectory_path(tmp_path: Path) -> None:
    topology = tmp_path / "min.pdb"
    _write_pdb(topology)

    with pytest.raises(FileNotFoundError, match="trajectory file not found"):
        frame_edit.run_frame_edit(
            _args(tmp_path, topology=str(topology), traj=str(tmp_path / "missing.xtc"))
        )


def test_run_frame_edit_reports_missing_output_directory(tmp_path: Path) -> None:
    topology = tmp_path / "min.pdb"
    traj = tmp_path / "eq_npt.xtc"
    _write_pdb(topology)
    _write_xtc(
        traj,
        [np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)],
    )

    with pytest.raises(FileNotFoundError, match="output directory not found"):
        frame_edit.run_frame_edit(
            _args(
                tmp_path,
                topology=str(topology),
                traj=str(traj),
                out=str(tmp_path / "missing-dir" / "out.xtc"),
            )
        )


def test_run_frame_edit_writes_native_pdb_without_mdanalysis(tmp_path: Path) -> None:
    topology = tmp_path / "min.pdb"
    traj = tmp_path / "eq_npt.xtc"
    _write_pdb(topology)
    _write_xtc(
        traj,
        [np.array([[12.0, 13.5, 14.25], [15.0, 16.5, 17.25]], dtype=np.float32)],
        times=[0.75],
    )

    out_path = tmp_path / "frame.pdb"
    exit_code, payload = frame_edit.run_frame_edit(_args(tmp_path, out=str(out_path), index=0))

    assert exit_code == 0
    assert payload["outputs"] == [str(out_path)]
    text = out_path.read_text(encoding="ascii")
    assert "HEADER    GENERATED BY WARP-MD FRAMES" in text
    assert "TITLE     warp-md frame 0 time_ps=0.75" in text
    assert "REMARK     Generated by warp-md frames" in text
    assert "HETATM    1 CA   GLY A   7" in text
    assert "ATOM      2 CB   GLY A   7" in text
    assert text.rstrip().endswith("END")


def test_run_frame_edit_preserves_time_ps_for_trajectory_output(tmp_path: Path) -> None:
    topology = tmp_path / "min.pdb"
    traj = tmp_path / "eq_npt.xtc"
    _write_pdb(topology)
    frames = [
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
        np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]], dtype=np.float32),
    ]
    _write_xtc(traj, frames, times=[1.25, 2.5, 3.75])

    out_path = tmp_path / "subset.xtc"
    exit_code, payload = frame_edit.run_frame_edit(
        _args(tmp_path, out=str(out_path), begin=0, end=3, step=2)
    )

    assert exit_code == 0
    assert payload["written_frames"] == 2
    chunk = _open_chunk(topology, out_path, include_time=True)
    assert np.allclose(chunk["time_ps"], [1.25, 3.75])


def test_run_frame_edit_preserves_trr_extras_for_trr_output(tmp_path: Path) -> None:
    topology = tmp_path / "min.pdb"
    traj = tmp_path / "eq_npt.trr"
    _write_pdb(topology)
    frames = [
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
    ]
    velocities = [
        np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype=np.float32),
    ]
    forces = [
        np.array([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32),
        np.array([[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]], dtype=np.float32),
    ]
    _write_trr(traj, frames, [1.0, 2.0], velocities, forces, [0.25, 0.5])

    out_path = tmp_path / "subset.trr"
    exit_code, payload = frame_edit.run_frame_edit(
        _args(
            tmp_path,
            traj=str(traj),
            traj_format="trr",
            out=str(out_path),
            begin=0,
            end=2,
            step=1,
        )
    )

    assert exit_code == 0
    assert payload["written_frames"] == 2
    chunk = _open_chunk(
        topology,
        out_path,
        include_time=True,
        include_velocities=True,
        include_forces=True,
        include_lambda=True,
    )
    assert np.allclose(chunk["time_ps"], [1.0, 2.0])
    assert np.allclose(chunk["lambda_value"], [0.25, 0.5])
    assert np.allclose(chunk["velocities"][0], velocities[0])
    assert np.allclose(chunk["velocities"][1], velocities[1])
    assert np.allclose(chunk["forces"][0], forces[0])
    assert np.allclose(chunk["forces"][1], forces[1])
