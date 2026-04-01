from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .cli_api import _load_system, _load_trajectory

_STRUCTURE_OUTPUT_EXTENSIONS = {".gro", ".pdb"}
_TRAJECTORY_OUTPUT_EXTENSIONS = {".dcd", ".trr", ".xtc"}
_INPUT_TRAJECTORY_FORMATS = ["dcd", "xtc", "pdb", "pdbqt"]


@dataclass(frozen=True)
class _FrameRecord:
    index: int
    coords: np.ndarray
    box_lengths: np.ndarray | None
    box_matrix: np.ndarray | None


def add_frame_edit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-p",
        "--topology",
        "--pdbfile",
        dest="topology",
        required=True,
        help="Topology file (.pdb or .gro)",
    )
    parser.add_argument(
        "-t",
        "--traj",
        required=True,
        help="Input trajectory file (.dcd, .xtc, .pdb, .pdbqt)",
    )
    parser.add_argument(
        "-o",
        "--out",
        "--outfile",
        dest="out",
        required=True,
        help="Output file (.pdb, .gro, .dcd, .xtc, .trr)",
    )
    parser.add_argument(
        "-b",
        "--begin",
        type=int,
        default=0,
        help="Starting frame index",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        help="Ending frame index (exclusive)",
    )
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        default=1,
        help="Step size between frames",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        help="Single frame index to extract; overrides begin/end/step",
    )
    parser.add_argument(
        "--topology-format",
        choices=["pdb", "gro"],
        help="Override topology format",
    )
    parser.add_argument(
        "--traj-format",
        choices=_INPUT_TRAJECTORY_FORMATS,
        help="Override trajectory format",
    )
    parser.add_argument(
        "--traj-length-scale",
        type=float,
        help="DCD length scale (for example 10.0 for nm->A)",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        help="Frames per read chunk while scanning and writing",
    )


def _infer_format(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".")


def _count_frames(traj: Any, chunk_frames: int | None) -> int:
    if hasattr(traj, "count_frames"):
        return int(traj.count_frames(chunk_frames))
    total = 0
    while True:
        chunk = traj.read_chunk(
            max_frames=chunk_frames,
            include_box=False,
            include_box_matrix=False,
            include_time=False,
        )
        if chunk is None:
            break
        frames = int(chunk.get("frames", 0))
        if frames <= 0:
            break
        total += frames
    traj.reset()
    return total


def _resolve_single_index(index: int) -> tuple[list[int], dict[str, Any]]:
    if index < 0:
        raise ValueError("frame index must be >= 0")
    return [index], {"mode": "single", "index": index}


def _resolve_bounded_range(
    begin: int,
    end: int,
    step: int,
) -> tuple[int, int, dict[str, Any]]:
    if step <= 0:
        raise ValueError("step must be >= 1")
    begin_frame = max(0, begin)
    if begin_frame >= end:
        raise ValueError("begin frame is greater than or equal to end frame")
    return begin_frame, end, {
        "mode": "range",
        "begin": begin_frame,
        "end": end,
        "step": step,
    }


def _resolve_frame_indices(
    *,
    total_frames: int,
    begin: int,
    end: int | None,
    step: int,
    index: int | None,
) -> tuple[list[int], dict[str, Any]]:
    if total_frames <= 0:
        raise ValueError("trajectory has no frames")
    if step <= 0:
        raise ValueError("step must be >= 1")
    if index is not None:
        if index < 0 or index >= total_frames:
            raise ValueError(
                f"frame index {index} is out of range (0 to {total_frames - 1})"
            )
        return _resolve_single_index(index)

    begin_frame = max(0, begin)
    end_frame = total_frames if end is None else min(end, total_frames)
    if begin_frame >= end_frame:
        raise ValueError("begin frame is greater than or equal to end frame")
    frame_indices = list(range(begin_frame, end_frame, step))
    return frame_indices, {
        "mode": "range",
        "begin": begin_frame,
        "end": end_frame,
        "step": step,
    }


def _copy_optional_frame(data: Any, local_index: int) -> np.ndarray | None:
    if data is None:
        return None
    array = np.asarray(data)
    if array.size == 0:
        return None
    return np.array(array[local_index], copy=True)


def _iter_selected_frames(
    traj: Any,
    frame_indices: list[int],
    chunk_frames: int | None,
) -> Iterator[_FrameRecord]:
    if hasattr(traj, "read_frames"):
        payload = traj.read_frames(frame_indices, chunk_frames, True, True, False)
        if payload is None:
            raise ValueError("requested frames could not be read from trajectory")
        coords = np.asarray(payload["coords"], dtype=np.float32)
        source_indices = payload.get("source_indices", frame_indices)
        for local_index, absolute_index in enumerate(source_indices):
            yield _FrameRecord(
                index=int(absolute_index),
                coords=np.array(coords[local_index], copy=True),
                box_lengths=_copy_optional_frame(payload.get("box"), local_index),
                box_matrix=_copy_optional_frame(payload.get("box_matrix"), local_index),
            )
        return

    current_frame = 0
    target_idx = 0
    while target_idx < len(frame_indices):
        chunk = traj.read_chunk(
            max_frames=chunk_frames,
            include_box=True,
            include_box_matrix=True,
            include_time=False,
        )
        if chunk is None:
            break
        frames = int(chunk.get("frames", 0))
        if frames <= 0:
            break
        coords = np.asarray(chunk["coords"], dtype=np.float32)
        chunk_stop = current_frame + frames
        while target_idx < len(frame_indices) and frame_indices[target_idx] < chunk_stop:
            absolute_index = frame_indices[target_idx]
            local_index = absolute_index - current_frame
            yield _FrameRecord(
                index=absolute_index,
                coords=np.array(coords[local_index], copy=True),
                box_lengths=_copy_optional_frame(chunk.get("box"), local_index),
                box_matrix=_copy_optional_frame(chunk.get("box_matrix"), local_index),
            )
            target_idx += 1
        current_frame = chunk_stop
    if target_idx != len(frame_indices):
        raise ValueError("requested frames could not be read from trajectory")


def _read_single_frame(
    traj: Any,
    index: int,
    chunk_frames: int | None,
) -> _FrameRecord:
    if hasattr(traj, "read_frames"):
        payload = traj.read_frames([index], chunk_frames, True, True, False)
        if payload is None:
            raise ValueError(f"frame index {index} is out of range")
        source_indices = [int(value) for value in payload.get("source_indices", [index])]
        if index not in source_indices:
            raise ValueError(f"frame index {index} is out of range")
        local_index = source_indices.index(index)
        coords = np.asarray(payload["coords"], dtype=np.float32)
        return _FrameRecord(
            index=index,
            coords=np.array(coords[local_index], copy=True),
            box_lengths=_copy_optional_frame(payload.get("box"), local_index),
            box_matrix=_copy_optional_frame(payload.get("box_matrix"), local_index),
        )

    current_frame = 0
    while True:
        chunk = traj.read_chunk(
            max_frames=chunk_frames,
            include_box=True,
            include_box_matrix=True,
            include_time=False,
        )
        if chunk is None:
            break
        frames = int(chunk.get("frames", 0))
        if frames <= 0:
            break
        chunk_stop = current_frame + frames
        if index < chunk_stop:
            local_index = index - current_frame
            coords = np.asarray(chunk["coords"], dtype=np.float32)
            return _FrameRecord(
                index=index,
                coords=np.array(coords[local_index], copy=True),
                box_lengths=_copy_optional_frame(chunk.get("box"), local_index),
                box_matrix=_copy_optional_frame(chunk.get("box_matrix"), local_index),
            )
        current_frame = chunk_stop

    raise ValueError(f"frame index {index} is out of range")


def _iter_frame_range(
    traj: Any,
    begin: int,
    end: int,
    step: int,
    chunk_frames: int | None,
) -> Iterator[_FrameRecord]:
    if hasattr(traj, "read_frame_range"):
        payload = traj.read_frame_range(begin, end, step, chunk_frames, True, True, False)
        if payload is None:
            raise ValueError("begin frame is greater than or equal to end frame")
        coords = np.asarray(payload["coords"], dtype=np.float32)
        source_indices = payload.get("source_indices", [])
        if not source_indices:
            raise ValueError("begin frame is greater than or equal to end frame")
        for local_index, absolute_index in enumerate(source_indices):
            yield _FrameRecord(
                index=int(absolute_index),
                coords=np.array(coords[local_index], copy=True),
                box_lengths=_copy_optional_frame(payload.get("box"), local_index),
                box_matrix=_copy_optional_frame(payload.get("box_matrix"), local_index),
            )
        return

    frame_indices = list(range(begin, end, step))
    yield from _iter_selected_frames(traj, frame_indices, chunk_frames)


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 90.0
    cosine = float(np.dot(a, b) / denom)
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def _dimensions_from_box(
    box_lengths: np.ndarray | None,
    box_matrix: np.ndarray | None,
) -> np.ndarray | None:
    if box_matrix is not None:
        matrix = np.asarray(box_matrix, dtype=np.float32)
        if matrix.shape != (3, 3):
            return None
        a_vec, b_vec, c_vec = matrix
        a = float(np.linalg.norm(a_vec))
        b = float(np.linalg.norm(b_vec))
        c = float(np.linalg.norm(c_vec))
        if min(a, b, c) <= 0.0:
            return None
        alpha = _angle_degrees(b_vec, c_vec)
        beta = _angle_degrees(a_vec, c_vec)
        gamma = _angle_degrees(a_vec, b_vec)
        return np.asarray([a, b, c, alpha, beta, gamma], dtype=np.float32)
    if box_lengths is not None:
        lengths = np.asarray(box_lengths, dtype=np.float32)
        if lengths.shape != (3,):
            return None
        if np.any(lengths <= 0.0):
            return None
        return np.asarray(
            [float(lengths[0]), float(lengths[1]), float(lengths[2]), 90.0, 90.0, 90.0],
            dtype=np.float32,
        )
    return None


class _MDAnalysisTrajectorySink:
    def __init__(self, parent: "_MDAnalysisFrameWriter", writer: Any) -> None:
        self._parent = parent
        self._writer = writer

    def write_frame(self, frame: _FrameRecord) -> None:
        self._parent._apply_frame(frame)
        self._writer.write(self._parent._atoms)


class _MDAnalysisFrameWriter:
    def __init__(self, topology_path: str, *, topology_format: str | None, n_atoms: int) -> None:
        try:
            import MDAnalysis as mda
        except ImportError as exc:  # pragma: no cover - import depends on local env
            raise RuntimeError(
                "warp-md frames requires MDAnalysis for writing output files"
            ) from exc

        kwargs = {}
        if topology_format:
            kwargs["topology_format"] = topology_format
        universe = mda.Universe(topology_path, **kwargs)
        atoms = universe.atoms
        if atoms.n_atoms != n_atoms:
            raise ValueError(
                f"topology atom count {atoms.n_atoms} does not match trajectory atom count {n_atoms}"
            )
        self._mda = mda
        self._universe = universe
        self._atoms = atoms

    def _apply_frame(self, frame: _FrameRecord) -> None:
        self._atoms.positions = frame.coords
        dimensions = _dimensions_from_box(frame.box_lengths, frame.box_matrix)
        if dimensions is not None:
            self._universe.dimensions = dimensions

    def write_structure(self, path: str, frame: _FrameRecord) -> None:
        self._apply_frame(frame)
        self._atoms.write(path)

    @contextmanager
    def open_trajectory(self, path: str) -> Iterator[_MDAnalysisTrajectorySink]:
        with self._mda.Writer(path, self._atoms.n_atoms) as writer:
            yield _MDAnalysisTrajectorySink(self, writer)


def _make_writer(
    topology_path: str,
    *,
    topology_format: str | None,
    n_atoms: int,
) -> _MDAnalysisFrameWriter:
    return _MDAnalysisFrameWriter(
        topology_path,
        topology_format=topology_format,
        n_atoms=n_atoms,
    )


def run_frame_edit(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    topology_format = args.topology_format or _infer_format(args.topology)
    traj_format = args.traj_format or _infer_format(args.traj)

    system = _load_system({"path": args.topology, "format": topology_format})
    traj = _load_trajectory(
        {
            "path": args.traj,
            "format": traj_format,
            "length_scale": args.traj_length_scale,
        },
        system,
    )
    total_frames: int | None
    expected_frames: int | None
    if args.index is not None:
        frame_indices, selection = _resolve_single_index(args.index)
        total_frames = None
        expected_frames = 1
        selected_frames = iter([_read_single_frame(traj, args.index, args.chunk_frames)])
    elif args.end is not None:
        begin_frame, end_frame, selection = _resolve_bounded_range(
            args.begin,
            args.end,
            args.step,
        )
        total_frames = None
        expected_frames = len(range(begin_frame, end_frame, args.step))
        frame_indices = None
        selected_frames = _iter_frame_range(
            traj,
            begin_frame,
            end_frame,
            args.step,
            args.chunk_frames,
        )
    else:
        total_frames = _count_frames(traj, args.chunk_frames)
        frame_indices, selection = _resolve_frame_indices(
            total_frames=total_frames,
            begin=args.begin,
            end=args.end,
            step=args.step,
            index=None,
        )
        expected_frames = len(frame_indices)
        traj.reset()
        selected_frames = _iter_selected_frames(traj, frame_indices, args.chunk_frames)

    out_path = Path(args.out)
    ext = out_path.suffix.lower()
    if ext not in _STRUCTURE_OUTPUT_EXTENSIONS and ext not in _TRAJECTORY_OUTPUT_EXTENSIONS:
        raise ValueError(f"unsupported output extension: {ext}")

    writer = _make_writer(
        args.topology,
        topology_format=topology_format,
        n_atoms=int(system.n_atoms()),
    )

    written_frames = 0
    outputs: list[str] = []

    if ext in _TRAJECTORY_OUTPUT_EXTENSIONS:
        with writer.open_trajectory(str(out_path)) as sink:
            for frame in selected_frames:
                sink.write_frame(frame)
                written_frames += 1
        outputs.append(str(out_path))
        output_mode = "trajectory"
    else:
        output_mode = "single_structure" if expected_frames == 1 else "structure_series"
        for frame in selected_frames:
            if output_mode == "single_structure":
                target = out_path
            else:
                target = out_path.with_name(f"{out_path.stem}_{frame.index}{ext}")
            writer.write_structure(str(target), frame)
            outputs.append(str(target))
            written_frames += 1

    if frame_indices is not None and written_frames != len(frame_indices):
        raise ValueError(
            f"expected to write {len(frame_indices)} frames but wrote {written_frames}"
        )

    return 0, {
        "status": "ok",
        "command": "frames",
        "topology": {"path": args.topology, "format": topology_format},
        "trajectory": {
            "path": args.traj,
            "format": traj_format,
            "length_scale": args.traj_length_scale,
            "total_frames": total_frames,
        },
        "selection": selection,
        "written_frames": written_frames,
        "output_mode": output_mode,
        "outputs": outputs,
    }
