from __future__ import annotations

from typing import Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np

from ._chunk_io import read_chunk_fields


FrameItem = Tuple[int, np.ndarray, Optional[np.ndarray], Optional[float]]


def normalize_frame_indices(frame_indices: Optional[Sequence[int]], n_frames: int) -> Optional[Sequence[int]]:
    if frame_indices is None:
        return None
    out = []
    for raw in frame_indices:
        idx = int(raw)
        if idx < 0:
            idx = n_frames + idx
        if 0 <= idx < n_frames:
            out.append(idx)
    return out


def infer_n_atoms(system, traj=None) -> int:
    if hasattr(system, "n_atoms"):
        try:
            n = int(system.n_atoms())
            if n >= 0:
                return n
        except Exception:
            pass
    if traj is not None and hasattr(traj, "n_atoms"):
        try:
            n = int(traj.n_atoms())
            if n >= 0:
                return n
        except Exception:
            pass
    if hasattr(system, "atom_table"):
        try:
            atoms = system.atom_table()
            for key in ("name", "resid", "mass", "resname", "chain_id"):
                values = atoms.get(key)
                if values is not None:
                    return int(len(values))
        except Exception:
            pass
    raise ValueError("unable to infer atom count from system/trajectory")


def _read_all_coords_fast_into(traj, chunk_frames: int) -> Optional[np.ndarray]:
    if not hasattr(traj, "read_chunk_into"):
        return None
    try:
        n_atoms = int(traj.n_atoms())
    except Exception:
        return None
    if n_atoms <= 0:
        return np.empty((0, 0, 3), dtype=np.float64)

    coords_buf = np.empty((chunk_frames, n_atoms, 3), dtype=np.float32)
    parts = []
    while True:
        try:
            read = int(traj.read_chunk_into(coords_buf, None, None, max_frames=chunk_frames))
        except (AttributeError, TypeError):
            return None
        if read <= 0:
            break
        parts.append(coords_buf[:read].astype(np.float64, copy=True))
    if not parts:
        return np.empty((0, n_atoms, 3), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def _iter_frames_coords_only_fast(
    traj,
    chunk_frames: int,
) -> Optional[Iterator[FrameItem]]:
    if not hasattr(traj, "read_chunk_into"):
        return None
    try:
        n_atoms = int(traj.n_atoms())
    except Exception:
        return None
    if n_atoms <= 0:
        return iter(())
    coords_buf = np.empty((chunk_frames, n_atoms, 3), dtype=np.float32)

    def _generator() -> Iterator[FrameItem]:
        global_idx = 0
        while True:
            read = int(traj.read_chunk_into(coords_buf, None, None, max_frames=chunk_frames))
            if read <= 0:
                break
            coords = coords_buf[:read].astype(np.float64, copy=False)
            for i in range(coords.shape[0]):
                yield global_idx, coords[i], None, None
                global_idx += 1

    return _generator()


def _read_all_frames(
    traj,
    chunk_frames: int,
    include_box: bool,
    include_time: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if not include_box and not include_time:
        fast_coords = _read_all_coords_fast_into(traj, chunk_frames)
        if fast_coords is not None:
            return fast_coords, None, None

    coords_chunks = []
    box_chunks = []
    time_chunks = []
    chunk = read_chunk_fields(
        traj,
        chunk_frames,
        include_box=include_box,
        include_time=include_time,
    )
    while chunk is not None:
        coords_chunks.append(np.asarray(chunk["coords"], dtype=np.float64))
        if include_box:
            box = chunk.get("box")
            if box is not None:
                box_chunks.append(np.asarray(box, dtype=np.float64))
        if include_time:
            t = chunk.get("time_ps")
            if t is None:
                t = chunk.get("time")
            if t is not None:
                time_chunks.append(np.asarray(t, dtype=np.float64))
        chunk = read_chunk_fields(
            traj,
            chunk_frames,
            include_box=include_box,
            include_time=include_time,
        )
    if not coords_chunks:
        return np.empty((0, 0, 3), dtype=np.float64), None, None
    coords = np.concatenate(coords_chunks, axis=0)
    box = np.concatenate(box_chunks, axis=0) if box_chunks else None
    time = np.concatenate(time_chunks, axis=0) if time_chunks else None
    return coords, box, time


def iter_frames(
    traj,
    chunk_frames: Optional[int] = None,
    frame_indices: Optional[Sequence[int]] = None,
    include_box: bool = False,
    include_time: bool = False,
) -> Iterator[FrameItem]:
    """Yield trajectory frames as (global_index, coords, box_row, time_scalar)."""
    max_chunk = max(1, int(chunk_frames or 128))

    if frame_indices is not None:
        coords, box, time = _read_all_frames(traj, max_chunk, include_box=include_box, include_time=include_time)
        selected = normalize_frame_indices(frame_indices, coords.shape[0])
        if selected is None:
            return
        for src_idx in selected:
            b = box[src_idx] if box is not None else None
            t = float(time[src_idx]) if time is not None else None
            yield int(src_idx), coords[src_idx], b, t
        return

    if not include_box and not include_time:
        fast_iter = _iter_frames_coords_only_fast(traj, max_chunk)
        if fast_iter is not None:
            yield from fast_iter
            return

    global_idx = 0
    chunk = read_chunk_fields(
        traj,
        max_chunk,
        include_box=include_box,
        include_time=include_time,
    )
    while chunk is not None:
        coords = np.asarray(chunk["coords"], dtype=np.float64)
        box = np.asarray(chunk["box"], dtype=np.float64) if include_box and chunk.get("box") is not None else None
        times = None
        if include_time:
            t = chunk.get("time_ps")
            if t is None:
                t = chunk.get("time")
            if t is not None:
                times = np.asarray(t, dtype=np.float64)
        for i in range(coords.shape[0]):
            b = box[i] if box is not None else None
            t = float(times[i]) if times is not None else None
            yield global_idx, coords[i], b, t
            global_idx += 1
        chunk = read_chunk_fields(
            traj,
            max_chunk,
            include_box=include_box,
            include_time=include_time,
        )
