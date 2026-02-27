# Usage:
# from warp_md.analysis.velocity import get_velocity
# v = get_velocity(traj, system, mask="name CA", frame_indices=[0, 5])

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._chunk_io import read_chunk_fields


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_indices(system, mask: Optional[Union[str, Sequence[int]]]) -> np.ndarray:
    if mask is None or mask in ("", "*", "all"):
        sel = system.select(_all_resid_mask(system))
        return np.asarray(list(sel.indices), dtype=np.int64)
    if isinstance(mask, str):
        sel = system.select(mask)
        return np.asarray(list(sel.indices), dtype=np.int64)
    return np.asarray([int(i) for i in mask], dtype=np.int64)


def _read_all(traj, chunk_frames: Optional[int]):
    max_chunk = chunk_frames or 128
    coords_list = []
    time_list = []
    chunk = read_chunk_fields(traj, max_chunk, include_time=True)
    if chunk is None:
        return None, None
    while chunk is not None:
        coords_list.append(np.asarray(chunk["coords"], dtype=np.float64))
        time = chunk.get("time")
        if time is None:
            time = chunk.get("time_ps")
        if time is not None:
            time_list.append(np.asarray(time, dtype=np.float64))
        chunk = read_chunk_fields(traj, max_chunk, include_time=True)
    coords = np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))
    times = np.concatenate(time_list, axis=0) if time_list else None
    return coords, times


def get_velocity(
    traj,
    system,
    mask: Optional[Union[str, Sequence[int]]] = None,
    frame_indices: Optional[Sequence[int]] = None,
    length_scale: float = 1.0,
    time_scale: float = 1.0,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Finite-difference velocities from coordinates (no stored velocities)."""
    coords, times = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0, 0, 3), dtype=np.float32)
    coords = coords * float(length_scale)

    n_frames = coords.shape[0]
    if frame_indices is not None:
        select = []
        for i in frame_indices:
            j = int(i)
            if j < 0:
                j = n_frames + j
            if 0 <= j < n_frames:
                select.append(j)
        coords = coords[select]
        if times is not None:
            times = times[select]
        n_frames = coords.shape[0]

    idx = _selection_indices(system, mask)
    sel = coords[:, idx, :]
    if n_frames == 0:
        return np.empty((0, sel.shape[1], 3), dtype=np.float32)

    vel = np.zeros_like(sel, dtype=np.float64)
    prev = sel[0]
    prev_time = times[0] if times is not None and times.size > 0 else None
    for f in range(n_frames):
        cur = sel[f]
        if f == 0:
            vel[f] = 0.0
        else:
            if times is not None and times.size > f and prev_time is not None:
                dt = (times[f] - prev_time) * float(time_scale)
                if not np.isfinite(dt) or dt == 0.0:
                    dt = 1.0 * float(time_scale)
            else:
                dt = 1.0 * float(time_scale)
            vel[f] = (cur - prev) / dt
        prev = cur
        if times is not None and times.size > f:
            prev_time = times[f]
    return vel.astype(np.float32)


__all__ = ["get_velocity"]
