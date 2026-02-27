# Usage:
# from warp_md.analysis.autoimage import autoimage
# new_traj = autoimage(traj, system, mask="all")

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._chunk_io import read_chunk_fields
from .trajectory import ArrayTrajectory


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_indices(system, mask: Optional[str]) -> np.ndarray:
    if mask in ("", "*", "all", None):
        mask = None
    if mask:
        sel = system.select(mask)
    else:
        sel = system.select(_all_resid_mask(system))
    return np.asarray(list(sel.indices), dtype=np.int64)


def _read_all(traj, chunk_frames: Optional[int]):
    max_chunk = chunk_frames or 128
    coords_list = []
    box_list = []
    time_list = []
    chunk = read_chunk_fields(traj, max_chunk, include_box=True, include_time=True)
    if chunk is None:
        return None, None, None
    while chunk is not None:
        coords_list.append(np.asarray(chunk["coords"], dtype=np.float64))
        box = chunk.get("box")
        if box is not None:
            box_list.append(np.asarray(box, dtype=np.float64))
        time = chunk.get("time") or chunk.get("time_ps")
        if time is not None:
            time_list.append(np.asarray(time, dtype=np.float64))
        chunk = read_chunk_fields(traj, max_chunk, include_box=True, include_time=True)
    coords = np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))
    box = np.concatenate(box_list, axis=0) if box_list else None
    time = np.concatenate(time_list, axis=0) if time_list else None
    return coords, box, time


def autoimage(
    traj,
    system,
    mask: str = "",
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
) -> ArrayTrajectory:
    """Move selected atoms into primary unit cell (orthorhombic only)."""
    coords, box, time = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return ArrayTrajectory(coords.astype(np.float32), box=box, time_ps=time)
    if box is None:
        raise ValueError("autoimage requires box lengths")

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
        box = box[select]
        if time is not None:
            time = time[select]
        n_frames = coords.shape[0]

    idx = _selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")

    out = coords.copy()
    for f in range(n_frames):
        lx, ly, lz = box[f]
        if lx <= 0 or ly <= 0 or lz <= 0:
            continue
        sel = out[f, idx, :]
        sel[:, 0] -= np.floor(sel[:, 0] / lx) * lx
        sel[:, 1] -= np.floor(sel[:, 1] / ly) * ly
        sel[:, 2] -= np.floor(sel[:, 2] / lz) * lz
        out[f, idx, :] = sel

    return ArrayTrajectory(out.astype(np.float32), box=box, time_ps=time)


__all__ = ["autoimage"]
