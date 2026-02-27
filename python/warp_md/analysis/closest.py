# Usage:
# from warp_md.analysis.closest import closest, closest_atom
# idx = closest_atom(system, coords, point=(0,0,0))

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ._chunk_io import read_chunk_fields
from .trajectory import ArrayTrajectory


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_indices(system, mask: str) -> np.ndarray:
    if mask in ("", "*", "all", None):
        sel = system.select(_all_resid_mask(system))
    else:
        sel = system.select(mask)
    return np.asarray(list(sel.indices), dtype=np.int64)


def closest_atom(system, frame: np.ndarray, point=(0.0, 0.0, 0.0), mask: str = "") -> int:
    idx = _selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")
    coords = frame[idx]
    diff = coords - np.asarray(point, dtype=np.float64)[None, :]
    dist2 = np.sum(diff * diff, axis=1)
    return int(idx[int(np.argmin(dist2))])


def closest(
    traj,
    system,
    mask: str = "*",
    solvent_mask: Optional[str] = None,
    n_solvents: int = 10,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
):
    """Keep only closest n_solvents atoms (basic parity)."""
    coords, box, time = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return ArrayTrajectory(coords.astype(np.float32), box=box, time_ps=time)

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
        if box is not None:
            box = box[select]
        if time is not None:
            time = time[select]

    target_idx = _selection_indices(system, mask)
    if solvent_mask:
        solvent_idx = _selection_indices(system, solvent_mask)
    else:
        solvent_idx = _selection_indices(system, mask)

    if target_idx.size == 0 or solvent_idx.size == 0:
        raise ValueError("selection resolved to empty set")

    out_coords = []
    for f in range(coords.shape[0]):
        frame = coords[f]
        target = frame[target_idx]
        solvent = frame[solvent_idx]
        # compute min distance from each solvent atom to any target atom
        diff = solvent[:, None, :] - target[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        min_dist = np.min(dist2, axis=1)
        order = np.argsort(min_dist)
        keep_idx = solvent_idx[order[: max(0, int(n_solvents))]]
        keep = frame[keep_idx]
        out_coords.append(keep)

    out_coords = np.asarray(out_coords, dtype=np.float32)
    return ArrayTrajectory(out_coords, box=None, time_ps=time)


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


__all__ = ["closest", "closest_atom"]
