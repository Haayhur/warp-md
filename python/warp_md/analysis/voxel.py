# Usage:
# from warp_md.analysis.voxel import count_in_voxel
# occupants = count_in_voxel(traj, system, mask="name O", voxel_cntr=(0,0,0), voxel_size=5)

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ._chunk_io import read_chunk_fields


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


def _read_all(traj, chunk_frames: Optional[int]):
    max_chunk = chunk_frames or 128
    coords_list = []
    chunk = read_chunk_fields(traj, max_chunk)
    if chunk is None:
        return None
    while chunk is not None:
        coords_list.append(np.asarray(chunk["coords"], dtype=np.float64))
        chunk = read_chunk_fields(traj, max_chunk)
    return np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))


def _in_voxel(center: Tuple[float, float, float], xyz: np.ndarray, delta: float) -> np.ndarray:
    return (
        (xyz[:, 0] >= center[0] - delta)
        & (xyz[:, 0] <= center[0] + delta)
        & (xyz[:, 1] >= center[1] - delta)
        & (xyz[:, 1] <= center[1] + delta)
        & (xyz[:, 2] >= center[2] - delta)
        & (xyz[:, 2] <= center[2] + delta)
    )


def count_in_voxel(
    traj,
    system,
    mask: str = "",
    voxel_cntr: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    voxel_size: float = 5.0,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
):
    """Return atom indices in voxel for each frame."""
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return []

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

    idx = _selection_indices(system, mask)
    if idx.size == 0:
        return [[] for _ in range(coords.shape[0])]

    delta = float(voxel_size) / 2.0
    out = []
    for frame in coords:
        sel = frame[idx]
        inside = _in_voxel(voxel_cntr, sel, delta)
        out.append(idx[inside].tolist())
    return out


__all__ = ["count_in_voxel"]
