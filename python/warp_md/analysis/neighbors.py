# Usage:
# from warp_md.analysis.neighbors import search_neighbors
# out = search_neighbors(traj, system, mask="resid 1:10", distance=3.0)

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._chunk_io import read_chunk_fields


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


def search_neighbors(
    traj,
    system,
    mask: str = "",
    distance: float = 3.0,
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "dataset",
    chunk_frames: Optional[int] = None,
):
    """Find neighbors within cutoff (per-frame indices)."""
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return [] if dtype == "dataset" else []

    n_frames = coords.shape[0]
    if frame_indices is not None:
        select = []
        for i in frame_indices:
            j = int(i)
            if j < 0:
                j = n_frames + j
            if 0 <= j < n_frames:
                select.append(j)
        frames = select
    else:
        frames = list(range(n_frames))

    out = []
    cutoff_sq = float(distance) * float(distance)
    for frame_idx in frames:
        system_positions = coords[frame_idx]
        atoms = system.atom_table()
        # use system selection for mask (no PBC); select indices
        sel = system.select(mask)
        indices = np.asarray(list(sel.indices), dtype=np.int64)
        if indices.size == 0:
            out.append({str(frame_idx): np.asarray([], dtype=np.int64)})
            continue
        # neighbors: atoms within cutoff to any selected atom
        sel_pos = system_positions[indices]
        diff = system_positions[:, None, :] - sel_pos[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        within = np.any(dist2 <= cutoff_sq, axis=1)
        neighbors = np.where(within)[0].astype(np.int64)
        out.append({str(frame_idx): neighbors})

    if dtype == "dataset":
        return out
    return out


__all__ = ["search_neighbors"]
