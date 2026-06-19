# Usage:
# from warp_md.analysis.neighbors import search_neighbors
# out = search_neighbors(traj, system, mask="resid 1:10", distance=3.0)

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._chunk_io import read_chunk_fields
from ._runtime import (
    load_native_symbol,
    native_inputs,
    native_selection,
    normalize_frame_indices,
    reset_traj,
)


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


def _frame_labels(traj, frame_indices: Optional[Sequence[int]], rows: int, chunk_frames: Optional[int]):
    if frame_indices is None:
        return list(range(rows))
    raw = [int(value) for value in frame_indices]
    if all(value >= 0 for value in raw):
        return raw[:rows]
    if hasattr(traj, "count_frames"):
        try:
            n_frames = int(traj.count_frames(chunk_frames=chunk_frames))
            labels = normalize_frame_indices(raw, n_frames) or []
            if len(labels) == rows:
                return [int(value) for value in labels]
        except Exception:
            pass
    return list(range(rows))


def _format_neighbor_payload(
    payload,
    traj,
    frame_indices: Optional[Sequence[int]],
    dtype: str,
    chunk_frames: Optional[int],
):
    offsets = np.asarray(payload["offsets"], dtype=np.uint64)
    indices = np.asarray(payload["indices"], dtype=np.int64)
    counts = np.asarray(payload["counts"], dtype=np.uint32)
    rows = int(payload["frames"])
    labels = _frame_labels(traj, frame_indices, rows, chunk_frames)
    dtype_key = str(dtype).lower()
    if dtype_key in ("flat", "native"):
        return {
            "offsets": offsets,
            "indices": indices,
            "counts": counts,
            "frame_indices": np.asarray(labels, dtype=np.int64),
        }
    if dtype_key in ("count", "counts"):
        return counts.astype(np.int64, copy=False)
    out = []
    for row in range(rows):
        start = int(offsets[row])
        end = int(offsets[row + 1])
        out.append({str(labels[row]): indices[start:end]})
    return out


def _native_search_neighbors(
    traj,
    system,
    mask: str,
    distance: float,
    frame_indices: Optional[Sequence[int]],
    dtype: str,
    chunk_frames: Optional[int],
):
    plan_cls = load_native_symbol("SearchNeighborListPlan")
    if plan_cls is None:
        return None
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        return None
    if not reset_traj(native_traj):
        raise RuntimeError("failed to reset native trajectory")
    target = native_selection(system, native_system, mask)
    probe = native_selection(system, native_system, "all")
    plan = plan_cls(target, probe, float(distance), pbc="none")
    kwargs = {"chunk_frames": chunk_frames, "device": "auto"}
    if frame_indices is not None:
        kwargs["frame_indices"] = [int(value) for value in frame_indices]
    payload = plan.run(native_traj, native_system, **kwargs)
    return _format_neighbor_payload(payload, traj, frame_indices, dtype, chunk_frames)


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
    if distance < 0.0:
        raise ValueError("distance must be non-negative")

    native_out = _native_search_neighbors(
        traj,
        system,
        mask,
        distance,
        frame_indices,
        dtype,
        chunk_frames,
    )
    if native_out is not None:
        return native_out

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
