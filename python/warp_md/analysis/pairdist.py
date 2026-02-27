# Usage:
# from warp_md.analysis.pairdist import pairdist
# out = pairdist(traj, system, mask='@CA', delta=0.1)

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


def _apply_frame_indices(coords: np.ndarray, frame_indices: Optional[Sequence[int]]) -> np.ndarray:
    if frame_indices is None:
        return coords
    n_frames = coords.shape[0]
    select = []
    for i in frame_indices:
        j = int(i)
        if j < 0:
            j = n_frames + j
        if 0 <= j < n_frames:
            select.append(j)
    return coords[select]


def _pairwise_distances(sel_a: np.ndarray, sel_b: np.ndarray, same: bool) -> np.ndarray:
    if same:
        n = sel_a.shape[0]
        if n < 2:
            return np.empty((0,), dtype=np.float64)
        i_idx, j_idx = np.triu_indices(n, k=1)
        diffs = sel_a[i_idx] - sel_a[j_idx]
    else:
        diffs = sel_a[:, None, :] - sel_b[None, :, :]
        diffs = diffs.reshape(-1, 3)
    return np.linalg.norm(diffs, axis=1)


def pairdist(
    traj,
    system,
    mask: str = "*",
    mask2: str = "",
    delta: float = 0.1,
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "dict",
    chunk_frames: Optional[int] = None,
):
    """Compute pair distance histogram."""
    if delta <= 0.0:
        raise ValueError("delta must be positive")

    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {"bin_centers": empty, "hist": empty, "n_frames": 0}

    coords = _apply_frame_indices(coords, frame_indices)
    if coords.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {"bin_centers": empty, "hist": empty, "n_frames": 0}

    idx_a = _selection_indices(system, mask)
    idx_b = _selection_indices(system, mask2) if mask2 else idx_a
    same = not mask2 or np.array_equal(idx_a, idx_b)

    if idx_a.size == 0 or idx_b.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {"bin_centers": empty, "hist": empty, "n_frames": int(coords.shape[0])}

    all_dists = []
    for frame in coords:
        sel_a = frame[idx_a]
        sel_b = frame[idx_b]
        dists = _pairwise_distances(sel_a, sel_b, same)
        if dists.size:
            all_dists.append(dists)
    if not all_dists:
        empty = np.empty((0,), dtype=np.float32)
        return {"bin_centers": empty, "hist": empty, "n_frames": int(coords.shape[0])}

    dists = np.concatenate(all_dists, axis=0)
    max_dist = float(np.max(dists))
    edges = np.arange(0.0, max_dist + delta, delta, dtype=np.float64)
    if edges.size < 2:
        edges = np.array([0.0, delta], dtype=np.float64)
    hist, _ = np.histogram(dists, bins=edges)
    centers = (edges[:-1] + edges[1:]) * 0.5

    out = {
        "bin_centers": centers.astype(np.float32),
        "hist": hist.astype(np.float32),
        "n_frames": int(coords.shape[0]),
    }
    if dtype == "dict":
        return out
    return out


__all__ = ["pairdist"]
