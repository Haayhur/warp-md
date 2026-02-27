# Usage:
# from warp_md.analysis.watershell import watershell
# counts = watershell(traj, system, solute_mask='!:WAT', solvent_mask=':WAT', lower=3.4, upper=5.0)

from __future__ import annotations

from typing import Optional, Sequence

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
    box_list = []
    chunk = read_chunk_fields(traj, max_chunk, include_box=True)
    if chunk is None:
        return None, None
    while chunk is not None:
        coords_list.append(np.asarray(chunk["coords"], dtype=np.float64))
        box = chunk.get("box")
        if box is not None:
            box_list.append(np.asarray(box, dtype=np.float64))
        chunk = read_chunk_fields(traj, max_chunk, include_box=True)
    coords = np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))
    boxes = np.concatenate(box_list, axis=0) if box_list else None
    return coords, boxes


def _apply_frame_indices(coords, boxes, frame_indices: Optional[Sequence[int]]):
    if frame_indices is None:
        return coords, boxes
    n_frames = coords.shape[0]
    select = []
    for i in frame_indices:
        j = int(i)
        if j < 0:
            j = n_frames + j
        if 0 <= j < n_frames:
            select.append(j)
    coords = coords[select]
    if boxes is not None:
        boxes = boxes[select]
    return coords, boxes


def _min_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return delta - np.round(delta / box) * box


def watershell(
    traj,
    system,
    solute_mask: str,
    solvent_mask: str = ":WAT",
    lower: float = 3.4,
    upper: float = 5.0,
    image: bool = True,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Count solvent atoms in a distance shell around solute atoms."""
    if not solute_mask:
        raise ValueError("solute_mask is required")
    if lower < 0.0 or upper <= 0.0 or upper <= lower:
        raise ValueError("upper must be greater than lower and both positive")

    coords, boxes = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0,), dtype=np.float32)

    coords, boxes = _apply_frame_indices(coords, boxes, frame_indices)
    if coords.size == 0:
        return np.empty((0,), dtype=np.float32)

    solute_idx = _selection_indices(system, solute_mask)
    solvent_idx = _selection_indices(system, solvent_mask)
    if solute_idx.size == 0 or solvent_idx.size == 0:
        return np.zeros(coords.shape[0], dtype=np.float32)

    counts = np.zeros(coords.shape[0], dtype=np.float64)
    for f in range(coords.shape[0]):
        solute = coords[f, solute_idx, :]
        solvent = coords[f, solvent_idx, :]
        if solute.size == 0 or solvent.size == 0:
            continue
        diffs = solvent[:, None, :] - solute[None, :, :]
        if image and boxes is not None:
            box = boxes[f]
            if box is not None and np.all(box > 0.0):
                diffs = _min_image(diffs, box)
        dists = np.linalg.norm(diffs, axis=2)
        min_dist = np.min(dists, axis=1)
        counts[f] = np.sum((min_dist >= lower) & (min_dist <= upper))

    return counts.astype(np.float32)


__all__ = ["watershell"]
