# Usage:
# from warp_md.analysis.rotation import rotation_matrix
# mats = rotation_matrix(traj, system, mask="name CA", ref=0)

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._chunk_io import read_chunk_fields


RefLike = Union[int, str]


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


def _mass_weights(system, indices: np.ndarray, mass: bool) -> np.ndarray:
    if not mass:
        return np.ones(indices.size, dtype=np.float64)
    atoms = system.atom_table()
    masses = atoms.get("mass", [])
    if not masses:
        return np.ones(indices.size, dtype=np.float64)
    w = np.asarray(masses, dtype=np.float64)
    return w[indices]


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


def _kabsch(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, float]:
    if x.shape != y.shape:
        raise ValueError("selection and reference selection must have same size")
    if x.size == 0:
        return np.eye(3, dtype=np.float64), 0.0
    w = weights.astype(np.float64)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        w = np.ones_like(w)
        wsum = float(np.sum(w))
    w = w / wsum
    cx = np.sum(x * w[:, None], axis=0)
    cy = np.sum(y * w[:, None], axis=0)
    x0 = x - cx
    y0 = y - cy
    h = (x0 * w[:, None]).T @ y0
    u, _s, vt = np.linalg.svd(h, full_matrices=False)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    x_rot = x0 @ r.T
    diff = x_rot - y0
    rmsd = float(np.sqrt((diff * diff).sum() / x.shape[0])) if x.shape[0] > 0 else 0.0
    return r, rmsd


def rotation_matrix(
    traj,
    system,
    mask: str = "",
    ref: RefLike = 0,
    mass: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    with_rmsd: bool = False,
):
    """Compute rotation matrices to align frames to reference.

    Returns (n_frames, 3, 3) or (matrices, rmsd) when with_rmsd=True.
    """
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    n_frames = coords.shape[0]
    if n_frames == 0:
        mats = np.empty((0, 3, 3), dtype=np.float32)
        if with_rmsd:
            return mats, np.empty((0,), dtype=np.float32)
        return mats

    sel_idx = _selection_indices(system, mask)
    if sel_idx.size == 0:
        raise ValueError("selection resolved to empty set")
    weights = _mass_weights(system, sel_idx, mass)

    ref_coords = None
    ref_kind = ref
    if isinstance(ref, str):
        key = ref.strip().lower()
        if key in ("topology", "top", "topo"):
            if hasattr(system, "positions0"):
                ref_coords = system.positions0()
        if ref_coords is None and key in ("frame0", "first", "0", "topology", "top", "topo"):
            ref_kind = 0

    if ref_coords is None:
        if isinstance(ref_kind, int):
            ref_index = ref_kind
            if ref_index < 0:
                ref_index = n_frames + ref_index
            if ref_index < 0 or ref_index >= n_frames:
                raise ValueError("ref index out of range")
            ref_coords = coords[ref_index]
        else:
            ref_coords = coords[0]
    else:
        ref_coords = np.asarray(ref_coords, dtype=np.float64)

    frame_set = None
    if frame_indices is not None:
        frame_set = set()
        for idx in frame_indices:
            i = int(idx)
            if i < 0:
                i = n_frames + i
            if 0 <= i < n_frames:
                frame_set.add(i)

    mats = np.zeros((n_frames, 3, 3), dtype=np.float64)
    rmsd_vals = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        if frame_set is not None and i not in frame_set:
            mats[i] = np.eye(3, dtype=np.float64)
            rmsd_vals[i] = 0.0
            continue
        r, rmsd = _kabsch(coords[i][sel_idx], ref_coords[sel_idx], weights)
        mats[i] = r
        rmsd_vals[i] = rmsd

    mats = mats.astype(np.float32)
    if with_rmsd:
        return mats, rmsd_vals.astype(np.float32)
    return mats


__all__ = ["rotation_matrix"]
