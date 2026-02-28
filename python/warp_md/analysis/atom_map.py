# Usage:
# from warp_md.analysis.atom_map import atom_map
# mask, rmsd = atom_map(traj, system, ref=0, rmsfit=True)

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._chunk_io import read_chunk_fields
from .rmsd import _kabsch_rmsd, _read_all, _selection_indices

RefLike = Union[int, np.ndarray, object]


def atom_map(
    traj,
    system,
    ref: RefLike,
    rmsfit: bool = False,
    mask: str = "",
    chunk_frames: Optional[int] = None,
):
    """Nearest-neighbor atom mapping between traj and reference.

    Returns (mask_string, rmsd_array if rmsfit else empty array).
    """
    coords, _box = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return "", np.empty((0,), dtype=np.float32)

    ref_coords = _resolve_ref(ref, coords)
    idx = _selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")

    target = coords[0][idx]
    ref_sel = ref_coords[idx]
    if rmsfit:
        # align target to reference before mapping
        target = _align_coords(target, ref_sel)

    mapping = _nearest_mapping(target, ref_sel)
    mask_str = "@" + ",".join(str(int(i) + 1) for i in mapping)

    if not rmsfit:
        return mask_str, np.empty((0,), dtype=np.float32)

    # compute RMSD for each frame with mapping applied
    out = np.zeros(coords.shape[0], dtype=np.float64)
    ref_mapped = ref_sel[mapping]
    for f in range(coords.shape[0]):
        cur = coords[f][idx][mapping]
        out[f] = _kabsch_rmsd(cur, ref_mapped)
    return mask_str, out.astype(np.float32)


def _resolve_ref(ref: RefLike, coords: np.ndarray) -> np.ndarray:
    if isinstance(ref, int):
        idx = ref
        if idx < 0:
            idx = coords.shape[0] + idx
        if idx < 0 or idx >= coords.shape[0]:
            raise ValueError("ref index out of range")
        return coords[idx]
    if isinstance(ref, np.ndarray):
        if ref.ndim != 2 or ref.shape[1] != 3:
            raise ValueError("ref coords must be (n_atoms, 3)")
        return ref.astype(np.float64)
    if hasattr(ref, "read_chunk"):
        chunk = read_chunk_fields(ref, 1)
        if chunk is None:
            raise ValueError("reference has no frames")
        coords_ref = np.asarray(chunk["coords"], dtype=np.float64)
        if coords_ref.shape[0] == 0:
            raise ValueError("reference has no frames")
        return coords_ref[0]
    raise ValueError("unsupported ref type")


def _align_coords(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # Kabsch align target to ref (no weights)
    if target.shape != ref.shape:
        raise ValueError("target/ref shape mismatch")
    cx = target.mean(axis=0)
    cy = ref.mean(axis=0)
    x0 = target - cx
    y0 = ref - cy
    h = x0.T @ y0
    u, _s, vt = np.linalg.svd(h, full_matrices=False)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return x0 @ r.T + cy


def _nearest_mapping(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # map each target atom to nearest ref atom
    mapping = np.zeros(target.shape[0], dtype=np.int64)
    for i in range(target.shape[0]):
        diff = ref - target[i]
        dist2 = np.sum(diff * diff, axis=1)
        mapping[i] = int(np.argmin(dist2))
    return mapping


__all__ = ["atom_map"]
